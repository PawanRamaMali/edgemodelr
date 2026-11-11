// Main bindings file - should compile cleanly

#include <Rcpp.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "llama.h"
#include "llama-grammar.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "r_output_redirect.h"

// End of includes

// Forward declaration for CPU backend registration
extern "C" {
  void ggml_backend_register(ggml_backend_reg_t reg);
  ggml_backend_reg_t ggml_backend_cpu_reg(void);
}

using namespace Rcpp;

// Global variable to control logging
static bool g_logging_enabled = false;

// Global variable to control console output suppression (for CRAN compliance)
bool g_suppress_console_output = true;

// Path to an optional CUDA backend DLL, set by edge_use_cuda_backend_internal()
// Must be set BEFORE the first call to ensure_llama_initialized()
static std::string g_cuda_backend_path;

// Tracks whether the CUDA backend DLL was successfully loaded by ggml_backend_load()
static bool g_cuda_backend_loaded = false;

// Custom log callback to suppress output
void quiet_log_callback(ggml_log_level level, const char * text, void * user_data) {
  // Only output critical errors using R's error system, suppress all other output
  if (g_logging_enabled && level >= GGML_LOG_LEVEL_ERROR) {
    // Use R's warning system instead of direct stderr output
    Rcpp::warning(std::string("llama.cpp error: ") + text);
  }
  // Otherwise, completely suppress output
}

// Thread that R runs on; recorded at package load. Only this thread may
// signal an R error.
static std::thread::id g_r_main_thread;

// ggml abort callback. Without one, GGML_ABORT/GGML_ASSERT end in
// raise(SIGABRT) (see ggml_abort in ggml.c) and take down the R session.
// On R's thread we throw instead: the exception unwinds through ggml_abort
// (C sources are built with -fexceptions) and through llama.cpp, which
// either catches it (e.g. llama_model_load returns an error) or lets it
// reach the BEGIN_RCPP/END_RCPP wrapper in RcppExports.cpp, which turns it
// into an ordinary R error. On ggml worker threads there is no safe way to
// reach R, so we return and ggml_abort falls back to raise(SIGABRT).
static void r_abort_callback(const char * error_message) {
  if (std::this_thread::get_id() == g_r_main_thread) {
    throw std::runtime_error(std::string("llama.cpp internal error: ") + error_message);
  }
}

// Feed the prompt to the model in chunks of at most n_batch tokens.
// llama_decode() asserts n_tokens <= n_batch, so a single call with the whole
// prompt fails for any prompt longer than n_batch (512 at n_ctx = 2048) even
// though it fits in the context. llama_batch_get_one() leaves positions unset,
// so each chunk continues from the previous one; only the last token of the
// final chunk produces logits, which is the one sampling needs.
static bool decode_prompt(llama_context * ctx, std::vector<llama_token> & tokens) {
  const size_t n_batch = std::max<uint32_t>(1, llama_n_batch(ctx));
  for (size_t i = 0; i < tokens.size(); i += n_batch) {
    const size_t n = std::min(n_batch, tokens.size() - i);
    llama_batch chunk = llama_batch_get_one(tokens.data() + i, (int32_t)n);
    if (llama_decode(ctx, chunk)) return false;
  }
  return true;
}

// Each completion starts from an empty KV cache. llama_batch_get_one() leaves
// positions unset, so without this a prompt is appended after everything the
// previous call left in the cache: answers depend on earlier calls, and once
// the cache is full every later call fails.
static void clear_kv_cache(llama_context * ctx) {
  llama_memory_t mem = llama_get_memory(ctx);
  if (mem) {
    llama_memory_clear(mem, true);
  }
}

// Owns a llama_grammar created with llama_grammar_init_impl().
struct GrammarDeleter {
  void operator()(llama_grammar * g) const { if (g) llama_grammar_free_impl(g); }
};
typedef std::unique_ptr<llama_grammar, GrammarDeleter> grammar_ptr;

// The text the grammar allows next when it allows exactly one character at
// each step: keys, separators and the rest of an enum value once it is
// unambiguous. It stops where the grammar branches, may end, or accepts a
// character class. The grammar's stacks are restored before returning.
static std::string grammar_forced_text(llama_grammar & g, size_t max_chars = 64) {
  std::string out;
  if (g.partial_utf8.n_remain > 0) return out;
  const llama_grammar_stacks saved = g.stacks;
  while (out.size() < max_chars && !g.stacks.empty()) {
    uint32_t chr = 0;
    bool forced = true;
    for (size_t i = 0; i < g.stacks.size() && forced; ++i) {
      const llama_grammar_stack & st = g.stacks[i];
      if (st.empty()) { forced = false; break; }  // the grammar may end here
      const llama_grammar_element * pos = st.back();
      if (pos->type != LLAMA_GRETYPE_CHAR ||
          pos[1].type == LLAMA_GRETYPE_CHAR_ALT ||
          pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
        forced = false;
      } else if (i == 0) {
        chr = pos->value;
      } else if (pos->value != chr) {
        forced = false;
      }
    }
    if (!forced || chr == 0 || chr > 0x7F) break;  // ASCII only: no partial UTF-8
    out.push_back(static_cast<char>(chr));
    llama_grammar_accept(&g, chr);
  }
  g.stacks = saved;
  return out;
}

// Frees the sampler chain on every exit path, including R errors, user
// interrupts and ggml aborts that unwind out of the generation loop.
struct SamplerDeleter {
  void operator()(llama_sampler * s) const { if (s) llama_sampler_free(s); }
};
typedef std::unique_ptr<llama_sampler, SamplerDeleter> sampler_ptr;

// Sampling settings shared by the generation functions. R builds them as a
// named list (see .sampling_options() in R/api.R); missing fields keep these
// defaults, which switch every optional step off.
struct SamplingOptions {
  double temperature = 0.8;
  double top_p = 0.95;
  int top_k = 0;              // 0: off
  double min_p = 0.0;         // 0: off
  double repeat_penalty = 1.0;  // 1: off
  int repeat_last_n = 64;
  uint32_t seed = 12345;
  std::vector<std::string> stop;
};

static SamplingOptions read_sampling(const List & s) {
  SamplingOptions o;
  if (s.containsElementNamed("temperature")) o.temperature = as<double>(s["temperature"]);
  if (s.containsElementNamed("top_p")) o.top_p = as<double>(s["top_p"]);
  if (s.containsElementNamed("top_k")) o.top_k = as<int>(s["top_k"]);
  if (s.containsElementNamed("min_p")) o.min_p = as<double>(s["min_p"]);
  if (s.containsElementNamed("repeat_penalty")) o.repeat_penalty = as<double>(s["repeat_penalty"]);
  if (s.containsElementNamed("repeat_last_n")) o.repeat_last_n = as<int>(s["repeat_last_n"]);
  if (s.containsElementNamed("seed")) o.seed = static_cast<uint32_t>(as<double>(s["seed"]));
  if (s.containsElementNamed("stop") && !Rf_isNull(s["stop"])) {
    o.stop = as<std::vector<std::string>>(s["stop"]);
  }
  if (o.temperature < 0.0 || o.temperature > 2.0) stop("Temperature must be between 0.0 and 2.0");
  if (o.top_p <= 0.0 || o.top_p > 1.0) stop("top_p must be between 0.0 and 1.0");
  if (o.top_k < 0) stop("top_k must be 0 or more");
  if (o.min_p < 0.0 || o.min_p >= 1.0) stop("min_p must be at least 0 and less than 1");
  if (o.repeat_penalty <= 0.0) stop("repeat_penalty must be positive");
  for (const std::string & st : o.stop) {
    if (st.empty()) stop("stop strings must not be empty");
  }
  return o;
}

// The sampling steps after any grammar, in llama.cpp's usual order. The
// repeat penalty comes first so greedy decoding can use it too.
// temperature = 0 means greedy decoding, so the same prompt always gives the
// same text; otherwise top-k, top-p, min-p, temperature and a seeded draw.
static void add_sampling_steps(llama_sampler * chain, const llama_vocab * vocab,
                               const SamplingOptions & o) {
  if (o.repeat_penalty != 1.0 && o.repeat_last_n != 0) {
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(
      llama_vocab_n_tokens(vocab), o.repeat_last_n,
      static_cast<float>(o.repeat_penalty), 0.0f, 0.0f));
  }
  if (o.temperature <= 0.0) {
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return;
  }
  if (o.top_k > 0) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(o.top_k));
  }
  if (o.top_p < 1.0) {
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(static_cast<float>(o.top_p), 1));
  }
  if (o.min_p > 0.0) {
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(static_cast<float>(o.min_p), 1));
  }
  llama_sampler_chain_add(chain, llama_sampler_init_temp(static_cast<float>(o.temperature)));
  llama_sampler_chain_add(chain, llama_sampler_init_dist(o.seed));
}

// Where the first stop string starts in text, searching only matches that
// end after position `from` (text before it was checked already), or npos.
static size_t find_stop(const std::string & text, const std::vector<std::string> & stops,
                        size_t from) {
  size_t best = std::string::npos;
  for (const std::string & st : stops) {
    const size_t start = from > st.size() ? from - st.size() : 0;
    const size_t pos = text.find(st, start);
    if (pos < best) best = pos;
  }
  return best;
}

// Length of the longest end of text that could be the start of a stop
// string. Streaming holds that much back until the next token decides it.
static size_t stop_holdback(const std::string & text, const std::vector<std::string> & stops) {
  size_t hold = 0;
  for (const std::string & st : stops) {
    const size_t max_len = std::min(text.size(), st.size() - 1);
    for (size_t len = max_len; len > hold; --len) {
      if (text.compare(text.size() - len, len, st, 0, len) == 0) {
        hold = len;
        break;
      }
    }
  }
  return hold;
}

// Helper function to ensure initialization is done
static void ensure_llama_initialized() {
  static bool initialized = false;
  if (!initialized) {
    // Set up quiet logging
    llama_log_set(quiet_log_callback, NULL);

    // Load CUDA (or other GPU) backend if configured via edge_use_cuda_backend_internal()
    if (!g_cuda_backend_path.empty()) {
      ggml_backend_reg_t cuda_reg = ggml_backend_load(g_cuda_backend_path.c_str());
      if (cuda_reg) {
        g_cuda_backend_loaded = true;
      } else if (g_logging_enabled) {
        Rcpp::warning("Failed to load GPU backend from: " + g_cuda_backend_path +
                      ". Falling back to CPU.");
      }
    }

    // Load all available backends (CPU always present, GPU backends via dynamic loading)
    ggml_backend_load_all();

    // Initialize llama backend
    llama_backend_init();

    // Register CPU backend explicitly (belt-and-suspenders in case load_all missed it)
    ggml_backend_register(ggml_backend_cpu_reg());

    initialized = true;
  }
}

// [[Rcpp::export]]
bool edge_use_cuda_backend_internal(std::string path) {
  // Store the path for use during next ensure_llama_initialized() call.
  // Note: if initialization has already run, this has no effect — the user
  // must restart R or call edge_reload_backends_internal() to apply.
  g_cuda_backend_path = path;
  return true;
}

// [[Rcpp::export]]
std::string edge_cuda_backend_path_internal() {
  return g_cuda_backend_path;
}

// [[Rcpp::export]]
bool edge_cuda_backend_loaded_internal() {
  return g_cuda_backend_loaded;
}

struct EdgeModelContext {
  struct llama_model* model = NULL;
  struct llama_context* ctx = NULL;

  // Prompt cache: the tokens in the KV cache (sequence 0), in order. Only
  // trusted while cache_valid is true; it is cleared before every decode and
  // set again after, so an interrupted decode never leaves a stale list.
  bool prompt_cache = false;
  bool cache_valid = false;
  std::vector<llama_token> cached_tokens;

  // LoRA adapters loaded with the model, and the scale each one is applied
  // at. The model owns the adapters and frees them with itself.
  std::vector<llama_adapter_lora *> loras;
  std::vector<float> lora_scales;

  // Prompts edge_completion_batch_internal() may run at once (n_parallel).
  int n_parallel = 1;

  EdgeModelContext() = default;

  // Copy constructor and assignment deleted to prevent double-free
  EdgeModelContext(const EdgeModelContext&) = delete;
  EdgeModelContext& operator=(const EdgeModelContext&) = delete;

  ~EdgeModelContext() {
    cleanup();
  }

  void cleanup() {
    if (ctx) {
      llama_free(ctx);
      ctx = NULL;
    }
    if (model) {
      llama_model_free(model);
      model = NULL;
    }
  }

  bool is_valid() const {
    return model != NULL && ctx != NULL;
  }

  // Additional safety check for pointers
  bool is_safe() const {
    try {
      return is_valid() &&
             llama_n_ctx(ctx) > 0 &&
             llama_model_n_ctx_train(model) > 0;
    } catch (...) {
      return false;
    }
  }
};

// Sets the context's LoRA adapters to the loaded ones at their scales. An
// adapter at scale 0 is left out instead of being computed for nothing.
static bool apply_loras(EdgeModelContext & e) {
  std::vector<llama_adapter_lora *> active;
  std::vector<float> scales;
  for (size_t i = 0; i < e.loras.size(); ++i) {
    if (e.lora_scales[i] != 0.0f) {
      active.push_back(e.loras[i]);
      scales.push_back(e.lora_scales[i]);
    }
  }
  return llama_set_adapters_lora(e.ctx, active.empty() ? nullptr : active.data(),
                                 active.size(), active.empty() ? nullptr : scales.data()) == 0;
}

// Puts a prompt into the KV cache. With the prompt cache on, the longest
// prefix the prompt shares with what the cache holds is kept and only the
// rest is decoded. Otherwise, or when the prefix cannot be reused, the cache
// is cleared first, so a completion never sees an earlier call's text.
static bool load_prompt(EdgeModelContext & e, std::vector<llama_token> & tokens) {
  llama_memory_t mem = llama_get_memory(e.ctx);
  size_t keep = 0;
  if (e.prompt_cache && e.cache_valid && mem && !tokens.empty()) {
    const size_t n = std::min(e.cached_tokens.size(), tokens.size());
    while (keep < n && e.cached_tokens[keep] == tokens[keep]) ++keep;
    // Always decode the last prompt token: its logits pick the next token.
    if (keep == tokens.size()) --keep;
    // Some caches (recurrent models) cannot drop just the tail.
    if (keep > 0 && !llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(keep), -1)) keep = 0;
    // Sliding-window layers may already have dropped early prefix tokens.
    const int32_t n_swa = llama_model_n_swa(e.model);
    if (keep > 0 && n_swa > 0 &&
        llama_memory_seq_pos_min(mem, 0) > std::max<llama_pos>(0, static_cast<llama_pos>(keep) - n_swa)) {
      keep = 0;
    }
  }
  if (keep == 0) clear_kv_cache(e.ctx);

  e.cache_valid = false;
  e.cached_tokens.resize(keep);
  std::vector<llama_token> rest(tokens.begin() + keep, tokens.end());
  if (!decode_prompt(e.ctx, rest)) return false;
  e.cached_tokens.insert(e.cached_tokens.end(), rest.begin(), rest.end());
  e.cache_valid = true;
  return true;
}

// Decodes one generated token and records it in the prompt cache.
static bool decode_token(EdgeModelContext & e, llama_token token) {
  e.cache_valid = false;
  llama_batch batch = llama_batch_get_one(&token, 1);
  if (llama_decode(e.ctx, batch)) return false;
  e.cached_tokens.push_back(token);
  e.cache_valid = true;
  return true;
}

// The KV cache types edge_load_model() accepts. These are the ones the CPU
// flash attention kernels support; another type would reach a GGML_ASSERT.
static ggml_type kv_cache_type(const std::string& name) {
  static const std::pair<const char*, ggml_type> types[] = {
    {"f16", GGML_TYPE_F16},   {"bf16", GGML_TYPE_BF16}, {"q8_0", GGML_TYPE_Q8_0},
    {"q5_1", GGML_TYPE_Q5_1}, {"q5_0", GGML_TYPE_Q5_0}, {"q4_1", GGML_TYPE_Q4_1},
    {"q4_0", GGML_TYPE_Q4_0}, {"iq4_nl", GGML_TYPE_IQ4_NL}};
  for (const auto& t : types) {
    if (name == t.first) return t.second;
  }
  stop("Unknown KV cache type: " + name);
}

// [[Rcpp::export]]
SEXP edge_load_model_internal(std::string model_path, int n_ctx, int n_gpu_layers, int n_threads, bool flash_attn, bool embeddings, bool prompt_cache,
                              std::vector<std::string> lora_paths,
                              std::vector<double> lora_scales,
                              int n_parallel,
                              std::string cache_type_k = "f16",
                              std::string cache_type_v = "f16") {
  try {
    // Ensure llama is properly initialized
    ensure_llama_initialized();
    // Checked before the model loads, so a bad name does not leak it.
    const ggml_type type_k = kv_cache_type(cache_type_k);
    const ggml_type type_v = kv_cache_type(cache_type_v);
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    // Reject anything that is not a GGUF file before handing it to the loader.
    // The loader is not hardened against arbitrary input: on clang 23 a plain
    // text file makes it read unmapped memory and take the R session down with
    // it, rather than returning null. Checking the magic here keeps a bad path
    // from ever reaching that code.
    {
      std::ifstream probe(model_path, std::ios::binary);
      if (!probe.good()) {
        stop("Model file does not exist or is not readable: " + model_path);
      }
      char magic[4] = {0, 0, 0, 0};
      probe.read(magic, 4);
      if (probe.gcount() != 4 || std::string(magic, 4) != "GGUF") {
        stop("Not a GGUF file: " + model_path +
             "\nThe file does not start with the GGUF magic header. "
             "If this came from Ollama, pass the blob file rather than the manifest.");
      }
    }

    struct llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
      // Check if file exists
      std::ifstream file(model_path);
      if (!file.good()) {
        stop("Model file does not exist or is not readable: " + model_path);
      }

      // Enhanced diagnostics for GGUF files
      std::string diagnostic_msg = "Failed to load GGUF model from: " + model_path +
                                   ". The file exists but llama.cpp cannot parse it.\n";

      // Check if it looks like a GGUF file
      file.seekg(0);
      char magic[4];
      if (file.read(magic, 4) && std::string(magic, 4) == "GGUF") {
        diagnostic_msg += "File has valid GGUF magic header. Possible issues:\n";
        diagnostic_msg += "- Incompatible GGUF version (try a different llama.cpp version)\n";
        diagnostic_msg += "- Model architecture not supported\n";
        diagnostic_msg += "- File corruption during download\n";
        diagnostic_msg += "- Insufficient memory (try smaller n_ctx or n_gpu_layers=0)\n";
      } else {
        diagnostic_msg += "File does not have GGUF magic header. This is not a valid GGUF file.\n";
        diagnostic_msg += "For Ollama models, ensure you're using the correct blob file path.\n";
      }

      stop(diagnostic_msg);
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;

    // Adaptive batch size optimization for small models
    // Small models benefit from larger batches relative to context size
    int optimal_batch;
    if (n_ctx <= 512) {
      optimal_batch = std::min(512, n_ctx);  // Very small context: use up to full context
    } else if (n_ctx <= 2048) {
      optimal_batch = std::min(512, n_ctx / 2);  // Small models: use 1/2 context
    } else if (n_ctx <= 4096) {
      optimal_batch = std::min(1024, n_ctx / 4);  // Medium models: use 1/4 context
    } else {
      optimal_batch = std::min(2048, n_ctx / 4);  // Large context: cap at 2048
    }
    ctx_params.n_batch = optimal_batch;

    // Threads: R passes one per physical core by default; 0 means all hardware
    // threads. Generating one token at a time is fastest on physical cores
    // (hyperthreads halve it), while prompt batches still gain from every
    // hardware thread.
    int hardware_threads = std::max(1, (int)std::thread::hardware_concurrency());
    int effective_threads = (n_threads > 0) ? std::min(n_threads, hardware_threads) : hardware_threads;
    ctx_params.n_threads = effective_threads;
    ctx_params.n_threads_batch = hardware_threads;
    // flash_attn was a bool in older llama.cpp; since b8179 it is an enum
    ctx_params.flash_attn_type = flash_attn
      ? LLAMA_FLASH_ATTN_TYPE_ENABLED
      : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // Quantized V needs flash attention; edge_load_model() checks that.
    ctx_params.type_k = type_k;
    ctx_params.type_v = type_v;
    ctx_params.embeddings = embeddings;
    if (embeddings) {
      // edge_embeddings() decodes several texts at once, one sequence each.
      // Pooled embeddings need a whole sequence in one micro-batch, and a
      // unified KV cache lets every sequence use the whole context.
      ctx_params.n_seq_max = 16;
      ctx_params.kv_unified = true;
      ctx_params.n_ubatch = ctx_params.n_batch;
    }
    if (n_parallel > 1) {
      // edge_completion_batch_internal() runs up to n_parallel prompts as
      // separate sequences. A unified KV cache lets them share the whole
      // context instead of n_ctx / n_parallel each.
      ctx_params.n_seq_max = std::max<uint32_t>(ctx_params.n_seq_max, static_cast<uint32_t>(n_parallel));
      ctx_params.kv_unified = true;
    }
    
    struct llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
      llama_model_free(model);
      if (cache_type_k != "f16" || cache_type_v != "f16") {
        stop("Failed to create context for model with cache_type_k = \"" + cache_type_k +
             "\" and cache_type_v = \"" + cache_type_v + "\". The model's attention "
             "head size may not fit this cache type, or the model may need the "
             "same type for both; try \"q8_0\" for both, or \"f16\".");
      }
      stop("Failed to create context for model");
    }
    
    auto edge_ctx = std::make_unique<EdgeModelContext>();
    edge_ctx->model = model;
    edge_ctx->prompt_cache = prompt_cache;
    edge_ctx->ctx = ctx;
    edge_ctx->n_parallel = std::max(1, n_parallel);

    // LoRA adapters. On any failure edge_ctx frees the context and the model,
    // and the model frees the adapters loaded so far.
    for (size_t i = 0; i < lora_paths.size(); ++i) {
      const std::string & path = lora_paths[i];
      std::ifstream probe(path, std::ios::binary);
      char magic[4] = {0, 0, 0, 0};
      probe.read(magic, 4);
      if (!probe.good() || std::string(magic, 4) != "GGUF") {
        stop("LoRA adapter is not a readable GGUF file: " + path);
      }
      llama_adapter_lora * adapter = llama_adapter_lora_init(model, path.c_str());
      if (!adapter) {
        stop("Failed to load LoRA adapter: " + path +
             "\nThe adapter must be a GGUF LoRA made for this model's architecture.");
      }
      edge_ctx->loras.push_back(adapter);
      edge_ctx->lora_scales.push_back(
        static_cast<float>(i < lora_scales.size() ? lora_scales[i] : 1.0));
    }
    if (!edge_ctx->loras.empty() && !apply_loras(*edge_ctx)) {
      stop("Failed to apply LoRA adapters to the context");
    }

    XPtr<EdgeModelContext> ptr(edge_ctx.release(), true);
    ptr.attr("class") = "edge_model_context";
    
    return ptr;
  } catch (const std::exception& e) {
    stop("Error loading model: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
std::string edge_completion_internal(SEXP model_ptr, std::string prompt, int n_predict, List sampling, double timeout_seconds = 0.0) {
  const auto started = std::chrono::steady_clock::now();
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) {
      stop("Invalid model context");
    }

    XPtr<EdgeModelContext> edge_ctx(model_ptr);

    if (!edge_ctx->is_valid() || edge_ctx->ctx == nullptr || edge_ctx->model == nullptr) {
      stop("Invalid model context or null pointers");
    }

    // Validate input parameters
    if (n_predict <= 0) {
      stop("n_predict must be positive");
    }
    const SamplingOptions opts = read_sampling(sampling);

    // Get vocabulary from model
    const struct llama_vocab* vocab = llama_model_get_vocab(edge_ctx->model);
    if (!vocab) {
      stop("Failed to get vocabulary from model");
    }

    // Tokenize the prompt - first get the number of tokens needed
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), NULL, 0, true, true);
    
    if (n_prompt_tokens <= 0) {
      stop("Failed to determine prompt token count");
    }
    
    // Allocate space for tokens and tokenize
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true) < 0) {
      stop("Failed to tokenize prompt");
    }

    // Validate prompt fits in context window
    int n_ctx = llama_n_ctx(edge_ctx->ctx);
    if (n_prompt_tokens >= n_ctx) {
      stop("Prompt too long (" + std::to_string(n_prompt_tokens) + " tokens) for context size (" +
           std::to_string(n_ctx) + "). Shorten the prompt or increase n_ctx in edge_load_model().");
    }

    // Process the prompt
    if (!load_prompt(*edge_ctx, prompt_tokens)) {
      stop("Failed to process prompt");
    }

    std::string result;  // Only collect generated text, not prompt
    result.reserve(n_predict * 8);
    
    // Create a sampler chain for better token generation
    auto sampler_chain_params = llama_sampler_chain_default_params();
    sampler_ptr sampler_owner(llama_sampler_chain_init(sampler_chain_params));
    llama_sampler * sampler = sampler_owner.get();

    add_sampling_steps(sampler, vocab, opts);

    // Generate tokens using the new sampler API
    for (int i = 0; i < n_predict; ++i) {
      // Lets Esc/Ctrl-C stop a long generation.
      Rcpp::checkUserInterrupt();
      if (timeout_seconds > 0) {
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
        if (elapsed.count() >= timeout_seconds) {
          std::ostringstream msg;
          msg << "Generation timed out after " << timeout_seconds << " seconds";
          throw std::runtime_error(msg.str());
        }
      }

      // Sample next token using the sampler chain
      llama_token new_token = llama_sampler_sample(sampler, edge_ctx->ctx, -1);
      
      // Check if it's end of generation
      if (llama_vocab_is_eog(vocab, new_token)) {
        break;
      }
      
      // Convert token to text with dynamic buffer allocation
      std::vector<char> piece(512);  // Start with larger buffer
      int n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);

      // If buffer too small, resize and retry
      if (n_chars < 0) {
        piece.resize(static_cast<size_t>(std::abs(n_chars)) + 1);  // +1 for null terminator
        n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
      }

      if (n_chars > 0) {
        const size_t before = result.size();
        result.append(piece.data(), n_chars);
        // A stop string ends the text where it starts; it is not returned.
        if (!opts.stop.empty()) {
          const size_t at = find_stop(result, opts.stop, before);
          if (at != std::string::npos) {
            result.resize(at);
            break;
          }
        }
      } else if (n_chars < 0) {
        // If still failing, log warning and skip this token
        warning("Failed to convert token to text, skipping token");
        continue;
      }

      // llama_sampler_sample already called accept — do not call it again.

      // Prepare next batch with the new token

      // Process the new token
      if (!decode_token(*edge_ctx, new_token)) {
        break;
      }
    }


    return result;
    
  } catch (const std::exception& e) {
    stop("Error during completion: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
void edge_free_model_internal(SEXP model_ptr) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) {
      warning("Invalid model pointer type");
      return;
    }

    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    if (edge_ctx.get() != nullptr) {
      edge_ctx->cleanup();
    }

  } catch (const std::exception& e) {
    warning("Error freeing model: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
bool is_valid_model_internal(SEXP model_ptr) {
  try {
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    return edge_ctx->is_valid();
  } catch (...) {
    return false;
  }
}

// The end of the longest prefix of s[0, upto) that does not stop inside a
// UTF-8 character. A character can span several tokens, and each streamed
// piece must be valid text on its own.
static size_t utf8_boundary(const std::string& s, size_t upto) {
  size_t i = upto;
  size_t cont = 0;
  while (i > 0 && cont < 3 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) {
    --i;
    ++cont;
  }
  if (i == 0) return upto;
  const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
  size_t len = 1;
  if ((lead & 0xE0) == 0xC0) len = 2;
  else if ((lead & 0xF0) == 0xE0) len = 3;
  else if ((lead & 0xF8) == 0xF0) len = 4;
  return (len > 1 && cont + 1 < len) ? i - 1 : upto;
}

// [[Rcpp::export]]
List edge_completion_stream_internal(SEXP model_ptr, std::string prompt, Function callback, int n_predict, List sampling) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) {
      stop("Invalid model context");
    }
    
    XPtr<EdgeModelContext> edge_ctx(model_ptr);

    if (!edge_ctx->is_valid() || edge_ctx->ctx == nullptr || edge_ctx->model == nullptr) {
      stop("Invalid model context or null pointers");
    }

    // Validate input parameters
    if (n_predict <= 0) {
      stop("n_predict must be positive");
    }
    const SamplingOptions opts = read_sampling(sampling);

    // Get vocabulary from model
    const struct llama_vocab* vocab = llama_model_get_vocab(edge_ctx->model);
    if (!vocab) {
      stop("Failed to get vocabulary from model");
    }

    // Tokenize the prompt - first get the number of tokens needed
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), NULL, 0, true, true);
    
    if (n_prompt_tokens <= 0) {
      stop("Failed to determine prompt token count");
    }
    
    // Allocate space for tokens and tokenize
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true) < 0) {
      stop("Failed to tokenize prompt");
    }

    // Validate prompt fits in context window
    int n_ctx = llama_n_ctx(edge_ctx->ctx);
    if (n_prompt_tokens >= n_ctx) {
      stop("Prompt too long (" + std::to_string(n_prompt_tokens) + " tokens) for context size (" +
           std::to_string(n_ctx) + "). Shorten the prompt or increase n_ctx in edge_load_model().");
    }

    // Process the prompt
    if (!load_prompt(*edge_ctx, prompt_tokens)) {
      stop("Failed to process prompt");
    }

    std::string full_response;  // Track generated text only
    std::vector<std::string> tokens_generated;
    int tokens_count = 0;
    int n_generated = 0;  // sampled tokens, not callback calls
    bool stopped_early = false;

    // Create sampler chain for streaming
    auto sampler_chain_params = llama_sampler_chain_default_params();
    sampler_ptr sampler_owner(llama_sampler_chain_init(sampler_chain_params));
    llama_sampler * sampler = sampler_owner.get();

    add_sampling_steps(sampler, vocab, opts);

    // Text is passed to the callback once no stop string can start in it
    // and it ends on a whole character. Without stop strings that is
    // usually every token as soon as it is generated.
    size_t sent = 0;
    auto send = [&](size_t upto, int position, bool flush = false) -> bool {
      if (!flush) upto = utf8_boundary(full_response, upto);
      if (upto <= sent) return true;
      const std::string text = full_response.substr(sent, upto - sent);
      sent = upto;
      tokens_generated.push_back(text);
      tokens_count++;
      try {
        List callback_data = List::create(
          Named("token") = text,
          Named("position") = position,
          Named("is_final") = false,
          Named("total_tokens") = tokens_count
        );
        SEXP result = callback(callback_data);
        // Check if callback wants to stop early
        if (is<LogicalVector>(result)) {
          LogicalVector stop_signal = as<LogicalVector>(result);
          if (stop_signal.length() > 0 && stop_signal[0] == false) return false;
        }
      } catch (const std::exception& e) {
        warning("Callback error: " + std::string(e.what()));
      }
      return true;
    };

    // Generate tokens one by one and stream them with proper sampling
    int position = 0;
    bool callback_stopped = false;
    for (int i = 0; i < n_predict; ++i) {
      // Lets Esc/Ctrl-C stop a long generation.
      Rcpp::checkUserInterrupt();
      position = i + 1;

      // Sample next token using the sampler chain
      llama_token new_token = llama_sampler_sample(sampler, edge_ctx->ctx, -1);
      
      // Check if it's end of generation
      if (llama_vocab_is_eog(vocab, new_token)) {
        stopped_early = true;
        break;
      }
      n_generated++;
      
      // Convert token to text with dynamic buffer allocation
      std::vector<char> piece(512);  // Start with larger buffer
      int n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);

      // If buffer too small, resize and retry
      if (n_chars < 0) {
        piece.resize(static_cast<size_t>(std::abs(n_chars)) + 1);  // +1 for null terminator
        n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
      }

      if (n_chars > 0) {
        const size_t before = full_response.size();
        full_response.append(piece.data(), n_chars);
        if (!opts.stop.empty()) {
          const size_t at = find_stop(full_response, opts.stop, before);
          if (at != std::string::npos) {
            full_response.resize(at);
            send(at, position);
            stopped_early = true;
            break;
          }
        }
        const size_t hold = opts.stop.empty() ? 0 : stop_holdback(full_response, opts.stop);
        if (!send(full_response.size() - hold, position)) {
          stopped_early = true;
          callback_stopped = true;
          break;
        }
      } else if (n_chars < 0) {
        // If still failing, log warning and skip this token
        warning("Failed to convert token to text, skipping token");
        continue;
      }

      // llama_sampler_sample already called accept — do not call it again.

      // Process the new token
      if (!decode_token(*edge_ctx, new_token)) {
        stopped_early = true;
        break;
      }
    }
    // Text held back for a stop string that never came.
    if (!callback_stopped) send(full_response.size(), position, true);

    // Send final callback
    try {
      List final_callback_data = List::create(
        Named("token") = "",
        Named("position") = tokens_count,
        Named("is_final") = true,
        Named("total_tokens") = tokens_count,
        Named("full_response") = full_response,
        Named("stopped_early") = stopped_early
      );
      callback(final_callback_data);
    } catch (const std::exception& e) {
      warning("Final callback error: " + std::string(e.what()));
    }


    // Return summary information
    return List::create(
      Named("full_response") = full_response,
      Named("tokens_generated") = tokens_generated,
      Named("total_tokens") = tokens_count,
      Named("stopped_early") = stopped_early,
      Named("original_prompt") = prompt,
      Named("n_generated") = n_generated,
      Named("hit_limit") = !stopped_early
    );
    
  } catch (const std::exception& e) {
    stop("Error during streaming completion: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
std::string edge_completion_grammar_internal(SEXP model_ptr, std::string prompt, std::string grammar_str, std::string grammar_root, int n_predict, List sampling, bool jump_forward = true) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) {
      stop("Invalid model context");
    }

    XPtr<EdgeModelContext> edge_ctx(model_ptr);

    if (!edge_ctx->is_valid() || edge_ctx->ctx == nullptr || edge_ctx->model == nullptr) {
      stop("Invalid model context or null pointers");
    }

    if (n_predict <= 0) stop("n_predict must be positive");
    const SamplingOptions opts = read_sampling(sampling);

    const struct llama_vocab* vocab = llama_model_get_vocab(edge_ctx->model);
    if (!vocab) stop("Failed to get vocabulary from model");

    // Tokenize
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), NULL, 0, true, true);
    if (n_prompt_tokens <= 0) stop("Failed to determine prompt token count");

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(), prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true) < 0) {
      stop("Failed to tokenize prompt");
    }

    // Validate prompt fits in context window
    int n_ctx = llama_n_ctx(edge_ctx->ctx);
    if (n_prompt_tokens >= n_ctx) {
      stop("Prompt too long (" + std::to_string(n_prompt_tokens) + " tokens) for context size (" +
           std::to_string(n_ctx) + "). Shorten the prompt or increase n_ctx in edge_load_model().");
    }

    // Process prompt
    if (!load_prompt(*edge_ctx, prompt_tokens)) stop("Failed to process prompt");

    // The grammar is kept out of the sampler chain, as in llama.cpp's common
    // sampler: sample with the chain first and check only that one token
    // against the grammar. The grammar runs over the whole vocabulary only
    // when the token is rejected. Checking all of a 150k-token vocabulary at
    // every step made grammar completions several times slower.
    auto sampler_chain_params = llama_sampler_chain_default_params();
    sampler_ptr sampler_owner(llama_sampler_chain_init(sampler_chain_params));
    llama_sampler * sampler = sampler_owner.get();
    add_sampling_steps(sampler, vocab, opts);

    sampler_ptr grammar_owner;
    llama_sampler * grammar_sampler = nullptr;
    if (!grammar_str.empty()) {
      grammar_sampler = llama_sampler_init_grammar(vocab, grammar_str.c_str(), grammar_root.c_str());
      if (!grammar_sampler) {
        stop("Failed to parse GBNF grammar. Check grammar syntax.");
      }
      grammar_owner.reset(grammar_sampler);
    }

    // A second copy of the grammar, kept in step with the sampler's, to look
    // ahead for text the grammar forces. Forced text is decoded as one batch
    // instead of one token per step: decoding k tokens together costs little
    // more than decoding one on CPU, and JSON keys and separators make up a
    // large share of a short structured answer.
    grammar_ptr shadow;
    if (grammar_sampler && jump_forward) {
      shadow.reset(llama_grammar_init_impl(vocab, grammar_str.c_str(), grammar_root.c_str(),
                                           false, nullptr, 0, nullptr, 0));
    }
    const int n_ctx_max = llama_n_ctx(edge_ctx->ctx);

    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> candidates(n_vocab);
    auto all_candidates = [&]() {
      const float * logits = llama_get_logits_ith(edge_ctx->ctx, -1);
      for (int t = 0; t < n_vocab; ++t) candidates[t] = llama_token_data{t, logits[t], 0.0f};
      return llama_token_data_array{candidates.data(), static_cast<size_t>(n_vocab), -1, false};
    };

    // Generate tokens (only collect generated text, not prompt)
    std::string result;
    result.reserve(n_predict * 8);

    for (int i = 0; i < n_predict; ++i) {
      Rcpp::checkUserInterrupt();

      if (shadow) {
        const std::string forced = grammar_forced_text(*shadow);
        if (!forced.empty()) {
          std::vector<llama_token> ft(forced.size() + 1);
          int n_ft = llama_tokenize(vocab, forced.c_str(), (int32_t)forced.size(), ft.data(),
                                    (int32_t)ft.size(), false, false);
          // Only when the tokens spell the forced text exactly and still fit.
          std::string spelled;
          // The last token is left for the model to sample: it may merge with
          // what follows (the model writes {" as one token, not { then ").
          // Forcing a split the model would not make changes its output.
          if (n_ft > 0) n_ft -= 1;
          if (n_ft > 0) {
            ft.resize(n_ft);
            for (llama_token tk : ft) {
              char buf[256];
              const int n = llama_token_to_piece(vocab, tk, buf, sizeof(buf), 0, false);
              if (n < 0) { spelled.clear(); break; }
              spelled.append(buf, n);
            }
          }
          if (n_ft > 0 && !spelled.empty() && forced.compare(0, spelled.size(), spelled) == 0 && i + n_ft <= n_predict &&
              static_cast<int>(edge_ctx->cached_tokens.size()) + n_ft < n_ctx_max) {
            for (llama_token tk : ft) {
              llama_sampler_accept(grammar_sampler, tk);
              llama_grammar_accept_impl(*shadow, tk);
              llama_sampler_accept(sampler, tk);
            }
            const size_t before = result.size();
            result += spelled;
            if (!opts.stop.empty()) {
              const size_t at = find_stop(result, opts.stop, before);
              if (at != std::string::npos) { result.resize(at); break; }
            }
            edge_ctx->cache_valid = false;
            llama_batch fb = llama_batch_get_one(ft.data(), n_ft);
            if (llama_decode(edge_ctx->ctx, fb)) break;
            edge_ctx->cached_tokens.insert(edge_ctx->cached_tokens.end(), ft.begin(), ft.end());
            edge_ctx->cache_valid = true;
            i += n_ft - 1;
            continue;
          }
        }
      }

      // Sample, then accept the token once into each sampler. (Accepting
      // into the grammar twice would advance its state twice.)
      llama_token_data_array cur = all_candidates();
      llama_sampler_apply(sampler, &cur);
      llama_token new_token = cur.data[cur.selected].id;
      if (grammar_sampler) {
        llama_token_data single = {new_token, 1.0f, 0.0f};
        llama_token_data_array one = {&single, 1, -1, false};
        llama_sampler_apply(grammar_sampler, &one);
        if (std::isinf(single.logit) && single.logit < 0) {
          cur = all_candidates();
          llama_sampler_apply(grammar_sampler, &cur);
          llama_sampler_apply(sampler, &cur);
          new_token = cur.data[cur.selected].id;
        }
        llama_sampler_accept(grammar_sampler, new_token);
        if (shadow && !llama_vocab_is_eog(vocab, new_token)) llama_grammar_accept_impl(*shadow, new_token);
      }
      llama_sampler_accept(sampler, new_token);

      if (llama_vocab_is_eog(vocab, new_token)) break;

      std::vector<char> piece(512);
      int n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
      if (n_chars < 0) {
        piece.resize(static_cast<size_t>(std::abs(n_chars)) + 1);
        n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
      }
      if (n_chars > 0) {
        const size_t before = result.size();
        result.append(piece.data(), n_chars);
        if (!opts.stop.empty()) {
          const size_t at = find_stop(result, opts.stop, before);
          if (at != std::string::npos) { result.resize(at); break; }
        }
      }

      if (!decode_token(*edge_ctx, new_token)) break;
    }

    return result;

  } catch (const std::exception& e) {
    stop("Error during grammar completion: " + std::string(e.what()));
  }
}

// Owns a llama_batch from llama_batch_init(), freed on every exit path.
struct BatchOwner {
  llama_batch b;
  BatchOwner(int32_t n_tokens, int32_t n_seq) : b(llama_batch_init(n_tokens, 0, n_seq)) {}
  ~BatchOwner() { llama_batch_free(b); }
  BatchOwner(const BatchOwner &) = delete;
  BatchOwner & operator=(const BatchOwner &) = delete;
};

static void batch_add(llama_batch & b, llama_token token, llama_pos pos, llama_seq_id seq, bool logits) {
  const int j = b.n_tokens++;
  b.token[j] = token;
  b.pos[j] = pos;
  b.n_seq_id[j] = 1;
  b.seq_id[j][0] = seq;
  b.logits[j] = logits ? 1 : 0;
}

// One prompt being generated in a batch.
struct BatchSeq {
  size_t index = 0;           // position in the prompts vector
  sampler_ptr chain;
  sampler_ptr grammar;
  llama_token last = 0;       // sampled, not yet decoded
  llama_pos pos = 0;          // position of `last`
  int generated = 0;
  bool done = false;
  std::string text;
};

// Samples from output `idx` of the last decode. With a grammar, the token the
// chain picks is checked alone first and the whole vocabulary is filtered
// only when the grammar rejects it, as in edge_completion_grammar_internal().
static llama_token batch_sample(BatchSeq & s, llama_context * ctx, int32_t idx,
                                std::vector<llama_token_data> & candidates) {
  if (!s.grammar) return llama_sampler_sample(s.chain.get(), ctx, idx);
  const float * logits = llama_get_logits_ith(ctx, idx);
  const int n_vocab = static_cast<int>(candidates.size());
  auto fill = [&]() {
    for (int t = 0; t < n_vocab; ++t) candidates[t] = llama_token_data{t, logits[t], 0.0f};
    return llama_token_data_array{candidates.data(), static_cast<size_t>(n_vocab), -1, false};
  };
  llama_token_data_array cur = fill();
  llama_sampler_apply(s.chain.get(), &cur);
  llama_token token = cur.data[cur.selected].id;
  llama_token_data single = {token, 1.0f, 0.0f};
  llama_token_data_array one = {&single, 1, -1, false};
  llama_sampler_apply(s.grammar.get(), &one);
  if (std::isinf(single.logit) && single.logit < 0) {
    cur = fill();
    llama_sampler_apply(s.grammar.get(), &cur);
    llama_sampler_apply(s.chain.get(), &cur);
    token = cur.data[cur.selected].id;
  }
  llama_sampler_accept(s.grammar.get(), token);
  llama_sampler_accept(s.chain.get(), token);
  return token;
}

// Adds a sampled token to a sequence's text. Returns false once the
// sequence is finished: end of generation, a stop string or n_predict.
static bool batch_take(BatchSeq & s, llama_token token, const llama_vocab * vocab,
                       const SamplingOptions & opts, int n_predict) {
  s.generated++;
  if (llama_vocab_is_eog(vocab, token)) return false;
  std::vector<char> piece(512);
  int n_chars = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
  if (n_chars < 0) {
    piece.resize(static_cast<size_t>(std::abs(n_chars)) + 1);
    n_chars = llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
  }
  if (n_chars > 0) {
    const size_t before = s.text.size();
    s.text.append(piece.data(), n_chars);
    if (!opts.stop.empty()) {
      const size_t at = find_stop(s.text, opts.stop, before);
      if (at != std::string::npos) {
        s.text.resize(at);
        return false;
      }
    }
  }
  if (s.generated >= n_predict) return false;
  s.last = token;
  return true;
}

// Generates a completion for each prompt, several at a time: each prompt is
// its own sequence in the KV cache, and every step decodes one token for
// each unfinished sequence in a single batch. A step for k sequences costs
// little more than a step for one on CPU, since generation is limited by
// reading the weights. Needs a context made with n_parallel > 1; as many
// prompts run together as fit in n_parallel and in the context (prompt plus
// n_predict tokens each).
// [[Rcpp::export]]
std::vector<std::string> edge_completion_batch_internal(SEXP model_ptr, std::vector<std::string> prompts,
                                                        int n_predict, List sampling,
                                                        std::vector<double> seeds,
                                                        std::string grammar_str,
                                                        std::string grammar_root) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    if (!edge_ctx->is_valid()) stop("Invalid model context or null pointers");
    if (n_predict <= 0) stop("n_predict must be positive");
    SamplingOptions opts = read_sampling(sampling);
    if (seeds.size() != prompts.size()) stop("Expected one seed per prompt");

    llama_context * ctx = edge_ctx->ctx;
    const llama_vocab * vocab = llama_model_get_vocab(edge_ctx->model);
    const int n_ctx = static_cast<int>(llama_n_ctx(ctx));
    const int n_batch = std::max<int>(1, static_cast<int>(llama_n_batch(ctx)));
    const int n_seq_max = std::max(1, std::min(edge_ctx->n_parallel,
                                               static_cast<int>(llama_n_seq_max(ctx))));

    std::vector<std::vector<llama_token>> toks(prompts.size());
    for (size_t p = 0; p < prompts.size(); ++p) {
      const std::string & text = prompts[p];
      const int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), NULL, 0, true, true);
      if (n <= 0) stop("Failed to determine prompt token count");
      toks[p].resize(n);
      if (llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks[p].data(), n, true, true) < 0) {
        stop("Failed to tokenize prompt");
      }
      if (n >= n_ctx) {
        stop("Prompt " + std::to_string(p + 1) + " is too long (" + std::to_string(n) +
             " tokens) for context size (" + std::to_string(n_ctx) +
             "). Shorten the prompt or increase n_ctx in edge_load_model().");
      }
    }

    // The sequences share the KV cache, so what the prompt cache held is gone.
    edge_ctx->cache_valid = false;
    edge_ctx->cached_tokens.clear();

    std::vector<std::string> results(prompts.size());
    std::vector<llama_token_data> candidates;
    if (!grammar_str.empty()) candidates.resize(llama_vocab_n_tokens(vocab));
    BatchOwner owner(std::max(n_batch, n_seq_max), 1);
    llama_batch & batch = owner.b;

    size_t next = 0;
    while (next < prompts.size()) {
      // As many prompts as fit, each with room for its whole answer.
      std::vector<BatchSeq> seqs;
      int budget = 0;
      while (next < prompts.size() && static_cast<int>(seqs.size()) < n_seq_max) {
        const int need = static_cast<int>(toks[next].size()) + n_predict;
        if (!seqs.empty() && budget + need > n_ctx) break;
        budget += need;
        seqs.emplace_back();
        seqs.back().index = next++;
      }
      clear_kv_cache(ctx);

      // Each prompt is decoded on its own sequence, and its first token is
      // sampled straight away, before the next decode replaces the logits.
      for (size_t s = 0; s < seqs.size(); ++s) {
        BatchSeq & q = seqs[s];
        Rcpp::checkUserInterrupt();
        SamplingOptions o = opts;
        o.seed = static_cast<uint32_t>(seeds[q.index]);
        q.chain.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        add_sampling_steps(q.chain.get(), vocab, o);
        if (!grammar_str.empty()) {
          q.grammar.reset(llama_sampler_init_grammar(vocab, grammar_str.c_str(), grammar_root.c_str()));
          if (!q.grammar) stop("Failed to parse GBNF grammar. Check grammar syntax.");
        }
        const std::vector<llama_token> & tk = toks[q.index];
        for (size_t i = 0; i < tk.size(); i += n_batch) {
          const size_t n = std::min(static_cast<size_t>(n_batch), tk.size() - i);
          batch.n_tokens = 0;
          for (size_t k = 0; k < n; ++k) {
            batch_add(batch, tk[i + k], static_cast<llama_pos>(i + k),
                      static_cast<llama_seq_id>(s), i + k == tk.size() - 1);
          }
          if (llama_decode(ctx, batch)) stop("Failed to process prompt");
        }
        q.pos = static_cast<llama_pos>(tk.size());
        const llama_token first = batch_sample(q, ctx, batch.n_tokens - 1, candidates);
        q.done = !batch_take(q, first, vocab, opts, n_predict);
      }

      // Then one token per unfinished sequence per decode.
      std::vector<size_t> active;
      while (true) {
        Rcpp::checkUserInterrupt();
        active.clear();
        batch.n_tokens = 0;
        for (size_t s = 0; s < seqs.size(); ++s) {
          if (seqs[s].done) continue;
          batch_add(batch, seqs[s].last, seqs[s].pos, static_cast<llama_seq_id>(s), true);
          seqs[s].pos++;
          active.push_back(s);
        }
        if (active.empty()) break;
        if (llama_decode(ctx, batch)) break;  // out of room: keep what was generated
        for (size_t j = 0; j < active.size(); ++j) {
          BatchSeq & q = seqs[active[j]];
          const llama_token token = batch_sample(q, ctx, static_cast<int32_t>(j), candidates);
          q.done = !batch_take(q, token, vocab, opts, n_predict);
        }
      }
      for (BatchSeq & q : seqs) results[q.index] = q.text;
    }
    clear_kv_cache(ctx);
    return results;
  } catch (const std::exception & e) {
    stop("Error during batch completion: " + std::string(e.what()));
  }
}

// What a saved state depends on: the model, and the LoRA adapters at their
// scales, since they change every layer's keys and values.
static std::string state_fingerprint(const EdgeModelContext & e) {
  char desc[256] = {0};
  llama_model_desc(e.model, desc, sizeof(desc));
  std::ostringstream fp;
  fp << desc << "|params=" << llama_model_n_params(e.model)
     << "|embd=" << llama_model_n_embd(e.model)
     << "|layers=" << llama_model_n_layer(e.model)
     << "|vocab=" << llama_vocab_n_tokens(llama_model_get_vocab(e.model));
  for (size_t i = 0; i < e.loras.size(); ++i) fp << "|lora" << i << "=" << e.lora_scales[i];
  return fp.str();
}

static const char STATE_MAGIC[8] = {'E', 'D', 'G', 'E', 'K', 'V', '0', '1'};

// Saves the prompt cache (the KV cache of sequence 0 and its tokens) to a
// file. Returns the number of tokens saved.
// [[Rcpp::export]]
int edge_save_state_internal(SEXP model_ptr, std::string path) {
  if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
  XPtr<EdgeModelContext> edge_ctx(model_ptr);
  if (!edge_ctx->is_valid()) stop("Invalid model context");
  if (!edge_ctx->prompt_cache) {
    stop("Saving the state needs a model loaded with prompt_cache = TRUE");
  }
  if (!edge_ctx->cache_valid || edge_ctx->cached_tokens.empty()) {
    stop("Nothing to save: the cache is empty. Run a completion first.");
  }
  const size_t size = llama_state_seq_get_size(edge_ctx->ctx, 0);
  std::vector<uint8_t> data(size);
  const size_t written = llama_state_seq_get_data(edge_ctx->ctx, data.data(), size, 0);
  if (written == 0) stop("Failed to copy the KV cache");

  const std::string fp = state_fingerprint(*edge_ctx);
  const uint32_t fp_len = static_cast<uint32_t>(fp.size());
  const uint32_t n_tokens = static_cast<uint32_t>(edge_ctx->cached_tokens.size());
  const uint64_t n_data = static_cast<uint64_t>(written);

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.good()) stop("Cannot write to " + path);
  out.write(STATE_MAGIC, sizeof(STATE_MAGIC));
  out.write(reinterpret_cast<const char *>(&fp_len), sizeof(fp_len));
  out.write(fp.data(), fp_len);
  out.write(reinterpret_cast<const char *>(&n_tokens), sizeof(n_tokens));
  out.write(reinterpret_cast<const char *>(edge_ctx->cached_tokens.data()),
            static_cast<std::streamsize>(n_tokens * sizeof(llama_token)));
  out.write(reinterpret_cast<const char *>(&n_data), sizeof(n_data));
  out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(written));
  out.close();
  if (!out.good()) stop("Failed to write " + path);
  return static_cast<int>(n_tokens);
}

// Loads a state saved with edge_save_state_internal() into the prompt cache,
// so the next prompt that starts with the saved tokens skips them. Returns
// the number of tokens loaded.
// [[Rcpp::export]]
int edge_load_state_internal(SEXP model_ptr, std::string path) {
  if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
  XPtr<EdgeModelContext> edge_ctx(model_ptr);
  if (!edge_ctx->is_valid()) stop("Invalid model context");
  if (!edge_ctx->prompt_cache) {
    stop("Loading a state needs a model loaded with prompt_cache = TRUE");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in.good()) stop("Cannot read " + path);
  char magic[8] = {0};
  in.read(magic, sizeof(magic));
  if (!in.good() || std::memcmp(magic, STATE_MAGIC, sizeof(magic)) != 0) {
    stop("Not a state file saved by edge_save_state(): " + path);
  }
  uint32_t fp_len = 0;
  in.read(reinterpret_cast<char *>(&fp_len), sizeof(fp_len));
  if (!in.good() || fp_len > 4096) stop("Corrupt state file: " + path);
  std::string fp(fp_len, '\0');
  in.read(&fp[0], fp_len);
  if (fp != state_fingerprint(*edge_ctx)) {
    stop("The state was saved with a different model or different LoRA scales: " + path);
  }
  uint32_t n_tokens = 0;
  in.read(reinterpret_cast<char *>(&n_tokens), sizeof(n_tokens));
  if (!in.good() || n_tokens == 0 || n_tokens >= llama_n_ctx(edge_ctx->ctx)) {
    stop("The saved state does not fit in this context (n_ctx = " +
         std::to_string(llama_n_ctx(edge_ctx->ctx)) + "): " + path);
  }
  std::vector<llama_token> tokens(n_tokens);
  in.read(reinterpret_cast<char *>(tokens.data()), static_cast<std::streamsize>(n_tokens * sizeof(llama_token)));
  uint64_t n_data = 0;
  in.read(reinterpret_cast<char *>(&n_data), sizeof(n_data));
  if (!in.good() || n_data == 0 || n_data > (uint64_t(1) << 36)) stop("Corrupt state file: " + path);
  std::vector<uint8_t> data(static_cast<size_t>(n_data));
  in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(n_data));
  if (!in.good()) stop("Corrupt state file: " + path);

  clear_kv_cache(edge_ctx->ctx);
  edge_ctx->cache_valid = false;
  edge_ctx->cached_tokens.clear();
  if (llama_state_seq_set_data(edge_ctx->ctx, data.data(), data.size(), 0) == 0) {
    clear_kv_cache(edge_ctx->ctx);
    stop("Failed to load the state into the context: " + path);
  }
  edge_ctx->cached_tokens = tokens;
  edge_ctx->cache_valid = true;
  return static_cast<int>(n_tokens);
}

// [[Rcpp::export]]
NumericMatrix edge_embeddings_internal(SEXP model_ptr, std::vector<std::string> texts, bool normalize = true) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) {
      stop("Invalid model context");
    }

    XPtr<EdgeModelContext> edge_ctx(model_ptr);

    if (!edge_ctx->is_valid() || edge_ctx->ctx == nullptr || edge_ctx->model == nullptr) {
      stop("Invalid model context or null pointers");
    }

    if (texts.empty()) stop("texts must not be empty");

    const struct llama_vocab* vocab = llama_model_get_vocab(edge_ctx->model);
    if (!vocab) stop("Failed to get vocabulary from model");

    // The embedding buffers hold n_embd_out values, which differs from
    // n_embd for models that project their output, such as EmbeddingGemma2.
    const int n_embd = llama_model_n_embd_out(edge_ctx->model);
    if (n_embd <= 0) stop("Failed to get embedding dimension from model");
    // A reranker pools to n_cls_out scores, not n_embd values, so reading an
    // embedding from it would run past the end of the buffer.
    if (llama_pooling_type(edge_ctx->ctx) == LLAMA_POOLING_TYPE_RANK) {
      stop("This model is a reranker: it scores query and document pairs and "
           "does not produce embeddings");
    }

    const int n_texts = static_cast<int>(texts.size());
    NumericMatrix result(n_texts, n_embd);

    // Tokenize every text first. A text longer than one batch is cut to the
    // batch size, since llama_decode() takes at most n_batch tokens.
    const int n_batch = static_cast<int>(llama_n_batch(edge_ctx->ctx));
    std::vector<std::vector<llama_token>> toks(n_texts);
    for (int t = 0; t < n_texts; ++t) {
      const std::string& text = texts[t];
      const int n_tokens = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), NULL, 0, true, true);
      if (n_tokens <= 0) {
        warning("Failed to tokenize text at index " + std::to_string(t + 1) + ", skipping");
        continue;
      }
      toks[t].resize(n_tokens);
      if (llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), toks[t].data(), (int32_t)n_tokens, true, true) < 0) {
        warning("Failed to tokenize text at index " + std::to_string(t + 1) + ", skipping");
        toks[t].clear();
        continue;
      }
      if (n_tokens > n_batch) toks[t].resize(n_batch);
    }

    // Several texts go through one decode, each as its own sequence, as many
    // as fit in a batch. One decode per text left most of the CPU idle on
    // short texts such as codelist terms.
    const int n_seq = std::max<int>(1, static_cast<int>(llama_n_seq_max(edge_ctx->ctx)));
    const bool has_encoder = llama_model_has_encoder(edge_ctx->model);
    const enum llama_pooling_type pooling = llama_pooling_type(edge_ctx->ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);

    auto store = [&](int t, const float * embd) {
      double norm = 0.0;
      if (normalize) {
        for (int i = 0; i < n_embd; ++i) norm += static_cast<double>(embd[i]) * embd[i];
        norm = std::sqrt(norm);
      }
      for (int i = 0; i < n_embd; ++i) {
        result(t, i) = !normalize ? embd[i] : (norm > 0.0 ? embd[i] / norm : 0.0);
      }
    };

    int t = 0;
    while (t < n_texts) {
      Rcpp::checkUserInterrupt();
      llama_memory_t mem = llama_get_memory(edge_ctx->ctx);
      if (mem) llama_memory_clear(mem, true);
      edge_ctx->cache_valid = false;
      edge_ctx->cached_tokens.clear();

      // Fill the batch: texts t .. t + group - 1, one sequence each.
      std::vector<int> members, last_index;
      batch.n_tokens = 0;
      while (t < n_texts && static_cast<int>(members.size()) < n_seq) {
        const auto & tk = toks[t];
        if (tk.empty()) { ++t; continue; }
        if (!members.empty() && batch.n_tokens + static_cast<int>(tk.size()) > n_batch) break;
        const int seq = static_cast<int>(members.size());
        for (size_t i = 0; i < tk.size(); ++i) {
          const int j = batch.n_tokens++;
          batch.token[j] = tk[i];
          batch.pos[j] = static_cast<llama_pos>(i);
          batch.n_seq_id[j] = 1;
          batch.seq_id[j][0] = seq;
          batch.logits[j] = 1;
        }
        members.push_back(t);
        last_index.push_back(batch.n_tokens - 1);
        ++t;
      }
      if (members.empty()) break;

      const int rc = has_encoder ? llama_encode(edge_ctx->ctx, batch) : llama_decode(edge_ctx->ctx, batch);
      if (rc != 0) {
        for (int m : members) warning("Failed to process text at index " + std::to_string(m + 1) + ", skipping");
        continue;
      }

      for (size_t s = 0; s < members.size(); ++s) {
        const float * embd = nullptr;
        if (pooling != LLAMA_POOLING_TYPE_NONE) {
          embd = llama_get_embeddings_seq(edge_ctx->ctx, static_cast<llama_seq_id>(s));
        }
        if (!embd) embd = llama_get_embeddings_ith(edge_ctx->ctx, last_index[s]);
        if (embd) {
          store(members[s], embd);
        } else {
          warning("Failed to extract embeddings for text at index " + std::to_string(members[s] + 1));
        }
      }
    }
    llama_batch_free(batch);

    return result;

  } catch (const std::exception& e) {
    stop("Error during embedding extraction: " + std::string(e.what()));
  }
}

// Changes the scales of the LoRA adapters loaded with the model. The KV
// cache was computed with the old weights, so it is cleared.
// [[Rcpp::export]]
Rcpp::NumericVector edge_lora_scales_internal(SEXP model_ptr, Rcpp::Nullable<Rcpp::NumericVector> scales = R_NilValue) {
  if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
  XPtr<EdgeModelContext> edge_ctx(model_ptr);
  if (!edge_ctx->is_valid()) stop("Invalid model context");
  if (scales.isNotNull()) {
    Rcpp::NumericVector s(scales);
    if (static_cast<size_t>(s.size()) != edge_ctx->loras.size()) {
      stop("Expected " + std::to_string(edge_ctx->loras.size()) +
           " LoRA scale(s), one per loaded adapter, got " + std::to_string(s.size()));
    }
    for (R_xlen_t i = 0; i < s.size(); ++i) edge_ctx->lora_scales[i] = static_cast<float>(s[i]);
    clear_kv_cache(edge_ctx->ctx);
    edge_ctx->cache_valid = false;
    edge_ctx->cached_tokens.clear();
    if (!apply_loras(*edge_ctx)) stop("Failed to apply LoRA adapters to the context");
  }
  return Rcpp::NumericVector(edge_ctx->lora_scales.begin(), edge_ctx->lora_scales.end());
}

// [[Rcpp::export]]
int edge_n_parallel_internal(SEXP model_ptr) {
  if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
  XPtr<EdgeModelContext> edge_ctx(model_ptr);
  if (!edge_ctx->is_valid()) stop("Invalid model context");
  return edge_ctx->n_parallel;
}

// [[Rcpp::export]]
int edge_model_n_embd_internal(SEXP model_ptr) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    if (!edge_ctx->is_valid()) stop("Invalid model context");
    return llama_model_n_embd_out(edge_ctx->model);
  } catch (const std::exception& e) {
    stop("Error getting embedding dimension: " + std::string(e.what()));
  }
}

// Tokens of text with the model's tokenizer, parsing special tokens the way
// the completion functions do.
// [[Rcpp::export]]
IntegerVector edge_tokenize_internal(SEXP model_ptr, std::string text, bool add_special = true) {
  if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
  XPtr<EdgeModelContext> edge_ctx(model_ptr);
  if (!edge_ctx->is_valid() || edge_ctx->model == nullptr) stop("Invalid model context");
  const struct llama_vocab* vocab = llama_model_get_vocab(edge_ctx->model);
  if (!vocab) stop("Failed to get vocabulary from model");
  const int n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), NULL, 0, add_special, true);
  if (n <= 0) return IntegerVector(0);
  std::vector<llama_token> tokens(n);
  if (llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), tokens.data(), n, add_special, true) < 0) {
    stop("Failed to tokenize text");
  }
  return IntegerVector(tokens.begin(), tokens.end());
}

// [[Rcpp::export]]
std::string edge_chat_apply_template_internal(SEXP model_ptr, List messages, bool add_generation_prompt = true) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) {
      stop("Invalid model context");
    }
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    if (!edge_ctx->is_valid()) stop("Invalid model context");

    // Get the model's chat template
    const char* tmpl = llama_model_chat_template(edge_ctx->model, NULL);
    std::string tmpl_str = tmpl ? std::string(tmpl) : "";

    int n_msg = messages.size();
    std::vector<llama_chat_message> chat(n_msg);
    std::vector<std::string> roles(n_msg);
    std::vector<std::string> contents(n_msg);

    for (int i = 0; i < n_msg; ++i) {
      List msg = messages[i];
      roles[i] = as<std::string>(msg["role"]);
      contents[i] = as<std::string>(msg["content"]);
      chat[i].role = roles[i].c_str();
      chat[i].content = contents[i].c_str();
    }

    // First call to get required buffer size
    int32_t needed = llama_chat_apply_template(
      tmpl_str.empty() ? NULL : tmpl_str.c_str(),
      chat.data(), n_msg, add_generation_prompt, NULL, 0);

    if (needed < 0) {
      // Template not supported, fall back to generic ChatML format
      std::string result;
      for (int i = 0; i < n_msg; ++i) {
        result += "<|im_start|>" + roles[i] + "\n" + contents[i] + "<|im_end|>\n";
      }
      if (add_generation_prompt) {
        result += "<|im_start|>assistant\n";
      }
      return result;
    }

    std::vector<char> buf(needed + 1);
    llama_chat_apply_template(
      tmpl_str.empty() ? NULL : tmpl_str.c_str(),
      chat.data(), n_msg, add_generation_prompt, buf.data(), buf.size());

    return std::string(buf.data(), needed);
  } catch (const std::exception& e) {
    stop("Error applying chat template: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
std::string edge_model_chat_template_internal(SEXP model_ptr) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    if (!edge_ctx->is_valid()) stop("Invalid model context");

    const char* tmpl = llama_model_chat_template(edge_ctx->model, NULL);
    return tmpl ? std::string(tmpl) : "";
  } catch (const std::exception& e) {
    stop("Error getting chat template: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
void set_llama_logging(bool enabled) {
  g_logging_enabled = enabled;
  // Also control general console output suppression
  g_suppress_console_output = !enabled;
  // Re-set the callback to ensure it takes effect
  llama_log_set(quiet_log_callback, NULL);
}

// Test hook (not exported from NAMESPACE): fires GGML_ABORT on R's thread so
// the test suite can check it becomes an R error instead of a crash.
// [[Rcpp::export]]
void ggml_abort_test_internal() {
  GGML_ABORT("edgemodelr abort-callback self-test");
}

// Package initialization function - called when the package is loaded
// [[Rcpp::init]]
void edgemodelr_init(DllInfo *dll) {
  // Don't initialize llama.cpp during package loading to avoid segfaults
  // Initialization will happen lazily when first needed

  // Installing the abort callback only stores a function pointer, so it is
  // safe here, and it covers every later ggml call.
  g_r_main_thread = std::this_thread::get_id();
  ggml_set_abort_callback(r_abort_callback);
}