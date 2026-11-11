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
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>

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

// Thread that R runs on; recorded at package load. Only this thread may
// signal an R error or warning.
static std::thread::id g_r_main_thread;

// Custom log callback to suppress output
void quiet_log_callback(ggml_log_level level, const char * text, void * user_data) {
  // Only output critical errors using R's error system, suppress all other output
  if (g_logging_enabled && level >= GGML_LOG_LEVEL_ERROR &&
      std::this_thread::get_id() == g_r_main_thread) {
    // Use R's warning system instead of direct stderr output
    Rcpp::warning(std::string("llama.cpp error: ") + text);
  }
  // Otherwise, completely suppress output
}

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

struct ServeEngine;
static void engine_detach(ServeEngine * engine);

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

  // The serving engine using this context, if any (see ServeEngine).
  ServeEngine * engine = nullptr;

  EdgeModelContext() = default;

  // Copy constructor and assignment deleted to prevent double-free
  EdgeModelContext(const EdgeModelContext&) = delete;
  EdgeModelContext& operator=(const EdgeModelContext&) = delete;

  ~EdgeModelContext() {
    cleanup();
  }

  void cleanup() {
    // An engine thread may be decoding with the context: stop it first.
    if (engine) engine_detach(engine);
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

static void check_no_engine(const EdgeModelContext & e);

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
    check_no_engine(*edge_ctx);

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
    check_no_engine(*edge_ctx);

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
    check_no_engine(*edge_ctx);

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
static llama_token sample_tokens(llama_sampler * chain_s, llama_sampler * grammar_s,
                                 llama_context * ctx, int32_t idx,
                                 std::vector<llama_token_data> & candidates,
                                 bool plain_greedy = false) {
  if (!grammar_s) return llama_sampler_sample(chain_s, ctx, idx);
  const float * logits = llama_get_logits_ith(ctx, idx);
  const int n_vocab = static_cast<int>(candidates.size());
  auto fill = [&]() {
    for (int t = 0; t < n_vocab; ++t) candidates[t] = llama_token_data{t, logits[t], 0.0f};
    return llama_token_data_array{candidates.data(), static_cast<size_t>(n_vocab), -1, false};
  };
  llama_token_data_array cur = fill();
  llama_sampler_apply(chain_s, &cur);
  llama_token token = cur.data[cur.selected].id;
  llama_token_data single = {token, 1.0f, 0.0f};
  llama_token_data_array one = {&single, 1, -1, false};
  llama_sampler_apply(grammar_s, &one);
  if (std::isinf(single.logit) && single.logit < 0 && plain_greedy) {
    // Greedy with no other step: the answer is the allowed token with the
    // highest logit, so look among the top ones before the whole vocabulary,
    // which is slow with large vocabularies (262k tokens for Gemma 3).
    const size_t k = std::min<size_t>(512, static_cast<size_t>(n_vocab));
    cur = fill();
    // Ties go to the lower id, as in llama.cpp's greedy sampler.
    auto by_logit = [](const llama_token_data & a, const llama_token_data & b) {
      return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };
    std::nth_element(cur.data, cur.data + (k - 1), cur.data + n_vocab, by_logit);
    std::sort(cur.data, cur.data + k, by_logit);
    llama_token_data_array top = {cur.data, k, -1, true};
    llama_sampler_apply(grammar_s, &top);
    for (size_t j = 0; j < k; ++j) {
      if (!(std::isinf(top.data[j].logit) && top.data[j].logit < 0)) {
        token = top.data[j].id;
        llama_sampler_accept(grammar_s, token);
        llama_sampler_accept(chain_s, token);
        return token;
      }
    }
  }
  if (std::isinf(single.logit) && single.logit < 0) {
    cur = fill();
    llama_sampler_apply(grammar_s, &cur);
    llama_sampler_apply(chain_s, &cur);
    token = cur.data[cur.selected].id;
  }
  llama_sampler_accept(grammar_s, token);
  llama_sampler_accept(chain_s, token);
  return token;
}

static llama_token batch_sample(BatchSeq & s, llama_context * ctx, int32_t idx,
                                std::vector<llama_token_data> & candidates) {
  return sample_tokens(s.chain.get(), s.grammar.get(), ctx, idx, candidates);
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
    check_no_engine(*edge_ctx);
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

// ---------------------------------------------------------------------------
// Serving engine. One background thread runs many requests on one context,
// the way llama.cpp's server does: each slot is a sequence in a unified KV
// cache, and every step decodes the next token of each generating slot plus
// prompt tokens of slots still reading their prompt, in a single batch. On
// CPU a step for many sequences costs little more than a step for one, so
// total throughput grows with the number of slots. A finished slot keeps its
// tokens, and a later request that starts the same way (the same system
// prompt, or the next turn of the same chat) skips that part of its prompt.
//
// The thread never calls R. R submits requests, polls them and cancels them;
// everything both sides touch is guarded by the engine's mutex. While an
// engine runs, the model's context belongs to it, and the direct completion
// functions refuse to use it (check_no_engine).

struct ServeEngine;
// Engines that have a running thread. Only R's thread touches this.
static std::vector<ServeEngine *> g_engines;

static double engine_now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct EngineRequest {
  int id = 0;
  std::vector<llama_token> prompt;
  int n_predict = 0;
  SamplingOptions opts;
  bool plain_greedy = false;   // temperature 0 and no repeat penalty
  sampler_ptr chain;
  sampler_ptr grammar;
  grammar_ptr shadow;        // the grammar's state, to find text it forces
  // Guarded by ServeEngine::mu.
  int status = 0;            // 0 queued, 1 running, 2 done
  bool cancel = false;
  bool released = false;     // R no longer wants it: forget it when it ends
  std::string text;
  int n_generated = 0;
  int n_cached = 0;          // prompt tokens reused from a slot
  std::string finish;        // stop, length, cancelled, error
  std::string error;
  double t_submit = 0, t_start = 0, t_first = 0, t_end = 0;
};

struct EngineSlot {
  llama_seq_id seq = 0;
  std::shared_ptr<EngineRequest> req;   // null when idle
  std::vector<llama_token> cache;       // tokens in this sequence's KV cells
  size_t n_prompt_done = 0;             // prompt tokens of req in the cache
  llama_token last = 0;                 // sampled, not yet decoded
  bool pending = false;
  std::vector<llama_token> forced;      // forced by the grammar, after last
  int32_t i_batch = -1;                 // row of this slot's logits
  double last_used = 0;
};

struct ServeEngine {
  EdgeModelContext * owner = nullptr;
  llama_context * ctx = nullptr;
  const llama_vocab * vocab = nullptr;
  int n_ctx = 0;
  int n_batch = 0;
  int max_queue = 0;
  double t_started = 0;

  std::mutex mu;
  std::condition_variable cv;
  std::deque<std::shared_ptr<EngineRequest>> queue;
  std::map<int, std::shared_ptr<EngineRequest>> requests;
  std::vector<EngineSlot> slots;
  std::thread worker;
  bool stopping = false;
  int next_id = 1;

  // Counters, guarded by mu.
  long long n_done = 0, n_failed = 0, n_cancelled = 0;
  long long tokens_generated = 0, tokens_prompt = 0, tokens_reused = 0;
  double busy_seconds = 0;
  double sample_seconds = 0;
  long long tokens_forced = 0;
  long long kv_tokens = 0;   // tokens in the slots' caches, updated by the worker
  int n_active = 0;

  std::vector<llama_token_data> candidates;   // worker only

  ~ServeEngine() { stop(); }

  void stop() {
    g_engines.erase(std::remove(g_engines.begin(), g_engines.end(), this), g_engines.end());
    {
      std::lock_guard<std::mutex> lock(mu);
      stopping = true;
    }
    cv.notify_all();
    if (worker.joinable()) worker.join();
    std::lock_guard<std::mutex> lock(mu);
    for (auto & kv : requests) {
      if (kv.second->status != 2) {
        kv.second->status = 2;
        kv.second->finish = "error";
        kv.second->error = "The engine was stopped";
        kv.second->t_end = engine_now();
      }
    }
    queue.clear();
    for (EngineSlot & s : slots) {
      s.req.reset();
      s.cache.clear();
    }
    if (ctx) clear_kv_cache(ctx);
    if (owner) owner->engine = nullptr;
    owner = nullptr;
    ctx = nullptr;
  }

  // Ends a request; mu must be held.
  void finish_locked(EngineSlot & s, const std::string & why, const std::string & err = "") {
    std::shared_ptr<EngineRequest> r = s.req;
    if (!r) return;
    r->status = 2;
    r->finish = why;
    r->error = err;
    r->t_end = engine_now();
    if (why == "error") n_failed++;
    else if (why == "cancelled") n_cancelled++;
    else n_done++;
    s.req.reset();
    s.pending = false;
    s.last_used = r->t_end;
    n_active--;
    if (r->released) requests.erase(r->id);
  }

  void drop_seq(EngineSlot & s) {
    llama_memory_seq_rm(llama_get_memory(ctx), s.seq, -1, -1);
    s.cache.clear();
  }

  // Moves queued requests into free slots, first come first served, while
  // the KV cache has room for each one's prompt and longest answer.
  // mu must be held.
  void admit_locked() {
    llama_memory_t mem = llama_get_memory(ctx);
    while (!queue.empty()) {
      std::shared_ptr<EngineRequest> r = queue.front();
      if (r->cancel) {
        queue.pop_front();
        r->status = 2;
        r->finish = "cancelled";
        r->t_end = engine_now();
        n_cancelled++;
        if (r->released) requests.erase(r->id);
        continue;
      }
      // The idle slot sharing the longest prefix with the prompt; ties go to
      // the one unused longest.
      int best = -1;
      size_t best_keep = 0;
      for (size_t i = 0; i < slots.size(); ++i) {
        const EngineSlot & s = slots[i];
        if (s.req) continue;
        size_t keep = 0;
        const size_t n = std::min(s.cache.size(), r->prompt.size());
        while (keep < n && s.cache[keep] == r->prompt[keep]) ++keep;
        if (best < 0 || keep > best_keep ||
            (keep == best_keep && s.last_used < slots[best].last_used)) {
          best = static_cast<int>(i);
          best_keep = keep;
        }
      }
      if (best < 0) return;   // every slot is busy
      EngineSlot & slot = slots[best];

      // Room: running requests hold their prompt plus their longest answer;
      // idle slots hold what they cached, and give it up when needed.
      const long long need = static_cast<long long>(r->prompt.size()) + r->n_predict;
      long long used = 0;
      for (size_t i = 0; i < slots.size(); ++i) {
        if (static_cast<int>(i) == best) continue;
        const EngineSlot & s = slots[i];
        used += s.req ? static_cast<long long>(s.req->prompt.size()) + s.req->n_predict
                      : static_cast<long long>(s.cache.size());
      }
      while (used + need > n_ctx) {
        int lru = -1;
        for (size_t i = 0; i < slots.size(); ++i) {
          if (static_cast<int>(i) == best || slots[i].req || slots[i].cache.empty()) continue;
          if (lru < 0 || slots[i].last_used < slots[lru].last_used) lru = static_cast<int>(i);
        }
        if (lru < 0) break;
        used -= static_cast<long long>(slots[lru].cache.size());
        drop_seq(slots[lru]);
      }
      if (used + need > n_ctx) return;   // wait for running requests to finish

      size_t keep = best_keep;
      if (keep == r->prompt.size() && keep > 0) --keep;   // the last token gives the logits
      if (keep > 0 && !llama_memory_seq_rm(mem, slot.seq, static_cast<llama_pos>(keep), -1)) keep = 0;
      const int32_t n_swa = llama_model_n_swa(owner->model);
      if (keep > 0 && n_swa > 0 &&
          llama_memory_seq_pos_min(mem, slot.seq) > std::max<llama_pos>(0, static_cast<llama_pos>(keep) - n_swa)) {
        keep = 0;
      }
      if (keep == 0) drop_seq(slot);
      slot.cache.resize(keep);
      slot.n_prompt_done = keep;
      slot.pending = false;
      slot.req = r;
      r->status = 1;
      r->n_cached = static_cast<int>(keep);
      r->t_start = engine_now();
      tokens_reused += static_cast<long long>(keep);
      n_active++;
      queue.pop_front();
    }
  }

  // Adds a sampled token to the slot's request. Returns false once it is
  // finished. mu must be held.
  bool take_locked(EngineSlot & s, llama_token token) {
    // Held here: finishing a released request drops it from the map.
    const std::shared_ptr<EngineRequest> keep = s.req;
    EngineRequest & r = *keep;
    if (r.n_generated == 0) r.t_first = engine_now();
    r.n_generated++;
    tokens_generated++;
    if (llama_vocab_is_eog(vocab, token)) {
      finish_locked(s, "stop");
      return false;
    }
    char buf[512];
    int n_chars = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
    if (n_chars > 0) {
      const size_t before = r.text.size();
      r.text.append(buf, n_chars);
      if (!r.opts.stop.empty()) {
        const size_t at = find_stop(r.text, r.opts.stop, before);
        if (at != std::string::npos) {
          r.text.resize(at);
          finish_locked(s, "stop");
          return false;
        }
      }
    }
    if (r.n_generated >= r.n_predict) {
      finish_locked(s, "length");
      return false;
    }
    s.last = token;
    s.pending = true;
    s.forced.clear();
    if (r.shadow) jump_forward_locked(s);
    return r.status != 2;
  }

  // Text the grammar allows in only one way (JSON keys and separators) is
  // taken without sampling and decoded with the next token in one step, as
  // edge_completion_grammar_internal() does. mu must be held.
  void jump_forward_locked(EngineSlot & s) {
    EngineRequest & r = *s.req;
    llama_grammar_accept_impl(*r.shadow, s.last);
    const std::string text = grammar_forced_text(*r.shadow);
    if (text.empty()) return;
    std::vector<llama_token> ft(text.size() + 1);
    int n_ft = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), ft.data(),
                              (int32_t)ft.size(), false, false);
    // The last token is left to the model: it may merge with what follows.
    if (n_ft > 0) n_ft -= 1;
    if (n_ft <= 0 || r.n_generated + n_ft >= r.n_predict) return;
    ft.resize(n_ft);
    std::string spelled;
    for (llama_token tk : ft) {
      char buf[256];
      const int n = llama_token_to_piece(vocab, tk, buf, sizeof(buf), 0, false);
      if (n < 0) return;
      spelled.append(buf, n);
    }
    if (spelled.empty() || text.compare(0, spelled.size(), spelled) != 0) return;
    for (llama_token tk : ft) {
      llama_sampler_accept(r.grammar.get(), tk);
      llama_grammar_accept_impl(*r.shadow, tk);
      llama_sampler_accept(r.chain.get(), tk);
    }
    r.n_generated += n_ft;
    tokens_generated += n_ft;
    tokens_forced += n_ft;
    const size_t before = r.text.size();
    r.text += spelled;
    if (!r.opts.stop.empty()) {
      const size_t at = find_stop(r.text, r.opts.stop, before);
      if (at != std::string::npos) {
        r.text.resize(at);
        finish_locked(s, "stop");
        return;
      }
    }
    s.forced = ft;
  }

  // One step: build the batch, decode it, sample. Returns false when there
  // was nothing to do.
  bool step(llama_batch & batch, std::vector<int> & row_slot) {
    {
      std::lock_guard<std::mutex> lock(mu);
      for (EngineSlot & s : slots) {
        if (s.req && s.req->cancel) finish_locked(s, "cancelled");
      }
      admit_locked();
    }
    batch.n_tokens = 0;
    row_slot.clear();
    long long n_prompt_tokens = 0;
    for (size_t i = 0; i < slots.size(); ++i) {
      EngineSlot & s = slots[i];
      s.i_batch = -1;
      if (!s.req || !s.pending) continue;
      batch_add(batch, s.last, static_cast<llama_pos>(s.cache.size()), s.seq, s.forced.empty());
      row_slot.push_back(static_cast<int>(i));
      s.cache.push_back(s.last);
      for (size_t k = 0; k < s.forced.size(); ++k) {
        batch_add(batch, s.forced[k], static_cast<llama_pos>(s.cache.size()), s.seq,
                  k + 1 == s.forced.size());
        row_slot.push_back(static_cast<int>(i));
        s.cache.push_back(s.forced[k]);
      }
      s.i_batch = batch.n_tokens - 1;
      s.forced.clear();
      s.pending = false;
    }
    for (size_t i = 0; i < slots.size() && batch.n_tokens < n_batch; ++i) {
      EngineSlot & s = slots[i];
      if (!s.req || s.n_prompt_done >= s.req->prompt.size()) continue;
      const std::vector<llama_token> & p = s.req->prompt;
      while (s.n_prompt_done < p.size() && batch.n_tokens < n_batch) {
        const bool last = s.n_prompt_done + 1 == p.size();
        if (last) s.i_batch = batch.n_tokens;
        batch_add(batch, p[s.n_prompt_done], static_cast<llama_pos>(s.cache.size()), s.seq, last);
        row_slot.push_back(static_cast<int>(i));
        s.cache.push_back(p[s.n_prompt_done]);
        s.n_prompt_done++;
        n_prompt_tokens++;
      }
    }
    if (batch.n_tokens == 0) return false;

    const double t0 = engine_now();
    const int32_t n_tokens = batch.n_tokens;
    int32_t view = std::min(n_tokens, static_cast<int32_t>(n_batch));
    for (int32_t i = 0; i < n_tokens;) {
      const int32_t n = std::min(view, n_tokens - i);
      llama_batch v = {n, batch.token + i, nullptr, batch.pos + i, batch.n_seq_id + i,
                       batch.seq_id + i, batch.logits + i};
      const int ret = llama_decode(ctx, v);
      if (ret == 1 && n > 1) {   // no room for this many at once: try fewer
        view = n / 2;
        continue;
      }
      if (ret != 0) {
        std::lock_guard<std::mutex> lock(mu);
        for (int32_t k = i; k < n_tokens; ++k) {
          EngineSlot & s = slots[row_slot[k]];
          if (!s.req) continue;
          finish_locked(s, "error", ret == 1 ? "The context is full; raise n_ctx" : "Failed to decode");
          drop_seq(s);
        }
        break;
      }
      for (size_t si = 0; si < slots.size(); ++si) {
        EngineSlot & s = slots[si];
        if (!s.req || s.i_batch < i || s.i_batch >= i + n) continue;
        const double ts = engine_now();
        const llama_token token = sample_tokens(s.req->chain.get(), s.req->grammar.get(), ctx,
                                                s.i_batch - i, candidates, s.req->plain_greedy);
        std::lock_guard<std::mutex> lock(mu);
        take_locked(s, token);
        sample_seconds += engine_now() - ts;
      }
      i += n;
    }
    std::lock_guard<std::mutex> lock(mu);
    busy_seconds += engine_now() - t0;
    tokens_prompt += n_prompt_tokens;
    kv_tokens = 0;
    for (const EngineSlot & sl : slots) kv_tokens += static_cast<long long>(sl.cache.size());
    return true;
  }

  void run() {
    // Room for a full batch of prompt tokens plus each slot's next token and
    // the text its grammar forces (grammar_forced_text() stops at 64 bytes).
    BatchOwner owner_batch(n_batch + static_cast<int>(slots.size()) * 66, 1);
    std::vector<int> row_slot;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return stopping || !queue.empty() || n_active > 0; });
        if (stopping) return;
      }
      try {
        if (!step(owner_batch.b, row_slot)) {
          // Nothing could run yet: wait for a submit, a cancel or a stop.
          std::unique_lock<std::mutex> lock(mu);
          cv.wait_for(lock, std::chrono::milliseconds(20));
        }
      } catch (const std::exception & e) {
        std::lock_guard<std::mutex> lock(mu);
        for (EngineSlot & s : slots) {
          finish_locked(s, "error", e.what());
          s.cache.clear();
        }
        clear_kv_cache(ctx);
      }
    }
  }
};

static void engine_detach(ServeEngine * engine) {
  if (engine) engine->stop();
}

static void check_no_engine(const EdgeModelContext & e) {
  if (e.engine) {
    stop("This model is serving requests through an engine (edge_api() or edge_engine()). "
         "Load a second copy with edge_load_model() for direct calls, or stop the engine first.");
  }
}

// Takes cancelled requests out of the queue right away, so they do not hold
// queue places. mu must be held.
static void drop_cancelled_queued(ServeEngine & e) {
  auto it = e.queue.begin();
  while (it != e.queue.end()) {
    std::shared_ptr<EngineRequest> r = *it;
    if (!r->cancel) {
      ++it;
      continue;
    }
    it = e.queue.erase(it);
    r->status = 2;
    r->finish = "cancelled";
    r->t_end = engine_now();
    e.n_cancelled++;
    if (r->released) e.requests.erase(r->id);
  }
}

struct EngineHandle {
  std::shared_ptr<ServeEngine> engine;
};

static void engine_handle_finalizer(SEXP p) {
  EngineHandle * h = static_cast<EngineHandle *>(R_ExternalPtrAddr(p));
  if (!h) return;
  R_ClearExternalPtr(p);
  delete h;
}

static ServeEngine & get_engine(SEXP engine_ptr) {
  if (TYPEOF(engine_ptr) != EXTPTRSXP) stop("Invalid engine");
  XPtr<EngineHandle> h(engine_ptr);
  if (!h.get() || !h->engine || !h->engine->ctx) stop("The engine is stopped");
  return *h->engine;
}

// [[Rcpp::export]]
SEXP edge_engine_start_internal(SEXP model_ptr, int n_slots, int max_queue) {
  if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
  XPtr<EdgeModelContext> edge_ctx(model_ptr);
  if (!edge_ctx->is_valid()) stop("Invalid model context");
  if (edge_ctx->engine) stop("This model already runs an engine");
  llama_context * ctx = edge_ctx->ctx;
  const int n_seq_max = static_cast<int>(llama_n_seq_max(ctx));
  if (n_slots < 1 || n_slots > n_seq_max) {
    stop("n_slots must be from 1 to " + std::to_string(n_seq_max) +
         "; load the model with n_parallel = n_slots");
  }
  if (max_queue < 1) stop("max_queue must be positive");

  auto engine = std::make_shared<ServeEngine>();
  engine->owner = edge_ctx.get();
  engine->ctx = ctx;
  engine->vocab = llama_model_get_vocab(edge_ctx->model);
  engine->n_ctx = static_cast<int>(llama_n_ctx(ctx));
  engine->n_batch = std::max<int>(n_slots, static_cast<int>(llama_n_batch(ctx)));
  engine->max_queue = max_queue;
  engine->t_started = engine_now();
  engine->candidates.resize(llama_vocab_n_tokens(engine->vocab));
  engine->slots.resize(n_slots);
  for (int i = 0; i < n_slots; ++i) engine->slots[i].seq = i;

  // Batches of a few tokens per slot behave like generation, which is
  // fastest on one thread per core: with every hardware thread (the batch
  // default) the engine ran several times slower, worst on a busy machine.
  llama_set_n_threads(ctx, llama_n_threads(ctx), llama_n_threads(ctx));

  // The prompt cache of the direct functions is gone once the slots use
  // the KV cache.
  clear_kv_cache(ctx);
  edge_ctx->cache_valid = false;
  edge_ctx->cached_tokens.clear();
  edge_ctx->engine = engine.get();
  ServeEngine * raw = engine.get();
  engine->worker = std::thread([raw] { raw->run(); });

  g_engines.push_back(raw);

  EngineHandle * h = new EngineHandle();
  h->engine = engine;
  // The handle keeps the model alive: it protects the model's pointer. Its
  // finalizer also runs when R exits, so the thread stops before the
  // process tears down what it uses.
  XPtr<EngineHandle> ptr(h, false, R_NilValue, model_ptr);
  R_RegisterCFinalizerEx(ptr, engine_handle_finalizer, TRUE);
  ptr.attr("class") = "edge_engine_ptr";
  return ptr;
}

// Stops every running engine; .onUnload calls it before the DLL goes away.
// [[Rcpp::export]]
int edge_engine_stop_all_internal() {
  const std::vector<ServeEngine *> running = g_engines;
  for (ServeEngine * e : running) e->stop();
  return static_cast<int>(running.size());
}

// [[Rcpp::export]]
int edge_engine_submit_internal(SEXP engine_ptr, std::string prompt, int n_predict, List sampling,
                                std::string grammar_str = "", std::string grammar_root = "root") {
  ServeEngine & e = get_engine(engine_ptr);
  if (n_predict <= 0) stop("n_predict must be positive");
  auto r = std::make_shared<EngineRequest>();
  r->opts = read_sampling(sampling);
  const int n = -llama_tokenize(e.vocab, prompt.c_str(), (int32_t)prompt.size(), NULL, 0, true, true);
  if (n <= 0) stop("Failed to determine prompt token count");
  r->prompt.resize(n);
  if (llama_tokenize(e.vocab, prompt.c_str(), (int32_t)prompt.size(), r->prompt.data(), n, true, true) < 0) {
    stop("Failed to tokenize prompt");
  }
  if (n >= e.n_ctx) {
    stop("Prompt too long (" + std::to_string(n) + " tokens) for the engine's context (" +
         std::to_string(e.n_ctx) + " tokens). Shorten the prompt or raise n_ctx.");
  }
  r->n_predict = std::min(n_predict, e.n_ctx - n);
  r->plain_greedy = r->opts.temperature <= 0.0 &&
                    (r->opts.repeat_penalty == 1.0 || r->opts.repeat_last_n == 0);
  r->chain.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
  add_sampling_steps(r->chain.get(), e.vocab, r->opts);
  if (!grammar_str.empty()) {
    r->grammar.reset(llama_sampler_init_grammar(e.vocab, grammar_str.c_str(), grammar_root.c_str()));
    if (!r->grammar) stop("Failed to parse GBNF grammar. Check grammar syntax.");
    r->shadow.reset(llama_grammar_init_impl(e.vocab, grammar_str.c_str(), grammar_root.c_str(),
                                            false, nullptr, 0, nullptr, 0));
  }
  {
    std::lock_guard<std::mutex> lock(e.mu);
    if (static_cast<int>(e.queue.size()) >= e.max_queue) {
      stop("queue_full: " + std::to_string(e.queue.size()) + " requests are waiting");
    }
    r->id = e.next_id++;
    r->t_submit = engine_now();
    e.requests[r->id] = r;
    e.queue.push_back(r);
  }
  e.cv.notify_one();
  return r->id;
}

static const char * engine_status_name(int status) {
  return status == 0 ? "queued" : status == 1 ? "running" : "done";
}

// The state of each request. Text of an unfinished request stops before a
// split UTF-8 character and before anything that may be the start of a stop
// string. A finished request stays until edge_engine_release_internal().
// [[Rcpp::export]]
List edge_engine_poll_internal(SEXP engine_ptr, IntegerVector ids) {
  if (TYPEOF(engine_ptr) != EXTPTRSXP) stop("Invalid engine");
  XPtr<EngineHandle> h(engine_ptr);
  if (!h.get() || !h->engine) stop("Invalid engine");
  ServeEngine & e = *h->engine;
  const R_xlen_t n = ids.size();
  CharacterVector status(n), text(n), finish(n), error(n);
  IntegerVector n_prompt(n), n_generated(n), n_cached(n), position(n);
  NumericVector queue_s(n), first_s(n), total_s(n);
  std::lock_guard<std::mutex> lock(e.mu);
  const double now = engine_now();
  for (R_xlen_t i = 0; i < n; ++i) {
    auto it = e.requests.find(ids[i]);
    if (it == e.requests.end()) {
      status[i] = "unknown";
      finish[i] = NA_STRING;
      error[i] = NA_STRING;
      position[i] = NA_INTEGER;
      queue_s[i] = first_s[i] = total_s[i] = NA_REAL;
      continue;
    }
    const EngineRequest & r = *it->second;
    status[i] = engine_status_name(r.status);
    if (r.status == 2) {
      text[i] = Rcpp::String(r.text, CE_UTF8);
    } else {
      size_t upto = r.text.size() - stop_holdback(r.text, r.opts.stop);
      upto = utf8_boundary(r.text, upto);
      text[i] = Rcpp::String(r.text.substr(0, upto), CE_UTF8);
    }
    finish[i] = r.status == 2 ? Rcpp::String(r.finish) : Rcpp::String(NA_STRING);
    error[i] = r.error.empty() ? Rcpp::String(NA_STRING) : Rcpp::String(r.error);
    n_prompt[i] = static_cast<int>(r.prompt.size());
    n_generated[i] = r.n_generated;
    n_cached[i] = r.n_cached;
    position[i] = NA_INTEGER;
    if (r.status == 0) {
      int k = 0;
      for (const auto & q : e.queue) {
        ++k;
        if (q->id == r.id) break;
      }
      position[i] = k;
    }
    const double started = r.status == 0 ? now : r.t_start;
    queue_s[i] = started - r.t_submit;
    first_s[i] = r.t_first > 0 ? r.t_first - r.t_submit : NA_REAL;
    total_s[i] = (r.status == 2 ? r.t_end : now) - r.t_submit;
  }
  return List::create(_["id"] = ids, _["status"] = status, _["text"] = text,
                      _["finish"] = finish, _["error"] = error,
                      _["prompt_tokens"] = n_prompt, _["completion_tokens"] = n_generated,
                      _["cached_tokens"] = n_cached, _["queue_position"] = position,
                      _["queue_seconds"] = queue_s, _["first_token_seconds"] = first_s,
                      _["total_seconds"] = total_s);
}

// Forgets finished requests; unfinished ones are cancelled and forgotten
// once they end.
// [[Rcpp::export]]
void edge_engine_release_internal(SEXP engine_ptr, IntegerVector ids) {
  if (TYPEOF(engine_ptr) != EXTPTRSXP) stop("Invalid engine");
  XPtr<EngineHandle> h(engine_ptr);
  if (!h.get() || !h->engine) return;
  ServeEngine & e = *h->engine;
  {
    std::lock_guard<std::mutex> lock(e.mu);
    for (R_xlen_t i = 0; i < ids.size(); ++i) {
      auto it = e.requests.find(ids[i]);
      if (it == e.requests.end()) continue;
      if (it->second->status == 2) {
        e.requests.erase(it);
      } else {
        it->second->cancel = true;
        it->second->released = true;
      }
    }
    drop_cancelled_queued(e);
  }
  e.cv.notify_one();
}

// [[Rcpp::export]]
void edge_engine_cancel_internal(SEXP engine_ptr, IntegerVector ids) {
  if (TYPEOF(engine_ptr) != EXTPTRSXP) stop("Invalid engine");
  XPtr<EngineHandle> h(engine_ptr);
  if (!h.get() || !h->engine) return;
  ServeEngine & e = *h->engine;
  {
    std::lock_guard<std::mutex> lock(e.mu);
    for (R_xlen_t i = 0; i < ids.size(); ++i) {
      auto it = e.requests.find(ids[i]);
      if (it != e.requests.end()) it->second->cancel = true;
    }
    drop_cancelled_queued(e);
  }
  e.cv.notify_one();
}

// [[Rcpp::export]]
List edge_engine_stats_internal(SEXP engine_ptr) {
  if (TYPEOF(engine_ptr) != EXTPTRSXP) stop("Invalid engine");
  XPtr<EngineHandle> h(engine_ptr);
  if (!h.get() || !h->engine) stop("Invalid engine");
  ServeEngine & e = *h->engine;
  std::lock_guard<std::mutex> lock(e.mu);
  return List::create(
    _["running"] = e.ctx != nullptr && e.worker.joinable(),
    _["slots"] = static_cast<int>(e.slots.size()),
    _["active"] = e.n_active,
    _["queued"] = static_cast<int>(e.queue.size()),
    _["max_queue"] = e.max_queue,
    _["n_ctx"] = e.n_ctx,
    _["kv_tokens"] = static_cast<double>(e.kv_tokens),
    _["requests_done"] = static_cast<double>(e.n_done),
    _["requests_failed"] = static_cast<double>(e.n_failed),
    _["requests_cancelled"] = static_cast<double>(e.n_cancelled),
    _["tokens_generated"] = static_cast<double>(e.tokens_generated),
    _["tokens_prompt"] = static_cast<double>(e.tokens_prompt),
    _["tokens_reused"] = static_cast<double>(e.tokens_reused),
    _["busy_seconds"] = e.busy_seconds,
    _["sample_seconds"] = e.sample_seconds,
    _["tokens_forced"] = static_cast<double>(e.tokens_forced),
    _["uptime_seconds"] = engine_now() - e.t_started);
}

// [[Rcpp::export]]
void edge_engine_stop_internal(SEXP engine_ptr) {
  if (TYPEOF(engine_ptr) != EXTPTRSXP) return;
  XPtr<EngineHandle> h(engine_ptr);
  if (h.get() && h->engine) h->engine->stop();
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
  check_no_engine(*edge_ctx);
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
  check_no_engine(*edge_ctx);
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
    check_no_engine(*edge_ctx);

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
    check_no_engine(*edge_ctx);
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

// Sets the threads used for one token at a time and for batches.
// [[Rcpp::export]]
void edge_set_threads_internal(SEXP model_ptr, int n_threads, int n_threads_batch) {
  if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
  XPtr<EdgeModelContext> edge_ctx(model_ptr);
  if (!edge_ctx->is_valid()) stop("Invalid model context");
  if (n_threads < 1 || n_threads_batch < 1) stop("Thread counts must be positive");
  check_no_engine(*edge_ctx);
  llama_set_n_threads(edge_ctx->ctx, n_threads, n_threads_batch);
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