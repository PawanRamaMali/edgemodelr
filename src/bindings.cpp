// Main bindings file - should compile cleanly

#include <Rcpp.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <cstdio>
#include <fstream>

#include "llama.h"
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

// [[Rcpp::export]]
SEXP edge_load_model_internal(std::string model_path, int n_ctx = 2048, int n_gpu_layers = 0, int n_threads = 0, bool flash_attn = true, bool embeddings = false) {
  try {
    // Ensure llama is properly initialized
    ensure_llama_initialized();
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

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

    // Thread configuration: use all hardware threads by default, allow user override
    int hardware_threads = std::max(1, (int)std::thread::hardware_concurrency());
    int effective_threads = (n_threads > 0) ? std::min(n_threads, hardware_threads) : hardware_threads;
    ctx_params.n_threads = effective_threads;
    ctx_params.n_threads_batch = hardware_threads;  // batch processing always benefits from max threads
    // flash_attn was a bool in older llama.cpp; b8179 changed to an enum
    ctx_params.flash_attn_type = flash_attn
      ? LLAMA_FLASH_ATTN_TYPE_ENABLED
      : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.embeddings = embeddings;
    
    struct llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
      llama_model_free(model);
      stop("Failed to create context for model");
    }
    
    auto edge_ctx = std::make_unique<EdgeModelContext>();
    edge_ctx->model = model;
    edge_ctx->ctx = ctx;
    
    XPtr<EdgeModelContext> ptr(edge_ctx.release(), true);
    ptr.attr("class") = "edge_model_context";
    
    return ptr;
  } catch (const std::exception& e) {
    stop("Error loading model: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
std::string edge_completion_internal(SEXP model_ptr, std::string prompt, int n_predict = 128, double temperature = 0.8, double top_p = 0.95) {
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
    if (temperature < 0.0 || temperature > 2.0) {
      stop("Temperature must be between 0.0 and 2.0");
    }
    if (top_p <= 0.0 || top_p > 1.0) {
      stop("top_p must be between 0.0 and 1.0");
    }

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

    // Create initial batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());

    // Process the prompt
    if (llama_decode(edge_ctx->ctx, batch)) {
      stop("Failed to process prompt");
    }

    std::string result;  // Only collect generated text, not prompt
    result.reserve(n_predict * 8);
    
    // Create a sampler chain for better token generation
    auto sampler_chain_params = llama_sampler_chain_default_params();
    auto * sampler = llama_sampler_chain_init(sampler_chain_params);

    // Add samplers in the right order for quality generation
    if (top_p < 1.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(top_p), 1));
    }
    if (temperature > 0.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(temperature)));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(12345)); // Random seed

    // Generate tokens using the new sampler API
    for (int i = 0; i < n_predict; ++i) {
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
        result.append(piece.data(), n_chars);
      } else if (n_chars < 0) {
        // If still failing, log warning and skip this token
        warning("Failed to convert token to text, skipping token");
        continue;
      }
      
      // Accept the token for sampling history
      llama_sampler_accept(sampler, new_token);

      // Prepare next batch with the new token
      batch = llama_batch_get_one(&new_token, 1);

      // Process the new token
      if (llama_decode(edge_ctx->ctx, batch)) {
        break;
      }
    }

    // Clean up sampler
    llama_sampler_free(sampler);

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

// [[Rcpp::export]]
List edge_completion_stream_internal(SEXP model_ptr, std::string prompt, Function callback, int n_predict = 128, double temperature = 0.8, double top_p = 0.95) {
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
    if (temperature < 0.0 || temperature > 2.0) {
      stop("Temperature must be between 0.0 and 2.0");
    }
    if (top_p <= 0.0 || top_p > 1.0) {
      stop("top_p must be between 0.0 and 1.0");
    }

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

    // Create initial batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());

    // Process the prompt
    if (llama_decode(edge_ctx->ctx, batch)) {
      stop("Failed to process prompt");
    }

    std::string full_response;  // Track generated text only
    std::vector<std::string> tokens_generated;
    int tokens_count = 0;
    bool stopped_early = false;

    // Create sampler chain for streaming
    auto sampler_chain_params = llama_sampler_chain_default_params();
    auto * sampler = llama_sampler_chain_init(sampler_chain_params);

    // Add samplers for quality generation
    if (top_p < 1.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(top_p), 1));
    }
    if (temperature > 0.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(temperature)));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(12345)); // Random seed

    // Generate tokens one by one and stream them with proper sampling
    for (int i = 0; i < n_predict; ++i) {
      // Sample next token using the sampler chain
      llama_token new_token = llama_sampler_sample(sampler, edge_ctx->ctx, -1);
      
      // Check if it's end of generation
      if (llama_vocab_is_eog(vocab, new_token)) {
        stopped_early = true;
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

      std::string token_text = "";
      if (n_chars > 0) {
        token_text = std::string(piece.data(), n_chars);
        full_response += token_text;
        tokens_generated.push_back(token_text);
        tokens_count++;
        
        // Call the R callback function with current token
        try {
          List callback_data = List::create(
            Named("token") = token_text,
            Named("position") = i + 1,
            Named("is_final") = false,
            Named("total_tokens") = tokens_count
          );
          
          SEXP result = callback(callback_data);
          
          // Check if callback wants to stop early
          if (is<LogicalVector>(result)) {
            LogicalVector stop_signal = as<LogicalVector>(result);
            if (stop_signal.length() > 0 && stop_signal[0] == false) {
              stopped_early = true;
              break;
            }
          }
        } catch (const std::exception& e) {
          warning("Callback error: " + std::string(e.what()));
        }
      } else if (n_chars < 0) {
        // If still failing, log warning and skip this token
        warning("Failed to convert token to text, skipping token");
        continue;
      }

      // Accept the token for sampling history
      llama_sampler_accept(sampler, new_token);

      // Prepare next batch with the new token
      batch = llama_batch_get_one(&new_token, 1);

      // Process the new token
      if (llama_decode(edge_ctx->ctx, batch)) {
        stopped_early = true;
        break;
      }
    }
    
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

    // Clean up sampler
    llama_sampler_free(sampler);

    // Return summary information
    return List::create(
      Named("full_response") = full_response,
      Named("tokens_generated") = tokens_generated,
      Named("total_tokens") = tokens_count,
      Named("stopped_early") = stopped_early,
      Named("original_prompt") = prompt
    );
    
  } catch (const std::exception& e) {
    stop("Error during streaming completion: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
std::string edge_completion_grammar_internal(SEXP model_ptr, std::string prompt, std::string grammar_str, std::string grammar_root, int n_predict = 512, double temperature = 0.3, double top_p = 0.95) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) {
      stop("Invalid model context");
    }

    XPtr<EdgeModelContext> edge_ctx(model_ptr);

    if (!edge_ctx->is_valid() || edge_ctx->ctx == nullptr || edge_ctx->model == nullptr) {
      stop("Invalid model context or null pointers");
    }

    if (n_predict <= 0) stop("n_predict must be positive");
    if (temperature < 0.0 || temperature > 2.0) stop("Temperature must be between 0.0 and 2.0");
    if (top_p <= 0.0 || top_p > 1.0) stop("top_p must be between 0.0 and 1.0");

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
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    if (llama_decode(edge_ctx->ctx, batch)) stop("Failed to process prompt");

    // Build sampler chain WITH grammar constraint
    auto sampler_chain_params = llama_sampler_chain_default_params();
    auto * sampler = llama_sampler_chain_init(sampler_chain_params);

    // Add grammar sampler first (constrains token selection)
    if (!grammar_str.empty()) {
      auto * grammar_sampler = llama_sampler_init_grammar(vocab, grammar_str.c_str(), grammar_root.c_str());
      if (!grammar_sampler) {
        llama_sampler_free(sampler);
        stop("Failed to parse GBNF grammar. Check grammar syntax.");
      }
      llama_sampler_chain_add(sampler, grammar_sampler);
    }

    // Add standard samplers
    if (top_p < 1.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(top_p), 1));
    }
    if (temperature > 0.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(temperature)));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(12345));

    // Generate tokens (only collect generated text, not prompt)
    std::string result;
    result.reserve(n_predict * 8);

    for (int i = 0; i < n_predict; ++i) {
      llama_token new_token = llama_sampler_sample(sampler, edge_ctx->ctx, -1);

      if (llama_vocab_is_eog(vocab, new_token)) break;

      std::vector<char> piece(512);
      int n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
      if (n_chars < 0) {
        piece.resize(static_cast<size_t>(std::abs(n_chars)) + 1);
        n_chars = llama_token_to_piece(vocab, new_token, piece.data(), static_cast<int32_t>(piece.size()), 0, true);
      }
      if (n_chars > 0) {
        result.append(piece.data(), n_chars);
      }

      llama_sampler_accept(sampler, new_token);
      batch = llama_batch_get_one(&new_token, 1);
      if (llama_decode(edge_ctx->ctx, batch)) break;
    }

    llama_sampler_free(sampler);
    return result;

  } catch (const std::exception& e) {
    stop("Error during grammar completion: " + std::string(e.what()));
  }
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

    const int n_embd = llama_model_n_embd(edge_ctx->model);
    if (n_embd <= 0) stop("Failed to get embedding dimension from model");

    const int n_texts = static_cast<int>(texts.size());
    NumericMatrix result(n_texts, n_embd);

    for (int t = 0; t < n_texts; ++t) {
      // Clear KV cache between texts
      llama_memory_t mem = llama_get_memory(edge_ctx->ctx);
      if (mem) {
        llama_memory_clear(mem, true);
      }

      // Tokenize
      const std::string& text = texts[t];
      const int n_tokens = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), NULL, 0, true, true);
      if (n_tokens <= 0) {
        warning("Failed to tokenize text at index " + std::to_string(t + 1) + ", skipping");
        continue;
      }

      std::vector<llama_token> tokens(n_tokens);
      if (llama_tokenize(vocab, text.c_str(), (int32_t)text.size(), tokens.data(), (int32_t)tokens.size(), true, true) < 0) {
        warning("Failed to tokenize text at index " + std::to_string(t + 1) + ", skipping");
        continue;
      }

      // Check if context size is sufficient
      int n_ctx = llama_n_ctx(edge_ctx->ctx);
      int tokens_to_process = std::min(n_tokens, n_ctx);

      // Create batch with tokens requesting output
      struct llama_batch batch = llama_batch_init(tokens_to_process, 0, 1);
      batch.n_tokens = tokens_to_process;
      for (int i = 0; i < tokens_to_process; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;  // request output for all tokens (needed for embeddings)
      }

      // Use encode for encoder models, decode for decoder-only (generative) models
      bool has_encoder = llama_model_has_encoder(edge_ctx->model);
      int rc;
      if (has_encoder) {
        rc = llama_encode(edge_ctx->ctx, batch);
      } else {
        rc = llama_decode(edge_ctx->ctx, batch);
      }
      if (rc != 0) {
        llama_batch_free(batch);
        warning("Failed to process text at index " + std::to_string(t + 1) + ", skipping");
        continue;
      }

      // Get embeddings - strategy depends on pooling type
      const float* embd = nullptr;

      enum llama_pooling_type pooling = llama_pooling_type(edge_ctx->ctx);
      if (pooling != LLAMA_POOLING_TYPE_NONE) {
        // Pooled models: get sequence-level embedding
        embd = llama_get_embeddings_seq(edge_ctx->ctx, 0);
      }
      if (!embd) {
        // Decoder-only / no pooling: get last token embedding
        embd = llama_get_embeddings_ith(edge_ctx->ctx, tokens_to_process - 1);
      }
      if (!embd) {
        // Final fallback: get all embeddings (first position)
        embd = llama_get_embeddings(edge_ctx->ctx);
      }

      if (embd) {
        if (normalize) {
          // L2 normalize
          double norm = 0.0;
          for (int i = 0; i < n_embd; ++i) {
            norm += static_cast<double>(embd[i]) * static_cast<double>(embd[i]);
          }
          norm = std::sqrt(norm);
          if (norm > 0.0) {
            for (int i = 0; i < n_embd; ++i) {
              result(t, i) = static_cast<double>(embd[i]) / norm;
            }
          } else {
            for (int i = 0; i < n_embd; ++i) {
              result(t, i) = 0.0;
            }
          }
        } else {
          for (int i = 0; i < n_embd; ++i) {
            result(t, i) = static_cast<double>(embd[i]);
          }
        }
      } else {
        warning("Failed to extract embeddings for text at index " + std::to_string(t + 1));
      }

      llama_batch_free(batch);
    }

    return result;

  } catch (const std::exception& e) {
    stop("Error during embedding extraction: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
int edge_model_n_embd_internal(SEXP model_ptr) {
  try {
    if (TYPEOF(model_ptr) != EXTPTRSXP) stop("Invalid model context");
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    if (!edge_ctx->is_valid()) stop("Invalid model context");
    return llama_model_n_embd(edge_ctx->model);
  } catch (const std::exception& e) {
    stop("Error getting embedding dimension: " + std::string(e.what()));
  }
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

// Package initialization function - called when the package is loaded
// [[Rcpp::init]]
void edgemodelr_init(DllInfo *dll) {
  // Don't initialize llama.cpp during package loading to avoid segfaults
  // Initialization will happen lazily when first needed
}