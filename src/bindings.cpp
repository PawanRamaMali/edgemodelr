// Suppress warnings from ggml/llama.cpp code
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable" 
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include <Rcpp.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <cstdio>

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "r_output_redirect.h"

// Restore compiler warnings
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

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

// Custom log callback to suppress output
void quiet_log_callback(ggml_log_level level, const char * text, void * user_data) {
  // Only output critical errors using R's error system, suppress all other output
  if (g_logging_enabled && level >= GGML_LOG_LEVEL_ERROR) {
    // Use R's warning system instead of direct stderr output
    Rcpp::warning(std::string("llama.cpp error: ") + text);
  }
  // Otherwise, completely suppress output
}

struct EdgeModelContext {
  struct llama_model* model = NULL;
  struct llama_context* ctx = NULL;
  
  EdgeModelContext() = default;
  
  ~EdgeModelContext() {
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
  }
  
  bool is_valid() const {
    return model != NULL && ctx != NULL;
  }
};

// [[Rcpp::export]]
SEXP edge_load_model_internal(std::string model_path, int n_ctx = 2048, int n_gpu_layers = 0) {
  try {
    // Set up quiet logging before any llama operations
    llama_log_set(quiet_log_callback, NULL);
    
    // Load all available backends (including CPU)
    ggml_backend_load_all();
    
    // Initialize llama backend
    llama_backend_init();
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    
    struct llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
      stop("Failed to load GGUF model from: " + model_path);
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = 512;
    ctx_params.n_threads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
    
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
    
    if (!edge_ctx->is_valid()) {
      stop("Invalid model context");
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
    
    // Create initial batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    
    // Process the prompt
    if (llama_decode(edge_ctx->ctx, batch)) {
      stop("Failed to process prompt");
    }
    
    std::string result = prompt;  // Start with the original prompt
    result.reserve(prompt.size() + n_predict * 4);
    
    // Generate tokens one by one using simple greedy sampling
    for (int i = 0; i < n_predict; ++i) {
      // Get logits from the context
      const float* logits = llama_get_logits_ith(edge_ctx->ctx, -1);
      if (!logits) {
        break;
      }
      
      // Get vocabulary size using the correct API
      const int n_vocab = llama_vocab_n_tokens(vocab);
      llama_token new_token = 0;
      float max_logit = logits[0];
      
      for (int j = 1; j < n_vocab; ++j) {
        if (logits[j] > max_logit) {
          max_logit = logits[j];
          new_token = j;
        }
      }
      
      // Check if it's end of generation
      if (llama_vocab_is_eog(vocab, new_token)) {
        break;
      }
      
      // Convert token to text
      char piece[256];
      int n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
      
      if (n_chars > 0) {
        result.append(piece, n_chars);
      }
      
      // Prepare next batch with the new token
      batch = llama_batch_get_one(&new_token, 1);
      
      // Process the new token
      if (llama_decode(edge_ctx->ctx, batch)) {
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
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    
    if (edge_ctx->ctx) {
      llama_free(edge_ctx->ctx);
      edge_ctx->ctx = NULL;
    }
    if (edge_ctx->model) {
      llama_model_free(edge_ctx->model);
      edge_ctx->model = NULL;
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
    
    if (!edge_ctx->is_valid()) {
      stop("Invalid model context");
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
    
    // Create initial batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    
    // Process the prompt
    if (llama_decode(edge_ctx->ctx, batch)) {
      stop("Failed to process prompt");
    }
    
    std::string full_response = prompt;  // Track full response
    std::vector<std::string> tokens_generated;
    int tokens_count = 0;
    bool stopped_early = false;
    
    // Generate tokens one by one and stream them
    for (int i = 0; i < n_predict; ++i) {
      // Get logits from the context
      const float* logits = llama_get_logits_ith(edge_ctx->ctx, -1);
      if (!logits) {
        stopped_early = true;
        break;
      }
      
      // Get vocabulary size using the correct API
      const int n_vocab = llama_vocab_n_tokens(vocab);
      llama_token new_token = 0;
      float max_logit = logits[0];
      
      // Simple greedy sampling (could be enhanced with temperature/top_p)
      for (int j = 1; j < n_vocab; ++j) {
        if (logits[j] > max_logit) {
          max_logit = logits[j];
          new_token = j;
        }
      }
      
      // Check if it's end of generation
      if (llama_vocab_is_eog(vocab, new_token)) {
        stopped_early = true;
        break;
      }
      
      // Convert token to text
      char piece[256];
      int n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
      
      std::string token_text = "";
      if (n_chars > 0) {
        token_text = std::string(piece, n_chars);
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
      }
      
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
void set_llama_logging(bool enabled) {
  g_logging_enabled = enabled;
  // Also control general console output suppression
  g_suppress_console_output = !enabled;
  // Re-set the callback to ensure it takes effect
  llama_log_set(quiet_log_callback, NULL);
}