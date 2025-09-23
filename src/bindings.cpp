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

    // Load all available backends (including CPU)
    ggml_backend_load_all();

    // Initialize llama backend
    llama_backend_init();

    // Register CPU backend explicitly
    ggml_backend_register(ggml_backend_cpu_reg());

    initialized = true;
  }
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
      stop("Failed to load GGUF model from: " + model_path + ". The file exists but llama.cpp cannot parse it. Check if it's a valid GGUF file.");
    }
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = std::min(2048, n_ctx / 4);  // Larger batch for better throughput
    ctx_params.n_threads = std::max(1, (int)std::thread::hardware_concurrency());  // Use all available cores
    
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
    result.reserve(prompt.size() + n_predict * 8);  // Reserve more space to avoid reallocations
    
    // Create a sampler chain for better token generation
    auto sampler_chain_params = llama_sampler_chain_default_params();
    auto * sampler = llama_sampler_chain_init(sampler_chain_params);

    // Add samplers in the right order for quality generation
    if (top_p < 1.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    }
    if (temperature > 0.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
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
      
      // Convert token to text
      char piece[256];
      int n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, true);
      
      if (n_chars > 0) {
        result.append(piece, n_chars);
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

    // Create sampler chain for streaming
    auto sampler_chain_params = llama_sampler_chain_default_params();
    auto * sampler = llama_sampler_chain_init(sampler_chain_params);

    // Add samplers for quality generation
    if (top_p < 1.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    }
    if (temperature > 0.0f) {
      llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
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
  // Call the helper to ensure initialization
  ensure_llama_initialized();
}