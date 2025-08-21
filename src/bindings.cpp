#include <Rcpp.h>
#include <memory>
#include <string>
#include <vector>
#include <thread>

#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

using namespace Rcpp;

struct EdgeModelContext {
  struct llama_model* model = nullptr;
  struct llama_context* ctx = nullptr;
  
  EdgeModelContext() = default;
  
  ~EdgeModelContext() {
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
  }
  
  bool is_valid() const {
    return model != nullptr && ctx != nullptr;
  }
};

// [[Rcpp::export]]
SEXP edge_load_model(std::string model_path, int n_ctx = 2048, int n_gpu_layers = 0) {
  try {
    // Initialize llama backend first
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
std::string edge_completion(SEXP model_ptr, std::string prompt, int n_predict = 128, double temperature = 0.8, double top_p = 0.95) {
  try {
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
    
    std::string result;
    result.reserve(n_predict * 4);
    
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
void edge_free_model(SEXP model_ptr) {
  try {
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    
    if (edge_ctx->ctx) {
      llama_free(edge_ctx->ctx);
      edge_ctx->ctx = nullptr;
    }
    if (edge_ctx->model) {
      llama_model_free(edge_ctx->model);
      edge_ctx->model = nullptr;
    }
    
  } catch (const std::exception& e) {
    warning("Error freeing model: " + std::string(e.what()));
  }
}

// [[Rcpp::export]]
bool is_valid_model(SEXP model_ptr) {
  try {
    XPtr<EdgeModelContext> edge_ctx(model_ptr);
    return edge_ctx->is_valid();
  } catch (...) {
    return false;
  }
}