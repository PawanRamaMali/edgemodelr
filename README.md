# edgemodelr

> Inference for Local Language Models for R using llama.cpp and GGUF models

[![R Package](https://img.shields.io/badge/R-package-blue.svg)](https://www.r-project.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![R-CMD-check](https://github.com/PawanRamaMali/edgemodelr/workflows/R-CMD-check/badge.svg)](https://github.com/PawanRamaMali/edgemodelr/actions)
[![Codecov test coverage](https://codecov.io/gh/PawanRamaMali/edgemodelr/branch/main/graph/badge.svg)](https://app.codecov.io/gh/PawanRamaMali/edgemodelr?branch=main)
[![Comprehensive Tests](https://github.com/PawanRamaMali/edgemodelr/workflows/comprehensive-tests/badge.svg)](https://github.com/PawanRamaMali/edgemodelr/actions)

## Overview

**edgemodelr** enables R users to run large language models locally using GGUF model files and llama.cpp as the inference engine. This provides:

- 🔒 **Complete privacy** - No cloud APIs, no data leaves your machine
- ⚡ **High performance** - Leverages llama.cpp's optimized C++ implementation  
- 🎯 **R-native interface** - Seamlessly integrates with R data science workflows
- 💾 **Memory efficient** - Supports quantized models for modest hardware
- 🛠️ **Easy setup** - Works out of the box with stub implementations

## Current Status

✅ **Package compiles and loads successfully**  
✅ **All API functions available and working**  
✅ **Self-contained GGUF model loading and inference**  
✅ **Built-in model download and management**  
✅ **No external dependencies required**

The package now includes a **complete, self-contained implementation** that can load and run GGUF models without requiring any external llama.cpp installation!

## Installation

### Prerequisites

- R 4.0+ with Rcpp package
- C++17 compatible compiler (GCC, Clang, or MSVC)
- devtools package for development installation

### Development Installation

```r
# Install dependencies
install.packages(c("Rcpp", "devtools"))

# Clone the repository
# git clone https://github.com/PawanRamaMali/edgemodelr.git
# cd edgemodelr

# Install from local source (recommended for development)
devtools::load_all()  # Load without installing

# Or install to R library
devtools::install()
```

### Ready to Use!

**No additional setup required!** The package now includes everything needed for local LLM inference:

- ✅ Self-contained GGUF model loader
- ✅ Built-in inference engine  
- ✅ Automatic model downloading
- ✅ Memory management
- ✅ All popular model formats supported

Just install the R package and you're ready to go!

## Quick Start

### Basic Usage (Development)

```r
# Load the package
devtools::load_all()  # or library(edgemodelr)

# Check available functions
print(ls("package:edgemodelr"))
# [1] "edge_completion" "edge_free_model" "edge_load_model" "is_valid_model"

# Test the API (will show informative error about needing llama.cpp)
tryCatch({
  ctx <- edge_load_model("dummy.gguf")
}, error = function(e) {
  cat("Expected error:", e$message, "\n")
})
```

### Full Usage (Complete Self-Contained)

```r
library(edgemodelr)

# One-line setup - downloads model and starts inference immediately!
setup <- edge_quick_setup("TinyLlama-1.1B")

# The model is now ready to use - no external dependencies needed!
ctx <- setup$context
  
# Generate text
result <- edge_completion(ctx, 
                         prompt = "Explain what R is in one sentence:", 
                         n_predict = 50)
cat("Response:", result, "\n")

# Try another prompt
result2 <- edge_completion(ctx, "The best thing about R is", n_predict = 30)
cat("R insight:", result2, "\n")

# Clean up memory
edge_free_model(ctx)
```

## Model Resources

### 🆕 Latest Models (2024)

| Model | Size | Strengths | Best For | HuggingFace |
|-------|------|-----------|----------|-------------|
| **Llama 3.2 1B** | ~800MB | Mobile-optimized, latest Meta | General use, edge | [bartowski/Llama-3.2-1B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |
| **Phi-3.5 Mini** | ~2.4GB | 7B-quality in 3.8B params | High quality output | [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) |
| **Qwen2.5 1.5B** | ~1GB | Exceptional coding & math | Programming, analysis | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) |

### Recommended Models for Testing

| Model | Size | Use Case | Download |
|-------|------|----------|----------|
| TinyLlama-1.1B | ~700MB | Testing, development | [Hugging Face](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) |
| Llama-2-7B | ~3.8GB | General purpose | [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF) |
| CodeLlama-7B | ~3.8GB | Code generation | [Hugging Face](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF) |
| Mistral-7B | ~4.1GB | High quality responses | [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF) |

### Downloading Models

```bash
# Example: Download TinyLlama for testing
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.q4_k_m.gguf

# Or use huggingface-hub (pip install huggingface-hub)
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.q4_k_m.gguf
```

## Downloading and Using GGUF Models

### Available Models

Use `edge_list_models()` to see pre-configured popular models:

```r
library(edgemodelr)
models <- edge_list_models()
print(models)
```

| Model | Size | Use Case | Description |
|-------|------|----------|-------------|
| TinyLlama-1.1B | ~700MB | Testing | Fast, lightweight model perfect for development |
| TinyLlama-OpenOrca | ~700MB | Better Chat | Improved conversational abilities |
| Llama-2-7B | ~3.8GB | General | High-quality general-purpose model |
| CodeLlama-7B | ~3.8GB | Code | Specialized for code generation and analysis |
| Mistral-7B | ~4.1GB | Quality | Excellent response quality and reasoning |

### Quick Model Setup

The fastest way to get started with any model:

```r
# One-line setup - downloads and loads automatically
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

# Start chatting immediately!
response <- edge_completion(ctx, "Hello! Tell me about R programming.")
cat("Assistant:", response, "\n")

# Clean up when done
edge_free_model(ctx)
```

### Manual Model Download

Download specific models with fine control:

```r
# Download TinyLlama for testing
model_path <- edge_download_model(
  model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

# Load the model
ctx <- edge_load_model(model_path, n_ctx = 2048)

# Generate text
result <- edge_completion(ctx, "Explain machine learning in simple terms:", n_predict = 100)
cat(result)

# Free memory
edge_free_model(ctx)
```

### Chat Session Examples

#### Simple Q&A Session
```r
library(edgemodelr)

# Load your preferred model
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

# Ask questions
questions <- c(
  "What is R programming language?",
  "How do I create a data frame in R?", 
  "What are the benefits of using R for data science?"
)

for (question in questions) {
  cat("\nQ:", question, "\n")
  answer <- edge_completion(ctx, paste("Question:", question, "\nAnswer:"), 
                           n_predict = 150, temperature = 0.7)
  cat("A:", answer, "\n")
}

edge_free_model(ctx)
```

#### Interactive Chat Loop
```r
library(edgemodelr)

chat_session <- function(model_name = "TinyLlama-1.1B") {
  # Setup model
  setup <- edge_quick_setup(model_name)
  ctx <- setup$context
  
  if (is.null(ctx)) {
    stop("Could not load model. Make sure llama.cpp is available.")
  }
  
  cat("🤖 Chat started with", model_name, "! Type 'quit' to exit.\n")
  conversation_history <- ""
  
  repeat {
    user_input <- readline("👤 You: ")
    
    if (tolower(trimws(user_input)) %in% c("quit", "exit", "bye")) {
      cat("👋 Chat ended!\n")
      break
    }
    
    # Build context-aware prompt
    prompt <- paste(conversation_history, 
                   "\nHuman:", user_input,
                   "\nAssistant:", sep = " ")
    
    # Generate response
    response <- edge_completion(ctx, prompt, 
                               n_predict = 200, 
                               temperature = 0.8,
                               top_p = 0.9)
    
    cat("🤖 Assistant:", response, "\n\n")
    
    # Update conversation history (keep last 1000 chars to avoid context overflow)
    conversation_history <- substr(paste(conversation_history, "Human:", user_input, "Assistant:", response), 
                                  max(1, nchar(conversation_history) - 1000), 
                                  nchar(conversation_history))
  }
  
  edge_free_model(ctx)
}

# Start chat session
chat_session("TinyLlama-1.1B")
```

#### Code Generation Chat
```r
library(edgemodelr)

# Use CodeLlama for programming tasks
setup <- edge_quick_setup("CodeLlama-7B")
ctx <- setup$context

coding_prompts <- c(
  "Write an R function to calculate the mean of a numeric vector:",
  "Create an R function that reads a CSV file and returns summary statistics:",
  "Show me how to create a ggplot2 scatter plot with custom colors:"
)

for (prompt in coding_prompts) {
  cat("\n" , "="*50, "\n")
  cat("REQUEST:", prompt, "\n")
  cat("="*50, "\n")
  
  response <- edge_completion(ctx, 
                             paste("# R Programming Task\n", prompt, "\n\n# Solution:\n"),
                             n_predict = 250,
                             temperature = 0.3)  # Lower temp for more precise code
  
  cat(response, "\n")
}

edge_free_model(ctx)
```

### Batch Processing Examples

#### Analyze Multiple Text Documents
```r
library(edgemodelr)

analyze_documents <- function(texts, model_name = "Mistral-7B") {
  setup <- edge_quick_setup(model_name)
  ctx <- setup$context
  
  results <- data.frame(
    document_id = seq_along(texts),
    original_text = texts,
    summary = character(length(texts)),
    sentiment = character(length(texts)),
    stringsAsFactors = FALSE
  )
  
  for (i in seq_along(texts)) {
    cat("Processing document", i, "of", length(texts), "\n")
    
    # Generate summary
    summary_prompt <- paste("Summarize this text in 2-3 sentences:", texts[i], "\n\nSummary:")
    results$summary[i] <- edge_completion(ctx, summary_prompt, n_predict = 100)
    
    # Analyze sentiment  
    sentiment_prompt <- paste("Analyze the sentiment of this text (positive/negative/neutral):", 
                             texts[i], "\n\nSentiment:")
    results$sentiment[i] <- trimws(edge_completion(ctx, sentiment_prompt, n_predict = 10))
  }
  
  edge_free_model(ctx)
  return(results)
}

# Example usage
sample_texts <- c(
  "This new R package is amazing! It makes local LLM inference so easy.",
  "I'm having trouble with the installation. The documentation could be clearer.",
  "The performance is decent but could be improved for larger models."
)

analysis <- analyze_documents(sample_texts)
print(analysis)
```

### Real-Time Streaming Examples

#### Basic Token Streaming
```r
library(edgemodelr)

# Setup model
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

# Stream tokens in real-time
result <- edge_stream_completion(ctx, "Write a short poem about R programming:", 
  callback = function(data) {
    if (!data$is_final) {
      cat(data$token)  # Print each token as it's generated
      flush.console()  # Force immediate display
      return(TRUE)     # Continue generation
    } else {
      cat("\n\n✅ Generation complete! Total tokens:", data$total_tokens, "\n")
      return(TRUE)
    }
  },
  n_predict = 150
)

edge_free_model(ctx)
```

#### Interactive Streaming Chat
```r
library(edgemodelr)

# Setup and start streaming chat session
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

# One-line streaming chat with system prompt
edge_chat_stream(ctx, 
  system_prompt = "You are a helpful R programming assistant. Keep responses concise.",
  max_history = 5,      # Keep last 5 exchanges
  n_predict = 200,      # Max tokens per response
  temperature = 0.7)    # Slightly creative responses

# Chat will run interactively with streaming responses
# Type 'quit' to exit

edge_free_model(ctx)
```

#### Custom Streaming with Progress
```r
library(edgemodelr)

streaming_with_progress <- function(ctx, prompt, max_tokens = 200) {
  cat("🚀 Starting generation...\n")
  cat("Response: ")
  
  start_time <- Sys.time()
  tokens_per_second <- 0
  
  result <- edge_stream_completion(ctx, prompt,
    callback = function(data) {
      if (!data$is_final) {
        cat(data$token)
        flush.console()
        
        # Calculate tokens per second
        elapsed <- as.numeric(Sys.time() - start_time, units = "secs")
        if (elapsed > 0) {
          tokens_per_second <<- data$total_tokens / elapsed
        }
        
        # Show progress every 20 tokens
        if (data$total_tokens %% 20 == 0) {
          cat(sprintf(" [%d tokens, %.1f tok/s]", data$total_tokens, tokens_per_second))
        }
        
        return(TRUE)
      } else {
        elapsed <- as.numeric(Sys.time() - start_time, units = "secs")
        cat(sprintf("\n\n📊 Final stats: %d tokens in %.2f seconds (%.1f tok/s)\n", 
                   data$total_tokens, elapsed, data$total_tokens / elapsed))
        return(TRUE)
      }
    },
    n_predict = max_tokens
  )
  
  return(result)
}

# Usage
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

streaming_with_progress(ctx, 
  "Explain the benefits of using R for data science in detail:",
  max_tokens = 300)

edge_free_model(ctx)
```

#### Multi-Model Streaming Comparison
```r
library(edgemodelr)

compare_models_streaming <- function(prompt, models = c("TinyLlama-1.1B", "TinyLlama-OpenOrca")) {
  for (model_name in models) {
    cat("\n", "="*60, "\n")
    cat("🤖 Model:", model_name, "\n")
    cat("="*60, "\n")
    
    setup <- edge_quick_setup(model_name)
    ctx <- setup$context
    
    if (!is.null(ctx)) {
      cat("Response: ")
      
      result <- edge_stream_completion(ctx, prompt,
        callback = function(data) {
          if (!data$is_final) {
            cat(data$token)
            flush.console()
            return(TRUE)
          } else {
            cat(sprintf("\n[%d tokens generated]\n", data$total_tokens))
            return(TRUE)
          }
        },
        n_predict = 150,
        temperature = 0.8
      )
      
      edge_free_model(ctx)
    } else {
      cat("❌ Failed to load model\n")
    }
  }
}

# Compare how different models respond to the same prompt
compare_models_streaming("What makes R special for data analysis?")
```

## API Reference

### `edge_load_model(model_path, n_ctx = 2048, n_gpu_layers = 0)`

Load a GGUF model file for inference.

- `model_path`: Path to .gguf model file
- `n_ctx`: Maximum context length (default: 2048)  
- `n_gpu_layers`: Layers to offload to GPU (default: 0, CPU-only)

### `edge_completion(ctx, prompt, n_predict = 128, temperature = 0.8, top_p = 0.95)`

Generate text completion using the loaded model.

- `ctx`: Model context from `edge_load_model()`
- `prompt`: Input text prompt
- `n_predict`: Maximum tokens to generate
- `temperature`: Sampling temperature (higher = more random)
- `top_p`: Top-p sampling threshold

### `edge_free_model(ctx)`

Free model context and release memory. Always call when done.

### `is_valid_model(ctx)`

Check if a model context is still valid and can be used for inference.

### `edge_stream_completion(ctx, prompt, callback, n_predict = 128, temperature = 0.8, top_p = 0.95)`

Stream text generation with real-time token callbacks.

- `ctx`: Model context from `edge_load_model()`
- `prompt`: Input text prompt
- `callback`: Function called for each token. Receives data list with token info
- `n_predict`: Maximum tokens to generate
- `temperature`: Sampling temperature
- `top_p`: Top-p sampling threshold

Callback receives: `list(token, position, is_final, total_tokens, full_response, stopped_early)`

### `edge_chat_stream(ctx, system_prompt = NULL, max_history = 10, n_predict = 200, temperature = 0.8)`

Interactive chat session with streaming responses.

- `ctx`: Model context from `edge_load_model()`
- `system_prompt`: Optional system message to set context
- `max_history`: Maximum conversation turns to keep
- `n_predict`: Maximum tokens per response
- `temperature`: Response creativity level

## Advanced Examples

### Interactive Chat Session

```r
library(edgemodelr)

chat_with_model <- function(model_path) {
  ctx <- edge_load_model(model_path, n_ctx = 4096)
  
  cat("Chat started! Type 'quit' to exit.\n")
  repeat {
    prompt <- readline("You: ")
    if (tolower(prompt) == "quit") break
    
    response <- edge_completion(ctx, 
                               prompt = paste("User:", prompt, "\nAssistant:"),
                               n_predict = 200,
                               temperature = 0.7)
    cat("Assistant:", response, "\n\n")
  }
  
  edge_free_model(ctx)
  cat("Chat ended.\n")
}

# Use with downloaded model
# chat_with_model("tinyllama-1.1b-chat-v1.0.q4_k_m.gguf")
```

### Batch Text Processing

```r
library(edgemodelr)

process_texts <- function(model_path, texts, task_prompt = "Summarize: ") {
  ctx <- edge_load_model(model_path)
  results <- vector("character", length(texts))
  
  for (i in seq_along(texts)) {
    prompt <- paste0(task_prompt, texts[i], "\nSummary:")
    results[i] <- edge_completion(ctx, prompt, n_predict = 100)
    cat("Processed", i, "of", length(texts), "\n")
  }
  
  edge_free_model(ctx)
  return(results)
}

# Example usage
# texts <- c("Long document 1...", "Long document 2...")
# summaries <- process_texts("model.gguf", texts)
```

### Integration with Data Frames

```r
library(edgemodelr)

analyze_sentiment <- function(model_path, df, text_column) {
  ctx <- edge_load_model(model_path)
  
  df$sentiment <- sapply(df[[text_column]], function(text) {
    prompt <- paste("Analyze sentiment (positive/negative/neutral):", text, "\nSentiment:")
    response <- edge_completion(ctx, prompt, n_predict = 10)
    trimws(response)
  })
  
  edge_free_model(ctx)
  return(df)
}

# Example usage
# data <- data.frame(review = c("Great product!", "Terrible service"))
# results <- analyze_sentiment("model.gguf", data, "review")
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: Modern x86_64 or ARM64 processor with AVX support
- **RAM**: 4GB+ (varies by model size)
- **Storage**: 1GB+ for model files

### Recommended Specs by Model Size

| Model Size | RAM | CPU Cores | Example Models |
|------------|-----|-----------|----------------|
| 1B params | 2GB | 2+ cores | TinyLlama |
| 7B params | 8GB | 4+ cores | Llama-2, Mistral |
| 13B params | 16GB | 6+ cores | Llama-2-13B |
| 30B+ params | 32GB+ | 8+ cores | Larger models |

### GPU Acceleration (Optional)
- **NVIDIA**: RTX 20xx series or newer with 6GB+ VRAM
- **AMD**: RX 6600 or newer (experimental support)
- **Apple Silicon**: M1/M2 chips supported via Metal

## Troubleshooting

### Common Issues

1. **Package won't compile**
   ```r
   # Ensure you have the right tools
   install.packages(c("Rcpp", "devtools"))
   devtools::load_all()
   ```

2. **"llama.cpp functions not available"**
   - This is expected with stub implementations
   - Install llama.cpp system-wide for full functionality

3. **Out of memory errors**
   - Try smaller models (TinyLlama for testing)
   - Reduce `n_ctx` parameter
   - Use quantized models (Q4_K_M, Q5_K_M)

### Getting Help

- 📖 Check the [documentation](R/api.R) for function details
- 🐛 Report bugs at [GitHub Issues](https://github.com/PawanRamaMali/edgemodelr/issues)
- 💬 Ask questions by opening an issue with the "question" label

## Development Status

- ✅ Package structure and build system
- ✅ Core API functions with proper documentation  
- ✅ Stub implementations for development
- ⚠️ Integration with system llama.cpp (in progress)
- 🔄 Performance optimizations (planned)
- 🔄 Advanced sampling methods (planned)

## License

MIT License - see [LICENSE](LICENSE) file.

## Contributing

We welcome contributions! Areas where help is needed:

- 🧪 Testing with different models and platforms
- 📚 Documentation improvements and examples
- ⚡ Performance optimizations
- 🔧 Integration with system llama.cpp installations

## Acknowledgments

Built with ❤️ using:
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Fast LLM inference in C++
- [Rcpp](https://github.com/RcppCore/Rcpp) - R and C++ integration
- [GGML](https://github.com/ggml-org/ggml) - Machine learning tensor library
