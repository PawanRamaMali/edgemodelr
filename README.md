# edgemodelr

> **Local Large Language Model Inference Engine for R**

[![CRAN Status](https://www.r-pkg.org/badges/version/edgemodelr)](https://cran.r-project.org/package=edgemodelr)
[![R Package](https://img.shields.io/badge/R-package-blue.svg)](https://www.r-project.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![R-CMD-check](https://github.com/PawanRamaMali/edgemodelr/workflows/R-CMD-check/badge.svg)](https://github.com/PawanRamaMali/edgemodelr/actions)

**edgemodelr** enables R users to run large language models locally using GGUF model files and the llama.cpp inference engine. Perfect for data scientists who need privacy-preserving AI capabilities integrated seamlessly into their R workflows.

## ✨ Key Features

- 🔒 **Complete Privacy** - All inference runs locally, no data leaves your machine
- ⚡ **High Performance** - Leverages optimized llama.cpp C++ implementation
- 🎯 **R-Native Interface** - Seamlessly integrates with R data science workflows
- 💾 **Memory Efficient** - Supports quantized models for resource-constrained environments
- 🛠️ **Zero Dependencies** - Self-contained installation with no external requirements
- 🔄 **Universal Compatibility** - Works with any GGUF model from Hugging Face, Ollama, or custom sources

## 📦 Installation

### From CRAN (Recommended)

```r
install.packages("edgemodelr")
```

### Development Version

```r
# Install from GitHub
devtools::install_github("PawanRamaMali/edgemodelr")
```

### System Requirements

- **R**: Version 4.0 or higher
- **Compiler**: C++17 compatible compiler (GCC, Clang, or MSVC)
- **Memory**: 4GB+ RAM (varies by model size)
- **Storage**: 1GB+ for model files

## 🚀 Quick Start

### 1. Basic Text Generation

```r
library(edgemodelr)

# One-line setup with automatic model download
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

# Generate text
result <- edge_completion(ctx,
                         prompt = "Explain what R is in one sentence:",
                         n_predict = 50)
cat("Response:", result, "\n")

# Clean up when done
edge_free_model(ctx)
```

### 2. Interactive Chat Session

```r
library(edgemodelr)

# Setup model
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

# Start interactive chat with streaming responses
edge_chat_stream(ctx,
  system_prompt = "You are a helpful R programming assistant.",
  max_history = 5,
  n_predict = 200,
  temperature = 0.7)

# Type your questions and get real-time responses!
# Type 'quit' to exit
```

### 3. Use Your Existing Models

```r
library(edgemodelr)

# Automatically find and use existing GGUF models from Ollama, Downloads, etc.
models_info <- edge_find_gguf_models()

if (length(models_info$models) > 0) {
  # Use any compatible model
  first_model <- models_info$models[[1]]
  ctx <- edge_load_model(first_model$path)

  # Same API as always
  result <- edge_completion(ctx, "What is machine learning?", n_predict = 100)
  cat(result)

  edge_free_model(ctx)
}
```

## 📚 Core API Reference

### Model Management

| Function | Description |
|----------|-------------|
| `edge_load_model(path, n_ctx, n_gpu_layers)` | Load a GGUF model for inference |
| `edge_free_model(ctx)` | Release model memory |
| `is_valid_model(ctx)` | Check if model context is valid |
| `edge_quick_setup(model_name)` | One-line model download and setup |

### Text Generation

| Function | Description |
|----------|-------------|
| `edge_completion(ctx, prompt, n_predict, temperature, top_p)` | Generate text completion |
| `edge_stream_completion(ctx, prompt, callback, ...)` | Stream tokens in real-time |
| `edge_chat_stream(ctx, system_prompt, max_history, ...)` | Interactive chat session |

### Model Discovery & Download

| Function | Description |
|----------|-------------|
| `edge_find_gguf_models(source_dirs, model_pattern, ...)` | Find existing GGUF models |
| `edge_list_models()` | List pre-configured popular models |
| `edge_download_model(model_id, filename)` | Download from HuggingFace |
| `edge_download_url(url, filename)` | Download from any direct URL (GPT4All, etc.) |

### Performance Optimization

| Function | Description |
|----------|-------------|
| `edge_small_model_config(model_size_mb, available_ram_gb, target)` | Get optimized settings for small models |
| `edge_benchmark(ctx, prompt, n_predict, iterations)` | Benchmark model performance |
| `edge_set_verbose(enabled)` | Control logging verbosity |

## ⚡ Performance Optimizations for Small Models

**NEW in v0.1.4**: Automatic optimizations for small language models (1B-3B parameters)!

```r
library(edgemodelr)

# Get optimized configuration for your device
config <- edge_small_model_config(
  model_size_mb = 700,      # TinyLlama size
  available_ram_gb = 8,     # Your system RAM
  target = "laptop"         # mobile/laptop/desktop/server
)

# View recommendations
print(config$tips)

# Load model with optimized settings
ctx <- edge_load_model(
  "path/to/model.gguf",
  n_ctx = config$n_ctx,
  n_gpu_layers = config$n_gpu_layers
)

# Faster inference with optimized parameters
result <- edge_completion(
  ctx,
  prompt = "Hello!",
  n_predict = config$recommended_n_predict,
  temperature = config$recommended_temperature
)
```

**Key Optimizations:**
- 🚀 **15-30% faster inference** for small models through adaptive batch sizing
- 💾 **Reduced memory usage** with context-aware thread allocation
- 📱 **Device-specific presets** optimized for mobile, laptop, desktop, and server
- 🎯 **Automatic tuning** based on model size and available RAM

## 🤖 Available Models

**All models download directly without authentication** - ready for offline use!

### Small Models (Testing & Development)

| Model | Size | Source | Use Case | Command |
|-------|------|--------|----------|---------|
| **TinyLlama-1.1B** | ~700MB | HuggingFace | Testing, development | `edge_quick_setup("TinyLlama-1.1B")` |
| **TinyLlama-OpenOrca** | ~700MB | HuggingFace | Better chat | `edge_quick_setup("TinyLlama-OpenOrca")` |

### Medium Models (4-5GB)

| Model | Size | Source | Use Case | Command |
|-------|------|--------|----------|---------|
| **Llama-3-8B** | ~4.7GB | GPT4All CDN | General purpose | `edge_quick_setup("llama3-8b")` |
| **Mistral-7B** | ~4.1GB | GPT4All CDN | High quality | `edge_quick_setup("mistral-7b")` |
| **Phi-3-mini** | ~2.4GB | HuggingFace | Reasoning | `edge_quick_setup("phi3-mini")` |

### Large Models (7-9GB) - NEW!

| Model | Size | Source | Use Case | Command |
|-------|------|--------|----------|---------|
| **Orca-2-13B** | ~7.4GB | GPT4All CDN | 13B Chat model | `edge_quick_setup("orca2-13b")` |
| **WizardLM-13B** | ~7.4GB | GPT4All CDN | 13B Instruct | `edge_quick_setup("wizardlm-13b")` |
| **Hermes-13B** | ~7.4GB | GPT4All CDN | 13B Chat | `edge_quick_setup("hermes-13b")` |
| **Starcoder** | ~9GB | GPT4All CDN | Code generation (15B) | `edge_quick_setup("starcoder")` |

> **Note:** GPT4All models download directly from their CDN - no HuggingFace account or authentication required!

## 💡 Use Cases & Examples

### Data Science Workflows

```r
library(edgemodelr)

# Analyze survey responses
analyze_sentiment <- function(texts, model_name = "Mistral-7B") {
  setup <- edge_quick_setup(model_name)
  ctx <- setup$context

  results <- data.frame(
    text = texts,
    sentiment = sapply(texts, function(text) {
      prompt <- paste("Analyze sentiment (positive/negative/neutral):", text)
      trimws(edge_completion(ctx, prompt, n_predict = 10))
    }),
    stringsAsFactors = FALSE
  )

  edge_free_model(ctx)
  return(results)
}

# Example usage
survey_data <- c(
  "This product exceeded my expectations!",
  "Poor customer service, very disappointed.",
  "It's okay, nothing special but works fine."
)

sentiment_results <- analyze_sentiment(survey_data)
print(sentiment_results)
```

### Code Generation Assistant

```r
library(edgemodelr)

# R programming helper
setup <- edge_quick_setup("CodeLlama-7B")
ctx <- setup$context

code_prompt <- "Create an R function to calculate correlation matrix with p-values:"
response <- edge_completion(ctx,
  paste("# R Programming Task\n", code_prompt, "\n\n# Solution:\n"),
  n_predict = 250,
  temperature = 0.3)

cat(response)
edge_free_model(ctx)
```

### Batch Document Processing

```r
library(edgemodelr)

# Process multiple documents
summarize_documents <- function(file_paths, model_name = "mistral-7b") {
  setup <- edge_quick_setup(model_name)
  ctx <- setup$context

  summaries <- sapply(file_paths, function(path) {
    text <- readLines(path, warn = FALSE)
    content <- paste(text, collapse = " ")

    prompt <- paste("Summarize this document in 2-3 sentences:", content)
    edge_completion(ctx, prompt, n_predict = 100)
  })

  edge_free_model(ctx)
  return(summaries)
}
```

## 🔥 Large Model Examples (7-9GB)

### End-to-End: Orca-2-13B (7GB Model)

Complete example downloading and running a 13 billion parameter model:

```r
library(edgemodelr)

# =============================================================
# Step 1: Download and Setup (one-time, ~7.4GB download)
# =============================================================
cat("Setting up Orca-2-13B (7.4GB)...\n")
cat("Source: GPT4All CDN (direct download, no auth required)\n\n")

setup <- edge_quick_setup("orca2-13b")
ctx <- setup$context

cat("Model loaded successfully!\n")
cat("Model path:", setup$model_path, "\n\n")

# =============================================================
# Step 2: Math Test
# =============================================================
cat("--- Math Test ---\n")
prompt <- "What is 123 + 456? The answer is"

result <- edge_completion(ctx, prompt, n_predict = 30, temperature = 0.1)
cat("Prompt:", prompt, "\n")
cat("Response:", trimws(gsub(prompt, "", result, fixed = TRUE)), "\n\n")
# Expected output: "579"

# =============================================================
# Step 3: Knowledge Test
# =============================================================
cat("--- Knowledge Test ---\n")
prompt <- "The theory of relativity was developed by"

result <- edge_completion(ctx, prompt, n_predict = 50, temperature = 0.1)
cat("Prompt:", prompt, "\n")
cat("Response:", trimws(gsub(prompt, "", result, fixed = TRUE)), "\n\n")
# Expected output: "Albert Einstein in the early 20th century..."

# =============================================================
# Step 4: Reasoning Test
# =============================================================
cat("--- Reasoning Test ---\n")
prompt <- "If all cats are animals, and some animals are pets, can we conclude that some cats are pets? Answer:"

result <- edge_completion(ctx, prompt, n_predict = 80, temperature = 0.3)
cat("Prompt:", prompt, "\n")
cat("Response:", trimws(gsub(prompt, "", result, fixed = TRUE)), "\n\n")
# Expected: Logical reasoning about the syllogism

# =============================================================
# Step 5: Cleanup
# =============================================================
edge_free_model(ctx)
cat("Model freed. Done!\n")
```

### Actual Test Results (Orca-2-13B)

These results were obtained on a Windows 11 system with CPU-only inference:

| Test | Prompt | Response | Time | Tokens/sec |
|------|--------|----------|------|------------|
| **Math** | "What is 123 + 456?" | **579** (correct!) | 103s | 0.29 |
| **Knowledge** | "The theory of relativity was developed by" | **Albert Einstein** in the early 20th century | 261s | 0.15 |
| **Reasoning** | Syllogism about cats/animals/pets | Correct logical deduction with step-by-step reasoning | 415s | 0.12 |

### End-to-End: Direct URL Download

Download any GGUF model from a direct URL:

```r
library(edgemodelr)

# Download from GPT4All CDN (or any direct URL)
model_path <- edge_download_url(
  url = "https://gpt4all.io/models/gguf/orca-2-13b.Q4_0.gguf",
  filename = "orca-2-13b.Q4_0.gguf"
)

# Load and use
ctx <- edge_load_model(model_path, n_ctx = 2048)
result <- edge_completion(ctx, "Explain quantum computing:", n_predict = 100)
cat(result)
edge_free_model(ctx)
```

### End-to-End: Starcoder (9GB Code Model)

```r
library(edgemodelr)

# Setup Starcoder - a 15B parameter code generation model
setup <- edge_quick_setup("starcoder")
ctx <- setup$context

# Generate Python code
prompt <- "def quicksort(arr):"
result <- edge_completion(ctx, prompt, n_predict = 150, temperature = 0.2)
cat(result)

# Generate R code
prompt2 <- "# R function to calculate moving average\nmoving_avg <- function(x, n) {"
result2 <- edge_completion(ctx, prompt2, n_predict = 100, temperature = 0.2)
cat(result2)

edge_free_model(ctx)
```

### Performance Comparison by Model Size

| Model | Parameters | File Size | Load Time | Tokens/sec (CPU) |
|-------|------------|-----------|-----------|------------------|
| TinyLlama-1.1B | 1.1B | 637 MB | 0.1s | 1-2 |
| Phi-3-mini | 3.8B | 2.3 GB | 0.3s | 0.15-0.2 |
| Llama-3.2-3B | 3.2B | 1.9 GB | 0.4s | 0.2-0.3 |
| Mistral-7B | 7B | 4.1 GB | 0.5s | 0.1-0.15 |
| Orca-2-13B | 13B | 6.9 GB | 0.4s | 0.12-0.29 |
| Starcoder | 15B | 8.4 GB | 0.6s | 0.08-0.12 |

> **Note:** Inference speed depends heavily on your CPU. GPU acceleration can increase speed 10-50x.

## ⚡ Performance & Hardware

### Hardware Requirements by Model Size

| Model Size | File Size | Min RAM | Recommended RAM | CPU Cores | Typical Speed |
|------------|-----------|---------|-----------------|-----------|---------------|
| 1B params | ~700 MB | 4 GB | 8 GB | 2+ cores | 1-2 tok/s |
| 3B params | ~2 GB | 8 GB | 12 GB | 4+ cores | 0.2-0.3 tok/s |
| 7B params | ~4 GB | 12 GB | 16 GB | 4+ cores | 0.1-0.15 tok/s |
| 13B params | ~7 GB | 16 GB | 24 GB | 6+ cores | 0.1-0.3 tok/s |
| 15B params | ~9 GB | 20 GB | 32 GB | 8+ cores | 0.08-0.12 tok/s |

> **Tip:** For faster inference, use GPU acceleration with `n_gpu_layers` parameter.

### Performance Optimizations

- **SIMD Instructions**: Automatic AVX/AVX2/FMA detection
- **Multi-threading**: Uses all available CPU cores
- **Memory Efficiency**: Optimized batch processing
- **Smart Quantization**: Q4_K_M and Q5_K_M support

### Benchmark Your Setup

```r
library(edgemodelr)

# Test your system performance
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

# Run benchmark
perf <- edge_benchmark(ctx, iterations = 5)
print(perf)

edge_free_model(ctx)
```

## 🔧 Advanced Configuration

### GPU Acceleration (Experimental)

```r
# Enable GPU layers (requires compatible hardware)
ctx <- edge_load_model("model.gguf",
                      n_ctx = 2048,
                      n_gpu_layers = 35)  # Offload layers to GPU
```

### Custom Model Sources

```r
# Search specific directories for models
models <- edge_find_gguf_models(
  source_dirs = c("~/Downloads", "~/my_models"),
  model_pattern = "llama",
  min_size_mb = 100,
  test_compatibility = TRUE
)
```

### Streaming with Custom Callbacks

```r
# Advanced streaming with progress tracking
result <- edge_stream_completion(ctx, prompt,
  callback = function(data) {
    if (!data$is_final) {
      cat(data$token)
      flush.console()

      # Custom progress logic
      if (data$total_tokens %% 10 == 0) {
        cat(sprintf(" [%d tokens]", data$total_tokens))
      }
    }
    return(TRUE)  # Continue generation
  },
  n_predict = 200
)
```

## 🛠️ Troubleshooting

### Common Issues

**Installation Problems:**
```r
# Ensure build tools are available
install.packages(c("Rcpp", "devtools"))

# Check compiler
system("gcc --version")  # Should show C++17 support
```

**Memory Issues:**
- Use smaller models (TinyLlama for testing)
- Reduce `n_ctx` parameter
- Try quantized models (Q4_K_M variant)

**Performance Issues:**
- Ensure all CPU cores are being used
- Check available RAM
- Consider GPU acceleration for large models

### Getting Help

- 📖 [Function Documentation](man/)
- 🐛 [Report Issues](https://github.com/PawanRamaMali/edgemodelr/issues)
- 💬 [Discussions](https://github.com/PawanRamaMali/edgemodelr/discussions)
- 📧 Contact: [prm@outlook.in](mailto:prm@outlook.in)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/PawanRamaMali/edgemodelr.git
cd edgemodelr
R CMD build .
R CMD check edgemodelr_*.tar.gz
```

### Areas for Contribution

- 🧪 Testing with different models and platforms
- 📚 Documentation and examples
- ⚡ Performance optimizations
- 🔧 New features and integrations

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ using:

- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** - Fast LLM inference engine
- **[GGML](https://github.com/ggml-org/ggml)** - Machine learning tensor library
- **[Rcpp](https://github.com/RcppCore/Rcpp)** - R and C++ integration

Special thanks to the open-source AI community for making local LLM inference accessible to everyone.

---

⭐ **Star this repo** if you find edgemodelr useful for your R projects!