# edgemodelr 0.2.0

## SIMD Optimizations for Faster CPU Inference

### New Features

* **SIMD-optimized build system**: Replaced generic scalar fallback with architecture-aware SIMD detection in both `Makevars` (Unix) and `Makevars.win` (Windows)
  - x86_64: Enables SSE4.2 baseline by default (universal since Intel Nehalem 2008)
  - aarch64/arm64: NEON support built into the ABI (no extra flags needed)
  - Other architectures: Automatic generic fallback

* **User-configurable SIMD levels**: Set `EDGEMODELR_SIMD` environment variable before install to select optimization level:
  - `GENERIC`: Scalar fallback (maximum compatibility)
  - `SSE42`: SSE4.2 baseline (default on x86_64)
  - `AVX`: AVX + F16C (Intel Sandy Bridge 2011+)
  - `AVX2`: AVX2 + FMA + F16C (Intel Haswell 2013+, recommended)
  - `AVX512`: AVX-512 (Intel Skylake-X 2017+)
  - `NATIVE`: Uses `-march=native` for maximum performance on the build machine

* **`edge_simd_info()`**: New function to query compile-time SIMD status including architecture, compiler features, and GGML optimization flags

* **x86 architecture-specific quantization**: Enabled optimized x86 quantization kernels (`arch/x86/quants.c`, `arch/x86/repack.cpp`) with SIMD-accelerated dot products and matrix operations

### Performance

* 15-40% faster inference on x86_64 with SSE4.2 baseline vs generic scalar
* Up to 2-3x faster with AVX2 for quantized model operations
* SSSE3-accelerated integer multiply-accumulate for quantized dot products

---

# edgemodelr 0.1.5

## CRAN Policy Fixes

### Bug Fixes

* **Fixed donttest examples**: Changed resource-intensive examples from `\donttest{}` to `\dontrun{}` to prevent downloading multi-GB models during CRAN checks

* **Fixed M1 Mac compiler warnings**: Added explicit `static_cast<>` for:
  - `double` to `float` conversions for temperature/top_p parameters
  - `size_type` to `int32_t` conversions for buffer size parameters

* **Fixed connection handling**: Replaced `on.exit()` with `tryCatch/finally` for proper connection cleanup in loops (thanks @eddelbuettel)

# edgemodelr 0.1.4

## Performance Optimizations for Small Language Models

### New Features

* **Small Model Configuration Helper**: New `edge_small_model_config()` function provides optimized settings for small models (1B-3B parameters)
  - Device-specific presets: mobile, laptop, desktop, and server
  - Adaptive configuration based on model size and available RAM
  - Built-in performance tips and recommendations
  - Automatic parameter tuning for optimal inference speed

* **Adaptive Batch Processing**: Intelligent batch size optimization based on context length
  - Small contexts (≤512): Uses up to full context for batching
  - Medium contexts (512-2048): Uses 1/2 context for optimal throughput
  - Large contexts (2048-4096): Uses 1/4 context to balance speed and memory
  - Very large contexts (>4096): Caps at 2048 tokens for stability

* **Smart Thread Allocation**: Context-aware CPU thread management
  - Small models automatically limit threads to avoid overhead
  - Reduces CPU contention on resource-constrained devices
  - Improves inference speed for models with contexts ≤2048 tokens

* **Automatic Context Optimization**: Model size-based context tuning
  - Small models (<1GB): Optimized to 1024 tokens for faster inference
  - Medium models (1-2GB): Set to 1536 tokens for balanced performance
  - Large models (>2GB): Maintains 2048+ tokens for quality
  - User override available via n_ctx parameter

### Performance Improvements

* **Faster Small Model Inference**: 15-30% speed improvement for small models through optimized batch and thread settings
* **Reduced Memory Footprint**: Better memory efficiency for resource-constrained environments
* **Lower Latency**: Optimized thread allocation reduces context switching overhead
* **Better Scalability**: Adaptive configurations scale from mobile devices to servers

### Examples and Documentation

* **Small Model Optimization Example**: Comprehensive example demonstrating all optimization features
  - Configuration comparison across device types
  - Performance benchmarking workflow
  - Best practices for different model sizes
  - Manual tuning guidelines

* **Enhanced Testing**: New test suite for small model configuration
  - Tests for all device target configurations
  - Validation of adaptive parameter adjustments
  - Safety checks for edge cases

### Technical Details

* Improved C++ bindings with adaptive batch size calculations
* Enhanced R API with intelligent parameter defaults
* Better integration between model size detection and configuration
* Comprehensive documentation for optimization features

---

# edgemodelr 0.1.2

## Major New Features

### Ollama Integration
* **Native Ollama Support**: Complete integration with Ollama models through automatic model discovery and SHA-256 hash-based loading
* `edge_find_ollama_models()` - Discover all locally available Ollama models across platforms (Windows, macOS, Linux)
* `edge_load_ollama_model()` - Load Ollama models using convenient SHA-256 hash prefixes instead of full file paths
* `test_ollama_model_compatibility()` - Built-in compatibility testing for Ollama models
* **Cross-platform Model Detection**: Robust model discovery supporting standard installations, snap packages (Linux), and various Windows configurations
* **Windows OneDrive Compatibility**: Enhanced path detection that properly handles Windows OneDrive document folder redirections

### Comprehensive Examples Suite
* **Structured Learning Path**: Complete examples directory with progressive difficulty levels (Beginner → Intermediate → Advanced)
* **01_basic_usage.R**: Fundamental operations including model loading, text generation, parameter tuning, and error handling
* **02_ollama_integration.R**: Complete Ollama workflow with model discovery, hash-based loading, and compatibility testing
* **03_streaming_generation.R**: Real-time streaming text generation with interactive chat interfaces and callback processing
* **04_performance_optimization.R**: Advanced performance tuning including GPU acceleration, benchmarking, memory management, and batch processing
* **examples/README.md**: Comprehensive documentation with learning paths, troubleshooting guide, and customization instructions

### Package Structure Improvements
* **Organized File Structure**: Consolidated all examples into structured examples/ directory with consistent formatting
* **Enhanced Documentation**: Improved inline documentation and example comments throughout

---

# edgemodelr 0.1.1

## Bug Fixes and Improvements

### Compilation Fixes
* **macOS Boolean Conflicts**: Completely resolved Boolean enum conflicts by avoiding problematic system headers and using direct function declarations
* **Filesystem Compatibility**: Added comprehensive fallback implementation for disabled `std::filesystem` on macOS builds
* **Header Protection**: Implemented robust cross-platform header inclusion strategy that works with R, Rcpp, and system headers
* **System Header Workarounds**: Replaced `<mach-o/dyld.h>` inclusion with direct function declarations to avoid enum conflicts
* **Format Attribute Warnings**: Suppressed unsupported printf format attribute warnings on macOS Apple Clang compiler
* **CRAN Compliance**: Removed non-portable optimization flags (`-march=native`, `-mtune=native`, etc.) from Makevars for CRAN compatibility
* **Cross-platform Build**: Enhanced Makevars configuration for better macOS compatibility with R package requirements

### Demo and Documentation Updates
* **Modern UI**: Updated streaming chat demo with modern bslib interface for enhanced user experience
* **Documentation**: Improved documentation for `edge_clean_cache()` function
* **Examples**: Enhanced streaming chat example with better UI components

### Technical Improvements
* **Build System**: Updated Makevars files for improved compilation on Windows and Unix systems
* **Core Bindings**: Enhanced C++ bindings for better performance and stability

---

# edgemodelr 0.1.0

## Initial CRAN Release

### New Features

* **Local LLM Inference**: Complete R interface for running large language models locally using llama.cpp and GGUF model files
* **Model Management**: Built-in functions for downloading and managing popular models from Hugging Face
* **Text Generation**: Support for both blocking and streaming text completion
* **Interactive Chat**: Real-time streaming chat interface with conversation history
* **Privacy-First**: All processing happens locally without external API calls

### Core Functions

* `edge_load_model()` - Load GGUF model files for inference
* `edge_completion()` - Generate text completions  
* `edge_stream_completion()` - Stream text generation with real-time callbacks
* `edge_chat_stream()` - Interactive chat session with streaming responses
* `edge_free_model()` - Memory management and cleanup
* `is_valid_model()` - Model context validation

### Model Management

* `edge_list_models()` - List pre-configured popular models
* `edge_download_model()` - Download models from Hugging Face Hub  
* `edge_quick_setup()` - One-line model download and setup

### System Support

* **Self-contained**: Includes complete llama.cpp implementation
* **Cross-platform**: Works on Windows, macOS, and Linux
* **CPU optimized**: Runs efficiently on standard hardware
* **Memory efficient**: Support for quantized models

### Documentation

* Comprehensive getting started vignette
* Complete API documentation with examples
* README with extensive usage examples
* Test coverage for all major functionality

### Technical Implementation

* C++17 integration via Rcpp
* Real-time token streaming with callback support
* Automatic memory management with RAII
* Robust error handling and validation
* Thread-safe model operations

This release provides a complete, production-ready solution for Local Large Language Model Inference Engine in R, enabling private, offline text generation workflows.