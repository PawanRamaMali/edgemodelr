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