# edgemodelr 0.1.1

## Bug Fixes and Improvements

### Compilation Fixes
* **macOS Boolean Conflicts**: Resolved Boolean type redefinition conflicts by defining `Rboolean` before including R headers on macOS
* **Header Inclusion Order**: Fixed `Rboolean` type definition order to prevent "unknown type name" errors in R headers
* **System Header Protection**: Added comprehensive guards around macOS system headers (`mach-o/dyld.h`, `mach/mach.h`) to prevent enum conflicts
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

This release provides a complete, production-ready solution for Inference for Local Language Models in R, enabling private, offline text generation workflows.