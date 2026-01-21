## Submission - edgemodelr 0.1.4

This is an update to the edgemodelr package with performance optimizations and improved Ollama integration.

### Changes in this version

#### New Features:
1. **Small Model Configuration Helper**: New `edge_small_model_config()` function for optimized settings on resource-constrained devices
2. **Adaptive Batch Processing**: Intelligent batch size optimization based on context length
3. **Smart Thread Allocation**: Context-aware CPU thread management
4. **Improved Ollama Integration**: Better GGUF version detection and error diagnostics
5. **GPT4All CDN Support**: Added `edge_download_url()` for direct model downloads

#### Bug Fixes:
1. Fixed resource leak in Ollama model compatibility testing
2. Improved error message truncation for better diagnostics
3. Fixed GGUF version detection using numeric conversion to avoid integer overflow

### Test environments
* local Windows 11 install, R 4.5.0
* GitHub Actions:
  - ubuntu-latest: R (release, devel, oldrel-1)
  - macOS-latest: R (release)
  - windows-latest: R (release)

### R CMD check results
There were no ERRORs or WARNINGs.

There was 1 NOTE:
* checking CRAN incoming feasibility ... NOTE

  Possibly misspelled words in DESCRIPTION:
    GGUF (9:73)
    cpp (10:19)
    llama (10:15, 13:22)

These are not misspelled words but technical terms:
- GGUF: A specific model file format used by llama.cpp
- cpp: Abbreviation for C++ as in "llama.cpp"
- llama: Part of the project name "llama.cpp"

### What this package does
This package enables R users to run large language models locally using the llama.cpp inference engine and GGUF model files. It provides complete privacy by keeping all computation local without requiring cloud APIs or internet connectivity.

### Dependencies
* Core dependencies: R >= 4.0, Rcpp >= 1.0.0, utils, tools
* SystemRequirements: C++17, GNU make or equivalent for building

### Third-party code attribution
The package includes code from:
- **llama.cpp** (MIT License) - Copyright (c) 2023-2024 The ggml authors
- **GGML library** (MIT License) - Copyright (c) 2023-2024 The ggml authors
- **YaRN RoPE implementation** (MIT License) - Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng
- **DRY sampler** (MIT License) - Author: pi6am (from Koboldcpp)
- **Z-algorithm** (Public Domain) - Author: Ivan Yurchenko

All attributions are documented in:
- DESCRIPTION (Authors@R field)
- LICENSE file
- inst/COPYRIGHTS

### Package size
The package includes a self-contained llama.cpp implementation (~56MB when installed) to provide complete functionality without external dependencies. This larger size is justified as it eliminates the need for users to separately install and configure llama.cpp.

### Platform-specific notes

#### macOS:
- Fixed enum conflicts between system headers and R
- Uses forward declarations for dyld functions
- Tested on both ARM64 and x86_64 architectures

#### Linux:
- Generic CPU implementation ensures portability
- No architecture-specific optimizations that could cause issues
- Tested on Ubuntu with both GCC and Clang

#### Windows:
- Compiles cleanly with Rtools
- E2E tests skipped on Windows CI due to memory constraints (tests work locally)

### Installation and compilation
The package compiles cleanly on all major platforms with proper C++17 support and includes comprehensive error handling for missing system requirements.
