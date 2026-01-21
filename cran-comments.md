## Resubmission - edgemodelr 0.1.4

This is a resubmission of edgemodelr which was previously archived on CRAN (2025-10-02).

### Addressing Previous Archive Issues

#### 1. Misrepresentation of authorship - FIXED
All third-party code contributors are now properly credited in:
- **DESCRIPTION**: Authors@R field includes all contributors with appropriate roles (aut, cph, ctb)
- **inst/COPYRIGHTS**: Detailed attribution for llama.cpp, GGML, YaRN RoPE, DRY sampler, and Z-algorithm

Contributors now credited:
- Georgi Gerganov (aut, cph) - Author of llama.cpp and GGML library
- The ggml authors (cph) - llama.cpp and GGML contributors
- Jeffrey Quesnelle & Bowen Peng (ctb, cph) - YaRN RoPE implementation
- pi6am (ctb) - DRY sampler from Koboldcpp
- Ivan Yurchenko (ctb) - Z-algorithm implementation

#### 2. Platform-dependent code - FIXED
- Removed all non-portable compiler flags (-march=native, -mtune=native)
- Unix/macOS (Makevars): Uses generic CPU implementation (-DGGML_CPU_GENERIC)
- Windows (Makevars.win): Uses x86 SIMD optimizations (clearly documented in file)
- SystemRequirements properly documents: "C++17, GNU make or equivalent for building"

#### 3. Repeated submissions - Acknowledged
We will wait for this submission to be fully processed before any further submissions.

### Changes in version 0.1.4

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
* local Windows 11 install, R 4.5.2
* GitHub Actions:
  - ubuntu-latest: R (release, devel, oldrel-1)
  - macOS-latest: R (release)
  - windows-latest: R (release)

### R CMD check results
There were no ERRORs or WARNINGs.

There were 2 NOTEs:
1. CRAN incoming feasibility - Package was previously archived (addressed above)
2. Unable to verify current time - Network/time sync issue, not a package problem

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
- Generic CPU implementation ensures portability
- Tested on both ARM64 and x86_64 architectures

#### Linux:
- Generic CPU implementation ensures portability
- No architecture-specific optimizations that could cause issues
- Tested on Ubuntu with both GCC and Clang

#### Windows:
- Uses x86 SIMD optimizations (SSE2) for better performance
- Compiles cleanly with Rtools
- E2E tests skipped on Windows CI due to memory constraints (tests work locally)

### Installation and compilation
The package compiles cleanly on all major platforms with proper C++17 support and includes comprehensive error handling for missing system requirements.
