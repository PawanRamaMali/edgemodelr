## Resubmission - edgemodelr 0.1.3

This is a resubmission addressing all issues from the previous CRAN check (2025-10-02).

### Changes in this version

#### Fixed CRAN Check Issues:
1. **Compiler warnings resolved**: Disabled `GGML_ATTRIBUTE_FORMAT` macro when building for R to eliminate unrecognized format function type warnings
2. **macOS compilation fixed**: Replaced `<mach-o/dyld.h>` inclusion with forward declarations to avoid enum conflicts between `DYLD_BOOL` and R's `Rboolean`
3. **Fedora Clang compatibility**: Implemented generic CPU configuration for cross-platform portability
4. **Author attribution**: Added all copyright holders and contributors (llama.cpp authors, GGML contributors, YaRN implementation authors, etc.)
5. **License compliance**: Updated LICENSE file and created inst/COPYRIGHTS with complete third-party attributions

#### Additional Improvements:
- Fixed integration test filtering issues in CI/CD
- Added Windows skip conditions for resource-intensive E2E tests
- Improved documentation and attribution

### Test environments
* local Windows 11 install, R 4.5.0
* GitHub Actions:
  - ubuntu-latest: R (release, devel)
  - macOS-latest: R (release, oldrel)
  - windows-latest: R (release)
* R-hub (via rhub2):
  - Ubuntu Linux 20.04.1 LTS, R-release, GCC
  - Fedora Linux, R-devel, clang, gfortran
  - Windows Server 2022, R-devel, 64 bit
  - macOS 10.13.6 High Sierra, R-release, CRAN's setup

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
- Tested on Ubuntu and Fedora with both GCC and Clang

#### Windows:
- Compiles cleanly with Rtools
- E2E tests skipped on Windows CI due to memory constraints (tests work locally)

### Installation and compilation
The package compiles cleanly on all major platforms with proper C++17 support and includes comprehensive error handling for missing system requirements. All previous CRAN compilation issues have been resolved.
