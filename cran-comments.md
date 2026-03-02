## Submission - edgemodelr 0.3.0

### Changes in 0.3.0

#### New Features
- Added `edge_install_cuda()` and `edge_install_cuda_toolkit()` for automatic
  CUDA GPU setup on Windows (copies nvcudart from the Windows DriverStore,
  downloads cublas from NVIDIA redistrib, extracts ggml-cuda backend from the
  llama.cpp release archive).
- Added `edge_reload_cuda()` and `edge_cuda_info()` for session-level CUDA
  management without restarting R.
- Updated bundled llama.cpp to build b8179 (GGML 0.9.7).

#### Bug Fixes
- Fixed 40-minute load time for Qwen3 / QWEN2 models: added a hand-written
  fast path `unicode_regex_split_custom_qwen2()` that avoids the degenerate
  O(2^n) backtracking in GCC's std::regex for the QWEN2 tokenizer pattern.
  Qwen3-14B now loads in 0.3 s (was 2422 s; 8000x speedup).

#### CRAN Compliance
- Removed `abort()` symbol from compiled objects (replaced with
  `raise(SIGABRT)` / `std::terminate()` under `#ifdef USING_R`).
- Guarded `fflush(stdout)`, `fprintf(stderr, ...)`, and `_Exit()` in
  `ggml.c` error/backtrace paths with `#ifndef USING_R`.
- Added `#define _GNU_SOURCE` to `ggml-cpu.c` (fixes `SCHED_BATCH` undeclared
  on Linux).
- Replaced explicit `printf()` calls in `ggml-quants.c` with `fprintf(stderr)`.
- `CXX_STD = CXX17` replaces non-portable `-std=c++17` in `PKG_CXXFLAGS`.
- `-fno-builtin-printf` suppresses `printf to puts` optimizations in ggml.

### Test environments
* Local: Windows 11, R 4.5.2, Rtools45 / GCC 14.3.0
* GitHub Actions:
  - ubuntu-latest: R release, R devel, R oldrel-1
  - macos-latest (ARM64): R release
  - windows-latest: R release

### R CMD check results (R CMD check --as-cran on tarball)

0 ERRORs. 0 WARNINGs. 2 NOTEs.

NOTE 1: CRAN incoming feasibility — new version number, no conflict.

NOTE 2: checking compiled code (macOS only)
Objects: ggml/ggml-quants.o, ggml/ggml-opt.o, llama/llama-grammar.o,
         llama/llama-impl.o, llama/unicode.o, ggml/ggml-cpu/ggml-cpu-c.o,
         ggml/ggml-cpu/unary-ops.o
These references come from `fprintf(stderr, ...)` calls in the bundled
llama.cpp and GGML upstream source (build b8179 / GGML 0.9.7). They appear
in non-fatal diagnostic paths (data validation, quantization checks, NUMA
warnings). The primary ggml.c has been patched to remove all stderr, stdout,
and _Exit references from R builds; the remaining objects are deep in the
upstream library and cannot be changed without forking the entire llama.cpp
codebase. At runtime these calls are suppressed through ggml's log-callback
API, which the R bindings replace with R's own output system.

### Third-party code
All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
