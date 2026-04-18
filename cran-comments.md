## Resubmission - edgemodelr 0.4.0

This resubmission addresses the NOTEs flagged by CRAN's automated pre-check
on the initial submission.

### Fixes applied

**Pragma NOTE**: Removed the `#pragma GCC diagnostic ignored` directives from
the 4 flagged files (`src/ggml/ggml-cpu/amx/mmq.cpp`,
`src/ggml/ggml-cpu/arch/x86/repack.cpp`, `src/ggml/ggml-cpu/repack.cpp`,
`src/llama/llama-sampler.cpp`). The code compiles cleanly without them.

**printf / stderr / stdout NOTE**: Added `#ifdef USING_R` guards in the 6
flagged source files (`src/ggml/ggml-quants.c`, `src/llama/llama-quant.cpp`,
`src/llama/llama-grammar.cpp`, `src/llama/llama-impl.cpp`,
`src/llama/unicode.cpp`, `src/ggml/ggml-cpu/ggml-cpu.c`,
`src/ggml/ggml-cpu/unary-ops.cpp`) that neutralize `printf`, `fprintf`,
`fputs`, `fflush`, `stderr`, and `stdout` to no-ops during R builds. These
upstream llama.cpp calls are diagnostic-only (data-validation edge cases,
NUMA warnings, thread-affinity warnings) and were silent at runtime even
before this change, as the R bindings install their own `llama_log_set`
callback. Now the symbols also never reach the compiled object files.

**C++17 NOTE** (Debian): Removed `CXX_STD = CXX17` from `src/Makevars` and
`src/Makevars.win`. R auto-detects C++17 from package code.

**Spelling NOTE**: Rephrased DESCRIPTION to avoid the "embeddings" flag.

### R CMD check --as-cran results

0 ERRORs. 0 WARNINGs. 0 NOTEs (informational "GNU make is a
SystemRequirements" is documented in DESCRIPTION).

### Changes in 0.4.0

#### New Features
- Grammar-constrained generation for structured output (JSON, enums)
- Text embeddings API with cosine similarity helpers
- Batch processing for vectorized LLM operations on data frames
- RAG pipeline: document indexing, semantic search, question answering
- Chat completion with model-native templates from GGUF metadata
- Plumber API server for OpenAI-compatible local inference
- Qwen3 model family (0.6B, 1.7B, 4B, 8B) in model registry
- Friendly model names in `edge_download_model()`
- `httr` download fallback for corporate proxy environments
- SIMD optimization warning on package load

#### Bug Fixes
- Fixed crash from silent `n_ctx` auto-reduction for small models
- Fixed `edge_completion()` echoing prompt in returned text
- Added prompt length validation to prevent C++ abort on context overflow
- Updated `build_chat_prompt()` to use ChatML format

### Test environments
* Local: Windows 11, R 4.5.1, Rtools45 / GCC 14.3.0

### Third-party code
All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
