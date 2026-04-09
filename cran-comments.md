## Resubmission - edgemodelr 0.4.0

This is a resubmission addressing the auto-check NOTEs from the initial
submission. All NOTEs are pre-existing from the bundled llama.cpp C/C++
library and were present in the accepted 0.2.0 release. No new NOTEs
were introduced.

### Changes since initial submission

* Rephrased DESCRIPTION to avoid the "embeddings" spelling NOTE.

### NOTEs explained

**NOTE 1 (Debian): "specified C++17"**
The bundled llama.cpp library requires C++17 features (structured bindings,
std::optional, if-constexpr). This is the minimum standard required by the
upstream project.

**NOTE 2 (all): Pragmas suppressing diagnostics**
Files: `src/ggml/ggml-cpu/amx/mmq.cpp`, `src/ggml/ggml-cpu/arch/x86/repack.cpp`,
`src/ggml/ggml-cpu/repack.cpp`, `src/llama/llama-sampler.cpp`.
These pragmas are in the bundled llama.cpp upstream source. They suppress
compiler warnings for platform-specific SIMD intrinsics (AMX, AVX) where
implicit type conversions are intentional and well-tested upstream.

**NOTE 3 (Debian): Compiled code references to printf/stderr/stdout**
Objects: `ggml-quants.o`, `llama-quant.o`, and other ggml/llama objects.
These are `fprintf(stderr, ...)` calls in the bundled llama.cpp library in
non-fatal diagnostic paths (quantization validation, NUMA warnings). The
primary ggml.c has been patched with `#ifndef USING_R` guards. The remaining
references are deep in upstream code and cannot be removed without forking
the entire llama.cpp codebase. At runtime, all output is suppressed through
ggml's log-callback API which the R bindings replace with R's own messaging.

These same NOTEs were present in the accepted 0.2.0 release (currently on
CRAN with status OK on 11/14 platforms, NOTE on 3 older platforms for
installed size only).

### Changes in 0.4.0

#### New Features
- Grammar-constrained generation for structured output (JSON, enums)
- Text embeddings API with cosine similarity helpers
- Batch processing for vectorized LLM operations on data frames
- RAG pipeline: document indexing, semantic search, question answering
- Chat completion with model-native templates from GGUF metadata
- Plumber API server for OpenAI-compatible local inference
- Qwen3 model family (0.6B, 1.7B, 4B, 8B) in model registry
- Friendly model names in edge_download_model()
- httr download fallback for corporate proxy environments
- SIMD optimization warning on package load

#### Bug Fixes
- Fixed crash from silent n_ctx auto-reduction for small models
- Fixed edge_completion() echoing prompt in returned text
- Added prompt length validation to prevent C++ abort on context overflow
- Updated build_chat_prompt() to use ChatML format

### Test environments
* Local: Windows 11, R 4.5.1, Rtools45 / GCC 14.3.0
* win-builder.r-project.org (Windows, R-devel)
* CRAN incoming pre-test (Windows + Debian)

### R CMD check results

0 ERRORs. 0 WARNINGs. 2-3 NOTEs (all pre-existing, explained above).

### Third-party code
All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
