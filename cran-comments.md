## Submission - edgemodelr 0.4.0

### Changes in 0.4.0

#### New Features
- Grammar-constrained generation (`edge_grammar_completion()`) for structured
  output using llama.cpp's native GBNF grammar sampler.
- JSON schema helper (`edge_json_grammar()`) converts R list schemas to GBNF.
- Structured data extraction (`edge_extract()`, `edge_extract_batch()`) with
  automatic JSON parsing.
- Text classification (`edge_classify()`) with grammar-constrained categories.
- Text embeddings API (`edge_embeddings()`) returning numeric matrix, with
  cosine similarity helpers (`edge_similarity()`, `edge_similarity_matrix()`).
- Batch processing (`edge_map()`) for vectorized LLM operations on data frames.
- RAG pipeline: `edge_index_documents()`, `edge_search()`, `edge_ask()` for
  retrieval-augmented generation over local documents.
- Chat completion with model-native templates (`edge_chat_completion()`) using
  `llama_chat_apply_template()` from GGUF metadata.

#### Bug Fixes
- Fixed crash from silent `n_ctx` auto-reduction for small models. Removed the
  auto-optimization that silently changed the user's context size, which caused
  segfaults when prompts exceeded the reduced context.
- Fixed `edge_completion()` echoing the prompt in the returned text. Now returns
  only the generated response.
- Added prompt length validation in all completion functions. Exceeding the
  context window now raises a recoverable R error instead of a C++ abort.
- Updated `build_chat_prompt()` to use ChatML as generic fallback (replacing
  `Human:/Assistant:` format) and accept optional model context for native
  template formatting.
- Lowered minimum `n_ctx` from 512 to 128 for short-task use cases.

### Test environments
* Local: Windows 11, R 4.5.1, Rtools45 / GCC 14.3.0
* R CMD check --as-cran (tarball)

### R CMD check results

0 ERRORs. 0 WARNINGs. 1 NOTE.

NOTE: checking pragmas in C/C++ headers and code
Files which contain pragma(s) suppressing diagnostics:
  src/ggml/ggml-cpu/amx/mmq.cpp
  src/ggml/ggml-cpu/arch/x86/repack.cpp
  src/ggml/ggml-cpu/repack.cpp
  src/llama/llama-sampler.cpp

These pragmas are in the bundled llama.cpp upstream source (build b8179,
GGML 0.9.7). They suppress compiler warnings for platform-specific SIMD
intrinsics (AMX, AVX) where the compiler may warn about implicit conversions
that are intentional and well-tested in the upstream project. These cannot be
removed without forking the llama.cpp codebase.

### Third-party code
All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
