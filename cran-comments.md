## Submission - edgemodelr 0.2.0

### Changes in 0.2.0

#### Bug Fixes
- Fixed SHA-256 checksum validation (was hashing the path string, not file contents)
- Fixed HuggingFace token authentication in downloads (now uses curl with proper Authorization header)
- Fixed model alias resolution with case-insensitive and partial prefix matching
- Fixed `edge_clean_cache()`: renamed `interactive` parameter to `ask` to avoid shadowing base R `interactive()`
- Fixed GGUF magic byte validation to use raw byte comparison throughout (`identical(header, charToRaw("GGUF"))`)
- Fixed `edge_completion()` and `edge_stream_completion()` to validate model context early
- Fixed `edge_chat_stream()` response extraction to use streamed response directly
- Added input validation to `build_chat_prompt()`

#### CRAN Compliance
- Default build is now fully portable (no `-msse4.2` or other non-portable flags); SIMD acceleration is opt-in via `EDGEMODELR_SIMD` environment variable
- All examples that download models use `\dontrun{}`
- Removed development-only files from the repository

### Test environments
* local Windows 11, R 4.5.1
* GitHub Actions:
  - ubuntu-latest: R (release, devel, oldrel-1)
  - macOS-latest: R (release)
  - windows-latest: R (release)

### R CMD check results
There were no ERRORs or WARNINGs.

There was 1 NOTE:
* checking installed package size ... NOTE
  installed size is ~8MB
  - Expected: the package bundles the llama.cpp inference engine (~56MB C/C++ source, ~8MB compiled)

### Third-party code
All bundled code (llama.cpp, GGML) is credited in DESCRIPTION Authors@R and inst/COPYRIGHTS.
