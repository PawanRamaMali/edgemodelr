## Update - edgemodelr 0.7.0

This release adds a serving engine with continuous batching and
`edge_api()`, an HTTP API compatible with the OpenAI chat completions API
for serving one model to many users, as well as tool calling, saved prompt
caches and an 'ellmer' chat backend. It also updates the bundled llama.cpp
from build b8179 (GGML 0.9.7) to v0.8.0, build b11960 (GGML 0.28.0), and
adds LoRA adapter loading. 0.5.0 and 0.6.0 were not submitted; their
changes are part of this release. See NEWS.md.

The serving engine runs generation on a background C++ thread. That thread
never calls the R API: R only submits requests and polls for results from
the main thread, and every engine is stopped and joined when the package is
unloaded.

The CRAN patches to the bundled code are carried over: no stdio, `abort()`
or `exit()` in compiled code, and no diagnostic pragmas.

### R CMD check --as-cran results

0 ERRORs. 0 WARNINGs. 0 NOTEs.

The C++ specification, installed size and "GNU make is a
SystemRequirements" INFOs are inherent to the bundled inference engine.

### Test environments

* Local: Windows 11, R 4.6.0, Rtools45
* GitHub Actions: ubuntu-latest (R devel, release, oldrel-1), macos-latest
  and macos-15-intel (R release), windows-latest (R release), clang with
  libc++, and an ASAN/UBSAN build

### Reverse dependencies

There are no reverse dependencies on CRAN.

### Third-party code

All bundled code (llama.cpp v0.8.0, build b11960, GGML 0.28.0) is credited
in DESCRIPTION Authors@R and inst/COPYRIGHTS.
