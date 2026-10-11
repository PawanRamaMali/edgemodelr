## Update - edgemodelr 0.4.5

This release fixes the WARNING on the current CRAN version (0.4.3), on
r-devel-linux-x86_64-debian-clang, r-devel-linux-x86_64-fedora-clang and
the M1mac additional check:

```
ggml/ggml-common.h:79:9: warning: keyword is hidden by macro definition [-Wkeyword-macro]
```

C23 made `static_assert` a keyword, and the bundled header defined a
macro with that name. The macro is now only defined for C standards
before C23. This fix was prepared as 0.4.4, which was not submitted;
0.4.5 contains it.

### Other changes since 0.4.3

* `edge_load_model()` checks the GGUF magic header before calling the
  bundled loader, so a file that is not a model gives an R error instead
  of a crash.
* Internal assertions in the bundled engine on R's thread now raise an R
  error instead of `SIGABRT`.
* Prompts longer than the batch size, the KV cache carrying over between
  calls, user interrupts during generation, and several R-level bugs in
  downloading, caching and document chunking are fixed. See NEWS.md.

### R CMD check --as-cran results

0 ERRORs. 0 WARNINGs. 0 NOTEs.

The C++ specification, installed size and "GNU make is a SystemRequirements"
INFOs are unchanged and are inherent to the bundled inference engine.

### Test environments

* Local: Windows 11, R 4.6.0, Rtools45

### Third-party code

All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
