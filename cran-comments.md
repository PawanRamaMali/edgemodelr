## Update - edgemodelr 0.6.0

This release updates the bundled llama.cpp from build b8179 (GGML 0.9.7)
to v0.8.0, build b11960 (GGML 0.28.0), and adds LoRA adapter loading,
faster defaults for CPU inference and several new functions. 0.5.0 was not
submitted; its changes are part of this release. See NEWS.md.

The CRAN patches to the bundled code are carried over: no stdio, `abort()`
or `exit()` in compiled code, and no diagnostic pragmas.

### R CMD check --as-cran results

0 ERRORs. 0 WARNINGs. 0 NOTEs.

The C++ specification, installed size (6.9 Mb, of which libs 6.3 Mb) and
"GNU make is a SystemRequirements" INFOs are inherent to the bundled
inference engine.

### Test environments

* Local: Windows 11, R 4.6.0, Rtools45
* GitHub Actions: ubuntu-latest (R devel, release, oldrel-1), macos-latest
  and macos-15-intel (R release), windows-latest (R release), clang with
  libc++, and an ASAN/UBSAN build

### Third-party code

All bundled code (llama.cpp v0.8.0, build b11960, GGML 0.28.0) is credited
in DESCRIPTION Authors@R and inst/COPYRIGHTS.
