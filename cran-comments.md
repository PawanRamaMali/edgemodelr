## Resubmission - edgemodelr 0.4.3

This addresses both points raised by Uwe Ligges on the 0.4.2 submission.
There are no user-visible changes to the R API.

### 1. Missing `<cstdlib>` in four translation units

The reported errors were:

```
llama/llama-mmap.cpp:303:19: error: use of undeclared identifier 'posix_memalign'
llama/llama-mmap.cpp:309:47: error: use of undeclared identifier 'free'
llama/llama-model-loader.cpp:512:9: error: use of undeclared identifier 'getenv'
llama/llama-vocab.cpp:2732:20: error: use of undeclared identifier 'strtol'
```

All three files now include `<cstdlib>`.

Rather than patch only the reported lines, every file in `src/` was
audited for the same defect: for each translation unit and header, the
set of standard headers reachable through its own includes and through
the package's local headers was computed, and the file body was scanned
for C library symbols that set does not cover.

That audit found a fourth file, `llama/llama-batch.cpp`, which calls
`getenv`, `atoi`, `malloc` and `free` with no `<cstdlib>` in scope. It
did not appear in the compiler report because the build stopped at
`llama-mmap.cpp` before reaching it. It would have failed this
resubmission had it not been fixed.

The audit reports five remaining hits, all verified by hand to be false
positives rather than defects:

* `ggml-backend-impl.h`, `ggml-backend.h`, `llama.h`: `free` appears as a
  struct member name and as a parameter name, never as a call.
* `ggml.h`: `abort()` appears inside `GGML_UNREACHABLE()`, which is
  defined only under `#ifndef NDEBUG`; the package always compiles with
  `-DNDEBUG`.
* `unicode.cpp`: `fprintf` appears inside `#ifndef USING_R`; the package
  always compiles with `-DUSING_R=1`.

### 2. CFLAGS and CXXFLAGS in subdirectory compilations

This was a real defect and is now fixed. The pattern rules for `ggml/`
and `llama/` invoked the compiler as

```make
	$(CXX) $(ALL_CPPFLAGS) $(GGML_CXXFLAGS) -c $< -o $@
```

where `GGML_CXXFLAGS` held only package-local flags. `$(ALL_CXXFLAGS)`,
which is where `CXXFLAGS` from the user or the site `Makevars` arrives,
was absent. Flags were therefore honoured for 6 of the 161 source files
and silently dropped for the other 155.

Every rule in `src/Makevars` and `src/Makevars.win` now reads

```make
	$(CXX) $(ALL_CPPFLAGS) $(ALL_CXXFLAGS) $(GGML_EXTRA_CXXFLAGS) -c $< -o $@
```

The engine-specific variables were renamed from `GGML_CXXFLAGS` to
`GGML_EXTRA_CXXFLAGS` (and likewise for the C and generic variants) so
that their role as additions rather than replacements is explicit at
every use site. Redundant `-fPIC` and `-DNDEBUG` entries were dropped
from them, since both now arrive through `$(ALL_*FLAGS)`.

Verified after the change: all 161 compile commands in the install log
carry `$(ALL_CFLAGS)` or `$(ALL_CXXFLAGS)`.

### C++ standard

`CXX_STD = CXX17` was added in 0.4.2 and is retained. The bundled
llama.cpp and GGML sources target C++17. This also resolved the
r-release-macos-x86_64 install failure, where `-std=gnu++20` combined
with the libc++ in MacOSX11.sdk broke `std::filesystem::path::u8string()`
handling in `ggml-backend-reg.cpp`.

### R CMD check --as-cran results

0 ERRORs. 0 WARNINGs. 0 NOTEs.

The installed size INFO and the "GNU make is a SystemRequirements" INFO
are unchanged and both are inherent to the bundled inference engine.

### Test environments

* Local: Windows 11, R 4.5.1, Rtools45 / GCC 14.3.0
* GitHub Actions: ubuntu-latest (devel/release/oldrel-1),
  windows-latest, macos-latest (ARM64), macos-15-intel (x86_64),
  macOS Strict (M1/ARM64), Sanitizers (ASAN/UBSAN),
  CRAN-ubuntu/windows/macos

All 161 C++ and 6 C translation units were additionally compiled under
both `-std=gnu++17` and `-std=gnu++20`.

No hosted CI image provides clang 23 or MacOSX11.sdk, so neither
reported configuration can be reproduced directly. The missing includes
were instead found and confirmed absent by the tree-wide audit described
above, which is repeatable and covers every file rather than only those
a build happened to reach before stopping.

### Third-party code

All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
