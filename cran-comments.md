## Submission - edgemodelr 0.4.2

This release fixes the installation failures reported on the CRAN check
page for 0.4.1. Both are compiler portability problems in the bundled
llama.cpp/GGML sources; there are no user-visible changes to the R API.

### Fix 1: r-release-macos-x86_64 (ERROR: installation failed)

```
ggml/ggml-backend-reg.cpp:125:29: error: no viable conversion from
'basic_string<char, char_traits<char>, allocator<char>>' to
'const basic_string<char8_t, char_traits<char8_t>, allocator<char8_t>>'
        const std::u8string u8str = path.u8string();
```

`path_str()` selected its conversion branch with
`#if defined(__cpp_lib_char8_t)`. That macro is not a reliable proxy for
the return type of `std::filesystem::path::u8string()`: the libc++
shipped in MacOSX11.sdk defines it when compiling with `-std=gnu++20`
while still returning `std::string` from `u8string()`, so the branch
written for `std::u8string` did not compile.

The function now deduces the return type with `auto` and converts
byte-wise via `reinterpret_cast<const char *>` on `data()`. This is
valid whether the element type is `char` or `char8_t`, and removes the
dependency on the feature-test macro entirely. The two code paths were
verified by compiling the file under both `-std=gnu++17` (where
`u8string()` returns `std::string`) and `-std=gnu++20` (where it returns
`std::u8string`).

### Fix 2: clang 23 (ERROR: installation failed)

```
ggml/gguf.cpp:847:94: error: use of undeclared identifier 'errno'
llama/llama-graph.h:84:48: error: use of undeclared identifier 'getenv'
llama/llama-graph.h:85:43: error: use of undeclared identifier 'atoi'
llama/llama-context.cpp:165:50: error: use of undeclared identifier 'getenv'
llama/llama-context.cpp:166:60: error: use of undeclared identifier 'atoi'
```

Three translation units used symbols without including the header that
declares them, relying on transitive includes that recent libc++
releases no longer provide. `ggml/gguf.cpp` now includes `<cerrno>`;
`llama/llama-graph.h` and `llama/llama-context.cpp` now include
`<cstdlib>`.

### R CMD check --as-cran results

0 ERRORs. 0 WARNINGs. 0 NOTEs (informational "GNU make is a
SystemRequirements" is documented in DESCRIPTION).

The installed size INFO (9.8Mb, of which libs is 9.2Mb) is inherent to
the bundled inference engine and unchanged from previous releases.

### Test environments

* Local: Windows 11, R 4.5.1, Rtools45 / GCC 14.3.0
* GitHub Actions: ubuntu-latest (devel/release/oldrel-1),
  windows-latest, macos-latest, macOS Strict (M1/ARM64), Sanitizers
  (ASAN/UBSAN), CRAN-ubuntu/windows/macos

Neither failing configuration is reproducible with GCC 14 / libstdc++,
so the fixes were validated by compiling the affected files under both
C++17 and C++20 and by confirming that the error mechanism in each case
is addressed at its source.

### Third-party code

All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
