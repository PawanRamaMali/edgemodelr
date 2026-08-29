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

### Fix 2: C++ standard pinned to C++17

Comparing the install logs across the macOS flavours isolates the
trigger precisely. r-oldrel-macos-x86_64 and r-release-macos-x86_64 use
the same compiler (Apple clang 14.0.3) and the same SDK
(MacOSX11.3.1.sdk) and differ only in the language standard R selects:

| Flavour                 | Standard    | 0.4.1 result |
|-------------------------|-------------|--------------|
| r-oldrel-macos-x86_64   | `gnu++17`   | OK           |
| r-release-macos-x86_64  | `gnu++20`   | ERROR        |
| r-release-macos-arm64   | `gnu++20`   | OK           |

The bundled llama.cpp and GGML sources target C++17, which is the
standard upstream builds and tests against. With no `CXX_STD` set, the
package inherited whichever default each platform applied, so the same
sources were compiled as C++17 under R 4.5 and as C++20 under R 4.6.

`src/Makevars` and `src/Makevars.win` now set `CXX_STD = CXX17`. This
puts r-release-macos-x86_64 on exactly the configuration under which
r-oldrel-macos-x86_64 already installs cleanly, and keeps every other
flavour on the dialect the vendored engine is written for.

### Fix 3: clang 23 (ERROR: installation failed)

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
  windows-latest, macos-latest (ARM64), macos-15-intel (x86_64),
  macOS Strict (M1/ARM64), Sanitizers (ASAN/UBSAN),
  CRAN-ubuntu/windows/macos

An x86_64 macOS job was added to the CI matrix for this release, since
the previous matrix covered only ARM64 on macOS. All 161 C++ translation
units were additionally compiled under both `-std=gnu++17` and
`-std=gnu++20` locally.

The specific SDK that failed (MacOSX11.sdk) is not available on any
hosted CI image, so the argument for Fix 1 rests on the mechanism rather
than on a reproduction: the feature-test macro no longer gates anything,
and the replacement expression is well formed for both possible return
types. Fix 2 removes the exposure independently by keeping that
configuration on C++17, which the CRAN check page already shows building
cleanly on the identical compiler and SDK.

### Third-party code

All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
