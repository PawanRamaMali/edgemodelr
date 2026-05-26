## Resubmission - edgemodelr 0.4.1

This resubmission addresses the compiled-code NOTE flagged by CRAN's
automated pre-check on 0.4.0 (Debian flavor):

```
File 'edgemodelr/libs/edgemodelr.so':
  Found 'stderr', possibly from 'stderr' (C)
    Objects: 'ggml/ggml.o', 'ggml/ggml-opt.o'
```

### Fix applied

The previous CRAN cleanup (0.4.0) added an `#ifdef USING_R` macro block
that neutralizes `printf`, `fprintf`, `fputs`, `fflush`, `stderr`, and
`stdout` to seven upstream files, but missed two:

* `src/ggml/ggml.c` — `ggml_log_callback_default()` (line 327) calls
  `fputs(text, stderr); fflush(stderr);`. The R bindings install their
  own log callback via `llama_log_set` before any logging happens, so
  this default callback is never invoked at runtime. Backtrace and
  diagnostic `fprintf(stderr, ...)` calls elsewhere in the file are
  similarly dead code because the R abort callback does `longjmp` first.

* `src/ggml/ggml-opt.cpp` — `ggml_opt_fit()` renders a training-progress
  bar to stderr (16 calls between lines 933 and 1072). The optimizer is
  not exposed through the R bindings; this code is dead in the package.

Both files now include the same `#ifdef USING_R` block used in the
seven files cleaned up in 0.4.0. The `stderr` symbol no longer appears
in either compiled object.

### R CMD check --as-cran results

0 ERRORs. 0 WARNINGs. 0 NOTEs (informational "GNU make is a
SystemRequirements" is documented in DESCRIPTION).

### Carried over from 0.4.0

This release also includes the grammar-constrained-generation fixes
that landed between submissions (issue #41):

* `edge_json_grammar()` previously emitted GBNF rule names containing
  underscores, which llama.cpp's grammar parser rejects (only
  `[a-zA-Z0-9-]` is allowed). Renamed to use hyphens.

* `llama_sampler_accept()` throws "Unexpected empty grammar stack" when
  a token completes the grammar. The binding now catches this and
  terminates cleanly, same as end-of-generation handling.

### Test environments

* Local: Windows 11, R 4.6.0, Rtools45 / GCC 14.3.0
* GitHub Actions: ubuntu-latest (devel/release/oldrel-1),
  windows-latest, macos-latest, macOS Strict (M1/ARM64), Sanitizers
  (ASAN/UBSAN), CRAN-ubuntu/windows/macos — all green on the
  submission commit.

### Third-party code

All bundled code (llama.cpp build b8179, GGML 0.9.7) is credited in
DESCRIPTION Authors@R and inst/COPYRIGHTS.
