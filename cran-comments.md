## Resubmission - edgemodelr 0.1.5

This is an urgent fix addressing CRAN's warning email (deadline: 2026-02-06).

### Issues Fixed in 0.1.5

#### 1. donttest Example Failure - FIXED
- Changed all `\donttest{}` to `\dontrun{}` for examples that download models
- This prevents the 4GB model downloads during `--run-donttest` checks
- Examples are still documented but won't run during CRAN checks

#### 2. M1 Mac Compiler Warnings - FIXED
From https://www.stats.ox.ac.uk/pub/bdr/M1mac/edgemodelr.log

Added explicit `static_cast<>` in `bindings.cpp`:
- Lines 243, 246, 390, 393: `static_cast<float>()` for temperature/top_p
- Lines 262, 267, 410, 415: `static_cast<int32_t>()` for buffer sizes

#### 3. Connection Handling - FIXED
- Fixed `on.exit()` issue in loops (reported by @eddelbuettel)
- Now uses `tryCatch/finally` for proper connection cleanup

### Test environments
* local Windows 11 install, R 4.5.2
* GitHub Actions:
  - ubuntu-latest: R (release, devel, oldrel-1)
  - macOS-latest: R (release)
  - windows-latest: R (release)

### R CMD check results
There were no ERRORs or WARNINGs.

There was 1 NOTE:
* checking CRAN incoming feasibility ... NOTE
  - New submission (version bump from 0.1.4 to 0.1.5)

### Previous Archive Issues (addressed in 0.1.4)

All third-party code contributors are properly credited in:
- DESCRIPTION: Authors@R field
- inst/COPYRIGHTS: Detailed attribution

Platform-dependent code is properly documented in SystemRequirements.
