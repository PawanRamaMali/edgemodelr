# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the edgemodelr R package.

## Workflows

### Core CI/CD Workflows

1. **R-CMD-check.yml** - Main package checking workflow
   - Runs `R CMD check` on multiple platforms (Windows, macOS, Linux)
   - Tests with multiple R versions (devel, release, oldrel-1)
   - Ensures package builds and passes standard R package checks

2. **comprehensive-tests.yml** - Extended testing with model downloads
   - Downloads actual model files for full integration testing
   - Runs the comprehensive test suite (`run_tests.R`)
   - Tests on all platforms with real inference workloads
   - Scheduled to run daily

3. **test-coverage.yml** - Code coverage analysis
   - Generates test coverage reports using `covr`
   - Uploads coverage to Codecov
   - Helps maintain test quality

### Documentation & Quality

4. **pkgdown.yml** - Package website generation
   - Builds package documentation website using pkgdown
   - Deploys to GitHub Pages
   - Updates on every push to main/master

5. **lint.yml** - Code quality and style checking
   - Runs `lintr` for R code quality
   - Checks code style with `styler`
   - Enforces consistent code formatting

### Release Management

6. **release.yml** - Automated releases
   - Triggered by version tags (e.g., `v0.1.0`)
   - Builds source and binary packages
   - Creates GitHub releases with installation instructions
   - Uploads package artifacts

## Status Badges

Add these badges to your README.md:

```markdown
[![R-CMD-check](https://github.com/PawanRamaMali/edgemodelr/workflows/R-CMD-check/badge.svg)](https://github.com/PawanRamaMali/edgemodelr/actions)
[![Codecov test coverage](https://codecov.io/gh/PawanRamaMali/edgemodelr/branch/main/graph/badge.svg)](https://app.codecov.io/gh/PawanRamaMali/edgemodelr?branch=main)
[![Comprehensive Tests](https://github.com/PawanRamaMali/edgemodelr/workflows/comprehensive-tests/badge.svg)](https://github.com/PawanRamaMali/edgemodelr/actions)
```

## Workflow Triggers

- **Push to main/master**: All workflows except release
- **Pull Requests**: All workflows except release and pkgdown deployment
- **Version Tags**: Release workflow only
- **Daily Schedule**: Comprehensive tests (2 AM UTC)
- **Manual Trigger**: pkgdown can be triggered manually

## Requirements

- Workflows automatically install required system dependencies
- C++17 compiler support is ensured on all platforms
- Model downloads use curl for testing with actual GGUF files
- Secrets required: `GITHUB_TOKEN` (automatically provided)