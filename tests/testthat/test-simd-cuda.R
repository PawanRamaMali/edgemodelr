# edge_simd_info() and the CUDA helpers. The CUDA tests never touch the
# network or a real GPU: downloads and the native backend calls are mocked,
# and the edgemodelr cache is redirected into tempdir().

test_that("edge_simd_info describes the build", {
  info <- edge_simd_info()
  expect_type(info, "list")
  expect_true(all(c("architecture", "compiler_features", "ggml_features",
                    "is_generic", "kernels") %in% names(info)))
  expect_type(info$architecture, "character")
  expect_length(info$architecture, 1L)
  expect_true(nzchar(info$architecture))
  expect_type(info$compiler_features, "character")
  expect_type(info$ggml_features, "character")
  expect_true(is.logical(info$is_generic) && length(info$is_generic) == 1L &&
                !is.na(info$is_generic))
  expect_true(info$kernels %in% c("avx2", "generic", "native"))
  if (info$is_generic) {
    expect_true("GGML_CPU_GENERIC" %in% info$ggml_features)
    expect_true(info$kernels %in% c("avx2", "generic"))
  } else {
    expect_identical(info$kernels, "native")
  }
  # Calling it twice gives the same answer.
  expect_identical(edge_simd_info(), info)
})

test_that("edge_cuda_info reports nothing installed in an empty cache", {
  local_temp_cache()
  local_mocked_bindings(
    edge_cuda_backend_path_internal = function() "",
    edge_cuda_backend_loaded_internal = function() FALSE,
    .package = "edgemodelr"
  )
  info <- edge_cuda_info()
  expect_identical(names(info), c("installed", "active", "path"))
  expect_false(info$installed)
  expect_false(info$active)
  expect_identical(info$path, NA_character_)
})

test_that("edge_cuda_info finds an installed backend and prefers the active one", {
  cuda_dir <- local_temp_cache()
  dir.create(cuda_dir, recursive = TRUE)
  dll <- file.path(cuda_dir, if (.Platform$OS.type == "windows")
    "ggml-cuda-13.4.dll" else "libggml-cuda.so")
  writeLines("fake", dll)

  local_mocked_bindings(
    edge_cuda_backend_path_internal = function() "",
    edge_cuda_backend_loaded_internal = function() FALSE,
    .package = "edgemodelr"
  )
  info <- edge_cuda_info()
  expect_true(info$installed)
  expect_false(info$active)
  expect_identical(normalizePath(info$path), normalizePath(dll))

  local_mocked_bindings(
    edge_cuda_backend_path_internal = function() "C:/elsewhere/ggml-cuda.dll",
    edge_cuda_backend_loaded_internal = function() TRUE,
    .package = "edgemodelr"
  )
  info <- edge_cuda_info()
  expect_true(info$active)
  expect_identical(info$path, "C:/elsewhere/ggml-cuda.dll")
})

test_that("edge_reload_cuda errors when no backend is installed", {
  local_temp_cache()
  expect_error(edge_reload_cuda(), "No CUDA backend found")
})

test_that("edge_reload_cuda warns and returns FALSE for a missing file", {
  local_temp_cache()
  missing <- file.path(tempdir(), "no-such-ggml-cuda.dll")
  expect_warning(res <- edge_reload_cuda(missing), "CUDA backend not found")
  expect_false(res)
})

test_that("edge_reload_cuda registers the default backend and extends PATH", {
  cuda_dir <- local_temp_cache()
  local_path_restore()
  dir.create(cuda_dir, recursive = TRUE)
  dll <- file.path(cuda_dir, if (.Platform$OS.type == "windows")
    "ggml-cuda-13.4.dll" else "libggml-cuda.so")
  writeLines("fake", dll)

  seen <- NULL
  local_mocked_bindings(
    edge_use_cuda_backend_internal = function(path) {
      seen <<- path
      TRUE
    },
    .package = "edgemodelr"
  )
  msgs <- capture_messages(res <- edge_reload_cuda())
  expect_match(paste(msgs, collapse = ""), "CUDA backend registered")
  expect_true(res)
  expect_identical(normalizePath(seen), normalizePath(dll))
  cuda_native <- normalizePath(cuda_dir, winslash = "\\", mustWork = FALSE)
  expect_true(grepl(cuda_native, Sys.getenv("PATH"), fixed = TRUE))

  # A failed registration is reported as FALSE, without the success message.
  local_mocked_bindings(
    edge_use_cuda_backend_internal = function(path) FALSE,
    .package = "edgemodelr"
  )
  expect_no_message(res <- edge_reload_cuda(dll))
  expect_false(res)
})

test_that("edge_install_cuda and edge_install_cuda_toolkit refuse non-Windows", {
  skip_on_os("windows")
  local_temp_cache()
  expect_error(edge_install_cuda(), "supports Windows only")
  expect_error(edge_install_cuda_toolkit(), "only needed on Windows")
})

test_that("edge_install_cuda skips the download when already installed", {
  skip_if_not(.Platform$OS.type == "windows")
  cuda_dir <- local_temp_cache()
  dir.create(cuda_dir, recursive = TRUE)
  dest <- file.path(cuda_dir, "ggml-cuda-12.4.dll")
  writeLines("fake", dest)

  activated <- NULL
  local_mocked_bindings(
    edge_activate_cuda = function(path) {
      activated <<- path
      invisible(TRUE)
    },
    .package = "edgemodelr"
  )
  local_mocked_bindings(
    download.file = function(...) stop("download.file must not be called"),
    .package = "utils"
  )
  msgs <- capture_messages(res <- edge_install_cuda("12.4"))
  expect_match(paste(msgs, collapse = ""), "already installed")
  expect_identical(res, dest)
  expect_identical(activated, dest)
})

test_that("edge_install_cuda downloads, extracts and renames the backend (mocked)", {
  skip_if_not(.Platform$OS.type == "windows")
  skip_if(!nzchar(Sys.which(Sys.getenv("R_ZIPCMD", "zip"))), "no zip program")
  cuda_dir <- local_temp_cache()

  # A small stand-in for the llama.cpp release archive.
  src <- tempfile("fake-release-")
  dir.create(src)
  on.exit(unlink(src, recursive = TRUE), add = TRUE)
  writeLines("cuda", file.path(src, "ggml-cuda.dll"))
  writeLines("base", file.path(src, "ggml-base.dll"))
  writeLines("ggml", file.path(src, "ggml.dll"))
  writeLines("other", file.path(src, "llama.dll"))
  fake_zip <- file.path(src, "release.zip")
  old_wd <- setwd(src)
  utils::zip(fake_zip, c("ggml-cuda.dll", "ggml-base.dll", "ggml.dll", "llama.dll"),
             flags = "-q")
  setwd(old_wd)

  requested <- NULL
  local_mocked_bindings(
    download.file = function(url, destfile, ...) {
      requested <<- url
      file.copy(fake_zip, destfile, overwrite = TRUE)
      invisible(0L)
    },
    .package = "utils"
  )
  local_mocked_bindings(
    edge_activate_cuda = function(path) invisible(TRUE),
    .package = "edgemodelr"
  )

  msgs <- capture_messages(res <- edge_install_cuda("13.4", llama_build = "b1"))
  expect_match(paste(msgs, collapse = ""), "CUDA backend installed")
  expect_identical(requested, paste0(
    "https://github.com/ggml-org/llama.cpp/releases/download/b1/",
    "llama-b1-bin-win-cuda-13.4-x64.zip"))
  expect_identical(res, file.path(cuda_dir, "ggml-cuda-13.4.dll"))
  expect_true(file.exists(res))
  expect_identical(readLines(res), "cuda")
  expect_true(file.exists(file.path(cuda_dir, "ggml-base.dll")))
  expect_true(file.exists(file.path(cuda_dir, "ggml.dll")))
  expect_false(file.exists(file.path(cuda_dir, "llama.dll")))
  expect_false(file.exists(file.path(cuda_dir, "ggml-cuda.dll")))
})

test_that("edge_install_cuda reports a failed download with its URL", {
  skip_if_not(.Platform$OS.type == "windows")
  local_temp_cache()
  local_mocked_bindings(
    download.file = function(...) stop("simulated outage"),
    .package = "utils"
  )
  expect_error(
    suppressMessages(edge_install_cuda("13.4", llama_build = "b1")),
    "Failed to download CUDA backend: simulated outage.*llama-b1-bin-win-cuda-13.4-x64.zip"
  )
})

test_that("edge_install_cuda stops when the archive has no CUDA backend", {
  skip_if_not(.Platform$OS.type == "windows")
  skip_if(!nzchar(Sys.which(Sys.getenv("R_ZIPCMD", "zip"))), "no zip program")
  local_temp_cache()
  src <- tempfile("fake-release-")
  dir.create(src)
  on.exit(unlink(src, recursive = TRUE), add = TRUE)
  writeLines("other", file.path(src, "llama.dll"))
  fake_zip <- file.path(src, "release.zip")
  old_wd <- setwd(src)
  utils::zip(fake_zip, "llama.dll", flags = "-q")
  setwd(old_wd)
  local_mocked_bindings(
    download.file = function(url, destfile, ...) {
      file.copy(fake_zip, destfile, overwrite = TRUE)
      invisible(0L)
    },
    .package = "utils"
  )
  expect_error(suppressMessages(edge_install_cuda("13.4")),
               "Could not find ggml-cuda DLL")
})

test_that("edge_install_cuda_toolkit does nothing when the runtime is present", {
  skip_if_not(.Platform$OS.type == "windows")
  cuda_dir <- local_temp_cache()
  dir.create(cuda_dir, recursive = TRUE)
  writeLines("fake", file.path(cuda_dir, "cublas64_13.dll"))
  writeLines("fake", file.path(cuda_dir, "nvcudart_hybrid64.dll"))
  local_mocked_bindings(
    download.file = function(...) stop("download.file must not be called"),
    .package = "utils"
  )
  msgs <- capture_messages(res <- edge_install_cuda_toolkit())
  expect_match(paste(msgs, collapse = ""), "already installed")
  expect_identical(res, cuda_dir)
  expect_setequal(list.files(cuda_dir), c("cublas64_13.dll", "nvcudart_hybrid64.dll"))
})
