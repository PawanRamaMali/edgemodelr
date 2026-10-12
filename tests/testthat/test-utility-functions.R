# Tests for utility functions that don't require model loading

# ============================================================================
# edge_clean_cache tests
# ============================================================================

test_that("edge_clean_cache handles non-existent cache directory", {

  result <- edge_clean_cache(
    cache_dir = file.path(tempdir(), "nonexistent_cache_dir_xyz"),
    ask = FALSE,
    verbose = FALSE
  )
  expect_equal(length(result), 0)
})

test_that("edge_clean_cache handles empty cache directory", {
  temp_cache <- file.path(tempdir(), "empty_cache_test")
  dir.create(temp_cache, showWarnings = FALSE)
  on.exit(unlink(temp_cache, recursive = TRUE))

  result <- edge_clean_cache(
    cache_dir = temp_cache,
    ask = FALSE,
    verbose = FALSE
  )
  expect_equal(length(result), 0)
})

test_that("edge_clean_cache respects max_age_days parameter", {
  temp_cache <- file.path(tempdir(), "cache_age_test")
  dir.create(temp_cache, showWarnings = FALSE, recursive = TRUE)
  on.exit(unlink(temp_cache, recursive = TRUE))


  # Create a test file
  test_file <- file.path(temp_cache, "test_model.gguf")
  writeLines("test content", test_file)

  # File is new, should not be deleted with default max_age_days=30

result <- edge_clean_cache(
    cache_dir = temp_cache,
    max_age_days = 30,
    ask = FALSE,
    verbose = FALSE
  )

  expect_true(file.exists(test_file))
})

test_that("edge_clean_cache accepts custom parameters", {
  temp_cache <- file.path(tempdir(), "cache_params_test")
  dir.create(temp_cache, showWarnings = FALSE)
  on.exit(unlink(temp_cache, recursive = TRUE))

  # Should not error with various parameter combinations
  expect_silent(edge_clean_cache(
    cache_dir = temp_cache,
    max_age_days = 7,
    max_size_mb = 100,
    ask = FALSE,
    verbose = FALSE
  ))
})

# ============================================================================
# edge_set_verbose tests
# ============================================================================

test_that("edge_set_verbose accepts boolean values", {
  # Should not error
  expect_silent(edge_set_verbose(TRUE))
  expect_silent(edge_set_verbose(FALSE))
})

test_that("edge_set_verbose returns NULL invisibly", {
  result <- edge_set_verbose(FALSE)
  expect_null(result)
})

# ============================================================================
# edge_benchmark tests (without model - error handling)
# ============================================================================

test_that("edge_benchmark requires valid model context", {
  expect_error(
    edge_benchmark(NULL),
    "Invalid model context"
  )
})

test_that("edge_benchmark rejects invalid context types", {
  expect_error(
    edge_benchmark("not_a_context"),
    "Invalid model context"
  )

  expect_error(
    edge_benchmark(list(ctx = NULL)),
    "Invalid model context"
  )
})

# ============================================================================
# edge_find_gguf_models tests
# ============================================================================

test_that("edge_find_gguf_models handles non-existent directories", {
  result <- edge_find_gguf_models(
    source_dirs = c("/nonexistent/path/xyz123"),
    verbose = FALSE
  )
  expect_null(result)
})

test_that("edge_find_gguf_models handles empty directories", {
  temp_dir <- file.path(tempdir(), "empty_model_dir")
  dir.create(temp_dir, showWarnings = FALSE)
  on.exit(unlink(temp_dir, recursive = TRUE))

  result <- edge_find_gguf_models(
    source_dirs = temp_dir,
    verbose = FALSE
  )
  # Should return NULL when no GGUF files found
  expect_null(result)
})

test_that("edge_find_gguf_models respects min_size_mb parameter", {
  temp_dir <- file.path(tempdir(), "small_files_test")
  dir.create(temp_dir, showWarnings = FALSE)
  on.exit(unlink(temp_dir, recursive = TRUE))

  # Create a small file (smaller than default 50MB threshold)
  small_file <- file.path(temp_dir, "small.gguf")
  writeLines("small content", small_file)

  result <- edge_find_gguf_models(
    source_dirs = temp_dir,
    min_size_mb = 50,
    verbose = FALSE
  )
  # Should not find files below threshold
  expect_null(result)
})

test_that("edge_find_gguf_models accepts model_pattern parameter", {
  temp_dir <- file.path(tempdir(), "pattern_test")
  dir.create(temp_dir, showWarnings = FALSE)
  on.exit(unlink(temp_dir, recursive = TRUE))

  # Should not error with pattern
  result <- edge_find_gguf_models(
    source_dirs = temp_dir,
    model_pattern = "llama",
    verbose = FALSE
  )
  expect_null(result)  # No files to find
})

# ============================================================================
# edge_find_ollama_models tests
# ============================================================================

test_that("edge_find_ollama_models handles missing Ollama installation", {
  # Use a directory that definitely doesn't exist
  result <- edge_find_ollama_models(
    ollama_dir = "/nonexistent/ollama/path/xyz123"
  )
  # If Ollama is not installed, should return NULL
  # (may also return models if Ollama IS installed on the system)
  expect_true(is.null(result) || is.list(result))
})

test_that("edge_find_ollama_models respects max_size_gb parameter", {
  # Should accept the parameter without error
  result <- edge_find_ollama_models(
    ollama_dir = "/nonexistent/path",
    max_size_gb = 5
  )
  expect_true(is.null(result) || is.list(result))
})

test_that("edge_find_ollama_models respects test_compatibility parameter", {
  result <- edge_find_ollama_models(
    ollama_dir = "/nonexistent/path",
    test_compatibility = FALSE
  )
  expect_true(is.null(result) || is.list(result))
})

# ============================================================================
# edge_load_ollama_model tests
# ============================================================================

test_that("edge_load_ollama_model errors with invalid hash", {
  # Should error when no Ollama models found
  expect_error(
    edge_load_ollama_model("nonexistent_hash_xyz123"),
    "No Ollama model"
  )
})

test_that("edge_load_ollama_model accepts optional parameters", {
  # Should still error (no models) but accept parameters
  expect_error(
    edge_load_ollama_model("xyz", n_ctx = 1024, n_gpu_layers = 0),
    "No Ollama model"
  )
})

# ============================================================================
# Fixes in 0.4.5
# ============================================================================

age_files <- function(paths, days) {
  old <- Sys.time() - days * 24 * 3600
  for (p in paths) Sys.setFileTime(p, old)
}

test_that("edge_clean_cache keeps the CUDA backend and checksum store", {
  temp_cache <- tempfile("cache_test")
  dir.create(temp_cache)
  on.exit(unlink(temp_cache, recursive = TRUE), add = TRUE)
  dir.create(file.path(temp_cache, "cuda"))
  model <- file.path(temp_cache, "old_model.gguf")
  dll <- file.path(temp_cache, "cuda", "ggml-cuda.dll")
  hashes <- file.path(temp_cache, "known_hashes.csv")
  for (p in c(model, dll, hashes)) writeLines("x", p)
  age_files(c(model, dll, hashes), 60)

  deleted <- edge_clean_cache(cache_dir = temp_cache, max_age_days = 30,
                              ask = FALSE, verbose = FALSE)

  expect_false(file.exists(model))
  expect_true(file.exists(dll))
  expect_true(file.exists(hashes))
  expect_equal(basename(deleted), "old_model.gguf")
})

test_that("edge_clean_cache evicts oldest files until under the size limit", {
  temp_cache <- tempfile("cache_test")
  dir.create(temp_cache)
  on.exit(unlink(temp_cache, recursive = TRUE), add = TRUE)
  big_old <- file.path(temp_cache, "big_old.gguf")
  small_new <- file.path(temp_cache, "small_new.gguf")
  writeBin(raw(600 * 1024), big_old)
  writeBin(raw(300 * 1024), small_new)
  age_files(big_old, 2)

  # 0.88 MB in total against a 0.5 MB limit: removing the oldest file is
  # enough, even though it is larger than the excess.
  edge_clean_cache(cache_dir = temp_cache, max_age_days = 30, max_size_mb = 0.5,
                   ask = FALSE, verbose = FALSE)

  expect_false(file.exists(big_old))
  expect_true(file.exists(small_new))
})

test_that("trust_first_use stores a hash for downloads without a model id", {
  temp_cache <- tempfile("cache_test")
  dir.create(temp_cache)
  on.exit(unlink(temp_cache, recursive = TRUE), add = TRUE)
  model <- file.path(temp_cache, "m.gguf")
  writeLines("model bytes", model)
  skip_if(is.na(edgemodelr:::.compute_sha256(model)), "no SHA-256 tool available")

  expect_true(edgemodelr:::.validate_model_checksum(
    model, model_id = NULL, filename = "m.gguf", cache_dir = temp_cache,
    trust_first_use = TRUE, verbose = FALSE
  ))
  known <- edgemodelr:::.get_known_hashes(temp_cache)
  expect_equal(nrow(known), 1L)
  expect_equal(known$filename, "m.gguf")

  # The stored hash is then used to verify the same file.
  expect_true(edgemodelr:::.validate_model_checksum(
    model, filename = "m.gguf", cache_dir = temp_cache, verbose = FALSE
  ))
})

test_that("edge_download_model passes checksum options through for model names", {
  seen <- NULL
  local_mocked_bindings(edge_download_url = function(...) {
    seen <<- list(...)
    "fake/path.gguf"
  })

  edge_download_model("TinyLlama-1.1B", expected_sha256 = "abc123",
                      trust_first_use = TRUE, verbose = FALSE)

  expect_equal(seen$expected_sha256, "abc123")
  expect_true(seen$trust_first_use)
  expect_true(seen$verify_checksum)
})

test_that(".chunk_text terminates and covers the whole text", {
  text <- strrep("Hello there. ", 100)
  chunks <- edgemodelr:::.chunk_text(text, chunk_size = 500L, overlap = 50L)
  expect_true(length(chunks) >= 3L && length(chunks) <= 5L)
  expect_true(endsWith(chunks[length(chunks)], "Hello there."))

  # Overlap as large as the chunk used to stop pos from advancing.
  chunks <- edgemodelr:::.chunk_text(strrep("abcdefghij", 30), chunk_size = 100L, overlap = 100L)
  expect_true(length(chunks) > 0L)
})
