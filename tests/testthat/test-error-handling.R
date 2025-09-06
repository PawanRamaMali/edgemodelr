test_that("Invalid file paths are handled properly", {
  # Non-existent file
  expect_error(
    edge_load_model("does_not_exist.gguf"),
    "Model file does not exist"
  )
  
  # Empty file path
  expect_error(
    edge_load_model(""),
    "Model file does not exist"
  )
  
  # NULL file path
  expect_error(
    edge_load_model(NULL),
    "invalid 'file' argument"
  )
  
  # Directory instead of file
  if (dir.exists("tests")) {
    expect_error(
      edge_load_model("tests"),
      "Failed to load GGUF model"
    )
  }
})

test_that("Invalid parameters are rejected", {
  # Test edge_load_model with invalid parameters
  dummy_path <- "dummy.gguf"  # Will fail anyway, but testing parameter validation
  
  # Negative context size - should fail due to missing file first
  expect_error(
    suppressWarnings(edge_load_model(dummy_path, n_ctx = -1)),
    "Model file does not exist"
  )
  
  # Zero context size - should fail due to missing file first
  expect_error(
    suppressWarnings(edge_load_model(dummy_path, n_ctx = 0)),
    "Model file does not exist"
  )
  
  # Extremely large context size - should fail due to missing file first
  expect_error(
    suppressWarnings(edge_load_model(dummy_path, n_ctx = 999999999)),
    "Model file does not exist"
  )
  
  # Negative GPU layers - should fail due to missing file first
  expect_error(
    suppressWarnings(edge_load_model(dummy_path, n_gpu_layers = -1)),
    "Model file does not exist"
  )
})

test_that("edge_completion handles errors gracefully", {
  # Test with NULL context
  expect_error(
    edge_completion(NULL, "Hello", n_predict = 5)
  )
  
  # Test with invalid context types
  invalid_contexts <- list("string", 123, list(), data.frame())
  for (ctx in invalid_contexts) {
    expect_error(
      edge_completion(ctx, "Hello", n_predict = 5)
    )
  }
  
  # Test with invalid prompt types
  expect_error(edge_completion(NULL, NULL, n_predict = 5))
  expect_error(edge_completion(NULL, 123, n_predict = 5))
  expect_error(edge_completion(NULL, c("a", "b"), n_predict = 5))
  expect_error(edge_completion(NULL, list("hello"), n_predict = 5))
})


test_that("Corrupted or invalid model files are handled", {
  # Create a fake GGUF file with wrong content
  fake_gguf <- "fake_model.gguf"
  if (!file.exists(fake_gguf)) {
    writeLines("This is not a real GGUF file", fake_gguf)
  }
  
  expect_error(
    edge_load_model(fake_gguf)
  )
  
  # Cleanup
  if (file.exists(fake_gguf)) {
    unlink(fake_gguf)
  }
})



test_that("Error messages are informative", {
  # Test that error messages contain useful information
  tryCatch({
    edge_load_model("clearly_nonexistent_file_xyz123.gguf")
  }, error = function(e) {
    # Error message should mention the file or path
    expect_true(
      grepl("file|path|exist|found", e$message, ignore.case = TRUE),
      info = paste("Error message should be informative:", e$message)
    )
  })
  
  # Test error messages for invalid contexts
  tryCatch({
    edge_completion("invalid_context", "Hello")
  }, error = function(e) {
    expect_true(
      grepl("context|model|invalid", e$message, ignore.case = TRUE),
      info = paste("Error message should mention context:", e$message)
    )
  })
})

test_that("is_valid_model handles all input types", {
  # Test with various invalid inputs
  expect_false(is_valid_model(NULL))
  expect_false(is_valid_model("string"))
  expect_false(is_valid_model(123))
  expect_false(is_valid_model(list()))
  expect_false(is_valid_model(data.frame()))
  expect_false(is_valid_model(TRUE))
  expect_false(is_valid_model(FALSE))
  expect_false(is_valid_model(c(1, 2, 3)))
})

test_that("edge_free_model handles invalid contexts gracefully", {
  # Should not error with NULL
  expect_silent(edge_free_model(NULL))
  
  # Should not error with invalid types
  expect_silent(edge_free_model("invalid"))
  expect_silent(edge_free_model(123))
  expect_silent(edge_free_model(list()))
  
  # Test with missing arguments
  expect_error(edge_free_model())
})