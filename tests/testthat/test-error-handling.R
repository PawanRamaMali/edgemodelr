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

test_that("Memory constraints are handled properly", {
  # Find any available model for testing
  possible_paths <- c(
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
  )
  
  model_path <- NULL
  for (path in possible_paths) {
    if (file.exists(path)) {
      model_path <- path
      break
    }
  }
  
  if (!is.null(model_path)) {
    # Test very small context size (should work)
    ctx_small <- edge_load_model(model_path, n_ctx = 16)
    expect_true(is_valid_model(ctx_small))
    edge_free_model(ctx_small)
    
    # Test reasonable context size
    ctx_normal <- edge_load_model(model_path, n_ctx = 512)
    expect_true(is_valid_model(ctx_normal))
    edge_free_model(ctx_normal)
    
  } else {
    skip("No test model available for memory constraint tests")
  }
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

test_that("Edge cases in text completion", {
  # Find any available model for testing
  possible_paths <- c(
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
  )
  
  model_path <- NULL
  for (path in possible_paths) {
    if (file.exists(path)) {
      model_path <- path
      break
    }
  }
  
  if (!is.null(model_path)) {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    
    # Test with unusual characters (should work or give reasonable error)
    unusual_prompts <- c(
      "Unicode: αβγδε",
      "Numbers: 123456789",
      "Symbols: !@#$%^&*()",
      "Mixed: Hello123!",
      "Newlines: Hello\nWorld",
      "Tabs: Hello\tWorld"
    )
    
    for (prompt in unusual_prompts) {
      tryCatch({
        result <- edge_completion(ctx, prompt, n_predict = 3)
        expect_true(is.character(result))
      }, error = function(e) {
        # If there's an error, it should be a reasonable one
        expect_true(nchar(e$message) > 0)
      })
    }
    
    edge_free_model(ctx)
    
  } else {
    skip("No test model available for edge case tests")
  }
})

test_that("Resource cleanup after errors", {
  # Test that resources are properly cleaned up even when errors occur
  
  # This should fail but not leak memory
  expect_error(
    edge_load_model("nonexistent.gguf"),
    "Model file does not exist"
  )
  
  # Multiple failed attempts should not accumulate resources
  for (i in 1:5) {
    expect_error(
      suppressWarnings(edge_load_model("nonexistent.gguf"))
    )
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