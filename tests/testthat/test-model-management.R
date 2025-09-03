test_that("edge_list_models provides valid model information", {
  models <- edge_list_models()
  
  # Check structure
  expect_true(is.data.frame(models))
  expect_true(nrow(models) > 0)
  
  # Check required columns
  required_cols <- c("name", "size", "model_id", "filename", "use_case")
  for (col in required_cols) {
    expect_true(col %in% colnames(models), 
                info = paste("Missing column:", col))
  }
  
  # Check data types
  expect_true(is.character(models$name))
  expect_true(is.character(models$size))
  expect_true(is.character(models$model_id))
  expect_true(is.character(models$filename))
  expect_true(is.character(models$use_case))
  
  # Check that all entries are non-empty
  expect_true(all(nchar(models$name) > 0))
  expect_true(all(nchar(models$size) > 0))
  expect_true(all(nchar(models$model_id) > 0))
  expect_true(all(nchar(models$filename) > 0))
  expect_true(all(nchar(models$use_case) > 0))
})

test_that("edge_download_model parameter validation", {
  # Test with invalid model_id
  expect_error(
    edge_download_model("", "test.gguf"),
    "model_id cannot be empty"
  )
  
  expect_error(
    edge_download_model(NULL, "test.gguf"),
    "model_id must be a string"
  )
  
  # Test with invalid filename
  expect_error(
    edge_download_model("test/model", ""),
    "filename cannot be empty"
  )
  
  expect_error(
    edge_download_model("test/model", NULL),
    "filename must be a string"
  )
})

test_that("edge_quick_setup parameter validation", {
  # Test with invalid model names
  expect_error(
    edge_quick_setup(""),
    "model_name cannot be empty"
  )
  
  expect_error(
    edge_quick_setup(NULL),
    "model_name cannot be empty"
  )
  
  # Test with non-existent model
  expect_error(
    edge_quick_setup("nonexistent_model_12345"),
    "Model.*not found"
  )
})

test_that("Model memory management works correctly", {
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
    # Test multiple load/free cycles
    for (i in 1:3) {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      expect_true(is_valid_model(ctx))
      
      # Use the model briefly
      result <- edge_completion(ctx, "Test", n_predict = 2)
      expect_true(is.character(result))
      
      # Free the model
      edge_free_model(ctx)
    }
    
    # Test that we can load multiple models (if system has enough memory)
    ctx1 <- edge_load_model(model_path, n_ctx = 128)
    expect_true(is_valid_model(ctx1))
    
    ctx2 <- edge_load_model(model_path, n_ctx = 128)
    expect_true(is_valid_model(ctx2))
    
    # Both should work independently
    result1 <- edge_completion(ctx1, "Hello", n_predict = 2)
    result2 <- edge_completion(ctx2, "Hi", n_predict = 2)
    
    expect_true(is.character(result1))
    expect_true(is.character(result2))
    
    # Cleanup
    edge_free_model(ctx1)
    edge_free_model(ctx2)
    
  } else {
    skip("No test model available for memory management tests")
  }
})

test_that("edge_free_model handles invalid contexts gracefully", {
  # These should not crash, just handle gracefully
  expect_silent(edge_free_model(NULL))
  expect_silent(edge_free_model("invalid"))
  expect_silent(edge_free_model(123))
})

test_that("Model format validation works", {
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
    # Test that GGUF file is properly recognized
    ctx <- edge_load_model(model_path, n_ctx = 256)
    expect_true(is_valid_model(ctx))
    edge_free_model(ctx)
  }
  
  # Test with non-GGUF files (should fail gracefully)
  if (file.exists("DESCRIPTION")) {
    expect_error(
      edge_load_model("DESCRIPTION", n_ctx = 256)
    )
  }
  
  # Test with binary files that aren't GGUF
  temp_binary <- tempfile(fileext = ".bin")
  tryCatch({
    writeBin(raw(1000), temp_binary)  # Create fake binary file
    expect_error(
      edge_load_model(temp_binary, n_ctx = 256)
    )
  }, finally = {
    # Always clean up, even if test fails
    if (file.exists(temp_binary)) {
      unlink(temp_binary)
    }
  })
})

test_that("edge_download_model validates inputs thoroughly", {
  temp_dir <- tempdir()
  
  # Test various invalid inputs
  expect_error(edge_download_model(123, "file.gguf"))
  expect_error(edge_download_model(c("a", "b"), "file.gguf"))
  expect_error(edge_download_model(list(), "file.gguf"))
  
  expect_error(edge_download_model("valid/model", 123))
  expect_error(edge_download_model("valid/model", c("a", "b")))
  expect_error(edge_download_model("valid/model", list()))
  
  # Test edge cases with cache_dir
  expect_error(edge_download_model("valid/model", "file.gguf", cache_dir = 123))
  expect_error(edge_download_model("valid/model", "file.gguf", cache_dir = c("a", "b")))
})

test_that("edge_quick_setup integration", {
  # Test parameter validation only - don't actually download models
  skip("Skipping edge_quick_setup integration test to avoid downloading models during testing")
})

test_that("Concurrent model operations", {
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
    # Test loading multiple models simultaneously
    contexts <- list()
    
    # Load multiple contexts
    for (i in 1:3) {
      contexts[[i]] <- edge_load_model(model_path, n_ctx = 128)
      expect_true(is_valid_model(contexts[[i]]))
    }
    
    # Test that all contexts work independently
    results <- list()
    for (i in 1:3) {
      results[[i]] <- edge_completion(contexts[[i]], paste("Test", i), n_predict = 2)
      expect_true(is.character(results[[i]]))
      expect_true(nchar(results[[i]]) > 0)
    }
    
    # Cleanup in random order to test robustness
    cleanup_order <- sample(1:3)
    for (i in cleanup_order) {
      edge_free_model(contexts[[i]])
    }
    
  } else {
    skip("No test model available for concurrent operations test")
  }
})

test_that("is_valid_model works with various inputs", {
  # Test with various invalid inputs
  invalid_inputs <- list(
    NULL,
    "",
    "string",
    123,
    list(),
    data.frame(),
    c(1, 2, 3),
    TRUE,
    FALSE,
    complex(1),
    as.raw(1:10)
  )
  
  for (input in invalid_inputs) {
    expect_false(is_valid_model(input),
                 info = paste("Should be invalid:", class(input)[1]))
  }
})

test_that("edge_free_model edge cases", {
  # Test with various invalid inputs
  invalid_inputs <- list(
    NULL,
    "",
    "string", 
    123,
    list(),
    data.frame(),
    c(1, 2, 3),
    TRUE,
    FALSE
  )
  
  # All should be handled gracefully without errors
  for (input in invalid_inputs) {
    expect_silent(edge_free_model(input))
  }
  
  # Test multiple calls on same invalid input
  expect_silent(edge_free_model(NULL))
  expect_silent(edge_free_model(NULL))
  expect_silent(edge_free_model("invalid"))
  expect_silent(edge_free_model("invalid"))
})

test_that("Memory and resource management", {
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
    # Test extreme context sizes
    tryCatch({
      # Very small context
      ctx_small <- edge_load_model(model_path, n_ctx = 16)
      expect_true(is_valid_model(ctx_small))
      edge_free_model(ctx_small)
      
      # Very large context (may fail on systems with limited memory)
      ctx_large <- edge_load_model(model_path, n_ctx = 8192)
      expect_true(is_valid_model(ctx_large))
      edge_free_model(ctx_large)
      
    }, error = function(e) {
      # Expected on systems with limited memory
      expect_true(grepl("memory|context|allocation", e$message, ignore.case = TRUE))
    })
    
    # Test rapid load/free cycles (memory leak detection)
    for (i in 1:10) {
      ctx <- edge_load_model(model_path, n_ctx = 64)
      expect_true(is_valid_model(ctx))
      edge_free_model(ctx)
    }
    
  } else {
    skip("No test model available for system integration tests")
  }
})

test_that("Platform-specific behavior", {
  # Test path handling across platforms
  if (.Platform$OS.type == "windows") {
    # Test Windows-specific path separators
    expect_error(edge_load_model("C:\\nonexistent\\model.gguf"))
    
    # Test UNC paths if applicable
    expect_error(edge_load_model("\\\\server\\share\\model.gguf"))
    
  } else {
    # Test Unix-specific paths
    expect_error(edge_load_model("/nonexistent/model.gguf"))
    expect_error(edge_load_model("~/nonexistent_model.gguf"))
  }
  
  # Test relative paths
  expect_error(edge_load_model("./nonexistent.gguf"))
  expect_error(edge_load_model("../nonexistent.gguf"))
})