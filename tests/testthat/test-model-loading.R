# Test 1: edge_list_models() returns a data frame
test_that("edge_list_models returns valid data", {
  models <- edge_list_models()
  expect_true(is.data.frame(models))
  expect_true(nrow(models) > 0)
  
  # Check all required columns exist
  required_cols <- c("name", "size", "model_id", "filename", "use_case")
  expect_true(all(required_cols %in% colnames(models)))
  
  # Check data types
  expect_true(is.character(models$name))
  expect_true(is.character(models$size))
  expect_true(is.character(models$model_id))
  expect_true(is.character(models$filename))
  expect_true(is.character(models$use_case))
  
  # Check for no empty values in critical columns
  expect_true(all(nchar(models$model_id) > 0))
  expect_true(all(nchar(models$filename) > 0))
  
  # Check filename format (should end with .gguf)
  expect_true(all(grepl("\\.gguf$", models$filename, ignore.case = TRUE)))
  
  # Check model_id format (should contain /)
  expect_true(all(grepl("/", models$model_id)))
})

# Test 2: edge_load_model with invalid inputs
test_that("edge_load_model handles invalid paths", {
  expect_error(
    edge_load_model("nonexistent_model.gguf"),
    "Model file does not exist"
  )
  
  # Test with empty string
  expect_error(
    edge_load_model(""),
    "Model file does not exist"
  )
  
  # Test with NULL
  expect_error(edge_load_model(NULL))
  
  # Test with directory instead of file
  temp_dir <- tempdir()
  expect_error(
    edge_load_model(temp_dir),
    "Model file does not exist"
  )
  
  # Test with non-GGUF file
  temp_file <- tempfile(fileext = ".txt")
  writeLines("not a model", temp_file)
  expect_warning(
    expect_error(edge_load_model(temp_file)),
    "should have .gguf extension"
  )
  unlink(temp_file)
})

# Test 2b: edge_load_model parameter validation
test_that("edge_load_model validates parameters", {
  # Test invalid n_ctx values
  expect_error(edge_load_model("nonexistent.gguf", n_ctx = -1))
  expect_error(edge_load_model("nonexistent.gguf", n_ctx = 0))
  
  # Test invalid n_gpu_layers values
  expect_error(edge_load_model("nonexistent.gguf", n_gpu_layers = -1))
  
  # Test extreme values
  expect_error(edge_load_model("nonexistent.gguf", n_ctx = 1e10)) # Too large
})

# Test 3: is_valid_model with invalid contexts
test_that("is_valid_model handles invalid contexts", {
  expect_false(is_valid_model(NULL))
  expect_false(is_valid_model("invalid"))
  expect_false(is_valid_model(123))
  expect_false(is_valid_model(list()))
  expect_false(is_valid_model(data.frame()))
  expect_false(is_valid_model(TRUE))
  expect_false(is_valid_model(FALSE))
  expect_false(is_valid_model(c(1, 2, 3)))
  
  # Test with missing arguments
  expect_error(is_valid_model())
})

# Test 3b: edge_free_model with invalid contexts
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

# Test 4: Model loading with real model (if available)
test_that("Model loading with real model (if available)", {
  # Try multiple possible model locations
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
    # Test successful model loading
    test_that("edge_load_model works with valid GGUF file", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      expect_true(!is.null(ctx))
      expect_true(inherits(ctx, "externalptr"))
      
      # Test model validation
      expect_true(is_valid_model(ctx))
      
      # Test that context is still valid after validation
      expect_true(is_valid_model(ctx))
      
      # Cleanup
      edge_free_model(ctx)
      
      # Test that context is invalid after cleanup
      expect_false(is_valid_model(ctx))
    })
    
    # Test different context sizes
    test_that("edge_load_model works with different context sizes", {
      # Very small context (edge case)
      ctx1 <- edge_load_model(model_path, n_ctx = 32)
      expect_true(is_valid_model(ctx1))
      edge_free_model(ctx1)
      
      # Small context
      ctx2 <- edge_load_model(model_path, n_ctx = 128)
      expect_true(is_valid_model(ctx2))
      edge_free_model(ctx2)
      
      # Medium context
      ctx3 <- edge_load_model(model_path, n_ctx = 512)
      expect_true(is_valid_model(ctx3))
      edge_free_model(ctx3)
      
      # Large context
      ctx4 <- edge_load_model(model_path, n_ctx = 1024)
      expect_true(is_valid_model(ctx4))
      edge_free_model(ctx4)
      
      # Very large context (may fail on low memory systems)
      tryCatch({
        ctx5 <- edge_load_model(model_path, n_ctx = 4096)
        expect_true(is_valid_model(ctx5))
        edge_free_model(ctx5)
      }, error = function(e) {
        # This is expected on systems with limited memory
        expect_true(grepl("memory|context", e$message, ignore.case = TRUE))
      })
    })
    
    # Test GPU layers parameter
    test_that("edge_load_model handles GPU layers parameter", {
      # Should work with 0 GPU layers (CPU only)
      ctx1 <- edge_load_model(model_path, n_ctx = 256, n_gpu_layers = 0)
      expect_true(is_valid_model(ctx1))
      edge_free_model(ctx1)
      
      # Test with small number of GPU layers (may or may not work)
      tryCatch({
        ctx2 <- edge_load_model(model_path, n_ctx = 256, n_gpu_layers = 1)
        expect_true(is_valid_model(ctx2))
        edge_free_model(ctx2)
      }, error = function(e) {
        # Expected if no GPU or unsupported GPU
        expect_true(grepl("gpu|cuda|opencl|metal", e$message, ignore.case = TRUE))
      })
    })
    
    # Test multiple model loading and cleanup
    test_that("Multiple model contexts can be managed", {
      ctx1 <- edge_load_model(model_path, n_ctx = 256)
      ctx2 <- edge_load_model(model_path, n_ctx = 256)
      
      expect_true(is_valid_model(ctx1))
      expect_true(is_valid_model(ctx2))
      
      # Cleanup one, other should still be valid
      edge_free_model(ctx1)
      expect_false(is_valid_model(ctx1))
      expect_true(is_valid_model(ctx2))
      
      edge_free_model(ctx2)
      expect_false(is_valid_model(ctx2))
    })
    
    # Test double cleanup (should be safe)
    test_that("Double cleanup is safe", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      expect_true(is_valid_model(ctx))
      
      edge_free_model(ctx)
      expect_false(is_valid_model(ctx))
      
      # Should not error
      expect_silent(edge_free_model(ctx))
      expect_false(is_valid_model(ctx))
    })
    
  } else {
    skip("No test model available for real model loading tests")
  }
})

# Test 5: Edge cases for model file validation
test_that("Model file validation edge cases", {
  # Test with very long path
  long_path <- paste0(rep("a", 1000), collapse = "")
  expect_error(edge_load_model(paste0(long_path, ".gguf")))
  
  # Test with special characters in path
  special_chars <- c("<", ">", ":", "\"", "|", "?", "*")
  for (char in special_chars) {
    if (.Platform$OS.type == "windows" && char %in% c("<", ">", ":", "\"", "|", "?", "*")) {
      path_with_special <- paste0("test", char, "model.gguf")
      expect_error(edge_load_model(path_with_special))
    }
  }
  
  # Test with Unicode characters
  unicode_path <- "test_\u4e2d\u6587.gguf"  # Chinese characters
  expect_error(edge_load_model(unicode_path))  # File doesn't exist
})