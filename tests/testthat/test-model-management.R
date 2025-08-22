test_that("Model management functions work correctly", {
  
  test_that("edge_list_models provides valid model information", {
    models <- edge_list_models()
    
    # Check structure
    expect_true(is.data.frame(models))
    expect_true(nrow(models) > 0)
    
    # Check required columns
    required_cols <- c("model_id", "filename", "description", "size_gb", "use_case")
    for (col in required_cols) {
      expect_true(col %in% colnames(models), 
                  info = paste("Missing column:", col))
    }
    
    # Check data types
    expect_true(is.character(models$model_id))
    expect_true(is.character(models$filename))
    expect_true(is.character(models$description))
    expect_true(is.numeric(models$size_gb))
    expect_true(is.character(models$use_case))
    
    # Check that all entries are non-empty
    expect_true(all(nchar(models$model_id) > 0))
    expect_true(all(nchar(models$filename) > 0))
    expect_true(all(nchar(models$description) > 0))
    expect_true(all(models$size_gb > 0))
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
      "model_name must be a string"
    )
    
    # Test with non-existent model
    expect_error(
      edge_quick_setup("nonexistent_model_12345"),
      "Model.*not found"
    )
  })
  
  # Test model memory management
  test_that("Model memory management works correctly", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
      # Test multiple load/free cycles
      for (i in 1:3) {
        ctx <- edge_load_model(model_path, n_ctx = 256)
        expect_true(is_valid_model(ctx))
        
        # Use the model briefly
        result <- edge_completion(ctx, "Test", n_predict = 2)
        expect_true(is.character(result))
        
        # Free the model
        edge_free_model(ctx)
        
        # Model should no longer be valid after freeing
        expect_false(is_valid_model(ctx))
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
  
  # Test model file format validation
  test_that("Model format validation works", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
      # Test that GGUF file is properly recognized
      ctx <- edge_load_model(model_path, n_ctx = 256)
      expect_true(is_valid_model(ctx))
      edge_free_model(ctx)
    }
    
    # Test with non-GGUF files (should fail gracefully)
    if (file.exists("DESCRIPTION")) {
      expect_error(
        edge_load_model("DESCRIPTION", n_ctx = 256),
        "Failed to load GGUF model"
      )
    }
  })
})

# Test helper functions
test_that("Helper functions work correctly", {
  
  test_that("is_valid_model works with various inputs", {
    # Test with various invalid inputs
    invalid_inputs <- list(
      NULL,
      "",
      "string",
      123,
      list(),
      data.frame(),
      c(1, 2, 3)
    )
    
    for (input in invalid_inputs) {
      expect_false(is_valid_model(input),
                   info = paste("Should be invalid:", class(input)))
    }
  })
})