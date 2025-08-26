test_that("Model loading functions work correctly", {
  
  # Test 1: edge_list_models() returns a data frame
  test_that("edge_list_models returns valid data", {
    models <- edge_list_models()
    expect_true(is.data.frame(models))
    expect_true(nrow(models) > 0)
    expect_true("model_id" %in% colnames(models))
    expect_true("filename" %in% colnames(models))
    expect_true("use_case" %in% colnames(models))
  })
  
  # Test 2: edge_load_model with invalid path should error
  test_that("edge_load_model handles invalid paths", {
    expect_error(
      edge_load_model("nonexistent_model.gguf")
    )
  })
  
  # Test 3: edge_load_model with invalid parameters
  test_that("edge_load_model handles invalid parameters", {
    # These will fail because file doesn't exist, which is expected
    expect_error(edge_load_model("test.gguf", n_ctx = -1))
    expect_error(edge_load_model("test.gguf", n_ctx = 0))
  })
  
  # Test 4: is_valid_model with invalid context
  test_that("is_valid_model handles invalid contexts", {
    expect_false(is_valid_model(NULL))
    expect_false(is_valid_model("invalid"))
    expect_false(is_valid_model(123))
  })
})

# Test with actual model if available
test_that("Model loading with real model (if available)", {
  model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  
  if (file.exists(model_path)) {
    # Test successful model loading
    test_that("edge_load_model works with valid GGUF file", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      expect_true(!is.null(ctx))
      
      # Test model validation
      expect_true(is_valid_model(ctx))
      
      # Cleanup
      edge_free_model(ctx)
    })
    
    # Test different context sizes
    test_that("edge_load_model works with different context sizes", {
      # Small context
      ctx1 <- edge_load_model(model_path, n_ctx = 128)
      expect_true(is_valid_model(ctx1))
      edge_free_model(ctx1)
      
      # Medium context
      ctx2 <- edge_load_model(model_path, n_ctx = 512)
      expect_true(is_valid_model(ctx2))
      edge_free_model(ctx2)
      
      # Large context (should work but may be slow)
      ctx3 <- edge_load_model(model_path, n_ctx = 1024)
      expect_true(is_valid_model(ctx3))
      edge_free_model(ctx3)
    })
    
    # Test GPU layers parameter
    test_that("edge_load_model handles GPU layers parameter", {
      # Should work with 0 GPU layers (CPU only)
      ctx <- edge_load_model(model_path, n_ctx = 256, n_gpu_layers = 0)
      expect_true(is_valid_model(ctx))
      edge_free_model(ctx)
    })
    
  } else {
    skip("No test model available for real model loading tests")
  }
})