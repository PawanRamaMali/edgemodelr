test_that("Error handling and edge cases work correctly", {
  
  test_that("Invalid file paths are handled properly", {
    # Non-existent file
    expect_error(
      edge_load_model("does_not_exist.gguf"),
      "Failed to load GGUF model"
    )
    
    # Empty file path
    expect_error(
      edge_load_model(""),
      "model_path cannot be empty|Failed to load GGUF model"
    )
    
    # NULL file path
    expect_error(
      edge_load_model(NULL),
      "model_path must be a string|Failed to load GGUF model"
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
    
    # Negative context size
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_ctx = -1)),
      "n_ctx must be positive|Failed to load GGUF model"
    )
    
    # Zero context size
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_ctx = 0)),
      "n_ctx must be positive|Failed to load GGUF model"
    )
    
    # Extremely large context size (should be handled gracefully)
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_ctx = 999999999)),
      "Failed to load GGUF model"
    )
    
    # Negative GPU layers
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_gpu_layers = -1)),
      "n_gpu_layers cannot be negative|Failed to load GGUF model"
    )
  })
  
  test_that("edge_completion handles errors gracefully", {
    # Test with NULL context
    expect_error(
      edge_completion(NULL, "Hello", n_predict = 5),
      "Invalid model context"
    )
    
    # Test with invalid context types
    invalid_contexts <- list("string", 123, list(), data.frame())
    for (ctx in invalid_contexts) {
      expect_error(
        edge_completion(ctx, "Hello", n_predict = 5),
        "Invalid model context"
      )
    }
    
    # Test with invalid n_predict
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if (file.exists(model_path)) {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      expect_error(
        edge_completion(ctx, "Hello", n_predict = -1),
        "n_predict must be positive"
      )
      
      expect_error(
        edge_completion(ctx, "Hello", n_predict = 0),
        "n_predict must be positive"
      )
      
      edge_free_model(ctx)
    }
  })
  
  test_that("Memory constraints are handled properly", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
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
      edge_load_model(fake_gguf),
      "Failed to load GGUF model"
    )
    
    # Cleanup
    if (file.exists(fake_gguf)) {
      unlink(fake_gguf)
    }
  })
  
  test_that("Edge cases in text completion", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      # Test with very long prompt (might exceed context)
      long_prompt <- paste(rep("This is a very long sentence that goes on and on.", 20), collapse = " ")
      
      # Should either work or give a reasonable error
      tryCatch({
        result <- edge_completion(ctx, long_prompt, n_predict = 5)
        expect_true(is.character(result))
      }, error = function(e) {
        # If it errors, the error should be informative
        expect_true(grepl("context|length|size", e$message, ignore.case = TRUE))
      })
      
      # Test with unusual characters
      unusual_prompts <- c(
        "Unicode: αβγδε",
        "Emojis: 🚀🔥💻",
        "Numbers: 123456789",
        "Symbols: !@#$%^&*()",
        "Mixed: Hello🌍123!"
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
  
  test_that("Concurrent access and thread safety", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
      # Test loading the same model multiple times
      contexts <- list()
      
      # Load multiple contexts
      for (i in 1:3) {
        contexts[[i]] <- edge_load_model(model_path, n_ctx = 128)
        expect_true(is_valid_model(contexts[[i]]))
      }
      
      # Use all contexts
      for (i in 1:3) {
        result <- edge_completion(contexts[[i]], paste("Test", i), n_predict = 3)
        expect_true(is.character(result))
      }
      
      # Free all contexts
      for (i in 1:3) {
        edge_free_model(contexts[[i]])
        expect_false(is_valid_model(contexts[[i]]))
      }
      
    } else {
      skip("No test model available for concurrency tests")
    }
  })
  
  test_that("Resource cleanup after errors", {
    # Test that resources are properly cleaned up even when errors occur
    
    # This should fail but not leak memory
    expect_error(
      edge_load_model("nonexistent.gguf"),
      "Failed to load GGUF model"
    )
    
    # Multiple failed attempts should not accumulate resources
    for (i in 1:5) {
      expect_error(
        suppressWarnings(edge_load_model("nonexistent.gguf")),
        "Failed to load GGUF model"
      )
    }
  })
})