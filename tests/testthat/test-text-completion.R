test_that("Text completion functions work correctly", {
  
  # Test with actual model if available
  model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  
  if (file.exists(model_path)) {
    # Load model for testing
    ctx <- edge_load_model(model_path, n_ctx = 512)
    
    test_that("edge_completion basic functionality", {
      # Test basic completion
      result <- edge_completion(ctx, "Hello", n_predict = 5)
      expect_true(is.character(result))
      expect_true(nchar(result) > 0)
      expect_true(nchar(result) > nchar("Hello"))
    })
    
    test_that("edge_completion with different prompts", {
      # Test various prompts
      prompts <- c(
        "The capital of France is",
        "Once upon a time",
        "2 + 2 =",
        "What is"
      )
      
      for (prompt in prompts) {
        result <- edge_completion(ctx, prompt, n_predict = 10)
        expect_true(is.character(result))
        expect_true(nchar(result) > nchar(prompt))
        expect_true(startsWith(result, prompt))
      }
    })
    
    test_that("edge_completion with different n_predict values", {
      prompt <- "Hello world"
      
      # Short completion
      result1 <- edge_completion(ctx, prompt, n_predict = 1)
      expect_true(is.character(result1))
      
      # Medium completion  
      result2 <- edge_completion(ctx, prompt, n_predict = 10)
      expect_true(is.character(result2))
      expect_true(nchar(result2) >= nchar(result1))
      
      # Longer completion
      result3 <- edge_completion(ctx, prompt, n_predict = 20)
      expect_true(is.character(result3))
      expect_true(nchar(result3) >= nchar(result2))
    })
    
    test_that("edge_completion with empty prompt", {
      # Empty string should still generate text
      result <- edge_completion(ctx, "", n_predict = 5)
      expect_true(is.character(result))
      expect_true(nchar(result) > 0)
    })
    
    test_that("edge_completion with long prompt", {
      # Test with a longer prompt
      long_prompt <- paste(rep("This is a test sentence.", 10), collapse = " ")
      result <- edge_completion(ctx, long_prompt, n_predict = 5)
      expect_true(is.character(result))
      expect_true(startsWith(result, long_prompt))
    })
    
    test_that("edge_completion parameter validation", {
      # Test invalid n_predict values
      expect_error(
        edge_completion(ctx, "Hello", n_predict = -1),
        "n_predict must be positive"
      )
      
      expect_error(
        edge_completion(ctx, "Hello", n_predict = 0),
        "n_predict must be positive"
      )
    })
    
    test_that("edge_completion with special characters", {
      # Test with various special characters
      special_prompts <- c(
        "Hello! How are you?",
        "What is 2+2=",
        "The price is $100.",
        "Email: user@example.com",
        "Path: C:\\Users\\test"
      )
      
      for (prompt in special_prompts) {
        result <- edge_completion(ctx, prompt, n_predict = 5)
        expect_true(is.character(result))
        expect_true(startsWith(result, prompt))
      }
    })
    
    test_that("edge_completion consistency", {
      # Test that the same prompt produces consistent results (within reason)
      prompt <- "The quick brown fox"
      results <- replicate(3, edge_completion(ctx, prompt, n_predict = 10))
      
      # All should be strings starting with the prompt
      for (result in results) {
        expect_true(is.character(result))
        expect_true(startsWith(result, prompt))
      }
    })
    
    # Cleanup
    edge_free_model(ctx)
    
  } else {
    skip("No test model available for text completion tests")
  }
})

test_that("edge_completion error handling", {
  # Test with invalid context
  test_that("edge_completion handles invalid contexts", {
    expect_error(
      edge_completion(NULL, "Hello", n_predict = 5),
      "Invalid model context"
    )
    
    expect_error(
      edge_completion("invalid", "Hello", n_predict = 5),
      "Invalid model context"
    )
  })
})