test_that("Streaming functions work correctly", {
  
  # Test 1: edge_stream_completion parameter validation
  test_that("edge_stream_completion validates parameters", {
    # Mock context
    mock_ctx <- list(valid = TRUE)
    class(mock_ctx) <- "edge_model_context"
    
    # Should error with invalid prompt
    expect_error(
      edge_stream_completion(mock_ctx, 123, function(x) TRUE),
      "Prompt must be a single character string"
    )
    
    expect_error(
      edge_stream_completion(mock_ctx, c("a", "b"), function(x) TRUE),
      "Prompt must be a single character string"
    )
    
    # Should error with invalid callback
    expect_error(
      edge_stream_completion(mock_ctx, "test", "not_a_function"),
      "Callback must be a function"
    )
    
    expect_error(
      edge_stream_completion(mock_ctx, "test", NULL),
      "Callback must be a function"
    )
  })
  
  # Test 2: edge_chat_stream parameter validation
  test_that("edge_chat_stream validates parameters", {
    # Should error with invalid context
    expect_error(
      edge_chat_stream(NULL),
      "Invalid model context"
    )
    
    expect_error(
      edge_chat_stream("invalid"),
      "Invalid model context"
    )
  })
  
  # Test 3: build_chat_prompt function
  test_that("build_chat_prompt formats correctly", {
    # Empty history
    expect_equal(build_chat_prompt(list()), "")
    
    # Single system message
    history1 <- list(
      list(role = "system", content = "You are helpful")
    )
    result1 <- build_chat_prompt(history1)
    expect_true(grepl("System: You are helpful", result1))
    expect_true(grepl("Assistant:$", result1))
    
    # Full conversation
    history2 <- list(
      list(role = "system", content = "You are helpful"),
      list(role = "user", content = "Hello"),
      list(role = "assistant", content = "Hi there!")
    )
    result2 <- build_chat_prompt(history2)
    expect_true(grepl("System: You are helpful", result2))
    expect_true(grepl("Human: Hello", result2))
    expect_true(grepl("Assistant: Hi there!", result2))
    expect_true(grepl("Assistant:$", result2))
  })
})

# Test with actual model if available (skip if not)
test_that("Streaming with real model (if available)", {
  model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  
  if (file.exists(model_path)) {
    # Test basic streaming
    test_that("edge_stream_completion works with real model", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      # Track callback calls
      callback_calls <- 0
      tokens_received <- character(0)
      final_data <- NULL
      
      result <- edge_stream_completion(ctx, "Hi",
        callback = function(data) {
          callback_calls <<- callback_calls + 1
          
          expect_true(is.list(data))
          expect_true("token" %in% names(data))
          expect_true("position" %in% names(data))
          expect_true("is_final" %in% names(data))
          expect_true("total_tokens" %in% names(data))
          
          if (!data$is_final) {
            expect_true(is.character(data$token))
            expect_true(data$position > 0)
            tokens_received <<- c(tokens_received, data$token)
          } else {
            expect_true("full_response" %in% names(data))
            expect_true("stopped_early" %in% names(data))
            final_data <<- data
          }
          
          return(TRUE)  # Continue generation
        },
        n_predict = 10
      )
      
      # Verify results
      expect_true(callback_calls > 1)  # Should have multiple calls
      expect_true(length(tokens_received) > 0)  # Should receive tokens
      expect_true(!is.null(final_data))  # Should receive final data
      expect_true(is.list(result))  # Should return result
      
      # Cleanup
      edge_free_model(ctx)
    })
    
    # Test early stopping
    test_that("edge_stream_completion supports early stopping", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      tokens_before_stop <- 0
      
      result <- edge_stream_completion(ctx, "Hello",
        callback = function(data) {
          if (!data$is_final) {
            tokens_before_stop <<- tokens_before_stop + 1
            # Stop after 3 tokens
            return(tokens_before_stop < 3)
          }
          return(TRUE)
        },
        n_predict = 20
      )
      
      # Should have stopped early
      expect_true(tokens_before_stop <= 3)
      expect_true(result$stopped_early)
      
      edge_free_model(ctx)
    })
    
  } else {
    skip("No test model available for streaming tests")
  }
})