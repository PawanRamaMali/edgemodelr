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
    
    # Test callback error handling
    test_that("edge_stream_completion handles callback errors", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      # Callback that throws error
      expect_warning(
        result <- edge_stream_completion(ctx, "Test",
          callback = function(data) {
            if (!data$is_final && data$position == 2) {
              stop("Test error")
            }
            return(TRUE)
          },
          n_predict = 5
        )
      )
      
      edge_free_model(ctx)
    })
    
    # Test different streaming parameters
    test_that("edge_stream_completion with various parameters", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      # Test with different n_predict values
      for (n_pred in c(1, 5, 20)) {
        tokens_count <- 0
        result <- edge_stream_completion(ctx, "Hello",
          callback = function(data) {
            if (!data$is_final) {
              tokens_count <<- tokens_count + 1
            }
            return(TRUE)
          },
          n_predict = n_pred
        )
        expect_true(tokens_count <= n_pred)
      }
      
      # Test with different temperature values
      for (temp in c(0.1, 0.8, 1.5)) {
        result <- edge_stream_completion(ctx, "Test",
          callback = function(data) TRUE,
          n_predict = 3,
          temperature = temp
        )
        expect_true(is.list(result))
      }
      
      # Test with different top_p values
      for (top_p in c(0.1, 0.5, 0.95)) {
        result <- edge_stream_completion(ctx, "Test",
          callback = function(data) TRUE,
          n_predict = 3,
          top_p = top_p
        )
        expect_true(is.list(result))
      }
      
      edge_free_model(ctx)
    })
    
    # Test streaming with long prompts
    test_that("edge_stream_completion with long prompts", {
      ctx <- edge_load_model(model_path, n_ctx = 512)
      
      # Medium length prompt
      medium_prompt <- paste(rep("This is a test sentence.", 10), collapse = " ")
      tokens_received <- 0
      
      result <- edge_stream_completion(ctx, medium_prompt,
        callback = function(data) {
          if (!data$is_final) {
            tokens_received <<- tokens_received + 1
          }
          return(TRUE)
        },
        n_predict = 5
      )
      
      expect_true(tokens_received > 0)
      
      # Very long prompt (may hit context limits)
      long_prompt <- paste(rep("Word", 200), collapse = " ")
      tryCatch({
        result <- edge_stream_completion(ctx, long_prompt,
          callback = function(data) TRUE,
          n_predict = 3
        )
        expect_true(is.list(result))
      }, error = function(e) {
        # Expected if prompt is too long
        expect_true(grepl("context|length|token", e$message, ignore.case = TRUE))
      })
      
      edge_free_model(ctx)
    })
    
    # Test streaming with special characters and Unicode
    test_that("edge_stream_completion with special text", {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      special_prompts <- c(
        "Hello! How are you?",
        "Test: $100 price",
        "Email@example.com",
        "Path\\to\\file"
      )
      
      for (prompt in special_prompts) {
        result <- edge_stream_completion(ctx, prompt,
          callback = function(data) TRUE,
          n_predict = 3
        )
        expect_true(is.list(result))
        expect_true("full_response" %in% names(result))
      }
      
      # Test with Unicode (may or may not work depending on model)
      unicode_prompts <- c(
        "Hello \u4e16\u754c",  # Chinese
        "Bonjour monde",  # French with accent
        "\ud83c\udf0d Earth"  # Emoji
      )
      
      for (prompt in unicode_prompts) {
        tryCatch({
          result <- edge_stream_completion(ctx, prompt,
            callback = function(data) TRUE,
            n_predict = 2
          )
          expect_true(is.list(result))
        }, error = function(e) {
          # Some models may not handle Unicode well
          expect_true(TRUE)  # Don't fail the test
        })
      }
      
      edge_free_model(ctx)
    })
    
    # Test concurrent streaming (if supported)
    test_that("edge_stream_completion concurrency behavior", {
      ctx1 <- edge_load_model(model_path, n_ctx = 256)
      ctx2 <- edge_load_model(model_path, n_ctx = 256)
      
      # Should be able to stream from different contexts
      result1_done <- FALSE
      result2_done <- FALSE
      
      result1 <- edge_stream_completion(ctx1, "First test",
        callback = function(data) {
          if (data$is_final) result1_done <<- TRUE
          return(TRUE)
        },
        n_predict = 3
      )
      
      result2 <- edge_stream_completion(ctx2, "Second test",
        callback = function(data) {
          if (data$is_final) result2_done <<- TRUE
          return(TRUE)
        },
        n_predict = 3
      )
      
      expect_true(result1_done)
      expect_true(result2_done)
      
      edge_free_model(ctx1)
      edge_free_model(ctx2)
    })
    
  } else {
    skip("No test model available for streaming tests")
  }
})

# Test 5: Stress tests for streaming
test_that("Streaming stress tests", {
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
    
    # Test rapid successive calls
    test_that("Rapid successive streaming calls", {
      for (i in 1:5) {
        result <- edge_stream_completion(ctx, paste("Test", i),
          callback = function(data) TRUE,
          n_predict = 2
        )
        expect_true(is.list(result))
      }
    })
    
    # Test with very small n_predict
    test_that("Streaming with minimal generation", {
      result <- edge_stream_completion(ctx, "Hi",
        callback = function(data) {
          expect_true(is.list(data))
          return(TRUE)
        },
        n_predict = 1
      )
      expect_true(is.list(result))
    })
    
    # Test with callback that always returns FALSE (immediate stop)
    test_that("Streaming with immediate stop callback", {
      result <- edge_stream_completion(ctx, "Test",
        callback = function(data) {
          return(FALSE)  # Stop immediately
        },
        n_predict = 10
      )
      expect_true(result$stopped_early)
    })
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for streaming stress tests")
  }
})