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

test_that("edge_stream_completion works with real model", {
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
  } else {
    skip("No test model available for streaming tests")
  }
})

test_that("edge_stream_completion supports early stopping", {
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
  } else {
    skip("No test model available for early stopping tests")
  }
})

test_that("edge_stream_completion handles callback errors", {
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
  } else {
    skip("No test model available for callback error tests")
  }
})

test_that("edge_stream_completion with various parameters", {
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
    
    # Test with different n_predict values
    for (n_pred in c(1, 5, 10)) {
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
  } else {
    skip("No test model available for parameter tests")
  }
})

test_that("Streaming with special text", {
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
    
    special_prompts <- c(
      "Hello! How are you?",
      "Test: $100 price",
      "Email@example.com"
    )
    
    for (prompt in special_prompts) {
      result <- edge_stream_completion(ctx, prompt,
        callback = function(data) TRUE,
        n_predict = 3
      )
      expect_true(is.list(result))
      expect_true("full_response" %in% names(result))
    }
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for special text tests")
  }
})

test_that("Streaming stress tests", {
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
    
    # Test rapid successive calls
    for (i in 1:5) {
      result <- edge_stream_completion(ctx, paste("Test", i),
        callback = function(data) TRUE,
        n_predict = 2
      )
      expect_true(is.list(result))
    }
    
    # Test with very small n_predict
    result <- edge_stream_completion(ctx, "Hi",
      callback = function(data) {
        expect_true(is.list(data))
        return(TRUE)
      },
      n_predict = 1
    )
    expect_true(is.list(result))
    
    # Test with callback that always returns FALSE (immediate stop)
    result <- edge_stream_completion(ctx, "Test",
      callback = function(data) {
        return(FALSE)  # Stop immediately
      },
      n_predict = 10
    )
    expect_true(result$stopped_early)
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for streaming stress tests")
  }
})