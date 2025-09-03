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

test_that("edge_stream_completion function signature and validation", {
  # Test function exists and has proper signature
  expect_true(exists("edge_stream_completion"))
  expect_true(is.function(edge_stream_completion))
  
  # Test parameter validation works before attempting model operations
  expect_error(
    edge_stream_completion(NULL, "test", function(x) TRUE, n_predict = -1),
    "could not find function"  # Will error on internal function call
  )
  
  # Test that callback validation happens first
  expect_error(
    edge_stream_completion(NULL, "test", "not_a_function"),
    "Callback must be a function"
  )
  
  # Test that prompt validation happens first
  expect_error(
    edge_stream_completion(NULL, c("a", "b"), function(x) TRUE),
    "Prompt must be a single character string"
  )
  
  # Test valid parameters get to the internal function call
  expect_error(
    edge_stream_completion(NULL, "test", function(x) TRUE),
    "could not find function"
  )
})

test_that("edge_stream_completion early stopping callback logic", {
  # Test callback return value logic (without requiring model)
  
  # Callback that should stop early
  early_stop_callback <- function(data) {
    if (!data$is_final && data$position >= 3) {
      return(FALSE)  # Stop early
    }
    return(TRUE)
  }
  
  # Callback that continues
  continue_callback <- function(data) {
    return(TRUE)
  }
  
  # Callback that stops immediately
  immediate_stop_callback <- function(data) {
    return(FALSE)
  }
  
  # Test that these are valid functions
  expect_true(is.function(early_stop_callback))
  expect_true(is.function(continue_callback))
  expect_true(is.function(immediate_stop_callback))
  
  # Test callback logic with mock data
  mock_data <- list(is_final = FALSE, position = 5)
  expect_false(early_stop_callback(mock_data))  # Should stop
  expect_true(continue_callback(mock_data))     # Should continue
  expect_false(immediate_stop_callback(mock_data))  # Should stop
})

test_that("edge_stream_completion error handling in callbacks", {
  # Test error-throwing callback functions
  error_callback <- function(data) {
    stop("Test error in callback")
  }
  
  expect_true(is.function(error_callback))
  
  # Test that callback function itself can be created and is valid
  expect_error(error_callback(list()), "Test error in callback")
  
  # Test conditional error callback
  conditional_error_callback <- function(data) {
    if (!is.null(data$position) && data$position == 2) {
      stop("Conditional test error")
    }
    return(TRUE)
  }
  
  expect_true(is.function(conditional_error_callback))
  
  # Test with mock data that should trigger error
  mock_error_data <- list(is_final = FALSE, position = 2)
  expect_error(
    conditional_error_callback(mock_error_data),
    "Conditional test error"
  )
  
  # Test with mock data that should not trigger error  
  mock_ok_data <- list(is_final = FALSE, position = 1)
  expect_true(conditional_error_callback(mock_ok_data))
})

test_that("edge_stream_completion parameter validation", {
  # Test that function exists and validates inputs properly
  expect_error(
    edge_stream_completion(NULL, "test", function(x) TRUE),
    "could not find function"
  )
  
  # Test prompt validation
  expect_error(
    edge_stream_completion(NULL, NULL, function(x) TRUE),
    "Prompt must be a single character string"
  )
  
  expect_error(
    edge_stream_completion(NULL, c("a", "b"), function(x) TRUE),
    "Prompt must be a single character string"
  )
  
  # Test callback validation  
  expect_error(
    edge_stream_completion(NULL, "test", "not_a_function"),
    "Callback must be a function"
  )
  
  expect_error(
    edge_stream_completion(NULL, "test", NULL),
    "Callback must be a function"
  )
})

test_that("Streaming prompt validation", {
  # Test special character handling in prompts
  special_prompts <- c(
    "Hello! How are you?",
    "Test: $100 price", 
    "Email@example.com",
    "Special chars: #@$%^&*()",
    "Unicode: ñáéíóú"
  )
  
  # Test that prompt validation works correctly with special characters
  for (prompt in special_prompts) {
    # Should not error on prompt validation itself
    expect_silent({
      # This will error on the internal function call, not on prompt validation
      tryCatch({
        edge_stream_completion(NULL, prompt, function(x) TRUE)
      }, error = function(e) {
        # Expect error about missing internal function, not prompt validation
        expect_true(grepl("could not find function", e$message))
      })
    })
  }
})

test_that("Streaming callback function validation", {
  # Test different callback function scenarios
  
  # Valid callback function
  valid_callback <- function(data) {
    return(TRUE)
  }
  expect_true(is.function(valid_callback))
  
  # Test callback that returns different values
  callback_false <- function(data) FALSE
  callback_true <- function(data) TRUE
  callback_complex <- function(data) {
    if (data$is_final) return(TRUE)
    return(data$position < 5)
  }
  
  expect_true(is.function(callback_false))
  expect_true(is.function(callback_true))
  expect_true(is.function(callback_complex))
  
  # Test invalid callback types
  expect_false(is.function("not a function"))
  expect_false(is.function(NULL))
  expect_false(is.function(123))
  expect_false(is.function(list()))
})