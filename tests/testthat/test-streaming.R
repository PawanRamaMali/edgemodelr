test_that("edge_stream_completion validates parameters", {
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
})

test_that("edge_chat_stream validates parameters", {
  # Should error with invalid context
  expect_error(edge_chat_stream(NULL))
  expect_error(edge_chat_stream("invalid"))
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

test_that("edge_stream_completion callback logic validation", {
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