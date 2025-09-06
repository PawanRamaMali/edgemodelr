test_that("edge_completion basic functionality", {
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
    # Load fresh model for this test
    ctx <- edge_load_model(model_path, n_ctx = 512)
    
    # Test basic completion
    result <- edge_completion(ctx, "Hello", n_predict = 5)
    expect_true(is.character(result))
    expect_true(nchar(result) > 0)
    expect_true(nchar(result) > nchar("Hello"))
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for basic functionality tests")
  }
})



test_that("edge_completion error handling", {
  # Test with invalid context
  expect_error(
    edge_completion(NULL, "Hello", n_predict = 5)
  )
  
  expect_error(
    edge_completion("invalid", "Hello", n_predict = 5)
  )

  expect_error(
    edge_completion(123, "Hello", n_predict = 5)
  )

  expect_error(
    edge_completion(list(), "Hello", n_predict = 5)
  )

  # Test with missing arguments
  expect_error(edge_completion())
  expect_error(edge_completion(NULL))
  
  # Test with invalid prompt types
  expect_error(edge_completion(NULL, NULL, n_predict = 5))
  expect_error(edge_completion(NULL, 123, n_predict = 5))
  expect_error(edge_completion(NULL, c("a", "b"), n_predict = 5))
  expect_error(edge_completion(NULL, list("hello"), n_predict = 5))
})




