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

test_that("edge_completion with different prompts", {
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
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for different prompts tests")
  }
})

test_that("edge_completion with different n_predict values", {
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
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for n_predict tests")
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

test_that("edge_completion with empty and unusual prompts", {
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
    
    # Empty string should still generate text
    result <- edge_completion(ctx, "", n_predict = 5)
    expect_true(is.character(result))
    expect_true(nchar(result) > 0)
    
    # Test with special characters
    special_prompts <- c(
      "Hello! How are you?",
      "What is 2+2=",
      "The price is $100.",
      "Email: user@example.com"
    )
    
    for (prompt in special_prompts) {
      result <- edge_completion(ctx, prompt, n_predict = 5)
      expect_true(is.character(result))
      expect_true(startsWith(result, prompt))
    }
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for empty/unusual prompt tests")
  }
})

test_that("edge_completion with different temperature and top_p settings", {
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
    
    prompt <- "The weather today is"
    
    # Low temperature (more deterministic)
    result_low <- edge_completion(ctx, prompt, n_predict = 10, temperature = 0.1)
    expect_true(is.character(result_low))
    expect_true(startsWith(result_low, prompt))
    
    # High temperature (more random)
    result_high <- edge_completion(ctx, prompt, n_predict = 10, temperature = 1.0)
    expect_true(is.character(result_high))
    expect_true(startsWith(result_high, prompt))
    
    # Different top_p values
    result_low_p <- edge_completion(ctx, prompt, n_predict = 5, top_p = 0.1)
    expect_true(is.character(result_low_p))
    expect_true(startsWith(result_low_p, prompt))
    
    result_high_p <- edge_completion(ctx, prompt, n_predict = 5, top_p = 0.95)
    expect_true(is.character(result_high_p))
    expect_true(startsWith(result_high_p, prompt))
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for temperature/top_p tests")
  }
})

test_that("Sequential completions work correctly", {
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
    
    prompts <- c("Hello", "World", "Test", "Final")
    
    for (prompt in prompts) {
      result <- edge_completion(ctx, prompt, n_predict = 3)
      expect_true(is.character(result))
      expect_true(startsWith(result, prompt))
    }
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for sequential completion tests")
  }
})

test_that("edge_completion boundary conditions", {
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
    ctx <- edge_load_model(model_path, n_ctx = 512)
    
    # Test minimum n_predict
    result_min <- edge_completion(ctx, "Hello", n_predict = 1)
    expect_true(is.character(result_min))
    expect_true(nchar(result_min) > nchar("Hello"))
    
    # Test various prompt lengths
    short_prompt <- "Hi"
    medium_prompt <- "This is a medium length prompt with several words"
    
    result_short <- edge_completion(ctx, short_prompt, n_predict = 5)
    result_medium <- edge_completion(ctx, medium_prompt, n_predict = 5)
    
    expect_true(is.character(result_short))
    expect_true(is.character(result_medium))
    expect_true(startsWith(result_short, short_prompt))
    expect_true(startsWith(result_medium, medium_prompt))
    
    # Test whitespace handling
    whitespace_prompts <- c(
      " ",  # Single space
      "   ",  # Multiple spaces
      "Hello World",  # Normal spaces
      "Hello\tWorld"  # Tab
    )
    
    for (prompt in whitespace_prompts) {
      result <- edge_completion(ctx, prompt, n_predict = 3)
      expect_true(is.character(result))
      expect_true(nchar(result) > 0)
    }
    
    edge_free_model(ctx)
  } else {
    skip("No test model available for boundary condition tests")
  }
})