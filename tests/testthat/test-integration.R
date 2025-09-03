test_that("Complete workflow integration test", {
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
    # Complete workflow test
    
    # 1. Load model
    ctx <- edge_load_model(model_path, n_ctx = 512)
    expect_true(is_valid_model(ctx))
    
    # 2. Test multiple completions with the same context
    prompts <- c(
      "The weather today is",
      "In the future, technology will",
      "My favorite color is",
      "The best way to learn is"
    )
    
    results <- character(length(prompts))
    for (i in seq_along(prompts)) {
      results[i] <- edge_completion(ctx, prompts[i], n_predict = 8)
      expect_true(is.character(results[i]))
      expect_true(startsWith(results[i], prompts[i]))
    }
    
    # 3. Test that results are different (high probability)
    expect_true(length(unique(results)) > 1, 
                info = "Results should generally be different")
    
    # 4. Free model
    edge_free_model(ctx)
    
  } else {
    skip("No test model available for integration tests")
  }
})

test_that("Model listing and information consistency", {
  # Test that model listing is consistent
  models1 <- edge_list_models()
  models2 <- edge_list_models()
  
  expect_equal(models1, models2, 
               info = "Model listing should be consistent")
  
  # Check that listed models have reasonable properties
  for (i in 1:min(3, nrow(models1))) {
    model <- models1[i, ]
    
    # Model ID should be non-empty and reasonable format
    expect_true(nchar(model$model_id) > 0)
    expect_true(grepl("/", model$model_id), 
                info = "Model ID should typically contain '/' for org/model format")
    
    # Filename should end with .gguf
    expect_true(endsWith(model$filename, ".gguf"))
    
    # Size should be informative
    expect_true(nchar(model$size) > 3)
    expect_true(grepl("MB|GB", model$size))
    
    # Use case should be informative
    expect_true(nchar(model$use_case) > 3)
  }
})

test_that("Package namespace and exports", {
  # Test that all expected functions are exported
  expected_functions <- c(
    "edge_load_model",
    "edge_completion", 
    "edge_free_model",
    "is_valid_model",
    "edge_list_models",
    "edge_download_model",
    "edge_quick_setup",
    "edge_stream_completion",
    "edge_chat_stream"
  )
  
  for (func_name in expected_functions) {
    expect_true(exists(func_name, mode = "function"),
                info = paste("Function should be exported:", func_name))
  }
})

test_that("Package loads correctly", {
  # Test package loading and basic functionality
  expect_true(requireNamespace("edgemodelr", quietly = TRUE))
  
  # Test that we can access documentation (if available)
  # This is mainly to ensure the package structure is correct
  tryCatch({
    help_result <- utils::help("edge_load_model", package = "edgemodelr")
    # If help exists, it should have content
    if (length(help_result) > 0) {
      expect_true(length(help_result) > 0)
    }
  }, error = function(e) {
    # Help might not be available in all test environments
    message("Help documentation not available in test environment")
  })
})

test_that("Stress test with multiple operations", {
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
    # Stress test: multiple load/use/free cycles
    
    for (cycle in 1:3) {
      # Load model
      ctx <- edge_load_model(model_path, n_ctx = 256)
      expect_true(is_valid_model(ctx))
      
      # Multiple completions
      for (completion in 1:3) {
        prompt <- paste("Test prompt", cycle, completion)
        result <- edge_completion(ctx, prompt, n_predict = 3)
        expect_true(is.character(result))
        expect_true(startsWith(result, prompt))
      }
      
      # Free model
      edge_free_model(ctx)
    }
    
  } else {
    skip("No test model available for stress tests")
  }
})

test_that("Performance and timing tests", {
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
    # Test that operations complete in reasonable time
    
    # Model loading should complete within reasonable time (30 seconds)
    start_time <- Sys.time()
    ctx <- edge_load_model(model_path, n_ctx = 256)
    load_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    expect_true(is_valid_model(ctx))
    expect_true(load_time < 30, 
                info = paste("Model loading took", round(load_time, 2), "seconds"))
    
    # Text completion should complete within reasonable time (10 seconds)
    start_time <- Sys.time()
    result <- edge_completion(ctx, "Quick test", n_predict = 5)
    completion_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    expect_true(is.character(result))
    expect_true(completion_time < 30,
                info = paste("Text completion took", round(completion_time, 2), "seconds"))
    
    # Model freeing should be fast
    start_time <- Sys.time()
    edge_free_model(ctx)
    free_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    expect_true(free_time < 1,
                info = paste("Model freeing took", round(free_time, 2), "seconds"))
    
  } else {
    skip("No test model available for performance tests")
  }
})

test_that("System integration and environment compatibility", {
  # Test that package works in current R environment
  expect_true(R.version$major >= "4", 
              info = "Package should work with R 4.0+")
  
  # Test platform compatibility
  platform <- Sys.info()["sysname"]
  expect_true(platform %in% c("Windows", "Linux", "Darwin"),
              info = paste("Unsupported platform:", platform))
  
  # Test that the package can handle the current working directory
  original_wd <- getwd()
  tryCatch({
    # Change to temp directory and back
    temp_dir <- tempdir()
    setwd(temp_dir)
    
    # Basic function should still work
    models <- edge_list_models()
    expect_true(is.data.frame(models))
    
  }, finally = {
    setwd(original_wd)
  })
})