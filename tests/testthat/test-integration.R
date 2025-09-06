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
    
    # 2. Test single completion
    result <- edge_completion(ctx, "Hello", n_predict = 5)
    expect_true(is.character(result))
    expect_true(startsWith(result, "Hello"))
    
    # 4. Free model
    edge_free_model(ctx)
    
  } else {
    skip("No test model available for integration tests")
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




