

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




