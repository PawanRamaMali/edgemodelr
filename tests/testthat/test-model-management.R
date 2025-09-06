
test_that("edge_download_model parameter validation", {
  # Test with invalid model_id
  expect_error(
    edge_download_model("", "test.gguf"),
    "model_id cannot be empty"
  )
  
  expect_error(
    edge_download_model(NULL, "test.gguf"),
    "model_id must be a string"
  )
  
  # Test with invalid filename
  expect_error(
    edge_download_model("test/model", ""),
    "filename cannot be empty"
  )
  
  expect_error(
    edge_download_model("test/model", NULL),
    "filename must be a string"
  )
})

test_that("edge_quick_setup parameter validation", {
  # Test with invalid model names
  expect_error(
    edge_quick_setup(""),
    "model_name cannot be empty"
  )
  
  expect_error(
    edge_quick_setup(NULL),
    "model_name cannot be empty"
  )
  
  # Test with non-existent model
  expect_error(
    edge_quick_setup("nonexistent_model_12345"),
    "Model.*not found"
  )
})


test_that("edge_free_model handles invalid contexts gracefully", {
  # These should not crash, just handle gracefully
  expect_silent(edge_free_model(NULL))
  expect_silent(edge_free_model("invalid"))
  expect_silent(edge_free_model(123))
})

test_that("Model format validation works", {
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
    # Test that GGUF file is properly recognized
    ctx <- edge_load_model(model_path, n_ctx = 256)
    expect_true(is_valid_model(ctx))
    edge_free_model(ctx)
  }
  
  # Test with non-GGUF files (should fail gracefully)
  if (file.exists("DESCRIPTION")) {
    expect_error(
      edge_load_model("DESCRIPTION", n_ctx = 256)
    )
  }
  
  # Test with binary files that aren't GGUF
  temp_binary <- tempfile(fileext = ".bin")
  tryCatch({
    writeBin(raw(1000), temp_binary)  # Create fake binary file
    expect_error(
      edge_load_model(temp_binary, n_ctx = 256)
    )
  }, finally = {
    # Always clean up, even if test fails
    if (file.exists(temp_binary)) {
      unlink(temp_binary)
    }
  })
})


test_that("edge_quick_setup integration", {
  # Test parameter validation only - don't actually download models
  skip("Skipping edge_quick_setup integration test to avoid downloading models during testing")
})





