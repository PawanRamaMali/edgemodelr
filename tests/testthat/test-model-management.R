
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








