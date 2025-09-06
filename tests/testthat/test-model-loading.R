# Test 1: edge_list_models() returns a data frame
test_that("edge_list_models returns valid data", {
  models <- edge_list_models()
  expect_true(is.data.frame(models))
  expect_true(nrow(models) > 0)
  
  # Check all required columns exist
  required_cols <- c("name", "size", "model_id", "filename", "use_case")
  expect_true(all(required_cols %in% colnames(models)))
  
  # Check data types
  expect_true(is.character(models$name))
  expect_true(is.character(models$size))
  expect_true(is.character(models$model_id))
  expect_true(is.character(models$filename))
  expect_true(is.character(models$use_case))
  
  # Check for no empty values in critical columns
  expect_true(all(nchar(models$model_id) > 0))
  expect_true(all(nchar(models$filename) > 0))
  
  # Check filename format (should end with .gguf)
  expect_true(all(grepl("\\.gguf$", models$filename, ignore.case = TRUE)))
  
  # Check model_id format (should contain /)
  expect_true(all(grepl("/", models$model_id)))
})

# Test 2: edge_load_model with invalid inputs
test_that("edge_load_model handles invalid paths", {
  expect_error(
    edge_load_model("nonexistent_model.gguf"),
    "Model file does not exist"
  )
  
  # Test with empty string
  expect_error(
    edge_load_model(""),
    "Model file does not exist"
  )
  
  # Test with NULL
  expect_error(edge_load_model(NULL))
  
  # Test with directory instead of file
  temp_dir <- tempdir()
  expect_error(
    edge_load_model(temp_dir)
  )
  
  # Test with non-GGUF file
  temp_file <- tempfile(fileext = ".txt")
  tryCatch({
    writeLines("not a model", temp_file)
    expect_warning(
      expect_error(edge_load_model(temp_file)),
      "should have .gguf extension"
    )
  }, finally = {
    if (file.exists(temp_file)) {
      unlink(temp_file)
    }
  })
})


# Test 3: is_valid_model with invalid contexts
test_that("is_valid_model handles invalid contexts", {
  expect_false(is_valid_model(NULL))
  expect_false(is_valid_model("invalid"))
  expect_false(is_valid_model(123))
  expect_false(is_valid_model(list()))
  expect_false(is_valid_model(data.frame()))
  expect_false(is_valid_model(TRUE))
  expect_false(is_valid_model(FALSE))
  expect_false(is_valid_model(c(1, 2, 3)))
  
  # Test with missing arguments - function returns FALSE instead of erroring
  result <- is_valid_model()
  expect_false(result)
})


# Test 4: Model loading with real model (if available)
test_that("Model loading with real model (if available)", {
  # Try multiple possible model locations
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
    # Test successful model loading
    ctx <- edge_load_model(model_path, n_ctx = 256)
    expect_true(!is.null(ctx))
    expect_true(inherits(ctx, "edge_model_context"))
    expect_true(is_valid_model(ctx))
    edge_free_model(ctx)
    
    
    
    
    
  } else {
    skip("No test model available for real model loading tests")
  }
})

