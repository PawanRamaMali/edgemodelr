# End-to-End Integration Tests
# These tests download a small model and perform actual inference

test_that("E2E: Download and load TinyLlama model", {
  skip_on_cran()
  skip_if_offline()

  # Use a small model for testing
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]

  expect_true(nrow(tiny_model) > 0, "TinyLlama model should be in catalog")

  # Create temporary directory for model download
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  dir.create(test_dir, showWarnings = FALSE, recursive = TRUE)

  # Download model
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    result <- edge_download_model(
      tiny_model$model_id[1],
      tiny_model$filename[1],
      dest_dir = test_dir
    )
    expect_true(file.exists(model_path), "Model should be downloaded")
  }

  # Load model
  ctx <- edge_load_model(model_path, n_ctx = 512, n_gpu_layers = 0)
  expect_true(is_valid_model(ctx), "Model context should be valid")

  # Clean up
  edge_free_model(ctx)
})


test_that("E2E: Basic text completion inference", {
  skip_on_cran()
  skip_if_offline()

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  # Ensure model is downloaded
  if (!file.exists(model_path)) {
    edge_download_model(
      tiny_model$model_id[1],
      tiny_model$filename[1],
      dest_dir = test_dir
    )
  }

  ctx <- edge_load_model(model_path, n_ctx = 512, n_gpu_layers = 0)

  # Test basic completion
  result <- edge_completion(
    ctx,
    prompt = "Q: What is 2+2? A:",
    n_predict = 20,
    temperature = 0.7
  )

  expect_true(is.character(result), "Result should be a character string")
  expect_true(nchar(result) > 0, "Result should not be empty")
  expect_true(nchar(result) < 1000, "Result should be reasonable length")

  # Clean up
  edge_free_model(ctx)
})


test_that("E2E: Multiple sequential completions", {
  skip_on_cran()
  skip_if_offline()

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_model(
      tiny_model$model_id[1],
      tiny_model$filename[1],
      dest_dir = test_dir
    )
  }

  ctx <- edge_load_model(model_path, n_ctx = 512, n_gpu_layers = 0)

  # Run multiple completions to test stability
  prompts <- c(
    "The color of the sky is",
    "2 + 2 equals",
    "R programming is"
  )

  for (prompt in prompts) {
    result <- edge_completion(ctx, prompt, n_predict = 10)
    expect_true(is.character(result), paste("Failed on prompt:", prompt))
    expect_true(nchar(result) > 0, paste("Empty result for prompt:", prompt))
  }

  # Clean up
  edge_free_model(ctx)
})


test_that("E2E: Streaming completion", {
  skip_on_cran()
  skip_if_offline()

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_model(
      tiny_model$model_id[1],
      tiny_model$filename[1],
      dest_dir = test_dir
    )
  }

  ctx <- edge_load_model(model_path, n_ctx = 512, n_gpu_layers = 0)

  # Test streaming completion
  chunks <- character()
  stream_result <- edge_stream_completion(
    ctx,
    prompt = "List three colors:",
    n_predict = 30,
    callback = function(chunk) {
      chunks <<- c(chunks, chunk)
    }
  )

  expect_true(length(chunks) > 0, "Should receive streaming chunks")
  full_text <- paste(chunks, collapse = "")
  expect_true(nchar(full_text) > 0, "Streamed text should not be empty")

  # Clean up
  edge_free_model(ctx)
})


test_that("E2E: edge_quick_setup integration", {
  skip_on_cran()
  skip_if_offline()

  # Use quick setup
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  setup <- edge_quick_setup("TinyLlama-1.1B", dest_dir = test_dir)

  expect_true(!is.null(setup), "Setup should return results")
  expect_true(!is.null(setup$context), "Context should be created")
  expect_true(is_valid_model(setup$context), "Context should be valid")
  expect_true(file.exists(setup$model_path), "Model file should exist")

  # Test inference with quick setup
  result <- edge_completion(
    setup$context,
    prompt = "Hello",
    n_predict = 15
  )

  expect_true(is.character(result), "Result should be character")
  expect_true(nchar(result) > 0, "Result should not be empty")

  # Clean up
  edge_free_model(setup$context)
})


test_that("E2E: Different temperature settings", {
  skip_on_cran()
  skip_if_offline()

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_model(
      tiny_model$model_id[1],
      tiny_model$filename[1],
      dest_dir = test_dir
    )
  }

  ctx <- edge_load_model(model_path, n_ctx = 512, n_gpu_layers = 0)

  # Test different temperature values
  temperatures <- c(0.1, 0.7, 1.0)
  prompt <- "The capital of France is"

  for (temp in temperatures) {
    result <- edge_completion(
      ctx,
      prompt = prompt,
      n_predict = 10,
      temperature = temp
    )
    expect_true(is.character(result), paste("Failed at temperature:", temp))
    expect_true(nchar(result) > 0, paste("Empty at temperature:", temp))
  }

  # Clean up
  edge_free_model(ctx)
})


test_that("E2E: Model reload and reuse", {
  skip_on_cran()
  skip_if_offline()

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_model(
      tiny_model$model_id[1],
      tiny_model$filename[1],
      dest_dir = test_dir
    )
  }

  # Load, use, and free
  ctx1 <- edge_load_model(model_path, n_ctx = 512)
  result1 <- edge_completion(ctx1, "Test", n_predict = 5)
  expect_true(nchar(result1) > 0)
  edge_free_model(ctx1)

  # Reload the same model
  ctx2 <- edge_load_model(model_path, n_ctx = 512)
  result2 <- edge_completion(ctx2, "Test", n_predict = 5)
  expect_true(nchar(result2) > 0)
  edge_free_model(ctx2)

  # Both should work independently
  expect_true(is.character(result1))
  expect_true(is.character(result2))
})
