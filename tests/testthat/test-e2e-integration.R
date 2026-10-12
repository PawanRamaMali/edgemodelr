# End-to-End Integration Tests
# These tests download a small model and perform actual inference

test_that("E2E: Download and load TinyLlama model", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  # Use a small model for testing
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]

  expect_true(nrow(tiny_model) > 0, "TinyLlama model should be in catalog")

  # Create temporary directory for model download
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  dir.create(test_dir, showWarnings = FALSE, recursive = TRUE)

  # Download model using direct URL
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    result <- edge_download_url(
      tiny_model$download_url[1],
      tiny_model$filename[1],
      cache_dir = test_dir
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
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  # Ensure model is downloaded
  if (!file.exists(model_path)) {
    edge_download_url(
      tiny_model$download_url[1],
      tiny_model$filename[1],
      cache_dir = test_dir
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
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_url(
      tiny_model$download_url[1],
      tiny_model$filename[1],
      cache_dir = test_dir
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
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_url(
      tiny_model$download_url[1],
      tiny_model$filename[1],
      cache_dir = test_dir
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
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  # Use quick setup
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  setup <- edge_quick_setup("TinyLlama-1.1B", cache_dir = test_dir)

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
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_url(
      tiny_model$download_url[1],
      tiny_model$filename[1],
      cache_dir = test_dir
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
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  # Setup model
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  model_path <- file.path(test_dir, tiny_model$filename[1])

  if (!file.exists(model_path)) {
    edge_download_url(
      tiny_model$download_url[1],
      tiny_model$filename[1],
      cache_dir = test_dir
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

test_that("E2E: prompts longer than n_batch are processed", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  dir.create(test_dir, showWarnings = FALSE, recursive = TRUE)
  model_path <- file.path(test_dir, tiny_model$filename[1])
  if (!file.exists(model_path)) {
    edge_download_url(tiny_model$download_url[1], tiny_model$filename[1],
                      cache_dir = test_dir)
  }

  # n_ctx = 513 gives n_batch = 256. This prompt is about 277 tokens: it
  # fits the context but used to trip GGML_ASSERT(n_tokens_all <= n_batch)
  # because the whole prompt went to llama_decode in one call. Keep it to
  # one short call: the generic CPU build processes prompts slowly, and the
  # sanitizer job runs this file under ASAN. All three completion paths
  # share decode_prompt(), so one path covers the fix.
  ctx <- edge_load_model(model_path, n_ctx = 513L, n_gpu_layers = 0)
  on.exit(edge_free_model(ctx), add = TRUE)
  long_prompt <- paste(
    rep("The patient reported a mild headache on day three.", 23),
    collapse = " "
  )

  out <- edge_completion(ctx, long_prompt, n_predict = 4L)
  expect_type(out, "character")
})

test_that("E2E: each completion starts from an empty context", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  dir.create(test_dir, showWarnings = FALSE, recursive = TRUE)
  model_path <- file.path(test_dir, tiny_model$filename[1])
  if (!file.exists(model_path)) {
    edge_download_url(tiny_model$download_url[1], tiny_model$filename[1],
                      cache_dir = test_dir)
  }

  # n_ctx = 256 fills after a few calls if the KV cache carries over.
  ctx <- edge_load_model(model_path, n_ctx = 256L, n_gpu_layers = 0)
  on.exit(edge_free_model(ctx), add = TRUE)

  first <- edge_completion(ctx, "The capital of France is", n_predict = 16L,
                           temperature = 0)
  edge_completion(ctx, "List three colours:", n_predict = 32L, temperature = 0)
  for (i in 1:8) {
    out <- edge_completion(ctx, "The capital of France is", n_predict = 16L,
                           temperature = 0)
  }
  expect_identical(out, first)
})

test_that("E2E: timeout_seconds stops a long generation", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading

  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  dir.create(test_dir, showWarnings = FALSE, recursive = TRUE)
  model_path <- file.path(test_dir, tiny_model$filename[1])
  if (!file.exists(model_path)) {
    edge_download_url(tiny_model$download_url[1], tiny_model$filename[1],
                      cache_dir = test_dir)
  }

  ctx <- edge_load_model(model_path, n_ctx = 2048L, n_gpu_layers = 0)
  on.exit(edge_free_model(ctx), add = TRUE)

  # The timeout is checked between tokens, so it fires after the prompt and
  # at least one token. Scale the bound by a short run on this machine:
  # under ASAN that alone takes close to a minute.
  prompt <- "Write a very long story about a dragon."
  short <- system.time(edge_completion(ctx, prompt, n_predict = 4L))[["elapsed"]]

  started <- proc.time()[["elapsed"]]
  expect_error(
    edge_completion(ctx, prompt, n_predict = 1500L, timeout_seconds = 1),
    "timed out"
  )
  expect_lt(proc.time()[["elapsed"]] - started, 30 + 10 * short)

  # The context is still usable afterwards.
  expect_type(edge_completion(ctx, "Hello", n_predict = 4L), "character")
})

test_that("E2E: temperature 0 is greedy and gives the same text every time", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows") # Windows CI has memory/segfault issues with model loading
  skip_if_not_installed("jsonlite")

  models <- edge_list_models()
  tiny_model <- models[models$name == "TinyLlama-1.1B", ]
  test_dir <- file.path(tempdir(), "edgemodelr_integration_tests")
  dir.create(test_dir, showWarnings = FALSE, recursive = TRUE)
  model_path <- file.path(test_dir, tiny_model$filename[1])
  if (!file.exists(model_path)) {
    edge_download_url(tiny_model$download_url[1], tiny_model$filename[1],
                      cache_dir = test_dir)
  }

  ctx <- edge_load_model(model_path, n_ctx = 512L, n_gpu_layers = 0)
  on.exit(edge_free_model(ctx), add = TRUE)

  prompt <- "List three primary colours:"
  first <- edge_completion(ctx, prompt, n_predict = 16L, temperature = 0)
  expect_identical(edge_completion(ctx, prompt, n_predict = 16L, temperature = 0),
                   first)

  # Grammar output stays valid and is just as repeatable. Enum fields only,
  # so the JSON always closes within n_predict.
  grammar <- edge_json_grammar(list(answer = c("yes", "no"),
                                    confidence = c("low", "medium", "high")))
  q <- "Is Paris the capital of France? Reply in JSON."
  a <- edge_grammar_completion(ctx, q, grammar, n_predict = 48L, temperature = 0)
  expect_identical(edge_grammar_completion(ctx, q, grammar, n_predict = 48L,
                                           temperature = 0), a)
  expect_true(jsonlite::fromJSON(a)$answer %in% c("yes", "no"))
})
