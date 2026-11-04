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

test_that("E2E: prompt_cache gives the same answers as an empty context", {
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

  cached <- edge_load_model(model_path, n_ctx = 512L, prompt_cache = TRUE)
  on.exit(edge_free_model(cached), add = TRUE)
  plain <- edge_load_model(model_path, n_ctx = 512L)
  on.exit(edge_free_model(plain), add = TRUE)

  # Prompts share a long start; the last one repeats the first, so it
  # follows a different prompt and reuses only the shared part.
  intro <- paste0("<|system|>\nYou answer geography questions with one short ",
                  "sentence.</s>\n<|user|>\n")
  questions <- c("What is the capital of France?", "What is the capital of Japan?",
                 "What is the capital of France?")
  prompts <- paste0(intro, questions, "</s>\n<|assistant|>\n")
  run <- function(ctx) {
    vapply(prompts, function(p) {
      edge_completion(ctx, p, n_predict = 12L, temperature = 0)
    }, character(1), USE.NAMES = FALSE)
  }
  with_cache <- run(cached)
  without <- run(plain)
  # The first call has nothing to reuse, so it must match exactly. Later
  # calls decode the prompt in different pieces, which rounds differently,
  # so they are checked for the right answer: a stale cache would answer
  # the previous question.
  expect_identical(with_cache[1], without[1])
  expect_match(with_cache[c(1, 3)], "Paris")
  expect_match(with_cache[2], "Tokyo")
  expect_false(grepl("Paris", with_cache[2]))
})

test_that("E2E: jump_forward gives valid JSON with the same fields", {
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

  ctx <- edge_load_model(model_path, n_ctx = 512L)
  on.exit(edge_free_model(ctx), add = TRUE)

  schema <- list(answer = c("yes", "no"), confidence = c("low", "medium", "high"))
  q <- "Is Paris the capital of France? Reply in JSON."
  for (compact in c(FALSE, TRUE)) {
    g <- edge_json_grammar(schema, compact = compact)
    on <- edge_grammar_completion(ctx, q, g, n_predict = 48L, temperature = 0)
    off <- edge_grammar_completion(ctx, q, g, n_predict = 48L, temperature = 0,
                                   jump_forward = FALSE)
    # Forced text is tokenized the standard way, so the two can differ in
    # the values the model picks, but never in the fields.
    on_json <- jsonlite::fromJSON(on)
    expect_identical(names(on_json), names(jsonlite::fromJSON(off)))
    expect_true(on_json$answer %in% c("yes", "no"))
    expect_true(on_json$confidence %in% c("low", "medium", "high"))
  }
})

test_that("E2E: edge_embeddings gives the same vectors in a batch as one by one", {
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

  ctx <- edge_load_model(model_path, n_ctx = 512L, embeddings = TRUE)
  on.exit(edge_free_model(ctx), add = TRUE)

  texts <- c("The cat sat on the mat.", "Blood pressure, systolic",
             "A much longer sentence about the weather in the mountains today.")
  batch <- edge_embeddings(ctx, texts)
  single <- do.call(rbind, lapply(texts, function(t) edge_embeddings(ctx, t)))
  expect_equal(dim(batch), c(3L, edge_model_n_embd(ctx)))
  # Batched matrix products round a little differently (cosine about 0.998
  # for short texts on Linux).
  expect_true(all(rowSums(batch * single) > 0.99))
  # Two copies of one text in a batch must not see each other.
  pair <- edge_embeddings(ctx, texts[c(1, 1)])
  expect_true(all(pair %*% single[1, ] > 0.9999))
})

test_that("E2E: LoRA adapters load, switch and reject a wrong model", {
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

  ctx <- edge_load_model(model_path, n_ctx = 512L)
  on.exit(edge_free_model(ctx), add = TRUE)
  expect_identical(edge_lora_scale(ctx), numeric(0))
  expect_error(edge_lora_scale(ctx, 1), "one per loaded adapter")

  # A model file is not a LoRA adapter: an R error, not a crash.
  expect_error(edge_load_model(model_path, n_ctx = 512L, lora = model_path),
               "Failed to load LoRA adapter")
  expect_true(is_valid_model(ctx))
})

test_that("E2E: seed, stop strings and the repeat penalty", {
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

  ctx <- edge_load_model(model_path, n_ctx = 512L)
  on.exit(edge_free_model(ctx), add = TRUE)
  p <- "<|user|>\nList five fruits, one per line.</s>\n<|assistant|>\n"

  # The same seed gives the same text, and so does set.seed().
  expect_identical(edge_completion(ctx, p, n_predict = 24L, seed = 7),
                   edge_completion(ctx, p, n_predict = 24L, seed = 7))
  set.seed(1)
  a <- edge_completion(ctx, p, n_predict = 24L)
  set.seed(1)
  expect_identical(edge_completion(ctx, p, n_predict = 24L), a)

  # Text ends where a stop string starts, also when streaming, and also
  # when the stop string spans several tokens.
  full <- edge_completion(ctx, p, n_predict = 40L, temperature = 0)
  skip_if(!grepl("\n", full), "greedy answer has no line break")
  first_line <- sub("\n.*", "", full)
  expect_identical(edge_completion(ctx, p, n_predict = 40L, temperature = 0,
                                   stop = "\n"), first_line)
  stop_str <- substr(full, nchar(first_line) - 2L, nchar(first_line) + 2L)
  pieces <- character(0)
  streamed <- edge_stream_completion(ctx, p, function(d) {
    if (!d$is_final) pieces <<- c(pieces, d$token)
    TRUE
  }, n_predict = 40L, temperature = 0, stop = stop_str)
  expect_identical(streamed$full_response, substr(full, 1L, nchar(first_line) - 3L))
  expect_identical(paste(pieces, collapse = ""), streamed$full_response)

  # Without stop strings the stream gives the same text as a completion.
  pieces <- character(0)
  streamed <- edge_stream_completion(ctx, p, function(d) {
    if (!d$is_final) pieces <<- c(pieces, d$token)
    TRUE
  }, n_predict = 40L, temperature = 0)
  expect_identical(paste(pieces, collapse = ""), full)

  # The repeat penalty stops a loop that greedy decoding falls into.
  loop <- "The cat sat. The cat sat. The cat sat."
  plain <- edge_completion(ctx, loop, n_predict = 24L, temperature = 0)
  penalised <- edge_completion(ctx, loop, n_predict = 24L, temperature = 0,
                               repeat_penalty = 1.5)
  expect_false(identical(plain, penalised))

  expect_type(edge_completion(ctx, p, n_predict = 8L, top_k = 40L,
                              min_p = 0.05, seed = 3), "character")
})

test_that("E2E: tool calls come back with the declared types", {
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

  ctx <- edge_load_model(model_path, n_ctx = 1024L)
  on.exit(edge_free_model(ctx), add = TRUE)
  tools <- list(
    edge_tool("get_weather", "Get the current weather for a city",
              list(city = "string", unit = c("celsius", "fahrenheit"))),
    edge_tool("add_numbers", "Add two numbers", list(a = "number", b = "number")))
  msgs <- list(list(role = "user", content = "What's the weather like in Paris?"))

  # A forced call is always one of the tools, with every argument typed.
  r <- edge_chat_completion(ctx, msgs, tools = tools, tool_choice = "required",
                            temperature = 0, n_predict = 64L)
  expect_null(r$content)
  expect_length(r$tool_calls, 1)
  call <- r$tool_calls[[1]]
  expect_true(call$name %in% c("get_weather", "add_numbers"))
  if (call$name == "get_weather") {
    expect_type(call$arguments$city, "character")
    expect_true(call$arguments$unit %in% c("celsius", "fahrenheit"))
  }
  r <- edge_chat_completion(ctx, msgs, tools = tools, tool_choice = "add_numbers",
                            temperature = 0, n_predict = 64L)
  expect_identical(r$tool_calls[[1]]$name, "add_numbers")
  expect_type(r$tool_calls[[1]]$arguments$a, "double")

  # In auto mode the result is a call or text, never both.
  r <- edge_chat_completion(ctx, msgs, tools = tools, temperature = 0, n_predict = 64L)
  expect_true(xor(is.null(r$content), length(r$tool_calls) == 0L))
  expect_identical(r$message$role, "assistant")
})

test_that("E2E: n_parallel runs a batch of prompts together", {
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

  words <- c("apple", "car", "river", "piano", "winter", "coffee")
  tmpl <- "<|user|>\nWrite one short sentence about {text}.</s>\n<|assistant|>\n"
  # A small context, so not every prompt fits at once.
  ctx <- edge_load_model(model_path, n_ctx = 256L, n_parallel = 4L)
  on.exit(edge_free_model(ctx), add = TRUE)
  out <- edge_map(ctx, c(words, words), tmpl, n_predict = 16L, temperature = 0,
                  progress = FALSE)
  expect_length(out, 12)
  expect_true(all(nzchar(out)))
  # Greedy answers do not depend on which prompts ran alongside. Batched
  # kernels round differently, so only the start is compared.
  expect_identical(substr(out[1:6], 1, 12), substr(out[7:12], 1, 12))

  s <- edge_map(ctx, words, tmpl, n_predict = 16L, seed = 3, stop = " ",
                progress = FALSE)
  expect_false(any(grepl(" ", s)))
  expect_identical(edge_map(ctx, words, tmpl, n_predict = 16L, seed = 3,
                            stop = " ", progress = FALSE), s)

  df <- edge_extract_batch(ctx, c("I love it!", "This is awful."),
                           list(sentiment = c("positive", "negative")),
                           n_predict = 16L, progress = FALSE)
  expect_identical(nrow(df), 2L)
  expect_true(all(df$sentiment %in% c("positive", "negative")))
  expect_type(edge_completion(ctx, "Hello", n_predict = 4L), "character")
})

test_that("E2E: the prompt cache can be saved and loaded", {
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

  prefix <- paste0("<|system|>\nYou answer geography questions with one short ",
                   "sentence.</s>\n<|user|>\n")
  p <- paste0(prefix, "What is the capital of France?</s>\n<|assistant|>\n")
  f <- tempfile(fileext = ".kv")
  on.exit(unlink(f), add = TRUE)

  a <- edge_load_model(model_path, n_ctx = 512L, prompt_cache = TRUE)
  first <- edge_completion(a, p, n_predict = 12L, temperature = 0)
  n <- edge_save_state(a, f)
  edge_free_model(a)
  expect_gt(n, 20)

  b <- edge_load_model(model_path, n_ctx = 512L, prompt_cache = TRUE)
  on.exit(edge_free_model(b), add = TRUE)
  expect_identical(edge_load_state(b, f), n)
  expect_identical(edge_completion(b, p, n_predict = 12L, temperature = 0), first)
  expect_match(edge_completion(b, paste0(prefix, "What is the capital of Japan?",
                                         "</s>\n<|assistant|>\n"),
                               n_predict = 12L, temperature = 0), "Tokyo")

  plain <- edge_load_model(model_path, n_ctx = 512L)
  on.exit(edge_free_model(plain), add = TRUE)
  expect_error(edge_load_state(plain, f), "prompt_cache = TRUE")
  bad <- tempfile()
  on.exit(unlink(bad), add = TRUE)
  writeBin(charToRaw("not a state file"), bad)
  expect_error(edge_load_state(b, bad), "Not a state file")
})
