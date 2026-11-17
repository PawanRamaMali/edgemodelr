# edge_serve() without opening a port: edge_api() and plumber::pr_run() are
# mocked, so only the argument plumbing and the startup banner are checked.

test_that("edge_serve passes its settings to edge_api and pr_run", {
  skip_if_not_installed("plumber")
  api_args <- NULL
  run_args <- NULL
  local_mocked_bindings(
    edge_api = function(model_path, ...) {
      api_args <<- c(list(model_path = model_path), list(...))
      structure(list(), class = "fake_router")
    },
    .package = "edgemodelr"
  )
  local_mocked_bindings(
    pr_run = function(pr, host, port, ...) {
      run_args <<- list(pr = pr, host = host, port = port)
      invisible(NULL)
    },
    .package = "plumber"
  )

  msgs <- capture_messages(
    edge_serve("models/my-model.gguf", host = "0.0.0.0", port = 9123,
               n_ctx = 1024L, n_slots = 2L, embeddings = TRUE, api_key = "k",
               n_threads = 2L)
  )
  expect_identical(api_args$model_path, "models/my-model.gguf")
  expect_identical(api_args$n_slots, 2L)
  expect_identical(api_args$n_ctx, 1024L)
  expect_true(api_args$embeddings)
  expect_identical(api_args$api_key, "k")
  expect_identical(api_args$n_gpu_layers, 0L)
  expect_identical(api_args$n_threads, 2L)
  expect_s3_class(run_args$pr, "fake_router")
  expect_identical(run_args$host, "0.0.0.0")
  expect_identical(run_args$port, 9123L)

  banner <- paste(msgs, collapse = "")
  expect_match(banner, "Model: my-model (2 slots)", fixed = TRUE)
  expect_match(banner, "POST 0.0.0.0:9123/v1/chat/completions", fixed = TRUE)
  expect_match(banner, "/v1/embeddings", fixed = TRUE)
  expect_match(banner, "Auth: API key required", fixed = TRUE)
})

test_that("edge_serve leaves embeddings and auth out of the banner by default", {
  skip_if_not_installed("plumber")
  local_mocked_bindings(
    edge_api = function(model_path, ...) "router",
    .package = "edgemodelr"
  )
  local_mocked_bindings(
    pr_run = function(pr, host, port, ...) invisible(NULL),
    .package = "plumber"
  )
  banner <- paste(capture_messages(edge_serve("m.gguf")), collapse = "")
  expect_match(banner, "Model: m (1 slot)", fixed = TRUE)
  expect_match(banner, "127.0.0.1:8080/v1/completions", fixed = TRUE)
  expect_false(grepl("/v1/embeddings", banner, fixed = TRUE))
  expect_false(grepl("Auth", banner, fixed = TRUE))
})

test_that("edge_serve stops before serving when the model is missing", {
  for (pkg in c("plumber", "jsonlite", "promises", "later")) skip_if_not_installed(pkg)
  ran <- FALSE
  local_mocked_bindings(
    pr_run = function(...) ran <<- TRUE,
    .package = "plumber"
  )
  expect_error(edge_serve(file.path(tempdir(), "no-such-model.gguf")),
               "Model file not found")
  expect_error(edge_serve(NA_character_), "model_path must be the path")
  expect_false(ran)
})
