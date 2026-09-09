# Completions must not depend on what was asked before them.
#
# The three generation loops build the prompt batch with llama_batch_get_one,
# which leaves the positions unset, so the allocator continues numbering from
# seq_pos_max + 1. Without clearing the cache first, every answer is
# conditioned on every earlier prompt in the session. That silently corrupts
# edge_map(), edge_classify(), edge_extract_batch() and edge_narrate(), which
# all issue one completion per element against a single context.
#
# Measured before the fix: asked "Name a fruit:" after two other questions,
# the model answered " fresh. Name a location: Paihia", continuing the earlier
# prompts instead of answering the current one.

# Resolve a model without assuming where it lives. The integration workflow
# downloads to its own directory, so looking only in the user cache would skip
# every test here while still reporting a green run.
ctx_iso_model <- function() {
  models <- edge_list_models()
  tiny <- models[models$name == "TinyLlama-1.1B", ]
  if (nrow(tiny) == 0) return(NA_character_)
  fname <- tiny$filename[1]

  runner_tmp <- Sys.getenv("RUNNER_TEMP", "")
  candidates <- c(
    file.path(tempdir(), "edgemodelr_integration_tests"),
    if (nzchar(runner_tmp)) file.path(runner_tmp, "edgemodelr_models"),
    if (nzchar(runner_tmp)) runner_tmp,
    tools::R_user_dir("edgemodelr", "cache")
  )
  for (d in candidates) {
    p <- file.path(d, fname)
    if (file.exists(p)) return(p)
  }

  # Nothing on disk, so fetch into the same place the other e2e tests use.
  d <- file.path(tempdir(), "edgemodelr_integration_tests")
  dir.create(d, showWarnings = FALSE, recursive = TRUE)
  try(edge_download_url(tiny$download_url[1], fname, cache_dir = d), silent = TRUE)
  p <- file.path(d, fname)
  if (file.exists(p)) p else NA_character_
}


test_that("a batch of prompts matches the same prompts on fresh contexts", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows")

  model <- ctx_iso_model()
  skip_if(is.na(model), "TinyLlama could not be resolved")

  qs <- c("Name a fruit:", "Name a country:", "Name a colour:")

  ctx <- edge_load_model(model, n_ctx = 512L)
  on.exit(edge_free_model(ctx), add = TRUE)
  batched <- vapply(qs, function(q) {
    edge_completion(ctx, q, n_predict = 10L, temperature = 0.8)
  }, character(1), USE.NAMES = FALSE)

  isolated <- vapply(qs, function(q) {
    c2 <- edge_load_model(model, n_ctx = 512L)
    on.exit(edge_free_model(c2), add = TRUE)
    edge_completion(c2, q, n_predict = 10L, temperature = 0.8)
  }, character(1), USE.NAMES = FALSE)

  expect_identical(batched, isolated)
})


test_that("repeating a prompt in a used context gives the same answer", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows")

  model <- ctx_iso_model()
  skip_if(is.na(model), "TinyLlama could not be resolved")

  ctx <- edge_load_model(model, n_ctx = 512L)
  on.exit(edge_free_model(ctx), add = TRUE)

  probe <- "The capital of France is"
  first <- edge_completion(ctx, probe, n_predict = 12L, temperature = 0.8)

  for (p in c("Write one word about oceans:", "Name a colour:", "Say hello:")) {
    invisible(edge_completion(ctx, p, n_predict = 12L, temperature = 0.8))
  }

  again <- edge_completion(ctx, probe, n_predict = 12L, temperature = 0.8)
  expect_identical(first, again)
})


test_that("many sequential completions do not exhaust the context", {
  skip_on_cran()
  skip_if_offline()
  skip_on_os("windows")

  model <- ctx_iso_model()
  skip_if(is.na(model), "TinyLlama could not be resolved")

  # A small context that would fill quickly if each call were appended to the
  # last. Before the cache was cleared per call this failed partway through
  # with "Failed to process prompt".
  ctx <- edge_load_model(model, n_ctx = 512L)
  on.exit(edge_free_model(ctx), add = TRUE)

  for (i in 1:12) {
    out <- edge_completion(ctx, paste0("Say the number ", i, ":"),
                           n_predict = 8L, temperature = 0.8)
    expect_type(out, "character")
  }
})
