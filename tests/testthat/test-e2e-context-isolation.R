# Completions must not depend on what was asked before them.
#
# The three generation loops build the prompt batch with llama_batch_get_one,
# which leaves the positions unset, so the allocator continues numbering from
# seq_pos_max + 1. Without clearing the cache first, every answer is
# conditioned on every earlier prompt in the session. That silently corrupts
# edge_map(), edge_classify(), edge_extract_batch() and edge_narrate(), which
# all issue one completion per element against a single context.
#
# These need a real model, so they only run when NOT_CRAN is set.

test_that("a batch of prompts matches the same prompts on fresh contexts", {
  skip_on_cran()
  skip_on_os("windows")

  model <- file.path(tools::R_user_dir("edgemodelr", "cache"),
                     "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
  skip_if_not(file.exists(model), "TinyLlama not in the cache")

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
  skip_on_os("windows")

  model <- file.path(tools::R_user_dir("edgemodelr", "cache"),
                     "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
  skip_if_not(file.exists(model), "TinyLlama not in the cache")

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
  skip_on_os("windows")

  model <- file.path(tools::R_user_dir("edgemodelr", "cache"),
                     "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
  skip_if_not(file.exists(model), "TinyLlama not in the cache")

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
