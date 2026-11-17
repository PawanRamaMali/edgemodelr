# edge_tokenize(): argument checks everywhere, and on TinyLlama that its
# counts match what generation reports as prompt tokens, for ASCII and UTF-8.

test_that("edge_tokenize validates its arguments", {
  expect_error(edge_tokenize(NULL, "hi"), "Invalid model context")
  expect_error(edge_tokenize(list(), "hi"), "Invalid model context")

  local_mocked_bindings(
    is_valid_model = function(ctx) TRUE,
    edge_tokenize_internal = function(ctx, text, add_special) {
      c(if (add_special) 1L, seq_len(nchar(text)) + 100L)
    },
    .package = "edgemodelr"
  )
  expect_error(edge_tokenize("ctx", c("a", "b")), "single character string")
  expect_error(edge_tokenize("ctx", NA_character_), "single character string")
  expect_error(edge_tokenize("ctx", 42), "single character string")
  expect_error(edge_tokenize("ctx", "a", add_special = NA), "TRUE or FALSE")
  expect_error(edge_tokenize("ctx", "a", add_special = "yes"), "TRUE or FALSE")
  expect_error(edge_tokenize("ctx", "a", add_special = c(TRUE, FALSE)), "TRUE or FALSE")
  expect_identical(edge_tokenize("ctx", "ab"), c(1L, 101L, 102L))
  expect_identical(edge_tokenize("ctx", "ab", add_special = FALSE), c(101L, 102L))
})

tok_texts <- c(
  ascii = "The quick brown fox jumps over the lazy dog.",
  utf8 = "Grüße aus Zürich: naïve café, 東京, emoji \U0001F600"
)

test_that("edge_tokenize returns token ids with and without the start token", {
  model <- coverage_tinyllama()
  skip_without_model(model, "TinyLlama")
  ctx <- edge_load_model(model, n_ctx = 512L, n_threads = 2L)
  on.exit(edge_free_model(ctx), add = TRUE)

  for (nm in names(tok_texts)) {
    txt <- tok_texts[[nm]]
    with_bos <- edge_tokenize(ctx, txt)
    without <- edge_tokenize(ctx, txt, add_special = FALSE)
    expect_type(with_bos, "integer")
    expect_false(anyNA(with_bos), label = nm)
    expect_true(all(with_bos >= 0L), label = nm)
    # TinyLlama's BOS is token 1, and add_special adds exactly that.
    expect_identical(with_bos[1], 1L, label = nm)
    expect_identical(with_bos[-1], without, label = nm)
    expect_identical(edge_tokenize(ctx, txt), with_bos, label = nm)
  }
  expect_identical(edge_tokenize(ctx, "", add_special = FALSE), integer(0))
  expect_identical(edge_tokenize(ctx, ""), 1L)

  # Multi-byte text takes more tokens than characters would suggest for the
  # CJK and emoji parts, but never zero.
  utf8_tokens <- edge_tokenize(ctx, "東京 \U0001F600", add_special = FALSE)
  expect_gt(length(utf8_tokens), 0L)
})

test_that("edge_tokenize counts match the prompt_tokens generation reports", {
  model <- coverage_tinyllama()
  skip_without_model(model, "TinyLlama")
  ctx <- edge_load_model(model, n_ctx = 1024L, n_threads = 2L, n_parallel = 2L)
  on.exit(edge_free_model(ctx), add = TRUE)

  prompts <- unname(c(tok_texts,
                      build_chat_prompt(list(list(role = "user", content = "Hi")),
                                        ctx = ctx)))
  expected <- vapply(prompts, function(p) length(edge_tokenize(ctx, p)), integer(1),
                     USE.NAMES = FALSE)

  engine <- edge_engine(ctx, n_slots = 2L)
  out <- tryCatch(
    edge_engine_generate(engine, prompts, n_predict = 2L, temperature = 0),
    finally = edge_engine_stop(engine)
  )
  expect_identical(as.integer(out$prompt_tokens), expected)
})

test_that("edge_tokenize treats latin1 and UTF-8 strings alike", {
  model <- coverage_tinyllama()
  skip_without_model(model, "TinyLlama")
  ctx <- edge_load_model(model, n_ctx = 256L, n_threads = 2L)
  on.exit(edge_free_model(ctx), add = TRUE)
  u <- "Zürich"
  l <- iconv(u, "UTF-8", "latin1")
  expect_identical(Encoding(l), "latin1")
  expect_identical(edge_tokenize(ctx, l), edge_tokenize(ctx, u))
})
