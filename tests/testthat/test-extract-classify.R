# edge_extract() and edge_classify(). The mocked tests check prompt
# construction, parsing and fallbacks without a model; the TinyLlama tests
# check the grammar keeps real output inside the schema.

test_that("edge_extract builds the prompt and parses the JSON", {
  skip_if_not_installed("jsonlite")
  seen <- list()
  local_mocked_bindings(
    edge_grammar_completion = function(ctx, prompt, grammar, n_predict, temperature) {
      seen <<- list(prompt = prompt, grammar = grammar, n_predict = n_predict,
                    temperature = temperature)
      ' {"sentiment": "positive", "stars": 5} trailing words'
    },
    edge_completion = function(...) stop("no draft pass expected"),
    .package = "edgemodelr"
  )
  schema <- list(sentiment = c("positive", "negative", "neutral"), stars = "integer")
  res <- edge_extract("ctx", "Loved it, five stars!", schema, n_predict = 64L)

  expect_identical(res, list(sentiment = "positive", stars = 5L))
  expect_identical(seen$grammar, edge_json_grammar(schema))
  expect_identical(seen$n_predict, 64L)
  expect_identical(seen$temperature, 0.2)
  expect_true(startsWith(seen$prompt, "### Instruction\nExtract the following fields"))
  expect_true(grepl("sentiment (one of: positive, negative, neutral)", seen$prompt, fixed = TRUE))
  expect_true(grepl("stars (integer)", seen$prompt, fixed = TRUE))
  expect_true(grepl("### Text\nLoved it, five stars!", seen$prompt, fixed = TRUE))
  expect_true(endsWith(seen$prompt, "### Response\n"))
  expect_false(grepl("### Notes", seen$prompt, fixed = TRUE))

  edge_extract("ctx", "x", schema, instruction = "Custom instruction.")
  expect_true(startsWith(seen$prompt, "### Instruction\nCustom instruction."))
})

test_that("edge_extract with draft = TRUE puts the draft notes in the prompt", {
  skip_if_not_installed("jsonlite")
  calls <- character()
  final_prompt <- NULL
  local_mocked_bindings(
    edge_completion = function(ctx, prompt, n_predict, temperature) {
      calls <<- c(calls, "draft")
      expect_true(grepl("field: value", prompt, fixed = TRUE))
      "  city: Oslo  "
    },
    edge_grammar_completion = function(ctx, prompt, grammar, n_predict, temperature) {
      calls <<- c(calls, "grammar")
      final_prompt <<- prompt
      '{"city": "Oslo"}'
    },
    .package = "edgemodelr"
  )
  res <- edge_extract("ctx", "I live in Oslo.", list(city = "string"), draft = TRUE)
  expect_identical(calls, c("draft", "grammar"))
  expect_identical(res, list(city = "Oslo"))
  expect_true(grepl("### Notes\ncity: Oslo\n\n### Response\n", final_prompt, fixed = TRUE))
})

test_that("edge_extract returns the raw text when there is no JSON", {
  local_mocked_bindings(
    edge_grammar_completion = function(...) "no json here",
    .package = "edgemodelr"
  )
  expect_identical(edge_extract("ctx", "t", list(a = "string")), "no json here")
})

test_that("edge_extract warns and returns raw text for broken JSON", {
  skip_if_not_installed("jsonlite")
  local_mocked_bindings(
    edge_grammar_completion = function(...) '{"a": }',
    .package = "edgemodelr"
  )
  expect_warning(res <- edge_extract("ctx", "t", list(a = "string")),
                 "JSON parsing failed")
  expect_identical(res, '{"a": }')
})

test_that("edge_extract validates text", {
  expect_error(edge_extract("ctx", c("a", "b"), list(a = "string")),
               "text must be a single character string")
  expect_error(edge_extract("ctx", 1, list(a = "string")),
               "text must be a single character string")
})

test_that("edge_classify returns the trimmed category, one per text", {
  prompts <- character()
  grammars <- character()
  local_mocked_bindings(
    edge_grammar_completion = function(ctx, prompt, grammar, n_predict, temperature) {
      prompts <<- c(prompts, prompt)
      grammars <<- c(grammars, grammar)
      if (grepl("great", prompt)) " positive\n" else "negative"
    },
    .package = "edgemodelr"
  )
  cats <- c("positive", "negative", "neutral")
  expect_identical(edge_classify("ctx", "This is great", cats), "positive")
  expect_identical(grammars[1], 'root ::= "positive" | "negative" | "neutral"\n')
  expect_true(grepl("one of these categories: positive, negative, neutral", prompts[1],
                    fixed = TRUE))
  expect_true(endsWith(prompts[1], "### Text\nThis is great\n\n### Category\n"))

  out <- edge_classify("ctx", c("great stuff", "awful", "great again"), cats)
  expect_identical(out, c("positive", "negative", "positive"))
  expect_null(names(out))

  edge_classify("ctx", "x", cats, instruction = "Pick one.")
  expect_true(startsWith(prompts[length(prompts)], "### Instruction\nPick one."))
})

test_that("edge_classify falls back to free generation when the grammar fails", {
  local_mocked_bindings(
    edge_grammar_completion = function(...) stop("grammar sampler failed"),
    edge_completion_internal = function(ctx, prompt, n_predict, sampling) {
      if (grepl("first", prompt)) paste0(prompt, " I think it is Negative overall")
      else if (grepl("second", prompt)) "neu"
      else "something else"
    },
    .package = "edgemodelr"
  )
  cats <- c("positive", "negative", "neutral")
  expect_identical(edge_classify("ctx", "the first text", cats), "negative")
  expect_identical(edge_classify("ctx", "the second text", cats), "neutral")
  expect_identical(edge_classify("ctx", "the third text", cats), "positive")
})

test_that("edge_classify validates categories", {
  expect_error(edge_classify("ctx", "t", "only-one"), "at least 2 options")
  expect_error(edge_classify("ctx", "t", 1:3), "at least 2 options")
})

test_that("edge_extract and edge_classify stay in the schema with TinyLlama", {
  skip_if_not_installed("jsonlite")
  model <- coverage_tinyllama()
  skip_without_model(model, "TinyLlama")
  ctx <- edge_load_model(model, n_ctx = 1024L, n_threads = 2L)
  on.exit(edge_free_model(ctx), add = TRUE)

  schema <- list(sentiment = c("positive", "negative", "neutral"), stars = "integer")
  res <- edge_extract(ctx, "Absolutely loved this phone. Five stars!", schema,
                      n_predict = 48L, temperature = 0)
  expect_type(res, "list")
  expect_setequal(names(res), names(schema))
  expect_true(res$sentiment %in% schema$sentiment)
  expect_true(is.numeric(res$stars) && res$stars == round(res$stars))

  cats <- c("positive", "negative", "neutral")
  one <- edge_classify(ctx, "I love it, best purchase ever!", cats, temperature = 0)
  expect_true(one %in% cats)
  many <- edge_classify(ctx, c("Great product!", "Terrible, it broke."), cats,
                        temperature = 0)
  expect_length(many, 2L)
  expect_true(all(many %in% cats))
})
