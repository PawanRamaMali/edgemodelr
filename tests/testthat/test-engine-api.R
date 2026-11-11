# edge_engine() and edge_api() pieces that need no model: argument checks,
# and the OpenAI layer run on a fake generator.

fake_gguf <- function() {
  path <- tempfile(fileext = ".gguf")
  writeBin(charToRaw("GGUF"), path)
  path
}

# Runs later's loop until the promise settles, then returns its value.
wait_for <- function(p) {
  value <- NULL
  err <- NULL
  done <- FALSE
  promises::then(p, function(v) {
    value <<- v
    done <<- TRUE
  }, onRejected = function(e) {
    err <<- e
    done <<- TRUE
  })
  for (i in 1:200) {
    later::run_now(0.01)
    if (done) break
  }
  if (!is.null(err)) stop(err)
  value
}

# A generator that answers like the engine and records what it was asked.
fake_gen <- function(reply = "Hello there, friend.", tool_reply = NULL) {
  calls <- list()
  gen <- list(
    prompt = function(messages) paste(vapply(messages, function(m) m$content, ""), collapse = "\n"),
    generate = function(prompt, n_predict, sampling, grammar = NULL) {
      calls[[length(calls) + 1L]] <<- list(prompt = prompt, grammar = grammar, sampling = sampling)
      text <- if (!is.null(grammar) && !is.null(tool_reply)) tool_reply else reply
      promises::promise_resolve(list(text = text, finish = "stop", prompt_tokens = 12L,
                                     completion_tokens = 5L, cached_tokens = 4L))
    })
  gen$calls <- function() calls
  gen
}

fake_res <- function() {
  res <- new.env()
  res$status <- 200L
  res$headers <- list()
  res$setHeader <- function(name, value) res$headers[[name]] <- value
  res
}

test_that("edge_api checks its arguments before loading anything", {
  skip_if_not_installed("plumber")
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  path <- fake_gguf()
  on.exit(unlink(path))
  expect_error(edge_api(NA_character_), "model_path must be")
  expect_error(edge_api("no-such-model.gguf"), "Model file not found")
  expect_error(edge_api(path, n_slots = 0), "n_slots must be")
  expect_error(edge_api(path, n_slots = 2.5), "n_slots must be")
  expect_error(edge_api(path, n_ctx = 10), "n_ctx must be at least 128")
  expect_error(edge_api(path, n_slots = 32, n_ctx = 4096), "more than the 32768")
})

test_that("edge_engine checks its arguments", {
  expect_error(edge_engine(NULL), "Invalid model context")
  expect_error(edge_engine_poll(list(), 1L), "engine must come from edge_engine")
  expect_error(edge_engine_submit("x", "prompt"), "engine must come from edge_engine")
})

test_that("streamed text is cut into pieces that add up to the text", {
  for (text in c("one two  three\n", "  lead", "x", "a\nb\n\n", "émoji 😊 ok")) {
    pieces <- edgemodelr:::.oa_pieces(text)
    expect_identical(paste(pieces, collapse = ""), text)
  }
  expect_gt(length(edgemodelr:::.oa_pieces("one two three")), 2)
  expect_identical(edgemodelr:::.oa_pieces(""), character())
})

test_that("the container CPU limit is NA or a positive number", {
  q <- edgemodelr:::.cpu_quota()
  expect_length(q, 1L)
  expect_true(is.na(q) || q > 0)
  expect_true(edgemodelr:::.default_threads() >= 0L)
})

test_that("a chat request becomes a chat completion", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  gen <- fake_gen()
  out <- wait_for(edgemodelr:::.oa_chat(gen, list(
    messages = list(list(role = "user", content = "Hi")), max_tokens = 20), "m"))
  r <- out$response
  expect_identical(r$object, "chat.completion")
  expect_identical(r$choices[[1]]$message$content, "Hello there, friend.")
  expect_identical(r$choices[[1]]$finish_reason, "stop")
  expect_identical(r$usage$total_tokens, 17L)
  expect_identical(r$usage$prompt_tokens_details$cached_tokens, 4L)
  expect_null(gen$calls()[[1]]$grammar)
})

test_that("a streamed chat comes in several chunks", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  out <- wait_for(edgemodelr:::.oa_chat(fake_gen(), list(
    messages = list(list(role = "user", content = "Hi")), stream = TRUE,
    stream_options = list(include_usage = TRUE)), "m"))
  deltas <- vapply(out$chunks, function(ch) {
    if (length(ch$choices)) ch$choices[[1]]$delta$content %||% "" else ""
  }, character(1))
  expect_identical(paste(deltas, collapse = ""), "Hello there, friend.")
  expect_gt(length(out$chunks), 4)
  expect_identical(out$chunks[[length(out$chunks)]]$usage$completion_tokens, 5L)
})

test_that("tools give a tool call, chosen greedily under a grammar", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  gen <- fake_gen(tool_reply = '{"tool": "get_weather", "arguments": {"city": "Rome"}}')
  tools <- list(list(type = "function", `function` = list(
    name = "get_weather", description = "Weather for a city",
    parameters = list(type = "object", properties = list(city = list(type = "string"))))))
  out <- wait_for(edgemodelr:::.oa_chat(gen, list(
    messages = list(list(role = "user", content = "Weather in Rome?")), tools = tools,
    temperature = 0.9), "m"))
  choice <- out$response$choices[[1]]
  expect_identical(choice$finish_reason, "tool_calls")
  expect_null(choice$message$content)
  call <- choice$message$tool_calls[[1]]
  expect_identical(call$`function`$name, "get_weather")
  expect_identical(jsonlite::fromJSON(call$`function`$arguments), list(city = "Rome"))
  expect_false(is.null(gen$calls()[[1]]$grammar))
  expect_identical(gen$calls()[[1]]$sampling$temperature, 0)
})

test_that("a tool reply of none falls back to a text answer", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  gen <- fake_gen(tool_reply = '{"tool": "none"}')
  tools <- list(list(type = "function", `function` = list(
    name = "get_weather", description = "Weather for a city",
    parameters = list(type = "object", properties = list(city = list(type = "string"))))))
  out <- wait_for(edgemodelr:::.oa_chat(gen, list(
    messages = list(list(role = "user", content = "Hello")), tools = tools), "m"))
  expect_identical(out$response$choices[[1]]$message$content, "Hello there, friend.")
  expect_length(gen$calls(), 2L)
  expect_null(gen$calls()[[2]]$grammar)
})

test_that("response_format json_schema sends a grammar", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  gen <- fake_gen(reply = '{"city": "Rome"}')
  rf <- list(type = "json_schema", json_schema = list(schema = list(
    type = "object", properties = list(city = list(type = "string")))))
  out <- wait_for(edgemodelr:::.oa_chat(gen, list(
    messages = list(list(role = "user", content = "City?")), response_format = rf), "m"))
  expect_identical(out$response$choices[[1]]$message$content, '{"city": "Rome"}')
  expect_match(gen$calls()[[1]]$grammar, "root")
})

test_that("completions use the prompt as given", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  gen <- fake_gen(reply = " Paris.")
  out <- wait_for(edgemodelr:::.oa_completion(gen, list(prompt = "The capital of France is"), "m"))
  expect_identical(out$response$choices[[1]]$text, " Paris.")
  expect_identical(gen$calls()[[1]]$prompt, "The capital of France is")
})

test_that("errors become 400, 503 with Retry-After, or 500", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  res <- fake_res()
  body <- edgemodelr:::.oa_handle(res, function() edgemodelr:::.bad_request("nope"))
  expect_identical(res$status, 400L)
  expect_identical(body$error$type, "invalid_request_error")

  res <- fake_res()
  busy <- structure(class = c("edge_busy", "error", "condition"),
                    list(message = "busy", call = NULL))
  body <- wait_for(edgemodelr:::.oa_handle(res, function() promises::promise_reject(busy)))
  expect_identical(res$status, 503L)
  expect_identical(res$headers[["Retry-After"]], "5")

  res <- fake_res()
  body <- wait_for(edgemodelr:::.oa_handle(res, function() promises::promise_reject(simpleError("boom"))))
  expect_identical(res$status, 500L)
  expect_identical(body$error$message, "boom")

  res <- fake_res()
  body <- wait_for(edgemodelr:::.oa_handle(res, function() list(response = list(ok = TRUE))))
  expect_true(body$ok)
})
