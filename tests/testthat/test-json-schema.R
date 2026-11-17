# JSON Schema to GBNF: grammar text, the JSON reader, tools with optional
# and nested arguments, and the OpenAI layer on a fake generator. The last
# tests parse each grammar in llama.cpp with a real model when one is on
# disk (EDGEMODELR_TEST_MODEL, or TinyLlama in the package cache).

parse_json <- function(x) jsonlite::fromJSON(x, simplifyVector = FALSE)

# A schema like the one ellmer's chat_structured() sends for
# type_object(items = type_array(type_object(...))).
order_schema <- function() {
  parse_json('{"type": "object", "description": "", "properties": {
    "items": {"type": "array", "description": "", "items": {
      "type": "object", "description": "", "properties": {
        "name": {"type": "string", "description": "Item name"},
        "qty": {"type": "integer", "description": ""},
        "size": {"type": "string", "enum": ["small", "large"]}},
      "required": ["name", "size"], "additionalProperties": false}},
    "rush": {"type": ["boolean", "null"]}},
    "required": ["items", "rush"], "additionalProperties": false}')
}

test_that("flat schemas give the same grammar as before", {
  g <- edge_json_grammar(list(a = c("x", "y"), b = "string"))
  expect_identical(g, paste0(
    'root ::= "{" ws "\\"a\\"" ws ":" ws field-1 ws "," ws "\\"b\\"" ws ":" ws field-2 ws "}"\n',
    'field-1 ::= "\\"x\\"" | "\\"y\\""\n',
    "field-2 ::= string\n",
    'ws ::= "" | " " | "\\n" [ \\t]{0,20}\n',
    'string ::= "\\"" [^"\\\\]* "\\""\n'))
  # The same fields as a JSON Schema give the same grammar.
  js <- list(type = "object", properties = list(a = list(type = "string", enum = list("x", "y")),
                                                b = list(type = "string")),
             required = list("a", "b"))
  expect_identical(edge_json_grammar(js), g)
  expect_identical(edge_json_grammar(jsonlite::toJSON(js, auto_unbox = TRUE)), g)
})

test_that("nested objects and arrays get their own rules", {
  g <- edge_json_grammar(order_schema(), compact = TRUE)
  expect_match(g, 'root ::= "{" ws "\\"items\\": " field-1 ", " "\\"rush\\": " field-2 ws "}"',
               fixed = TRUE)
  expect_match(g, 'field-1 ::= "[" ws ( field-1-item ( ", " field-1-item )* )? ws "]"',
               fixed = TRUE)
  # qty is optional: it may be left out, and the required size still follows.
  expect_match(g, 'field-1-item-rest-2 ::= ( ", " "\\"qty\\": " field-1-item-field-2 )? field-1-item-rest-3',
               fixed = TRUE)
  expect_match(g, 'field-1-item-rest-3 ::= ", " "\\"size\\": " field-1-item-field-3', fixed = TRUE)
  expect_match(g, "field-2 ::= boolean | null", fixed = TRUE)
  expect_match(g, 'null ::= "null"', fixed = TRUE)
  # Rule names only use letters, digits and dashes.
  names <- sub(" ::=.*", "", strsplit(g, "\n")[[1]])
  expect_true(all(grepl("^[A-Za-z0-9-]+$", names)))
  expect_false(anyDuplicated(names) > 0)
})

test_that("optional fields may all be left out, in order", {
  s <- list(type = "object", properties = list(a = list(type = "string"),
                                               b = list(type = "integer")),
            required = list())
  g <- edge_json_grammar(s, compact = TRUE)
  expect_match(g, 'root ::= "{" ws ( "\\"a\\": " field-1 rest-2 | "\\"b\\": " field-2 )? ws "}"',
               fixed = TRUE)
  expect_match(g, 'rest-2 ::= ( ", " "\\"b\\": " field-2 )?', fixed = TRUE)
})

test_that("enum, const, anyOf, refs, bounds and formats", {
  s <- parse_json('{"type": "object", "properties": {
    "n": {"enum": [1, 2.5, true, null, "a\\"b"]},
    "k": {"const": "fixed"},
    "v": {"anyOf": [{"type": "string"}, {"type": "number"}]},
    "p": {"$ref": "#/$defs/Point"},
    "q": {"$ref": "#/$defs/Point"},
    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
    "day": {"type": "string", "format": "date"},
    "at": {"type": "string", "format": "date-time"},
    "code": {"type": "string", "minLength": 2, "maxLength": 4},
    "any": {},
    "map": {"type": "object", "additionalProperties": {"type": "integer"}}},
    "$defs": {"Point": {"type": "object", "properties": {"x": {"type": "number"}}}}}')
  g <- edge_json_grammar(s)
  expect_match(g, 'field-1 ::= "1" | "2.5" | "true" | "null" | "\\"a\\\\\\"b\\""', fixed = TRUE)
  expect_match(g, 'field-2 ::= "\\"fixed\\""', fixed = TRUE)
  expect_match(g, "field-3 ::= string | number", fixed = TRUE)
  # One rule per definition, used by both fields.
  expect_match(g, "field-4 ::= ref-Point\n", fixed = TRUE)
  expect_match(g, "field-5 ::= ref-Point\n", fixed = TRUE)
  expect_identical(lengths(regmatches(g, gregexpr("\nref-Point ::=", g))), 1L)
  expect_match(g, 'field-6 ::= "[" ws string ( ws "," ws string ){0,2} ws "]"', fixed = TRUE)
  expect_match(g, 'field-7 ::= "\\"" date "\\""', fixed = TRUE)
  expect_match(g, 'field-8 ::= "\\"" date "T" time "\\""', fixed = TRUE)
  expect_match(g, 'field-9 ::= "\\"" [^"\\\\]{2,4} "\\""', fixed = TRUE)
  expect_match(g, "field-10 ::= json-value", fixed = TRUE)
  expect_match(g, 'field-11 ::= "{" ws ( string ws ":" ws integer', fixed = TRUE)
  expect_match(g, "\njson-value ::= ", fixed = TRUE)
})

test_that("keys are escaped in the grammar", {
  g <- edge_json_grammar(list(type = "object", properties = setNames(
    list(list(type = "string")), 'say "hi"\\')), compact = TRUE)
  expect_match(g, '"\\"say \\\\\\"hi\\\\\\"\\\\\\\\\\": " field-1', fixed = TRUE)
})

test_that("json_object and bare object schemas allow any object", {
  g <- edgemodelr:::.oa_response_grammar(list(type = "json_object"))
  expect_match(g, '^root ::= "\\{" ws \\( string ws ":" ws json-value')
  expect_match(edge_json_grammar(list(type = "object", properties = parse_json("{}"))),
               'root ::= "{}"', fixed = TRUE)
  expect_match(edge_json_grammar(list(type = "array", maxItems = 0)), 'root ::= "[" ws "]"',
               fixed = TRUE)
})

test_that("unsupported schemas are an error, not a wrong grammar", {
  obj <- function(p) list(type = "object", properties = list(a = p))
  expect_error(edge_json_grammar(obj(list(type = "string", pattern = "^a"))), "pattern")
  expect_error(edge_json_grammar(obj(list(type = "integer", minimum = 0))), "minimum")
  expect_error(edge_json_grammar(obj(list(type = "date"))), "unsupported type")
  expect_error(edge_json_grammar(obj(list(type = "array", maxItems = 1000))), "100")
  expect_error(edge_json_grammar(obj(list(type = "array", items = list(list(type = "string"))))),
               "tuples")
  expect_error(edge_json_grammar(obj(list(`$ref` = "#/$defs/Missing"))), "not defined")
  expect_error(edge_json_grammar(obj(list(`$ref` = "https://example.com/s.json"))), "local")
  expect_error(edge_json_grammar(obj(list(allOf = list(list(type = "string"), list(type = "string"))))),
               "allOf")
  rec <- list(type = "object", properties = list(node = list(`$ref` = "#/$defs/Node")),
              `$defs` = list(Node = list(type = "object", properties = list(
                child = list(anyOf = list(list(`$ref` = "#/$defs/Node"), list(type = "null")))))))
  expect_error(edge_json_grammar(rec), "recursive")
  expect_error(edge_json_grammar(list(type = "object", properties = list(a = list(type = "string")),
                                      required = list("b"))), "'b' is not in properties")
  expect_error(edge_json_grammar(list(a = "date")), "Unknown type")
  expect_error(edge_json_grammar(list()), "named list")
})

test_that("the JSON reader reads what the grammars allow", {
  p <- edgemodelr:::.json_parse
  expect_identical(p('{"a": 1, "b": [1.5, "x\\"y\\n\\u00e9"], "c": null, "d": {}, "e": [], "f": true}'),
                   list(a = 1L, b = list(1.5, "x\"y\n\u00e9"), c = NULL,
                        d = structure(list(), names = character()), e = list(), f = TRUE))
  expect_identical(p("-12"), -12L)
  expect_identical(p("1e3"), 1000)
  expect_identical(p('"\\ud83d\\ude00"'), "\U0001F600")
  expect_error(p('{"a": 1'))
  expect_error(p('{"a": "x'), "unterminated")
  expect_error(p('{"a": 1} x'), "after")
  expect_error(p("[1,]"))
})

test_that("tools take JSON Schema parameters with optional and nested arguments", {
  params <- order_schema()
  t <- edge_tool("order", "Place an order", params)
  expect_identical(t$required, c("items", "rush"))
  expect_true(is.list(t$parameters$items))
  expect_output(print(t), "items: array of object {name: string, qty?: integer, size: one of \"small\", \"large\"}",
                fixed = TRUE)
  w <- edge_tool("get_weather", "Weather", list(city = "string", unit = c("c", "f")),
                 required = "city")
  expect_output(print(w), 'unit: one of "c", "f" (optional)', fixed = TRUE)
  expect_error(edge_tool("f", "x", list(a = "string"), required = "b"), "required")
  expect_error(edge_tool("f", "x", list(a = list(type = "string", pattern = "x"))),
               "tool 'f'.a: the keyword 'pattern'", fixed = TRUE)

  tools <- edgemodelr:::.check_tools(list(t, w))
  g <- edgemodelr:::.tool_grammar(tools, NULL)
  expect_match(g, 't2-call ::= "\\"get_weather\\", \\"arguments\\": " "{" ws ( "\\"city\\": " t2-field-1 t2-rest-2 ) ws "}"',
               fixed = TRUE)
  expect_match(g, 't2-rest-2 ::= ( ", " "\\"unit\\": " t2-field-2 )?', fixed = TRUE)
  # Strings in calls leave out line breaks.
  expect_match(g, 'string ::= "\\"" [^"\\\\\\n]* "\\""', fixed = TRUE)

  parse <- edgemodelr:::.parse_tool_reply
  r <- parse('{"tool": "get_weather", "arguments": {"city": "Paris"}}', tools)
  expect_identical(r$tool_calls[[1]]$arguments, list(city = "Paris"))
  r <- parse(paste0('{"tool": "order", "arguments": {"items": [{"name": "tea", "qty": 2, ',
                    '"size": "large"}, {"name": "cake", "size": "small"}], "rush": null}}'), tools)
  args <- r$tool_calls[[1]]$arguments
  expect_identical(args$items[[1]], list(name = "tea", qty = 2L, size = "large"))
  expect_identical(args$items[[2]], list(name = "cake", size = "small"))
  expect_true("rush" %in% names(args) && is.null(args$rush))
  expect_warning(r <- parse('{"tool": "order", "arguments": {"items": [{"name": "te', tools),
                 "cut off")
  expect_length(r$tool_calls, 0)
  # A number field keeps doubles even when the model writes a whole number.
  n <- edgemodelr:::.check_tools(list(edge_tool("f", "x", list(x = "number"))))
  expect_identical(parse('{"tool": "f", "arguments": {"x": 2}}', n)$tool_calls[[1]]$arguments,
                   list(x = 2))
})

test_that("OpenAI tools keep optional and nested arguments", {
  tools <- parse_json('[{"type": "function", "function": {"name": "get_weather",
    "description": "Weather", "parameters": {"type": "object", "properties": {
      "city": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "description": "Unit"},
      "days": {"type": "array", "items": {"type": "integer"}}},
      "required": ["city"], "additionalProperties": false}}},
    {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "noop"}}]')
  out <- edgemodelr:::.oa_tools(tools)
  expect_identical(out[[1]]$required, "city")
  expect_identical(out[[1]]$notes, c("City name", "Unit", ""))
  expect_identical(out[[2]]$parameters, list())
  expect_identical(out[[3]]$parameters, list())
  g <- edgemodelr:::.tool_grammar(edgemodelr:::.check_tools(out), NULL)
  expect_match(g, 't2-call ::= "\\"ping\\", \\"arguments\\": " "{}"', fixed = TRUE)
  bad <- parse_json('[{"type": "function", "function": {"name": "f", "parameters": {
    "type": "object", "properties": {"x": {"type": "string", "pattern": "a+"}}}}}]')
  expect_error(edgemodelr:::.oa_tools(bad), "pattern", class = "edge_bad_request")
})

# Runs later's loop until the promise settles, then returns its value.
settle <- function(p) {
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

# A generator that gives the replies in turn and records each request.
scripted_gen <- function(...) {
  replies <- c(...)
  calls <- list()
  gen <- list(
    prompt = function(messages) paste(vapply(messages, function(m) m$content, ""), collapse = "\n"),
    generate = function(prompt, n_predict, sampling, grammar = NULL) {
      calls[[length(calls) + 1L]] <<- list(prompt = prompt, grammar = grammar)
      text <- replies[min(length(calls), length(replies))]
      promises::promise_resolve(list(text = text, finish = "stop", prompt_tokens = 12L,
                                     completion_tokens = 5L, cached_tokens = 0L))
    })
  gen$calls <- function() calls
  gen
}

test_that("a nested response_format sends its grammar", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  reply <- '{"items": [{"name": "tea", "size": "large"}], "rush": false}'
  gen <- scripted_gen(reply)
  rf <- list(type = "json_schema", json_schema = list(name = "order", strict = TRUE,
                                                      schema = order_schema()))
  out <- settle(edgemodelr:::.oa_chat(gen, list(
    messages = list(list(role = "user", content = "Tea, large")), response_format = rf), "m"))
  expect_identical(out$response$choices[[1]]$message$content, reply)
  expect_identical(gen$calls()[[1]]$grammar, edge_json_grammar(order_schema()))

  gen <- scripted_gen('{"a": 1}')
  settle(edgemodelr:::.oa_chat(gen, list(messages = list(list(role = "user", content = "x")),
                                         response_format = list(type = "json_object")), "m"))
  expect_match(gen$calls()[[1]]$grammar, "json-value")

  bad <- list(type = "json_schema", json_schema = list(schema = list(
    type = "object", properties = list(a = list(type = "string", pattern = "x")))))
  expect_error(edgemodelr:::.oa_chat(gen, list(messages = list(list(role = "user", content = "x")),
                                               response_format = bad), "m"),
               "pattern", class = "edge_bad_request")
})

test_that("a call may leave out optional arguments and nest values", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  tools <- parse_json('[{"type": "function", "function": {"name": "order",
    "description": "Order", "parameters": {"type": "object", "properties": {
      "items": {"type": "array", "items": {"type": "string"}},
      "note": {"type": "string"}}, "required": ["items"]}}}]')
  gen <- scripted_gen('{"tool": "order", "arguments": {"items": ["tea", "cake"]}}')
  out <- settle(edgemodelr:::.oa_chat(gen, list(
    messages = list(list(role = "user", content = "Tea and cake")), tools = tools), "m"))
  call <- out$response$choices[[1]]$message$tool_calls[[1]]
  expect_identical(call$`function`$arguments, '{"items":["tea","cake"]}')
  expect_match(gen$calls()[[1]]$prompt, "note: string (optional)", fixed = TRUE)
})

test_that("tools and response_format work together", {
  skip_if_not_installed("promises")
  skip_if_not_installed("later")
  tools <- list(list(type = "function", `function` = list(
    name = "get_weather", description = "Weather for a city",
    parameters = list(type = "object", properties = list(city = list(type = "string"))))))
  rf <- list(type = "json_schema", json_schema = list(schema = order_schema()))
  reply <- '{"items": [], "rush": null}'
  gen <- scripted_gen('{"tool": "none"}', reply)
  out <- settle(edgemodelr:::.oa_chat(gen, list(
    messages = list(list(role = "user", content = "Order nothing")), tools = tools,
    response_format = rf), "m"))
  expect_identical(out$response$choices[[1]]$message$content, reply)
  expect_length(gen$calls(), 2L)
  expect_match(gen$calls()[[1]]$grammar, "t1-call", fixed = TRUE)
  expect_identical(gen$calls()[[2]]$grammar, edge_json_grammar(order_schema()))
})

test_model_path <- function() {
  path <- Sys.getenv("EDGEMODELR_TEST_MODEL")
  if (!nzchar(path)) {
    path <- file.path(tools::R_user_dir("edgemodelr", "cache"),
                      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
  }
  path
}

test_that("llama.cpp accepts the grammars and their output parses", {
  skip_on_cran()
  skip_if_not_installed("jsonlite")
  path <- test_model_path()
  skip_if_not(file.exists(path), "no test model on disk")
  ctx <- edge_load_model(path, n_ctx = 1024L, n_threads = 2L)
  on.exit(edge_free_model(ctx), add = TRUE)

  schemas <- list(
    order_schema(),
    list(type = "object", properties = list(a = list(type = "string"), b = list(type = "integer")),
         required = list()),
    parse_json('{"type": "object", "properties": {"p": {"$ref": "#/definitions/P"},
      "when": {"type": "string", "format": "date"},
      "v": {"anyOf": [{"type": "number"}, {"type": "null"}]},
      "tags": {"type": "array", "items": {"enum": ["x", 1, true]}, "minItems": 1, "maxItems": 2}},
      "definitions": {"P": {"type": "object", "properties": {"x": {"const": 3}}}}}'),
    list(type = "object"),
    list(type = "array", items = list(type = "string"), maxItems = 3)
  )
  prompt <- "Write a JSON value.\n"
  for (s in schemas) {
    for (compact in c(FALSE, TRUE)) {
      g <- edge_json_grammar(s, compact = compact)
      out <- edge_grammar_completion(ctx, prompt, g, n_predict = 96L, temperature = 0)
      expect_true(is.character(out) && nzchar(out))
      # Cut off at n_predict is allowed; a finished value must be valid JSON.
      done <- tryCatch({
        jsonlite::fromJSON(out, simplifyVector = FALSE)
        TRUE
      }, error = function(e) FALSE)
      if (!done) expect_gte(nchar(out), 40L)
    }
  }

  tools <- edgemodelr:::.check_tools(list(
    edge_tool("order", "Order items", order_schema()),
    edge_tool("get_weather", "Weather for a city",
              list(city = "string", unit = c("celsius", "fahrenheit")), required = "city")))
  res <- edge_chat_completion(ctx, list(list(role = "user", content = "Weather in Paris?")),
                              tools = tools, tool_choice = "get_weather", n_predict = 64L)
  expect_length(res$tool_calls, 1L)
  expect_identical(res$tool_calls[[1]]$name, "get_weather")
  expect_true(is.character(res$tool_calls[[1]]$arguments$city))
})
