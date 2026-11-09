# The OpenAI translation used by edge_serve(); no model needed.

parse_json <- function(x) jsonlite::fromJSON(x, simplifyVector = FALSE)

test_that("OpenAI messages become edge_chat_completion messages", {
  skip_if_not_installed("jsonlite")
  msgs <- parse_json('[
    {"role": "developer", "content": "Be brief."},
    {"role": "user", "content": [{"type": "text", "text": "Weather in "},
                                 {"type": "text", "text": "Paris?"}]},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
      "function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\", \\"days\\": 2}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "24 C"}
  ]')
  out <- edgemodelr:::.oa_messages(msgs)
  expect_identical(out[[1]], list(role = "system", content = "Be brief."))
  expect_identical(out[[2]]$content, "Weather in Paris?")
  expect_identical(out[[3]]$content, '{"tool": "get_weather", "arguments": {"city": "Paris", "days": 2}}')
  expect_identical(out[[4]], list(role = "tool", name = "get_weather", content = "24 C"))
  expect_error(edgemodelr:::.oa_messages(list()), class = "edge_bad_request")
})

test_that("a call sent back reads the same as one the model made", {
  skip_if_not_installed("jsonlite")
  tools <- edgemodelr:::.check_tools(list(edge_tool("get_weather", "Weather",
    list(city = "string", days = "integer", hot = "boolean"))))
  text <- edgemodelr:::.oa_call_text("get_weather", list(city = "Rome", days = 3L, hot = TRUE))
  reply <- edgemodelr:::.parse_tool_reply(text, tools)
  expect_identical(reply$tool_calls[[1]]$arguments, list(city = "Rome", days = 3L, hot = TRUE))
  expect_identical(edgemodelr:::.oa_call_text("ping", list()), '{"tool": "ping", "arguments": {}}')
})

test_that("OpenAI tools become edge tools", {
  skip_if_not_installed("jsonlite")
  tools <- parse_json('[{"type": "function", "function": {"name": "get_weather",
    "description": "Get the weather", "parameters": {"type": "object", "properties": {
      "city": {"type": "string", "description": "City name"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
      "days": {"type": ["integer", "null"]}}}}}]')
  out <- edgemodelr:::.oa_tools(tools)
  expect_s3_class(out[[1]], "edge_tool")
  expect_identical(out[[1]]$parameters,
                   list(city = "string", unit = c("celsius", "fahrenheit"), days = "integer"))
  expect_identical(out[[1]]$notes, c("City name", "", ""))
  expect_null(edgemodelr:::.oa_tools(NULL))

  nested <- parse_json('[{"type": "function", "function": {"name": "f", "parameters": {
    "type": "object", "properties": {"xs": {"type": "array"}}}}}]')
  expect_error(edgemodelr:::.oa_tools(nested), class = "edge_bad_request")
})

test_that("an enum with one value stays an enum", {
  skip_if_not_installed("jsonlite")
  tools <- parse_json('[{"type": "function", "function": {"name": "f", "parameters": {
    "type": "object", "properties": {"unit": {"type": "string", "enum": ["celsius"]},
    "kind": {"type": "string", "enum": ["string"]}}}}}]')
  out <- edgemodelr:::.oa_tools(tools)
  expect_identical(out[[1]]$parameters, list(unit = I("celsius"), kind = I("string")))
  expect_match(edgemodelr:::.describe_tool(out[[1]]), 'unit: one of "celsius", kind: one of "string"')
  grammar <- edge_json_grammar(list(unit = I("celsius"), kind = I("string")))
  expect_match(grammar, '"\\"celsius\\""', fixed = TRUE)
  expect_false(grepl("^string ::=", grammar))
  expect_false(grepl("\nstring ::=", grammar))
})

test_that("tool_choice is read in both OpenAI forms", {
  expect_identical(edgemodelr:::.oa_tool_choice(NULL), "auto")
  expect_identical(edgemodelr:::.oa_tool_choice("required"), "required")
  expect_identical(edgemodelr:::.oa_tool_choice(list(type = "function",
                                                     `function` = list(name = "f"))), "f")
})

test_that("response_format json_schema becomes a JSON grammar", {
  skip_if_not_installed("jsonlite")
  expect_null(edgemodelr:::.oa_response_grammar(NULL))
  expect_null(edgemodelr:::.oa_response_grammar(list(type = "text")))
  rf <- parse_json('{"type": "json_schema", "json_schema": {"name": "person", "schema": {
    "type": "object", "properties": {"name": {"type": "string", "description": "Full name"},
    "age": {"type": "integer"}}, "required": ["name", "age"], "additionalProperties": false}}}')
  expect_identical(edgemodelr:::.oa_response_grammar(rf),
                   edge_json_grammar(list(name = "string", age = "integer")))
  expect_error(edgemodelr:::.oa_response_grammar(list(type = "json_object")),
               class = "edge_bad_request")
})

test_that("max_tokens takes the newer name and is checked", {
  expect_identical(edgemodelr:::.oa_max_tokens(list(), 256L), 256L)
  expect_identical(edgemodelr:::.oa_max_tokens(list(max_tokens = 10), 256L), 10L)
  expect_identical(edgemodelr:::.oa_max_tokens(list(max_completion_tokens = 7, max_tokens = 10), 256L), 7L)
  expect_error(edgemodelr:::.oa_max_tokens(list(max_tokens = 0), 256L), class = "edge_bad_request")
})

test_that("streamed chunks are server-sent events ending with DONE", {
  skip_if_not_installed("jsonlite")
  sse <- edgemodelr:::.oa_sse(list(
    list(choices = list(list(delta = list(content = "Hi"), finish_reason = NA))),
    list(choices = list(list(delta = edgemodelr:::.oa_empty_object(), finish_reason = "stop")))))
  expect_identical(sse, paste0(
    'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
    "data: [DONE]\n\n"))
})

test_that("chat_edge needs exactly one of model_path and base_url", {
  skip_if_not_installed("ellmer")
  expect_error(chat_edge(), "either model_path or base_url")
  expect_error(chat_edge("a.gguf", base_url = "http://127.0.0.1:1"), "either model_path or base_url")
})
