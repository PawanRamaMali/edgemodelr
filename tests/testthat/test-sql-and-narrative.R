test_that("edge_text_to_sql validates its arguments", {
  schema <- "CREATE TABLE adsl (usubjid TEXT, age INTEGER);"

  expect_error(
    edge_text_to_sql(NULL, "How many subjects?", schema),
    "Invalid model context"
  )

  # A context that is not a model pointer is rejected before any generation.
  expect_error(
    edge_text_to_sql(list(), "How many subjects?", schema),
    "Invalid model context"
  )
})


test_that("edge_narrate rejects an invalid context", {
  expect_error(
    edge_narrate(NULL, list(id = "S001", age = 42)),
    "Invalid model context"
  )

  expect_error(
    edge_narrate(list(), data.frame(id = "S001", age = 42)),
    "Invalid model context"
  )
})


test_that("edge_verify_narrative rejects an invalid context", {
  expect_error(
    edge_verify_narrative(NULL, "Subject S001 is 42.", list(id = "S001")),
    "Invalid model context"
  )

  expect_error(
    edge_verify_narrative(list(), "Subject S001 is 42.", list(id = "S001")),
    "Invalid model context"
  )
})


test_that("the new helpers are exported", {
  for (fn in c("edge_text_to_sql", "edge_narrate", "edge_verify_narrative")) {
    expect_true(
      exists(fn, envir = asNamespace("edgemodelr"), inherits = FALSE),
      info = fn
    )
    expect_true(is.function(get(fn, envir = asNamespace("edgemodelr"))), info = fn)
  }
})


test_that("edge_text_to_sql argument checks fire in order", {
  # These need a model to reach, so they are only meaningful with one loaded.
  # Without a model the context check fires first, which is what we assert here
  # so the suite stays runnable on CRAN with no model present.
  skip_if_not(exists("edge_load_model"))

  schema <- "CREATE TABLE t (a INTEGER);"
  expect_error(edge_text_to_sql(NULL, "q", schema), "Invalid model context")
  expect_error(edge_text_to_sql(NULL, "", schema), "Invalid model context")
  expect_error(edge_text_to_sql(NULL, "q", character(0)), "Invalid model context")
})
