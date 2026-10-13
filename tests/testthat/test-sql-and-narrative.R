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

test_that(".clean_sql_output strips fences in any case", {
  clean <- edgemodelr:::.clean_sql_output
  expect_equal(clean("```sql\nSELECT 1;\n```"), "SELECT 1;")
  expect_equal(clean("Here you go:\n```SQL\nSELECT 1;\n```"), "SELECT 1;")
  expect_equal(clean("SELECT 1; -- done. Explanation: trivial"), "SELECT 1;")
})


test_that(".clean_sql_output keeps semicolons inside quoted literals", {
  clean <- edgemodelr:::.clean_sql_output
  expect_equal(
    clean("SELECT * FROM notes WHERE text = 'started; resolved'; extra"),
    "SELECT * FROM notes WHERE text = 'started; resolved';"
  )
  expect_equal(
    clean("SELECT 'it''s; fine' AS x; chatter"),
    "SELECT 'it''s; fine' AS x;"
  )
  expect_equal(clean("SELECT \"a;b\" FROM t;"), "SELECT \"a;b\" FROM t;")
})


test_that("edge_verify_narrative literal check needs no model", {
  narrative <- paste(
    "Subject ABC-001-042 on Drug X had ALT of 285 U/L (ULN 55) on study",
    "day 28 after a total dose of 1,250 mg."
  )
  rec <- list(subject = "ABC-001-042", drug = "Drug X", alt = 285,
              uln = 55, study_day = 28L, dose = 1250)

  res <- edge_verify_narrative(NULL, narrative, rec, method = "literal")
  expect_true(res$ok)
  expect_null(res$extracted)
  expect_equal(res$literal$field, c("alt", "uln", "study_day", "dose"))
  expect_true(all(res$literal$found))
})


test_that("edge_verify_narrative flags numbers missing from the narrative", {
  # 42 only occurs inside the subject ID, which must not count as the value.
  res <- edge_verify_narrative(
    NULL, "Subject ABC-001-042 had ALT of 285 U/L.",
    list(alt = 285, age = 42), method = "literal"
  )
  expect_false(res$ok)
  expect_equal(res$literal$field[!res$literal$found], "age")
})


test_that("edge_verify_narrative accepts units written next to numbers", {
  res <- edge_verify_narrative(
    NULL, "Took 2.5mg daily; ALT 285U/L on day 28.",
    list(dose = 2.5, alt = 285, day = 28L), method = "literal"
  )
  expect_true(res$ok)

  # A longer number does not count as a shorter one.
  res <- edge_verify_narrative(
    NULL, "ALT was 2850 and dose 12.5 mg.", list(alt = 285, dose = 2.5),
    method = "literal"
  )
  expect_false(any(res$literal$found))
})


test_that("edge_verify_narrative flags unsupported and omitted terms", {
  rec <- list(drug = "Drug X", event = "Hepatotoxicity")
  terms <- c("Drug X", "Drug Y", "Hepatotoxicity", "Rash")

  hallucinated <- edge_verify_narrative(
    NULL, "The patient on Drug Y developed a rash.", rec,
    method = "literal", terms = terms
  )
  expect_false(hallucinated$ok)
  expect_setequal(hallucinated$unsupported_terms, c("Drug Y", "Rash"))
  expect_setequal(hallucinated$omitted_terms, c("Drug X", "Hepatotoxicity"))

  faithful <- edge_verify_narrative(
    NULL, "The patient on drug x developed hepatotoxicity.", rec,
    method = "literal", terms = terms
  )
  expect_true(faithful$ok)

  # Whole-word matching: "Drug XL" is not "Drug X".
  partial <- edge_verify_narrative(
    NULL, "The patient took Drug XL.", list(drug = "Drug XL"),
    method = "literal", terms = "Drug X"
  )
  expect_length(partial$unsupported_terms, 0L)
})


test_that("edge_verify_narrative validates terms and method", {
  expect_error(
    edge_verify_narrative(NULL, "x", list(a = 1), method = "literal",
                          terms = 1:3),
    "terms must be a character vector"
  )
  expect_error(
    edge_verify_narrative(NULL, "x", list(a = 1), method = "literal",
                          terms = c("Drug X", NA)),
    "without NA"
  )
  expect_error(
    edge_verify_narrative(NULL, "x", list(a = 1), method = "nope"),
    "should be one of"
  )
})
