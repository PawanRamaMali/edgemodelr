## ============================================================================
## edgemodelr — Pharma Text-to-SQL + Data-to-Text + Verification Demo
## ----------------------------------------------------------------------------
## Builds a small SDTM-style SQLite database (ADSL, ADAE, ADLB) in memory,
## then exercises:
##   1. edge_text_to_sql()    -- NL question -> SQL over CDISC-style tables
##   2. edge_narrate()        -- Structured lab record -> English narrative
##   3. edge_verify_narrative -- Numeric back-check against source values
##
## Choose your model below. Qwen3-1.7B is a good sweet spot for AE/lab work
## on a laptop CPU; for SQL accuracy on real CDISC schemas prefer Qwen3-4B
## or a dedicated text-to-SQL model (e.g. SQLCoder-7B-2 Q4_K_M).
## ============================================================================

options(warn = 1, edgemodelr.verbose = FALSE)
suppressPackageStartupMessages({
  library(edgemodelr)
  library(DBI)
  library(RSQLite)
})

cat("\n===== edgemodelr SQL + Narrative Pharma Demo =====\n")
cat("edgemodelr", as.character(packageVersion("edgemodelr")), "\n\n")

## ---------------------------------------------------------------------------
## 1. Build a small SDTM-style database
## ---------------------------------------------------------------------------
con <- dbConnect(SQLite(), ":memory:")

dbExecute(con, "
  CREATE TABLE ADSL (
    USUBJID TEXT PRIMARY KEY,
    AGE     INTEGER,
    SEX     TEXT,
    TRT01A  TEXT,
    SAFFL   TEXT
  );")
dbExecute(con, "
  CREATE TABLE ADAE (
    USUBJID TEXT,
    AETERM  TEXT,
    AESER   TEXT,
    AESEV   TEXT,
    AEREL   TEXT,
    ASTDY   INTEGER
  );")
dbExecute(con, "
  CREATE TABLE ADLB (
    USUBJID  TEXT,
    LBTESTCD TEXT,
    LBORRES  REAL,
    LBORNRHI REAL,
    LBDY     INTEGER,
    VISIT    TEXT,
    LBNRIND  TEXT
  );")

adsl <- data.frame(
  USUBJID = c("ABC-001-001", "ABC-001-002", "ABC-001-003",
              "ABC-001-004", "ABC-001-005", "ABC-001-006"),
  AGE     = c(67L, 45L, 52L, 71L, 38L, 59L),
  SEX     = c("M", "F", "F", "M", "F", "M"),
  TRT01A  = c("Drug X 100mg", "Drug Y 75mg/m2", "Drug Z 200mg",
              "Placebo",      "Drug X 100mg",  "Drug Y 75mg/m2"),
  SAFFL   = c("Y", "Y", "Y", "Y", "Y", "Y"),
  stringsAsFactors = FALSE
)
dbWriteTable(con, "ADSL", adsl, append = TRUE)

adae <- data.frame(
  USUBJID = c("ABC-001-001", "ABC-001-002", "ABC-001-003",
              "ABC-001-003", "ABC-001-005"),
  AETERM  = c("Headache", "Neutropenia", "Nausea",
              "Vomiting", "Fatigue"),
  AESER   = c("N", "N", "Y", "Y", "N"),
  AESEV   = c("SEVERE", "GRADE 3", "MODERATE", "MODERATE", "MILD"),
  AEREL   = c("PROBABLE", "DEFINITE", "POSSIBLE", "POSSIBLE", "UNLIKELY"),
  ASTDY   = c(1L, 14L, 3L, 3L, 21L),
  stringsAsFactors = FALSE
)
dbWriteTable(con, "ADAE", adae, append = TRUE)

adlb <- data.frame(
  USUBJID  = c("ABC-001-001", "ABC-001-002", "ABC-001-003",
               "ABC-001-005", "ABC-001-001"),
  LBTESTCD = c("ALT", "ANC", "AST", "ALT", "AST"),
  LBORRES  = c(285,    0.8,   142,    47,   190),
  LBORNRHI = c(55,     7.0,   40,     55,   40),
  LBDY     = c(28L, 14L, 5L, 14L, 28L),
  VISIT    = c("Week 4", "Week 2", "Day 5", "Week 2", "Week 4"),
  LBNRIND  = c("HIGH", "LOW", "HIGH", "NORMAL", "HIGH"),
  stringsAsFactors = FALSE
)
dbWriteTable(con, "ADLB", adlb, append = TRUE)

cat("SDTM tables loaded: ADSL=", nrow(adsl),
    " rows, ADAE=", nrow(adae),
    " rows, ADLB=", nrow(adlb), " rows.\n\n", sep = "")

## ---------------------------------------------------------------------------
## 2. Load the model
## ---------------------------------------------------------------------------
## Pre-cached path (overrides edge_quick_setup download). Set to NULL or remove
## to use edge_quick_setup() and download on first run.
cached_model <- "C:/Users/alpine/AppData/Local/edgemodelr-cache/Qwen3-1.7B-Q4_K_M.gguf"

model_name <- "Qwen3-1.7B"   # change to "Qwen3-4B" for better SQL fidelity
cat("Loading", model_name, "(this can take a minute on first run)...\n")

if (!is.null(cached_model) && file.exists(cached_model)) {
  ctx <- edge_load_model(cached_model, n_ctx = 4096L)
} else {
  setup <- edge_quick_setup(model_name)
  ctx <- setup$context
}
stopifnot(is_valid_model(ctx))
cat("Model loaded.\n\n")

## ---------------------------------------------------------------------------
## 3. Text-to-SQL
## ---------------------------------------------------------------------------
schema <- c(
  "CREATE TABLE ADSL (USUBJID TEXT, AGE INTEGER, SEX TEXT, TRT01A TEXT, SAFFL TEXT);",
  "CREATE TABLE ADAE (USUBJID TEXT, AETERM TEXT, AESER TEXT, AESEV TEXT, AEREL TEXT, ASTDY INTEGER);",
  "CREATE TABLE ADLB (USUBJID TEXT, LBTESTCD TEXT, LBORRES REAL, LBORNRHI REAL, LBDY INTEGER, VISIT TEXT, LBNRIND TEXT);"
)

questions <- c(
  "How many subjects in the safety analysis population are on Drug X 100mg?",
  "List the USUBJID and AETERM of all serious adverse events related to study drug.",
  "What is the highest ALT value recorded, and which subject does it belong to?"
)

cat("--- Text-to-SQL ---\n")
for (q in questions) {
  cat("\nQ: ", q, "\n", sep = "")
  out <- edge_text_to_sql(ctx, q, schema = schema, con = con,
                          dialect = "sqlite", n_predict = 200,
                          temperature = 0.1)
  cat("SQL: ", out$sql, "\n", sep = "")
  if (!is.na(out$error)) {
    cat("EXEC ERROR: ", out$error, "\n", sep = "")
  } else {
    cat("Result:\n")
    print(out$result)
  }
}

## ---------------------------------------------------------------------------
## 4. Data-to-text (lab abnormality narratives)
## ---------------------------------------------------------------------------
cat("\n\n--- Data-to-Text: Lab Abnormality Narratives ---\n")

abnormal <- dbGetQuery(con,
  "SELECT USUBJID, LBTESTCD, LBORRES, LBORNRHI, LBDY, VISIT, LBNRIND
     FROM ADLB
    WHERE LBNRIND <> 'NORMAL'")

narratives <- edge_narrate(ctx, abnormal,
                            max_words = 50,
                            n_predict = 150,
                            temperature = 0.2,
                            progress = TRUE)

for (i in seq_len(nrow(abnormal))) {
  cat("\nRecord ", i, ": ", abnormal$USUBJID[i],
      " / ", abnormal$LBTESTCD[i], "\n", sep = "")
  cat("Narrative: ", narratives[i], "\n", sep = "")
}

## ---------------------------------------------------------------------------
## 5. Numeric back-verification of the generated narratives
## ---------------------------------------------------------------------------
cat("\n\n--- Numeric Back-Verification ---\n")

verdicts <- vector("list", nrow(abnormal))
for (i in seq_len(nrow(abnormal))) {
  expected <- list(
    USUBJID  = abnormal$USUBJID[i],
    LBTESTCD = abnormal$LBTESTCD[i],
    LBORRES  = as.numeric(abnormal$LBORRES[i]),
    LBDY     = as.integer(abnormal$LBDY[i])
  )
  v <- edge_verify_narrative(ctx, narratives[i], expected = expected,
                              tolerance = 0.5)
  verdicts[[i]] <- v
  cat("\nRecord ", i, ": ", abnormal$USUBJID[i], " — ",
      if (v$ok) "PASS" else "FAIL", "\n", sep = "")
  if (!v$ok) {
    print(v$mismatches)
  }
}

n_pass <- sum(vapply(verdicts, function(v) isTRUE(v$ok), logical(1)))
cat(sprintf("\nVerification summary: %d / %d narratives passed numeric back-check.\n",
            n_pass, length(verdicts)))

## ---------------------------------------------------------------------------
## 6. Cleanup
## ---------------------------------------------------------------------------
edge_free_model(ctx)
dbDisconnect(con)
cat("\n===== Done =====\n")
