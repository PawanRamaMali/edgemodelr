## ============================================================================
## edgemodelr — Pharma Benchmark
## ----------------------------------------------------------------------------
## Measures real accuracy of small models on two pharma tasks:
##   (A) Text-to-SQL over a small SDTM-style DB — execution accuracy vs. gold
##   (B) Data-to-text narrative generation — round-trip field accuracy
## Across multiple models. Writes a CSV summary and prints comparison tables.
## ============================================================================

options(warn = 1, edgemodelr.verbose = FALSE)
suppressPackageStartupMessages({
  library(edgemodelr)
  library(DBI)
  library(RSQLite)
})

cache_root <- "C:/Users/alpine/AppData/Local/edgemodelr-cache"

MODELS <- list(
  list(name = "Qwen3-1.7B",     path = file.path(cache_root, "Qwen3-1.7B-Q4_K_M.gguf")),
  list(name = "Qwen3-4B",       path = file.path(cache_root, "Qwen3-4B-Q4_K_M.gguf"))
)
## TinyLlama was excluded after observed 130min+ hangs on multi-row SELECT
## queries; the 1.1B chat model is not viable for text-to-SQL on this schema.

cat("\n===== edgemodelr Pharma Benchmark =====\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Package:", as.character(packageVersion("edgemodelr")), "\n\n")

## ---------------------------------------------------------------------------
## Build the SDTM-style DB (shared across all models)
## ---------------------------------------------------------------------------
build_db <- function() {
  con <- dbConnect(SQLite(), ":memory:")
  dbExecute(con, "CREATE TABLE ADSL (USUBJID TEXT, AGE INTEGER, SEX TEXT, TRT01A TEXT, SAFFL TEXT);")
  dbExecute(con, "CREATE TABLE ADAE (USUBJID TEXT, AETERM TEXT, AESER TEXT, AESEV TEXT, AEREL TEXT, ASTDY INTEGER);")
  dbExecute(con, "CREATE TABLE ADLB (USUBJID TEXT, LBTESTCD TEXT, LBORRES REAL, LBORNRHI REAL, LBDY INTEGER, VISIT TEXT, LBNRIND TEXT);")

  dbWriteTable(con, "ADSL", data.frame(
    USUBJID = c("ABC-001-001","ABC-001-002","ABC-001-003","ABC-001-004",
                "ABC-001-005","ABC-001-006","ABC-001-007","ABC-001-008"),
    AGE     = c(67L, 45L, 52L, 71L, 38L, 59L, 28L, 64L),
    SEX     = c("M","F","F","M","F","M","F","M"),
    TRT01A  = c("Drug X 100mg","Drug Y 75mg/m2","Drug Z 200mg","Placebo",
                "Drug X 100mg","Drug Y 75mg/m2","Drug Z 200mg","Placebo"),
    SAFFL   = c("Y","Y","Y","Y","Y","Y","N","Y"),
    stringsAsFactors = FALSE), append = TRUE)

  dbWriteTable(con, "ADAE", data.frame(
    USUBJID = c("ABC-001-001","ABC-001-002","ABC-001-003","ABC-001-003",
                "ABC-001-005","ABC-001-006","ABC-001-001","ABC-001-008"),
    AETERM  = c("Headache","Neutropenia","Nausea","Vomiting",
                "Fatigue","Diarrhea","Dizziness","Rash"),
    AESER   = c("N","N","Y","Y","N","N","N","N"),
    AESEV   = c("SEVERE","GRADE 3","MODERATE","MODERATE","MILD","MILD","MILD","MILD"),
    AEREL   = c("PROBABLE","DEFINITE","POSSIBLE","POSSIBLE","UNLIKELY","POSSIBLE","PROBABLE","UNRELATED"),
    ASTDY   = c(1L, 14L, 3L, 3L, 21L, 7L, 1L, 10L),
    stringsAsFactors = FALSE), append = TRUE)

  dbWriteTable(con, "ADLB", data.frame(
    USUBJID  = c("ABC-001-001","ABC-001-002","ABC-001-003","ABC-001-005",
                 "ABC-001-001","ABC-001-006","ABC-001-007"),
    LBTESTCD = c("ALT","ANC","AST","ALT","AST","ALT","ANC"),
    LBORRES  = c(285, 0.8, 142, 47, 190, 38, 4.5),
    LBORNRHI = c(55, 7.0, 40, 55, 40, 55, 7.0),
    LBDY     = c(28L, 14L, 5L, 14L, 28L, 7L, 14L),
    VISIT    = c("Week 4","Week 2","Day 5","Week 2","Week 4","Week 1","Week 2"),
    LBNRIND  = c("HIGH","LOW","HIGH","NORMAL","HIGH","NORMAL","NORMAL"),
    stringsAsFactors = FALSE), append = TRUE)

  con
}

SCHEMA <- c(
  "CREATE TABLE ADSL (USUBJID TEXT, AGE INTEGER, SEX TEXT, TRT01A TEXT, SAFFL TEXT);",
  "CREATE TABLE ADAE (USUBJID TEXT, AETERM TEXT, AESER TEXT, AESEV TEXT, AEREL TEXT, ASTDY INTEGER);",
  "CREATE TABLE ADLB (USUBJID TEXT, LBTESTCD TEXT, LBORRES REAL, LBORNRHI REAL, LBDY INTEGER, VISIT TEXT, LBNRIND TEXT);"
)

## ---------------------------------------------------------------------------
## SQL test cases — question + gold SQL
## Execution accuracy = compare normalized result sets.
## (Trimmed sanity set: one per difficulty band. Full set in git history.)
## ---------------------------------------------------------------------------
SQL_TESTS <- list(
  list(id = "Q01", difficulty = "easy",
       question = "How many subjects are in the ADSL table?",
       gold = "SELECT COUNT(*) AS n FROM ADSL;"),
  list(id = "Q05", difficulty = "medium",
       question = "How many subjects are in the safety analysis population (SAFFL = 'Y')?",
       gold = "SELECT COUNT(*) AS n FROM ADSL WHERE SAFFL = 'Y';"),
  list(id = "Q10", difficulty = "hard",
       question = "For each treatment arm, how many subjects had at least one adverse event?",
       gold = "SELECT s.TRT01A, COUNT(DISTINCT s.USUBJID) AS n FROM ADSL s JOIN ADAE a ON s.USUBJID = a.USUBJID GROUP BY s.TRT01A ORDER BY s.TRT01A;")
)

## ---------------------------------------------------------------------------
## Narrative test cases — record + key fields to back-check
## ---------------------------------------------------------------------------
NARR_TESTS <- list(
  list(id = "N01",
       record = list(USUBJID="ABC-001-001", LBTESTCD="ALT", LBORRES=285,
                     LBORNRHI=55, LBDY=28L, VISIT="Week 4", LBNRIND="HIGH"),
       check = list(USUBJID="ABC-001-001", LBTESTCD="ALT", LBORRES=285,
                    LBDY=28L)),
  list(id = "N03",
       record = list(USUBJID="ABC-001-003", AETERM="Nausea", AESER="Y",
                     AESEV="MODERATE", AEREL="POSSIBLE", ASTDY=3L),
       check = list(USUBJID="ABC-001-003", AETERM="Nausea", ASTDY=3L))
)

## ---------------------------------------------------------------------------
## Helpers — result comparison
## ---------------------------------------------------------------------------
normalize_result <- function(df) {
  if (is.null(df) || nrow(df) == 0) {
    return("EMPTY")
  }
  df <- as.data.frame(df, stringsAsFactors = FALSE)
  # Coerce all cols to character, trim, lowercase for stable compare
  for (j in seq_along(df)) {
    df[[j]] <- tolower(trimws(as.character(df[[j]])))
  }
  # Sort rows for set-equality
  ord <- do.call(order, df)
  df <- df[ord, , drop = FALSE]
  rownames(df) <- NULL
  # Drop column names: compare values only (column names may differ)
  paste(apply(df, 1, paste, collapse = "|"), collapse = ";")
}

results_match <- function(got, gold) {
  identical(normalize_result(got), normalize_result(gold))
}

## ---------------------------------------------------------------------------
## Per-model evaluation
## ---------------------------------------------------------------------------
eval_model <- function(model) {
  cat("\n========== Model:", model$name, "==========\n")
  if (!file.exists(model$path)) {
    cat("  SKIP — model file not found at:", model$path, "\n")
    return(NULL)
  }
  cat("  Loading...\n")
  t_load_start <- Sys.time()
  ctx <- tryCatch(
    edge_load_model(model$path, n_ctx = 2048L),
    error = function(e) { cat("  LOAD ERROR:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(ctx)) return(NULL)
  cat(sprintf("  Loaded in %.1fs\n",
              as.numeric(difftime(Sys.time(), t_load_start, units = "secs"))))

  con <- build_db()
  on.exit({ try(edge_free_model(ctx), silent = TRUE); try(dbDisconnect(con), silent = TRUE) })

  ## ----- SQL tests -----
  cat("\n  --- Text-to-SQL ---\n")
  sql_rows <- vector("list", length(SQL_TESTS))
  for (k in seq_along(SQL_TESTS)) {
    tc <- SQL_TESTS[[k]]
    gold_result <- tryCatch(dbGetQuery(con, tc$gold), error = function(e) NULL)
    t0 <- Sys.time()
    out <- tryCatch(
      edge_text_to_sql(ctx, tc$question, schema = SCHEMA, con = con,
                       dialect = "sqlite", n_predict = 80, temperature = 0.1),
      error = function(e) list(sql = NA_character_, result = NULL, error = conditionMessage(e))
    )
    dt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    exec_ok <- is.na(out$error)
    matched <- exec_ok && results_match(out$result, gold_result)
    cat(sprintf("    %s [%s] %s  %.1fs  %s\n",
                tc$id, tc$difficulty,
                if (matched) "PASS" else if (exec_ok) "WRONG" else "ERROR",
                dt,
                if (matched) "" else paste0("SQL=", substr(gsub("\\s+", " ", out$sql), 1, 80))))
    sql_rows[[k]] <- data.frame(
      model = model$name, test = tc$id, difficulty = tc$difficulty,
      executed = exec_ok, correct = matched, seconds = round(dt, 1),
      sql = ifelse(is.na(out$sql), "", out$sql),
      stringsAsFactors = FALSE
    )
  }
  sql_df <- do.call(rbind, sql_rows)

  ## ----- Narrative tests -----
  cat("\n  --- Data-to-Text (narrative + back-verify) ---\n")
  narr_rows <- vector("list", length(NARR_TESTS))
  for (k in seq_along(NARR_TESTS)) {
    tc <- NARR_TESTS[[k]]
    t0 <- Sys.time()
    narrative <- tryCatch(
      edge_narrate(ctx, tc$record, max_words = 50,
                   n_predict = 90, temperature = 0.2, progress = FALSE),
      error = function(e) NA_character_
    )
    t_narr <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    if (is.na(narrative)) {
      narr_rows[[k]] <- data.frame(
        model = model$name, test = tc$id,
        narrative_ok = FALSE, fields_correct = 0L,
        fields_total = length(tc$check), pct = 0.0,
        seconds = round(t_narr, 1),
        stringsAsFactors = FALSE)
      cat(sprintf("    %s  ERROR\n", tc$id))
      next
    }

    t0 <- Sys.time()
    v <- tryCatch(
      edge_verify_narrative(ctx, narrative, expected = tc$check, tolerance = 0.5),
      error = function(e) list(ok = FALSE,
                               mismatches = data.frame(field = names(tc$check)))
    )
    t_verify <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

    fields_correct <- length(tc$check) - nrow(v$mismatches)
    pct <- 100 * fields_correct / length(tc$check)
    cat(sprintf("    %s  %d/%d fields correct (%.0f%%)  gen=%.1fs verify=%.1fs\n",
                tc$id, fields_correct, length(tc$check), pct, t_narr, t_verify))
    narr_rows[[k]] <- data.frame(
      model = model$name, test = tc$id,
      narrative_ok = nzchar(narrative),
      fields_correct = fields_correct,
      fields_total = length(tc$check),
      pct = round(pct, 1),
      seconds = round(t_narr + t_verify, 1),
      stringsAsFactors = FALSE)
  }
  narr_df <- do.call(rbind, narr_rows)

  list(sql = sql_df, narr = narr_df)
}

## ---------------------------------------------------------------------------
## Run benchmark over all models
## ---------------------------------------------------------------------------
all_results <- lapply(MODELS, eval_model)
all_results <- Filter(Negate(is.null), all_results)

sql_all  <- do.call(rbind, lapply(all_results, function(r) r$sql))
narr_all <- do.call(rbind, lapply(all_results, function(r) r$narr))

## ---------------------------------------------------------------------------
## Summary tables
## ---------------------------------------------------------------------------
cat("\n\n===================== SUMMARY =====================\n")

cat("\n--- Text-to-SQL: per-model accuracy ---\n")
sql_summary <- aggregate(cbind(correct, executed) ~ model, data = sql_all, FUN = sum)
sql_summary$n <- aggregate(correct ~ model, data = sql_all, FUN = length)$correct
sql_summary$exec_pct    <- round(100 * sql_summary$executed / sql_summary$n, 1)
sql_summary$correct_pct <- round(100 * sql_summary$correct  / sql_summary$n, 1)
sql_summary$mean_seconds <- round(aggregate(seconds ~ model, data = sql_all, FUN = mean)$seconds, 1)
print(sql_summary[, c("model","n","executed","correct","exec_pct","correct_pct","mean_seconds")],
      row.names = FALSE)

cat("\n--- Text-to-SQL: per-difficulty accuracy ---\n")
diff_summary <- aggregate(correct ~ model + difficulty, data = sql_all, FUN = mean)
diff_summary$correct <- round(100 * diff_summary$correct, 1)
diff_wide <- reshape(diff_summary, idvar = "model", timevar = "difficulty",
                     direction = "wide")
names(diff_wide) <- gsub("correct\\.", "", names(diff_wide))
print(diff_wide, row.names = FALSE)

cat("\n--- Narrative: per-model round-trip field accuracy ---\n")
narr_summary <- aggregate(cbind(fields_correct, fields_total) ~ model,
                          data = narr_all, FUN = sum)
narr_summary$pct <- round(100 * narr_summary$fields_correct / narr_summary$fields_total, 1)
narr_summary$mean_seconds <- round(aggregate(seconds ~ model, data = narr_all, FUN = mean)$seconds, 1)
print(narr_summary, row.names = FALSE)

## ---------------------------------------------------------------------------
## Persist raw results
## ---------------------------------------------------------------------------
out_dir <- "C:/GitHub/edgemodelr/examples/benchmark_results"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(sql_all,  file.path(out_dir, "sql_results.csv"),  row.names = FALSE)
write.csv(narr_all, file.path(out_dir, "narrative_results.csv"), row.names = FALSE)
cat("\nRaw results saved to:", out_dir, "\n")

cat("\n===== Done =====\n")
