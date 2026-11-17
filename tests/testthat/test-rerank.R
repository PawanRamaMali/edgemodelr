# edge_rerank(): argument checks without a model, then a live check with a
# reranker when one is on disk.

test_that("edge_rerank checks its arguments", {
  expect_error(edge_rerank(NULL, c("a", "b"), "d"), "query must be a single")
  expect_error(edge_rerank(NULL, NA_character_, "d"), "query must be a single")
  expect_error(edge_rerank(NULL, "q", 1:3), "documents must be a character")
  expect_error(edge_rerank(NULL, "q", c("a", NA)), "documents must be a character")
  expect_error(edge_rerank(NULL, "q", "d", top_n = 0), "top_n must be")
  expect_error(edge_rerank(NULL, "q", "d", top_n = 1.5), "top_n must be")
  expect_error(edge_rerank(NULL, "q", "d", truncate = NA), "truncate must be")
  expect_error(edge_rerank(NULL, "q", "d"), "Invalid model context")
})

test_that("edge_load_model checks reranking", {
  path <- tempfile(fileext = ".gguf")
  writeBin(charToRaw("GGUF"), path)
  on.exit(unlink(path))
  expect_error(edge_load_model(path, reranking = NA), "reranking must be TRUE or FALSE")
  expect_error(edge_load_model(path, reranking = "yes"), "reranking must be TRUE or FALSE")
})

test_that("rerank requests are checked before the model is used", {
  expect_error(edgemodelr:::.oa_rerank(NULL, list(documents = list("a")), "m"),
               "query must be a string", class = "edge_bad_request")
  expect_error(edgemodelr:::.oa_rerank(NULL, list(query = "q"), "m"),
               "documents must be", class = "edge_bad_request")
  expect_error(edgemodelr:::.oa_rerank(NULL, list(query = "q", documents = list(1)), "m"),
               "each document", class = "edge_bad_request")
  expect_error(edgemodelr:::.oa_rerank(NULL, list(query = "q", documents = list("a"), top_n = 0),
                                       "m"),
               "top_n", class = "edge_bad_request")
})

rerank_model <- function() {
  path <- file.path(Sys.getenv("LOCALAPPDATA", unset = path.expand("~")),
                    "edgemodelr-cache", "bge-reranker-v2-m3-Q4_K_M.gguf")
  path <- Sys.getenv("EDGEMODELR_RERANK_MODEL", unset = path)
  if (!file.exists(path)) skip("bge-reranker-v2-m3 model not available")
  path
}

test_that("a reranker puts the relevant document first", {
  skip_on_cran()
  rr <- edge_load_model(rerank_model(), n_ctx = 512L, n_threads = 2L, reranking = TRUE)
  on.exit(edge_free_model(rr))

  q <- "What is the capital of France?"
  docs <- c("Berlin is in Germany.", "Paris is the capital of France.", "Bananas are yellow.")
  r <- edge_rerank(rr, q, docs)
  expect_s3_class(r, "data.frame")
  expect_named(r, c("index", "document", "score"))
  expect_identical(r$index[1], 2L)
  expect_identical(r$document[1], "Paris is the capital of France.")
  expect_true(all(is.finite(r$score)))
  expect_false(is.unsorted(rev(r$score)))

  # Scores are deterministic, and the same in a batch as one at a time.
  again <- edge_rerank(rr, q, docs)
  expect_identical(again$score, r$score)
  one <- edge_rerank(rr, q, docs[2])
  expect_equal(one$score, r$score[1], tolerance = 1e-4)

  expect_identical(nrow(edge_rerank(rr, q, docs, top_n = 1)), 1L)
  expect_identical(nrow(edge_rerank(rr, q, character())), 0L)

  # A pair longer than a batch is cut, or refused when asked.
  long <- paste(rep("word", 1500), collapse = " ")
  expect_true(is.finite(edge_rerank(rr, q, long)$score))
  expect_error(edge_rerank(rr, q, long, truncate = FALSE), "more than the")

  expect_error(edge_embeddings(rr, "x"), "reranker")
})
