# edge_index_documents(), edge_search() and edge_ask().
#
# The first group replaces the model with a keyword-count embedder, so the
# retrieval logic is checked everywhere without a GGUF file. The second group
# runs on real models when they are on disk.

rag_vocab <- c("paris", "tower", "plant", "sunlight", "stock", "market",
               "football", "goal")

# One dimension per vocabulary word, plus a constant so no row is all zero.
fake_embed <- function(ctx, texts, normalize = TRUE) {
  m <- t(vapply(tolower(texts), function(x) {
    c(vapply(rag_vocab, function(w) {
      lengths(regmatches(x, gregexpr(w, x, fixed = TRUE)))
    }, numeric(1)), 0.1)
  }, numeric(length(rag_vocab) + 1L), USE.NAMES = FALSE))
  if (normalize) m <- m / sqrt(rowSums(m^2))
  m
}

rag_docs <- c(
  "The Eiffel Tower is a tower in Paris.",
  "A plant turns sunlight into sugar; every plant needs sunlight.",
  "The stock market fell as stock prices dropped across the market.",
  "He scored a goal in the football final, his third goal."
)

local_fake_model <- function(env = parent.frame()) {
  local_mocked_bindings(
    is_valid_model = function(ctx) identical(ctx, "fake-ctx"),
    edge_embeddings = fake_embed,
    .package = "edgemodelr",
    .env = env
  )
}

test_that("edge_index_documents indexes direct text", {
  local_fake_model()
  idx <- edge_index_documents(rag_docs, "fake-ctx", progress = FALSE)
  expect_s3_class(idx, "edge_index")
  expect_identical(idx$chunks, rag_docs)
  expect_identical(idx$n_chunks, 4L)
  expect_identical(idx$n_embd, length(rag_vocab) + 1L)
  expect_identical(dim(idx$embeddings), c(4L, length(rag_vocab) + 1L))
  expect_true(all(is.na(idx$sources)))
  expect_equal(rowSums(idx$embeddings^2), rep(1, 4))
  expect_output(print(idx), "Chunks: 4")
  expect_output(print(idx), "direct text input")
})

test_that("edge_index_documents reads a directory and records sources", {
  local_fake_model()
  dir <- tempfile("rag-docs-")
  dir.create(dir)
  on.exit(unlink(dir, recursive = TRUE), add = TRUE)
  writeLines(rag_docs[1], file.path(dir, "paris.txt"))
  writeLines(rag_docs[3], file.path(dir, "markets.txt"))
  writeLines("ignored", file.path(dir, "notes.md"))
  writeLines(character(), file.path(dir, "empty.txt"))

  msgs <- capture_messages(idx <- edge_index_documents(dir, "fake-ctx"))
  expect_match(paste(msgs, collapse = ""), "Reading 3 files")
  expect_match(paste(msgs, collapse = ""), "Index built: 2 chunks")
  expect_identical(idx$n_chunks, 2L)
  expect_setequal(basename(idx$sources), c("paris.txt", "markets.txt"))
  expect_output(print(idx), "Source files: 2")

  md <- edge_index_documents(dir, "fake-ctx", file_pattern = "*.md", progress = FALSE)
  expect_identical(md$chunks, "ignored")
  expect_error(edge_index_documents(dir, "fake-ctx", file_pattern = "*.pdf",
                                    progress = FALSE),
               "No files matching")
})

test_that("edge_index_documents chunks long text with overlap", {
  local_fake_model()
  long <- paste(rep("The tower in Paris is tall.", 40), collapse = " ")
  idx <- edge_index_documents(c(long, "short goal"), "fake-ctx",
                              chunk_size = 100L, chunk_overlap = 20L,
                              progress = FALSE)
  expect_gt(idx$n_chunks, 2L)
  expect_identical(idx$chunks[idx$n_chunks], "short goal")
  expect_true(all(nchar(idx$chunks) <= 100L))
  expect_identical(nrow(idx$embeddings), idx$n_chunks)
})

test_that("edge_index_documents validates its arguments", {
  local_fake_model()
  expect_error(edge_index_documents(rag_docs, NULL), "Invalid model context")
  expect_error(edge_index_documents(rag_docs, "fake-ctx", chunk_size = 0),
               "chunk_size must be a positive number")
  expect_error(edge_index_documents(rag_docs, "fake-ctx", chunk_size = NA),
               "chunk_size must be a positive number")
  expect_error(edge_index_documents(rag_docs, "fake-ctx", chunk_size = 10,
                                    chunk_overlap = 10),
               "chunk_overlap must be at least 0")
  expect_error(edge_index_documents(rag_docs, "fake-ctx", chunk_overlap = -1),
               "chunk_overlap must be at least 0")
  expect_error(edge_index_documents(character(), "fake-ctx", progress = FALSE),
               "source must be a directory path or a character vector")
  expect_error(edge_index_documents(c(42, 43), "fake-ctx", progress = FALSE),
               "source must be a directory path or a character vector")
})

test_that("edge_index_documents gives its own error for a single non-character source", {
  skip(paste("bug: a length-1 non-character source reaches dir.exists() first,",
             "which fails with 'invalid filename argument' instead of the",
             "function's own message (R/api.R:4415)"))
  local_fake_model()
  expect_error(edge_index_documents(42, "fake-ctx", progress = FALSE),
               "source must be a directory path or a character vector")
})

test_that("edge_search ranks the relevant document first", {
  local_fake_model()
  idx <- edge_index_documents(rag_docs, "fake-ctx", progress = FALSE)
  queries <- c("What is the tallest tower in Paris?" = 1L,
               "How does a plant use sunlight?" = 2L,
               "Why did the stock market fall?" = 3L,
               "Who scored the football goal?" = 4L)
  for (q in names(queries)) {
    res <- edge_search(idx, "fake-ctx", q, top_k = 2L)
    expect_identical(res$index[1], queries[[q]], label = q)
  }

  res <- edge_search(idx, "fake-ctx", "stock market", top_k = 4L)
  expect_identical(names(res), c("chunk", "score", "source", "index"))
  expect_identical(nrow(res), 4L)
  expect_identical(res$chunk, rag_docs[res$index])
  expect_false(is.unsorted(rev(res$score)))
  expect_true(all(res$score >= -1 - 1e-12 & res$score <= 1 + 1e-12))
  # top_k larger than the index is capped.
  expect_identical(nrow(edge_search(idx, "fake-ctx", "goal", top_k = 50L)), 4L)
})

test_that("edge_search validates its arguments", {
  local_fake_model()
  idx <- edge_index_documents(rag_docs, "fake-ctx", progress = FALSE)
  expect_error(edge_search(list(), "fake-ctx", "q"), "edge_index object")
  expect_error(edge_search(idx, NULL, "q"), "Invalid model context")
  expect_error(edge_search(idx, "fake-ctx", c("a", "b")), "single character string")
  expect_error(edge_search(idx, "fake-ctx", 1), "single character string")
})

test_that("edge_search scores are cosine similarities for an unnormalized index", {
  skip(paste("bug: edge_search() normalizes the query but not the index rows,",
             "so with edge_index_documents(normalize = FALSE) the scores are dot",
             "products, not the documented cosine similarities (R/api.R:4537)"))
  local_fake_model()
  idx <- edge_index_documents(rag_docs, "fake-ctx", normalize = FALSE, progress = FALSE)
  res <- edge_search(idx, "fake-ctx", "stock market", top_k = 4L)
  expect_true(all(res$score >= -1 - 1e-12 & res$score <= 1 + 1e-12))
})

test_that("edge_ask builds a grounded prompt and returns the answer with sources", {
  local_fake_model()
  seen_prompt <- NULL
  local_mocked_bindings(
    edge_completion_internal = function(ctx, prompt, n_predict, sampling) {
      seen_prompt <<- prompt
      paste0(prompt, "  It is in Paris.  ")
    },
    .package = "edgemodelr"
  )
  idx <- edge_index_documents(rag_docs, "fake-ctx", progress = FALSE)
  q <- "Which tower is in Paris?"

  ans <- edge_ask("fake-ctx", q, idx, top_k = 2L)
  expect_identical(ans, "It is in Paris.")

  res <- edge_ask("fake-ctx", q, idx, top_k = 2L, return_context = TRUE)
  expect_identical(names(res), c("answer", "context", "prompt"))
  expect_identical(res$answer, "It is in Paris.")
  expect_identical(nrow(res$context), 2L)
  expect_identical(res$context$index[1], 1L)
  expect_identical(res$prompt, seen_prompt)
  expect_true(grepl(paste0("[1] ", rag_docs[1]), res$prompt, fixed = TRUE))
  expect_true(grepl(paste0("### Question\n", q), res$prompt, fixed = TRUE))
  expect_true(grepl("based only on the provided context", res$prompt, fixed = TRUE))

  custom <- edge_ask("fake-ctx", q, idx, system_prompt = "Answer in French.",
                     return_context = TRUE)
  expect_true(startsWith(custom$prompt, "### System\nAnswer in French."))
})

test_that("edge_ask validates its arguments", {
  local_fake_model()
  idx <- edge_index_documents(rag_docs, "fake-ctx", progress = FALSE)
  expect_error(edge_ask(NULL, "q", idx), "Invalid model context")
  expect_error(edge_ask("fake-ctx", "q", list()), "edge_index object")
  expect_error(edge_ask("fake-ctx", c("a", "b"), idx), "single character string")
})

rag_real_docs <- c(
  "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
  "Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar.",
  "The stock market fell sharply after the central bank raised interest rates.",
  "Lionel Messi is an Argentine footballer who won the 2022 World Cup."
)

test_that("edge_search ranks the relevant document first with Qwen3-Embedding", {
  model <- coverage_qwen3_embedding()
  skip_without_model(model, "Qwen3-Embedding-0.6B")
  ctx <- edge_load_model(model, n_ctx = 512L, n_threads = 2L, embeddings = TRUE)
  on.exit(edge_free_model(ctx), add = TRUE)

  idx <- edge_index_documents(rag_real_docs, ctx, progress = FALSE)
  expect_identical(idx$n_chunks, 4L)
  expect_identical(idx$n_embd, edge_model_n_embd(ctx))
  queries <- c("Which landmark is in Paris?" = 1L,
               "How do plants make food from light?" = 2L,
               "What happened to share prices?" = 3L,
               "Who is a famous soccer player?" = 4L)
  for (q in names(queries)) {
    res <- edge_search(idx, ctx, q, top_k = 2L)
    expect_identical(res$index[1], queries[[q]], label = q)
    expect_gt(res$score[1], res$score[2])
  }
})

test_that("edge_ask answers from the index with TinyLlama", {
  model <- coverage_tinyllama()
  skip_without_model(model, "TinyLlama")
  # One context does both retrieval and generation.
  ctx <- edge_load_model(model, n_ctx = 1024L, n_threads = 2L, embeddings = TRUE)
  on.exit(edge_free_model(ctx), add = TRUE)

  idx <- edge_index_documents(rag_real_docs, ctx, progress = FALSE)
  res <- edge_ask(ctx, "Which landmark is in Paris?", idx, top_k = 2L,
                  n_predict = 24L, temperature = 0, return_context = TRUE)
  expect_type(res$answer, "character")
  expect_length(res$answer, 1L)
  expect_true(nzchar(res$answer))
  expect_identical(nrow(res$context), 2L)
  expect_true(all(res$context$index %in% seq_along(rag_real_docs)))
  expect_identical(res$context$chunk, rag_real_docs[res$context$index])
  expect_true(all(is.na(res$context$source)))
  expect_false(startsWith(res$answer, "### System"))

  ans <- edge_ask(ctx, "Which landmark is in Paris?", idx, top_k = 1L,
                  n_predict = 24L, temperature = 0)
  expect_type(ans, "character")
  expect_true(nzchar(ans))
})

test_that("edge_ask works with an index built by an embedding model", {
  skip(paste("bug: edge_ask() uses one ctx for both retrieval and generation,",
             "and a Qwen3-Embedding context cannot generate (GGML_ASSERT(logits != nullptr)",
             "in llama-sampler.cpp), so an embedding-model index cannot be used with",
             "edge_ask() (R/api.R:4584, R/api.R:4626)"))
  model <- coverage_qwen3_embedding()
  skip_without_model(model, "Qwen3-Embedding-0.6B")
  ctx <- edge_load_model(model, n_ctx = 512L, n_threads = 2L, embeddings = TRUE)
  on.exit(edge_free_model(ctx), add = TRUE)
  idx <- edge_index_documents(rag_real_docs, ctx, progress = FALSE)
  res <- edge_ask(ctx, "Which landmark is in Paris?", idx, top_k = 2L,
                  n_predict = 16L, temperature = 0, return_context = TRUE)
  expect_true(nzchar(res$answer))
})
