# edge_similarity() and edge_similarity_matrix() on known vectors.

test_that("edge_similarity gives the cosine of known vectors", {
  expect_equal(edge_similarity(c(1, 0), c(1, 0)), 1)
  expect_equal(edge_similarity(c(1, 0), c(0, 1)), 0)
  expect_equal(edge_similarity(c(1, 0), c(-1, 0)), -1)
  expect_equal(edge_similarity(c(1, 1), c(1, 0)), 1 / sqrt(2))
  expect_equal(edge_similarity(c(1, 2, 3), c(4, 5, 6)),
               32 / (sqrt(14) * sqrt(77)))
  # Integer input works the same as double.
  expect_equal(edge_similarity(1:3, 4:6), edge_similarity(c(1, 2, 3), c(4, 5, 6)))
})

test_that("edge_similarity ignores scale and is symmetric", {
  a <- c(0.3, -1.2, 2.5, 0.7)
  b <- c(-0.4, 0.9, 1.1, -2)
  expect_equal(edge_similarity(a, b), edge_similarity(b, a))
  expect_equal(edge_similarity(10 * a, 0.01 * b), edge_similarity(a, b))
  expect_equal(edge_similarity(a, -b), -edge_similarity(a, b))
  s <- edge_similarity(a, b)
  expect_true(s >= -1 && s <= 1)
})

test_that("edge_similarity returns 0 for a zero vector", {
  expect_identical(edge_similarity(c(0, 0, 0), c(1, 2, 3)), 0)
  expect_identical(edge_similarity(c(1, 2, 3), c(0, 0, 0)), 0)
})

test_that("edge_similarity validates its arguments", {
  expect_error(edge_similarity("a", 1), "must be numeric")
  expect_error(edge_similarity(1, list(1)), "must be numeric")
  expect_error(edge_similarity(c(1, 2), c(1, 2, 3)), "same length")
})

test_that("edge_similarity_matrix matches pairwise edge_similarity", {
  m <- rbind(c(1, 0, 0),
             c(1, 1, 0),
             c(0, 0, 2),
             c(-3, 0, 0))
  s <- edge_similarity_matrix(m)
  expect_true(is.matrix(s))
  expect_identical(dim(s), c(4L, 4L))
  for (i in 1:4) for (j in 1:4) {
    expect_equal(s[i, j], edge_similarity(m[i, ], m[j, ]))
  }
  expect_equal(s[1, 2], 1 / sqrt(2))
  expect_equal(s[1, 3], 0)
  expect_equal(s[1, 4], -1)
})

test_that("edge_similarity_matrix is symmetric with a unit diagonal", {
  set.seed(42)
  m <- matrix(rnorm(5 * 16), nrow = 5)
  s <- edge_similarity_matrix(m)
  expect_equal(s, t(s))
  expect_equal(diag(s), rep(1, 5))
  expect_true(all(s >= -1 - 1e-12 & s <= 1 + 1e-12))
  # Rescaling a row changes nothing.
  m2 <- m
  m2[2, ] <- m2[2, ] * 1000
  expect_equal(edge_similarity_matrix(m2), s)
})

test_that("edge_similarity_matrix handles a zero row and a single row", {
  m <- rbind(c(1, 2), c(0, 0))
  s <- edge_similarity_matrix(m)
  expect_equal(s[1, 1], 1)
  # A zero row is similar to nothing, itself included, as in edge_similarity().
  expect_equal(s[2, ], c(0, 0))
  expect_equal(s[, 2], c(0, 0))
  expect_equal(edge_similarity_matrix(matrix(c(3, 4), nrow = 1)), matrix(1))
})

test_that("edge_similarity_matrix validates its input", {
  expect_error(edge_similarity_matrix(c(1, 2, 3)), "numeric matrix")
  expect_error(edge_similarity_matrix(matrix("a", 2, 2)), "numeric matrix")
  expect_error(edge_similarity_matrix(data.frame(a = 1:2, b = 3:4)), "numeric matrix")
})

test_that("similarity on real embeddings ranks related texts higher", {
  model <- coverage_qwen3_embedding()
  skip_without_model(model, "Qwen3-Embedding-0.6B")
  ctx <- edge_load_model(model, n_ctx = 512L, n_threads = 2L, embeddings = TRUE)
  on.exit(edge_free_model(ctx), add = TRUE)

  embs <- edge_embeddings(ctx, c("a happy cat", "a joyful kitten", "the stock market crashed"))
  expect_gt(edge_similarity(embs[1, ], embs[2, ]), edge_similarity(embs[1, ], embs[3, ]))
  s <- edge_similarity_matrix(embs)
  expect_equal(s, t(s), tolerance = 1e-6)
  expect_equal(diag(s), rep(1, 3), tolerance = 1e-5)
  expect_equal(s[1, 2], edge_similarity(embs[1, ], embs[2, ]), tolerance = 1e-6)
})
