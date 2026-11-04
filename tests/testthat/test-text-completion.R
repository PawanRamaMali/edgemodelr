


test_that("edge_completion error handling", {
  # Test with invalid context
  expect_error(
    edge_completion(NULL, "Hello", n_predict = 5)
  )

  expect_error(
    edge_completion("invalid", "Hello", n_predict = 5)
  )

  expect_error(
    edge_completion(123, "Hello", n_predict = 5)
  )

  expect_error(
    edge_completion(list(), "Hello", n_predict = 5)
  )

  # Test with missing arguments
  expect_error(edge_completion())
  expect_error(edge_completion(NULL))

  # Test with invalid prompt types
  expect_error(edge_completion(NULL, NULL, n_predict = 5))
  expect_error(edge_completion(NULL, 123, n_predict = 5))
  expect_error(edge_completion(NULL, c("a", "b"), n_predict = 5))
  expect_error(edge_completion(NULL, list("hello"), n_predict = 5))
})

test_that("sampling arguments are checked before any generation", {
  so <- edgemodelr:::.sampling_options
  o <- so(0.8, 0.95, top_k = 40, min_p = 0.05, repeat_penalty = 1.1,
          seed = 42, stop = c("\n\n", "END"))
  expect_identical(o$top_k, 40L)
  expect_identical(o$seed, 42)
  expect_identical(o$stop, c("\n\n", "END"))

  # seed = NULL draws from R's RNG, so set.seed() makes it reproducible.
  set.seed(3)
  s1 <- so(0.8, 0.95)$seed
  set.seed(3)
  expect_identical(so(0.8, 0.95)$seed, s1)
  # Greedy decoding needs no seed and leaves the RNG alone.
  set.seed(3)
  before <- .Random.seed
  expect_identical(so(0, 0.95)$seed, 0)
  expect_identical(.Random.seed, before)

  expect_error(so(0.8, 0.95, top_k = -1), "top_k")
  expect_error(so(0.8, 0.95, top_k = 2.5), "top_k")
  expect_error(so(0.8, 0.95, min_p = 1), "min_p")
  expect_error(so(0.8, 0.95, repeat_penalty = 0), "repeat_penalty")
  expect_error(so(0.8, 0.95, seed = -1), "seed")
  expect_error(so(0.8, 0.95, seed = 2^32), "seed")
  expect_error(so(0.8, 0.95, seed = "a"), "seed")
  expect_error(so(0.8, 0.95, stop = ""), "stop")
  expect_error(so(0.8, 0.95, stop = NA_character_), "stop")
  expect_error(so(0.8, 0.95, stop = 1), "stop")
})
