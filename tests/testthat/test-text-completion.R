


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




