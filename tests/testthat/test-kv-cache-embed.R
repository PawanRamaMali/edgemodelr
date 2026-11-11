# KV cache types and embed_edge(); no model needed.

fake_gguf <- function() {
  path <- tempfile(fileext = ".gguf")
  writeBin(charToRaw("GGUF"), path)
  path
}

test_that("edge_load_model checks the KV cache types", {
  path <- fake_gguf()
  on.exit(unlink(path))
  expect_error(edge_load_model(path, cache_type_k = "q2_k"), "cache_type_k must be one of")
  expect_error(edge_load_model(path, cache_type_v = c("f16", "q8_0")), "cache_type_v must be one of")
  expect_error(edge_load_model(path, cache_type_v = "q8_0", flash_attn = FALSE),
               "quantized cache_type_v needs flash_attn = TRUE")
})

test_that("embed_edge without texts returns a function that survives a new session", {
  path <- fake_gguf()
  on.exit(unlink(path))
  fn <- embed_edge(model_path = path, n_ctx = 512L)
  expect_true(is.function(fn))
  expect_identical(names(formals(fn)), "x")
  expect_identical(environment(fn), baseenv())
  call <- body(fn)
  expect_identical(call[[1]], quote(edgemodelr::embed_edge))
  expect_identical(call$model_path, normalizePath(path, winslash = "/"))
  expect_identical(call$n_ctx, 512L)
  expect_true(call$normalize)

  # Saved and loaded, as ragnar does with a store's embed function.
  again <- unserialize(serialize(fn, NULL))
  expect_identical(deparse(again), deparse(fn))
  expect_identical(environment(again), baseenv())
})

test_that("embed_edge checks its arguments", {
  expect_error(embed_edge("a"), "model_path must be")
  expect_error(embed_edge(c("a", NA), "m.gguf"), "without NA")
  expect_error(embed_edge(data.frame(body = "a"), "m.gguf"), "text column")
  expect_error(embed_edge("a", "no-such-model.gguf"), "Model file not found")
})
