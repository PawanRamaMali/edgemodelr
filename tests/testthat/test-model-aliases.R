test_that("edge_resolve_model_name handles case-insensitive names and aliases", {
  models <- edge_list_models()

  expect_equal(edgemodelr:::edge_resolve_model_name("TinyLlama-1.1B", models), "TinyLlama-1.1B")
  expect_equal(edgemodelr:::edge_resolve_model_name("tinyllama-1.1b", models), "TinyLlama-1.1B")
  expect_equal(edgemodelr:::edge_resolve_model_name("Llama-3.2-3B", models), "llama3.2-3b")
  expect_equal(edgemodelr:::edge_resolve_model_name("phi-3-mini", models), "phi3-mini")
})

test_that("edge_cache_info returns expected fields", {
  info <- edge_cache_info()
  expect_true(all(c("cache_dir", "total_size_mb", "file_count") %in% names(info)))
})
