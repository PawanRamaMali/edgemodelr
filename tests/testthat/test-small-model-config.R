test_that("edge_small_model_config returns valid configuration", {
  # Test default laptop configuration
  config <- edge_small_model_config()

  expect_true(is.list(config))
  expect_true("n_ctx" %in% names(config))
  expect_true("n_gpu_layers" %in% names(config))
  expect_true("recommended_n_predict" %in% names(config))
  expect_true("recommended_temperature" %in% names(config))
  expect_true("tips" %in% names(config))

  # Check reasonable defaults
  expect_true(config$n_ctx >= 512)
  expect_true(config$n_ctx <= 4096)
  expect_true(config$n_gpu_layers >= 0)
  expect_true(config$recommended_temperature >= 0.0)
  expect_true(config$recommended_temperature <= 2.0)
})

test_that("edge_small_model_config handles different targets", {
  # Test all target configurations
  targets <- c("mobile", "laptop", "desktop", "server")

  for (target in targets) {
    config <- edge_small_model_config(target = target)
    expect_true(is.list(config))
    expect_true(config$n_ctx > 0)
  }

  # Test that mobile has smallest context
  mobile_config <- edge_small_model_config(target = "mobile")
  server_config <- edge_small_model_config(target = "server")

  expect_true(mobile_config$n_ctx < server_config$n_ctx)
  expect_true(mobile_config$recommended_n_predict < server_config$recommended_n_predict)
})

test_that("edge_small_model_config adjusts for model size", {
  # Small model should get larger context
  small_config <- edge_small_model_config(model_size_mb = 500, target = "laptop")

  # Large model should get smaller context
  large_config <- edge_small_model_config(model_size_mb = 3000, target = "laptop")

  # Base config for comparison
  base_config <- edge_small_model_config(target = "laptop")

  # Small model should have larger context than large model
  expect_true(small_config$n_ctx >= base_config$n_ctx)
  expect_true(large_config$n_ctx <= base_config$n_ctx)
})

test_that("edge_small_model_config adjusts for available RAM", {
  # Low RAM should reduce context
  low_ram_config <- edge_small_model_config(available_ram_gb = 2, target = "laptop")

  # High RAM should increase context
  high_ram_config <- edge_small_model_config(available_ram_gb = 32, target = "laptop")

  expect_true(low_ram_config$n_ctx <= 512)
  expect_true(high_ram_config$n_ctx >= low_ram_config$n_ctx)
})

test_that("edge_small_model_config handles invalid target gracefully", {
  # Should fall back to laptop with warning
  expect_warning(
    config <- edge_small_model_config(target = "invalid_target"),
    "Unknown target"
  )

  # Should still return valid config
  expect_true(is.list(config))
  expect_true(config$n_ctx > 0)
})

test_that("edge_small_model_config provides helpful tips", {
  config <- edge_small_model_config()

  expect_true(is.character(config$tips))
  expect_true(length(config$tips) > 0)

  # Check that tips contain useful information
  tips_text <- paste(config$tips, collapse = " ")
  expect_true(grepl("context", tips_text, ignore.case = TRUE))
})

test_that("edge_small_model_config values are within safe ranges", {
  # Test extreme cases
  configs <- list(
    edge_small_model_config(model_size_mb = 100, available_ram_gb = 1, target = "mobile"),
    edge_small_model_config(model_size_mb = 5000, available_ram_gb = 64, target = "server")
  )

  for (config in configs) {
    # Context size should be reasonable
    expect_true(config$n_ctx >= 512)
    expect_true(config$n_ctx <= 8192)

    # Temperature should be in valid range
    expect_true(config$recommended_temperature >= 0.0)
    expect_true(config$recommended_temperature <= 2.0)

    # Prediction length should be reasonable
    expect_true(config$recommended_n_predict >= 10)
    expect_true(config$recommended_n_predict <= 1000)
  }
})
