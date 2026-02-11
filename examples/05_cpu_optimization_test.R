# =============================================================================
# CPU Inference Optimization Test Script
# =============================================================================
# Tests the new threading, flash attention, and vectorization optimizations
# by benchmarking different configurations with real models.
#
# Usage: source("examples/05_cpu_optimization_test.R")
# =============================================================================

library(edgemodelr)

cat("=== edgemodelr CPU Optimization Test ===\n\n")

# --- Step 1: Check compile-time features ---
cat("--- Step 1: SIMD & Compile Features ---\n")
info <- edge_simd_info()
cat("Architecture:", info$architecture, "\n")
cat("Compiler features:", paste(info$compiler_features, collapse = ", "), "\n")
cat("GGML features:", paste(info$ggml_features, collapse = ", "), "\n")
cat("Generic mode:", info$is_generic, "\n\n")

# --- Step 2: Download test model ---
cat("--- Step 2: Download Test Model ---\n")
model_path <- tryCatch({
  setup <- edge_quick_setup("TinyLlama-1.1B", verbose = TRUE)
  setup$model_path
}, error = function(e) {
  cat("Auto-download failed:", e$message, "\n")
  cat("Trying direct URL download...\n")
  edge_download_url(
    url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
  )
})

if (!file.exists(model_path)) {
  stop("Model download failed. Please download a GGUF model manually.")
}

model_size_mb <- round(file.info(model_path)$size / (1024^2), 1)
cat("Model:", basename(model_path), "(", model_size_mb, "MB)\n")
cat("Hardware threads:", parallel::detectCores(), "\n\n")

# --- Step 3: Benchmark configurations ---
cat("--- Step 3: Benchmark Configurations ---\n\n")

test_prompt <- "Explain what a neural network is in simple terms:"
n_predict <- 50
iterations <- 3

configs <- list(
  list(name = "All threads + flash_attn ON",  n_threads = NULL, flash_attn = TRUE),
  list(name = "All threads + flash_attn OFF", n_threads = NULL, flash_attn = FALSE),
  list(name = "4 threads + flash_attn ON",    n_threads = 4L,   flash_attn = TRUE),
  list(name = "4 threads + flash_attn OFF",   n_threads = 4L,   flash_attn = FALSE),
  list(name = "2 threads + flash_attn ON",    n_threads = 2L,   flash_attn = TRUE),
  list(name = "1 thread + flash_attn ON",     n_threads = 1L,   flash_attn = TRUE)
)

results <- list()

for (i in seq_along(configs)) {
  cfg <- configs[[i]]
  cat(sprintf("[%d/%d] %s\n", i, length(configs), cfg$name))

  ctx <- tryCatch({
    edge_load_model(model_path, n_ctx = 1024L,
                    n_threads = cfg$n_threads,
                    flash_attn = cfg$flash_attn)
  }, error = function(e) {
    cat("  FAILED to load:", e$message, "\n\n")
    NULL
  })

  if (is.null(ctx)) next

  times <- numeric(iterations)
  for (j in seq_len(iterations)) {
    start <- Sys.time()
    result <- edge_completion(ctx, test_prompt, n_predict = n_predict, temperature = 0.0)
    end <- Sys.time()
    times[j] <- as.numeric(end - start, units = "secs")
  }

  edge_free_model(ctx)

  avg_time <- mean(times)
  avg_tps <- n_predict / avg_time

  results[[i]] <- data.frame(
    config = cfg$name,
    n_threads = if (is.null(cfg$n_threads)) "auto" else as.character(cfg$n_threads),
    flash_attn = cfg$flash_attn,
    avg_seconds = round(avg_time, 3),
    tokens_per_sec = round(avg_tps, 1),
    stringsAsFactors = FALSE
  )

  cat(sprintf("  avg: %.3fs | %.1f tokens/sec\n\n", avg_time, avg_tps))
}

# --- Step 4: Summary table ---
cat("--- Step 4: Results Summary ---\n\n")
summary_df <- do.call(rbind, results)

# Find baseline (first config)
baseline_tps <- summary_df$tokens_per_sec[1]
summary_df$speedup <- paste0(round(summary_df$tokens_per_sec / baseline_tps, 2), "x")

print(summary_df, row.names = FALSE)

cat("\n--- Test Complete ---\n")
cat("Baseline:", summary_df$config[1], "\n")
cat("Best config:", summary_df$config[which.max(summary_df$tokens_per_sec)], "\n")
cat("Best throughput:", max(summary_df$tokens_per_sec), "tokens/sec\n")
