# Small Model Optimization Example
# This example demonstrates how to use edgemodelr's optimization features
# for small language models (1B-3B parameters) to achieve maximum performance
# on resource-constrained devices.

library(edgemodelr)

# Example 1: Using edge_small_model_config() for optimal settings
# ----------------------------------------------------------------

cat("\n=== Example 1: Optimized Configuration ===\n")

# Get optimized configuration for a laptop with 8GB RAM
config <- edge_small_model_config(
  model_size_mb = 700,     # TinyLlama is ~700MB
  available_ram_gb = 8,
  target = "laptop"
)

cat("\nOptimized configuration:\n")
cat("Context size:", config$n_ctx, "tokens\n")
cat("GPU layers:", config$n_gpu_layers, "\n")
cat("Recommended generation length:", config$recommended_n_predict, "tokens\n")
cat("Recommended temperature:", config$recommended_temperature, "\n")

cat("\nOptimization tips:\n")
for (tip in config$tips) {
  cat("  -", tip, "\n")
}

# Example 2: Load a small model with optimized settings
# ------------------------------------------------------

cat("\n=== Example 2: Loading with Optimization ===\n")

# Quick setup automatically downloads and configures the model
setup <- edge_quick_setup("TinyLlama-1.1B")

if (!is.null(setup$context)) {
  ctx <- setup$context

  cat("\nModel loaded with automatic optimizations!\n")
  cat("Model size:", round(file.info(setup$model_path)$size / (1024^2), 1), "MB\n")

  # Example 3: Fast inference with optimized parameters
  # ---------------------------------------------------

  cat("\n=== Example 3: Fast Inference ===\n")

  # Use greedy decoding (temperature=0) for fastest results
  prompt <- "Q: What is 2+2? A:"

  cat("\nPrompt:", prompt, "\n")
  cat("Response: ")

  result <- edge_completion(
    ctx,
    prompt = prompt,
    n_predict = 50,        # Short response for speed
    temperature = 0.0      # Greedy decoding = fastest
  )

  cat(result, "\n")

  # Example 4: Balanced quality and speed
  # -------------------------------------

  cat("\n=== Example 4: Balanced Settings ===\n")

  prompt2 <- "Write a short greeting:"

  cat("\nPrompt:", prompt2, "\n")
  cat("Response: ")

  result2 <- edge_completion(
    ctx,
    prompt = prompt2,
    n_predict = config$recommended_n_predict,
    temperature = config$recommended_temperature
  )

  cat(result2, "\n")

  # Example 5: Benchmark the optimizations
  # --------------------------------------

  cat("\n=== Example 5: Performance Benchmark ===\n")

  perf <- edge_benchmark(ctx, prompt = "The quick brown fox", n_predict = 50, iterations = 2)

  cat("\nPerformance metrics:\n")
  cat("Average tokens per second:", round(perf$avg_tokens_per_second, 2), "\n")
  cat("Average time per token:", round(perf$avg_time_per_token * 1000, 2), "ms\n")
  cat("Total time:", round(perf$total_time, 2), "seconds\n")

  # Example 6: Compare different configurations
  # -------------------------------------------

  cat("\n=== Example 6: Configuration Comparison ===\n")

  targets <- c("mobile", "laptop", "desktop")

  cat("\nRecommended context sizes by device:\n")
  for (target in targets) {
    cfg <- edge_small_model_config(model_size_mb = 700, target = target)
    cat(sprintf("  %s: %d tokens\n", target, cfg$n_ctx))
  }

  # Clean up
  edge_free_model(ctx)
  cat("\n✓ Model freed successfully\n")

} else {
  cat("\nCouldn't load model. Please check your installation.\n")
}

# Example 7: Manual optimization for very small models
# ----------------------------------------------------

cat("\n=== Example 7: Manual Optimization Tips ===\n")

cat("\nFor models < 1GB (like TinyLlama):\n")
cat("  - Use n_ctx=1024 for faster inference\n")
cat("  - Use n_predict=50-100 for quick responses\n")
cat("  - Use temperature=0.0 for deterministic output\n")
cat("  - Keep prompts concise and clear\n")

cat("\nFor models 1-2GB (like Llama 3.2 1B):\n")
cat("  - Use n_ctx=1536 for balanced performance\n")
cat("  - Use n_predict=100-150 for good quality\n")
cat("  - Use temperature=0.7-0.8 for varied output\n")

cat("\nFor models 2-3GB (like Llama 3.2 3B):\n")
cat("  - Use n_ctx=2048 for better context\n")
cat("  - Use n_predict=150-200 for detailed responses\n")
cat("  - Use temperature=0.8 for creative output\n")

# Summary of optimizations
cat("\n=== Optimization Summary ===\n")
cat("\nKey optimizations in edgemodelr for small models:\n")
cat("1. Adaptive batch sizing based on context length\n")
cat("2. Optimal thread allocation for small models\n")
cat("3. Automatic context size tuning based on model size\n")
cat("4. Memory-efficient inference pipeline\n")
cat("5. Smart defaults for resource-constrained devices\n")

cat("\n✓ Example completed!\n\n")
