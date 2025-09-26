# ==============================================================================
# Performance Optimization Examples for edgemodelr
# ==============================================================================
# This file demonstrates techniques for optimizing performance, including
# GPU acceleration, memory management, benchmarking, and efficient workflows.

library(edgemodelr)

# ------------------------------------------------------------------------------
# Example 1: Benchmarking and Performance Testing
# ------------------------------------------------------------------------------

cat("Example 1: Benchmarking and Performance Testing\n")
cat("===============================================\n\n")

# Setup model for testing
ollama_info <- edge_find_ollama_models()
if (!is.null(ollama_info) && length(ollama_info$models) > 0) {
  model_hash <- substr(ollama_info$models[[1]]$sha256, 1, 8)
} else {
  cat("❌ No models found for benchmarking.\n")
  cat("   Please install Ollama and download a model: ollama pull llama3.2:latest\n")
  quit()
}

cat("Running comprehensive benchmark:\n")

# Use built-in benchmark function
ctx <- edge_load_ollama_model(model_hash, n_ctx = 1024, n_gpu_layers = 0)
benchmark_results <- edge_benchmark(
  ctx = ctx,
  prompt = "The quick brown fox jumps over the lazy dog",
  n_predict = 50,
  iterations = 3
)

cat("Benchmark Results:\n")
cat("==================\n")
print(benchmark_results)

edge_free_model(ctx)
cat("\n✅ Example 1 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 2: Context Length Optimization
# ------------------------------------------------------------------------------

cat("Example 2: Context Length Optimization\n")
cat("======================================\n\n")

prompt <- "Explain machine learning in simple terms"

# Test different context lengths
context_sizes <- c(256, 512, 1024, 2048)
performance_data <- data.frame(
  context_size = context_sizes,
  load_time = numeric(length(context_sizes)),
  generation_time = numeric(length(context_sizes)),
  tokens_per_second = numeric(length(context_sizes))
)

cat("Testing context length impact on performance:\n\n")

for (i in seq_along(context_sizes)) {
  n_ctx <- context_sizes[i]
  cat(sprintf("Testing n_ctx = %d...\n", n_ctx))

  # Time model loading
  start_time <- Sys.time()
  ctx <- edge_load_ollama_model(model_hash, n_ctx = n_ctx, n_gpu_layers = 0)
  load_time <- as.numeric(Sys.time() - start_time)

  # Time generation
  start_gen <- Sys.time()
  result <- edge_completion(ctx, prompt, n_predict = 30, temperature = 0.5)
  gen_time <- as.numeric(Sys.time() - start_gen)

  # Calculate tokens per second (approximate)
  tokens_per_sec <- 30 / gen_time

  # Store results
  performance_data[i, ] <- c(n_ctx, load_time, gen_time, tokens_per_sec)

  edge_free_model(ctx)
}

cat("\nPerformance Analysis:\n")
print(performance_data)

# Find optimal context size
optimal_idx <- which.max(performance_data$tokens_per_second)
optimal_ctx <- performance_data$context_size[optimal_idx]
cat(sprintf("\n🏆 Optimal context size: %d (%.1f tokens/sec)\n",
           optimal_ctx, performance_data$tokens_per_second[optimal_idx]))

cat("\n✅ Example 2 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 3: Memory Management and Cleanup
# ------------------------------------------------------------------------------

cat("Example 3: Memory Management and Cleanup\n")
cat("========================================\n\n")

# Monitor memory usage patterns
get_memory_info <- function() {
  if (.Platform$OS.type == "windows") {
    # Windows memory info (simplified)
    return("Memory monitoring not implemented for Windows in this example")
  } else {
    # Unix-like systems
    system("ps -o pid,vsz,rss -p `echo $$`", intern = FALSE)
  }
}

cat("Demonstrating memory management best practices:\n\n")

# Load multiple models and show memory impact
cat("1. Loading model...\n")
ctx1 <- edge_load_ollama_model(model_hash, n_ctx = 512, n_gpu_layers = 0)
cat("   Model 1 loaded\n")

# Generate some text
result1 <- edge_completion(ctx1, "Test prompt 1", n_predict = 20, temperature = 0.5)
cat("   Generated text 1:", substr(result1, 1, 50), "...\n")

# Load second instance
cat("\n2. Loading second model instance...\n")
ctx2 <- edge_load_ollama_model(model_hash, n_ctx = 512, n_gpu_layers = 0)
result2 <- edge_completion(ctx2, "Test prompt 2", n_predict = 20, temperature = 0.5)
cat("   Generated text 2:", substr(result2, 1, 50), "...\n")

# Demonstrate proper cleanup
cat("\n3. Cleaning up models...\n")
edge_free_model(ctx1)
cat("   Model 1 freed\n")

edge_free_model(ctx2)
cat("   Model 2 freed\n")

# Clear any caches
edge_clean_cache()
cat("   Cache cleared\n")

cat("\n💡 Memory Management Tips:\n")
cat("• Always call edge_free_model() when done\n")
cat("• Use edge_clean_cache() periodically\n")
cat("• Monitor memory usage in long-running applications\n")
cat("• Consider smaller context sizes for memory-constrained environments\n")

cat("\n✅ Example 3 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 4: GPU Acceleration (if available)
# ------------------------------------------------------------------------------

cat("Example 4: GPU Acceleration Testing\n")
cat("===================================\n\n")

# Test GPU availability and performance
gpu_layers_to_test <- c(0, 10, 20, -1)  # -1 means all layers
gpu_results <- list()

for (gpu_layers in gpu_layers_to_test) {
  layer_name <- if (gpu_layers == -1) "all" else as.character(gpu_layers)
  cat(sprintf("Testing with %s GPU layers...\n", layer_name))

  tryCatch({
    # Load model with GPU layers
    start_time <- Sys.time()
    ctx <- edge_load_ollama_model(model_hash, n_ctx = 512, n_gpu_layers = gpu_layers)
    load_time <- as.numeric(Sys.time() - start_time)

    # Test generation speed
    gen_start <- Sys.time()
    result <- edge_completion(ctx, "Count from 1 to 10", n_predict = 25, temperature = 0.1)
    gen_time <- as.numeric(Sys.time() - gen_start)

    gpu_results[[layer_name]] <- list(
      gpu_layers = gpu_layers,
      load_time = load_time,
      generation_time = gen_time,
      tokens_per_second = 25 / gen_time,
      success = TRUE
    )

    cat(sprintf("  ✅ Success: %.2f tokens/sec\n", 25 / gen_time))
    edge_free_model(ctx)

  }, error = function(e) {
    gpu_results[[layer_name]] <<- list(
      gpu_layers = gpu_layers,
      load_time = NA,
      generation_time = NA,
      tokens_per_second = NA,
      success = FALSE,
      error = e$message
    )
    cat(sprintf("  ❌ Failed: %s\n", e$message))
  })
}

cat("\nGPU Performance Summary:\n")
cat("========================\n")
for (name in names(gpu_results)) {
  result <- gpu_results[[name]]
  if (result$success) {
    cat(sprintf("%s layers: %.2f tokens/sec (load: %.2fs, gen: %.2fs)\n",
               name, result$tokens_per_second, result$load_time, result$generation_time))
  } else {
    cat(sprintf("%s layers: FAILED\n", name))
  }
}

# Find best GPU configuration
successful_results <- gpu_results[sapply(gpu_results, function(x) x$success)]
if (length(successful_results) > 0) {
  best_config <- successful_results[[which.max(sapply(successful_results, function(x) x$tokens_per_second))]]
  cat(sprintf("\n🚀 Best GPU config: %s layers (%.2f tokens/sec)\n",
             if (best_config$gpu_layers == -1) "all" else best_config$gpu_layers,
             best_config$tokens_per_second))
} else {
  cat("\n💻 GPU acceleration not available - using CPU only\n")
}

cat("\n✅ Example 4 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 5: Batch Processing Optimization
# ------------------------------------------------------------------------------

cat("Example 5: Batch Processing Optimization\n")
cat("========================================\n\n")

# Demonstrate efficient batch processing
prompts <- c(
  "What is R?",
  "Explain data.frame",
  "How to use ggplot2?",
  "What is machine learning?",
  "Describe linear regression"
)

cat("Comparing single vs batch processing:\n\n")

# Method 1: Load/unload model for each prompt (inefficient)
cat("Method 1: Load/unload for each prompt\n")
start_time <- Sys.time()
results_method1 <- list()

for (i in seq_along(prompts)) {
  ctx <- edge_load_ollama_model(model_hash, n_ctx = 512, n_gpu_layers = 0)
  results_method1[[i]] <- edge_completion(ctx, prompts[i], n_predict = 15, temperature = 0.3)
  edge_free_model(ctx)
}
method1_time <- as.numeric(Sys.time() - start_time)

# Method 2: Load once, process all (efficient)
cat("Method 2: Load once, process batch\n")
start_time <- Sys.time()
ctx <- edge_load_ollama_model(model_hash, n_ctx = 512, n_gpu_layers = 0)
results_method2 <- list()

for (i in seq_along(prompts)) {
  results_method2[[i]] <- edge_completion(ctx, prompts[i], n_predict = 15, temperature = 0.3)
}
edge_free_model(ctx)
method2_time <- as.numeric(Sys.time() - start_time)

# Compare results
cat("\nBatch Processing Results:\n")
cat("=========================\n")
cat(sprintf("Method 1 (load/unload): %.2f seconds\n", method1_time))
cat(sprintf("Method 2 (batch): %.2f seconds\n", method2_time))
cat(sprintf("Speedup: %.1fx faster\n", method1_time / method2_time))
cat(sprintf("Time per prompt: Method 1 = %.2fs, Method 2 = %.2fs\n",
           method1_time / length(prompts), method2_time / length(prompts)))

# Show some results
cat("\nSample Results:\n")
for (i in 1:min(3, length(prompts))) {
  cat(sprintf("%d. %s → %s\n", i, prompts[i], substr(results_method2[[i]], 1, 40), "...")
}

cat("\n✅ Example 5 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 6: Advanced Performance Monitoring
# ------------------------------------------------------------------------------

cat("Example 6: Advanced Performance Monitoring\n")
cat("==========================================\n\n")

# Create a performance monitoring function
monitor_performance <- function(ctx, prompt, n_predict = 30, iterations = 3) {
  results <- data.frame(
    iteration = 1:iterations,
    time_seconds = numeric(iterations),
    tokens_per_second = numeric(iterations),
    memory_before = numeric(iterations),
    memory_after = numeric(iterations)
  )

  for (i in 1:iterations) {
    # Simple memory monitoring (R objects)
    mem_before <- object.size(ls(envir = .GlobalEnv))

    # Time the generation
    start_time <- Sys.time()
    result <- edge_completion(ctx, prompt, n_predict = n_predict, temperature = 0.5)
    end_time <- Sys.time()

    mem_after <- object.size(ls(envir = .GlobalEnv))

    # Record metrics
    time_taken <- as.numeric(end_time - start_time)
    results[i, ] <- c(
      i,
      time_taken,
      n_predict / time_taken,
      as.numeric(mem_before),
      as.numeric(mem_after)
    )

    # Small delay between iterations
    Sys.sleep(0.1)
  }

  return(results)
}

# Run performance monitoring
cat("Running detailed performance analysis:\n")
ctx <- edge_load_ollama_model(model_hash, n_ctx = 512, n_gpu_layers = 0)

perf_data <- monitor_performance(
  ctx,
  prompt = "Describe the importance of data visualization",
  n_predict = 40,
  iterations = 5
)

cat("\nDetailed Performance Metrics:\n")
print(perf_data)

# Calculate statistics
cat("\nPerformance Statistics:\n")
cat("=======================\n")
cat(sprintf("Mean generation time: %.3f ± %.3f seconds\n",
           mean(perf_data$time_seconds), sd(perf_data$time_seconds)))
cat(sprintf("Mean tokens/second: %.2f ± %.2f\n",
           mean(perf_data$tokens_per_second), sd(perf_data$tokens_per_second)))
cat(sprintf("Best performance: %.2f tokens/second\n", max(perf_data$tokens_per_second)))
cat(sprintf("Worst performance: %.2f tokens/second\n", min(perf_data$tokens_per_second)))
cat(sprintf("Performance consistency: %.1f%% (CV)\n",
           (sd(perf_data$tokens_per_second) / mean(perf_data$tokens_per_second)) * 100))

edge_free_model(ctx)

cat("\n✅ Example 6 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Summary and Best Practices
# ------------------------------------------------------------------------------

cat("🎉 Performance Optimization Examples Complete!\n")
cat("==============================================\n\n")

cat("🚀 Key Performance Optimization Strategies:\n\n")

cat("📊 Benchmarking:\n")
cat("• Use edge_benchmark() for standardized performance testing\n")
cat("• Test different context sizes to find optimal balance\n")
cat("• Monitor tokens/second as primary performance metric\n")
cat("• Run multiple iterations to account for variability\n\n")

cat("🧠 Memory Management:\n")
cat("• Always call edge_free_model() when done with models\n")
cat("• Use edge_clean_cache() to clear temporary files\n")
cat("• Choose appropriate context length (n_ctx) for your use case\n")
cat("• Monitor memory usage in long-running applications\n\n")

cat("⚡ GPU Acceleration:\n")
cat("• Test different n_gpu_layers values (0, 10, 20, -1)\n")
cat("• GPU may not always be faster for small models\n")
cat("• Consider GPU memory limitations\n")
cat("• Fall back to CPU if GPU acceleration fails\n\n")

cat("🔄 Batch Processing:\n")
cat("• Load model once, process multiple prompts\n")
cat("• Avoid repeated model loading/unloading\n")
cat("• Consider model switching costs vs processing benefits\n")
cat("• Implement proper error handling for batch operations\n\n")

cat("📈 Monitoring & Profiling:\n")
cat("• Track generation time and throughput\n")
cat("• Monitor memory usage patterns\n")
cat("• Identify performance bottlenecks\n")
cat("• Use consistent test prompts for benchmarking\n\n")

cat("⚙️  Configuration Tips:\n")
cat(sprintf("• Optimal context size found: %d tokens\n", optimal_ctx))
if (exists("best_config") && !is.null(best_config)) {
  cat(sprintf("• Optimal GPU layers: %s\n",
             if (best_config$gpu_layers == -1) "all" else best_config$gpu_layers))
}
cat("• Batch processing provides significant speedups\n")
cat("• Performance monitoring helps identify optimization opportunities\n\n")

cat("Next: See 05_advanced_applications.R for complex use cases\n")