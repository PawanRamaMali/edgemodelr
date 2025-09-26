# ==============================================================================
# Ollama Integration Examples for edgemodelr
# ==============================================================================
# This file demonstrates how to use edgemodelr with Ollama models, including
# model discovery, loading by hash, and compatibility testing.

library(edgemodelr)

# ------------------------------------------------------------------------------
# Example 1: Discover Available Ollama Models
# ------------------------------------------------------------------------------

cat("Example 1: Discover Available Ollama Models\n")
cat("===========================================\n\n")

# Find all Ollama models on the system
ollama_info <- edge_find_ollama_models()

if (is.null(ollama_info)) {
  cat("❌ No Ollama installation found.\n")
  cat("   Install Ollama from: https://ollama.ai\n")
  cat("   Then download models with: ollama pull llama2\n\n")
  quit()
}

cat("📍 Ollama models directory:", ollama_info$ollama_path, "\n")
cat("📊 Total blob files found:", ollama_info$total_found, "\n")
cat("🤖 Valid GGUF models:", ollama_info$gguf_models, "\n\n")

if (ollama_info$gguf_models == 0) {
  cat("❌ No compatible GGUF models found.\n")
  cat("   Try downloading: ollama pull llama3.2:latest\n")
  quit()
}

# Show available models
cat("Available models:\n")
for (i in seq_along(ollama_info$models)) {
  model <- ollama_info$models[[i]]
  cat(sprintf("  %d. %s (%.1f GB)\n", i, model$name, model$size_gb))
  cat(sprintf("     SHA: %s\n", substr(model$sha256, 1, 12)))
  cat(sprintf("     Path: %s\n\n", basename(model$path)))
}

cat("✅ Example 1 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 2: Load Models by SHA Hash (Convenient Method)
# ------------------------------------------------------------------------------

cat("Example 2: Load Models by SHA Hash\n")
cat("==================================\n\n")

# Get the first available model's hash
model_hash <- substr(ollama_info$models[[1]]$sha256, 1, 8)
cat("Using model hash:", model_hash, "\n")

# Load model using convenient hash-based loading
ctx <- edge_load_ollama_model(model_hash, n_ctx = 1024, n_gpu_layers = 0)

# Test the model
prompts <- c(
  "The sky is",
  "In R programming,",
  "Machine learning is"
)

cat("\nGenerating text with Ollama model:\n")
for (prompt in prompts) {
  result <- edge_completion(ctx, prompt, n_predict = 15, temperature = 0.7)
  cat(sprintf("• %s → %s\n", prompt, result))
}

edge_free_model(ctx)
cat("\n✅ Example 2 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 3: Test Model Compatibility
# ------------------------------------------------------------------------------

cat("Example 3: Test Model Compatibility\n")
cat("===================================\n\n")

# Test all models for compatibility
cat("Testing model compatibility (this may take a moment)...\n")
compatible_models <- edge_find_ollama_models(test_compatibility = TRUE)

if (!is.null(compatible_models) && length(compatible_models$models) > 0) {
  cat("\n🎉 Compatible models found:\n")
  for (model in compatible_models$models) {
    if (isTRUE(model$compatible)) {
      cat(sprintf("✅ %s - Works perfectly!\n", model$name))
    }
  }
} else {
  cat("❌ No compatible models found.\n")
  cat("   Try downloading a different model version.\n")
}

cat("\n✅ Example 3 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 4: Advanced Ollama Model Usage
# ------------------------------------------------------------------------------

cat("Example 4: Advanced Ollama Model Usage\n")
cat("======================================\n\n")

# Load the best available model (first compatible one)
best_model <- NULL
for (model in ollama_info$models) {
  if (is.null(model$compatible)) {
    # Test compatibility if not already tested
    model$compatible <- tryCatch({
      temp_ctx <- edge_load_ollama_model(substr(model$sha256, 1, 8), n_ctx = 256, n_gpu_layers = 0)
      test_result <- edge_completion(temp_ctx, "Hi", n_predict = 3, temperature = 0.1)
      edge_free_model(temp_ctx)
      TRUE
    }, error = function(e) FALSE)
  }

  if (isTRUE(model$compatible)) {
    best_model <- model
    break
  }
}

if (!is.null(best_model)) {
  cat("Using best model:", best_model$name, "\n")

  # Load with different configurations
  configs <- list(
    list(name = "Small Context", n_ctx = 512, n_gpu_layers = 0),
    list(name = "Large Context", n_ctx = 2048, n_gpu_layers = 0)
  )

  for (config in configs) {
    cat(sprintf("\nTesting %s (n_ctx=%d):\n", config$name, config$n_ctx))

    ctx <- edge_load_ollama_model(
      substr(best_model$sha256, 1, 8),
      n_ctx = config$n_ctx,
      n_gpu_layers = config$n_gpu_layers
    )

    # Generate with different parameters
    prompt <- "Explain the benefits of R programming:"
    result <- edge_completion(ctx, prompt, n_predict = 30, temperature = 0.6)

    cat("Result:", result, "\n")
    edge_free_model(ctx)
  }
} else {
  cat("❌ No compatible models available for advanced testing.\n")
}

cat("\n✅ Example 4 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 5: Error Handling and Edge Cases
# ------------------------------------------------------------------------------

cat("Example 5: Error Handling and Edge Cases\n")
cat("========================================\n\n")

# Test non-existent hash
cat("Testing non-existent hash handling:\n")
tryCatch({
  ctx <- edge_load_ollama_model("nonexist", n_ctx = 512)
}, error = function(e) {
  cat("✅ Properly caught error:", e$message, "\n")
})

# Test ambiguous hash (if multiple models exist)
if (length(ollama_info$models) > 1) {
  cat("\nTesting hash disambiguation:\n")
  # Try with just one character (might be ambiguous)
  tryCatch({
    ctx <- edge_load_ollama_model(substr(ollama_info$models[[1]]$sha256, 1, 1))
  }, error = function(e) {
    cat("✅ Disambiguation working:", e$message, "\n")
  })
}

# Test with very specific hash
if (length(ollama_info$models) > 0) {
  specific_hash <- substr(ollama_info$models[[1]]$sha256, 1, 12)
  cat(sprintf("\nTesting specific hash (%s):\n", specific_hash))

  tryCatch({
    ctx <- edge_load_ollama_model(specific_hash, n_ctx = 256, n_gpu_layers = 0)
    result <- edge_completion(ctx, "Test", n_predict = 5, temperature = 0.1)
    cat("✅ Specific hash loading works:", result, "\n")
    edge_free_model(ctx)
  }, error = function(e) {
    cat("❌ Specific hash failed:", e$message, "\n")
  })
}

cat("\n✅ Example 5 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Summary and Best Practices
# ------------------------------------------------------------------------------

cat("🎉 Ollama Integration Examples Complete!\n")
cat("=======================================\n\n")
cat("Best practices for Ollama integration:\n\n")

cat("📋 Model Discovery:\n")
cat("• Use edge_find_ollama_models() to discover available models\n")
cat("• Check model compatibility with test_compatibility = TRUE\n")
cat("• Consider model size (GB) for your system resources\n\n")

cat("🔑 Hash-based Loading:\n")
cat("• Use edge_load_ollama_model() with partial SHA hashes\n")
cat("• Start with 8 characters, use more if models conflict\n")
cat("• Hashes are more convenient than full file paths\n\n")

cat("⚙️  Configuration:\n")
cat("• Set appropriate n_ctx based on your use case\n")
cat("• Use n_gpu_layers = 0 for CPU-only inference\n")
cat("• Test different temperature values for creativity\n\n")

cat("🛡️  Error Handling:\n")
cat("• Always use tryCatch() for robust applications\n")
cat("• Check for NULL returns from discovery functions\n")
cat("• Validate models before using in production\n\n")

cat("💡 Model Selection:\n")
available_models <- sapply(ollama_info$models, function(m) {
  sprintf("%s (%.1fGB)", substr(m$sha256, 1, 8), m$size_gb)
})
cat(sprintf("• Available hashes: %s\n", paste(available_models, collapse = ", ")))
cat("• Smaller models are faster but less capable\n")
cat("• Test multiple models to find the best fit\n\n")

cat("Next: See 03_streaming_generation.R for real-time text generation\n")