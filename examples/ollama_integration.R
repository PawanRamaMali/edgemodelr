# Ollama Integration Examples for edgemodelr
#
# This script demonstrates how to use edgemodelr with Ollama models.
# Ollama stores models as SHA-256 named blob files that are actually GGUF files.

library(edgemodelr)

cat("=== Ollama Integration with edgemodelr ===\n\n")

# Example 1: Find available Ollama models
cat("1. Finding Ollama models...\n")
ollama_info <- edge_find_ollama_models()

if (is.null(ollama_info)) {
  cat("❌ No Ollama installation found.\n")
  cat("Make sure Ollama is installed and has downloaded some models.\n")
  cat("Visit: https://ollama.ai/\n\n")
} else {
  cat("✅ Found Ollama installation at:", ollama_info$ollama_path, "\n")
  cat("Total blob files:", ollama_info$total_found, "\n")
  cat("GGUF models detected:", ollama_info$gguf_models, "\n\n")

  if (length(ollama_info$models) > 0) {
    cat("Available models:\n")
    for (i in seq_along(ollama_info$models)) {
      model <- ollama_info$models[[i]]
      cat(sprintf("  %d. %s (%s MB) - SHA: %s...%s\n",
                  i, model$name, model$size_mb,
                  substr(model$sha256, 1, 8),
                  substr(model$sha256, nchar(model$sha256)-7, nchar(model$sha256))))
    }
    cat("\n")

    # Example 2: Load a model using the traditional approach
    cat("2. Loading model using full path...\n")
    first_model <- ollama_info$models[[1]]

    tryCatch({
      ctx <- edge_load_model(first_model$path, n_ctx = 1024)
      cat("✅ Successfully loaded:", first_model$name, "\n")

      # Quick test
      cat("Testing inference...\n")
      result <- edge_completion(ctx, "The sky is", n_predict = 5, temperature = 0.1)
      cat("Test result:", result, "\n")

      edge_free_model(ctx)
      cat("✅ Model unloaded successfully\n\n")

    }, error = function(e) {
      cat("❌ Failed to load model:", e$message, "\n\n")
    })

    # Example 3: Load model using convenient hash-based loading
    cat("3. Loading model using partial SHA hash...\n")
    partial_hash <- substr(first_model$sha256, 1, 8)

    tryCatch({
      ctx <- edge_load_ollama_model(partial_hash, n_ctx = 1024)
      cat("✅ Successfully loaded using hash:", partial_hash, "\n")

      # More comprehensive test
      cat("Running chat test...\n")
      result <- edge_completion(ctx, "Hello! How are you today?",
                               n_predict = 20, temperature = 0.7)
      cat("Chat result:", result, "\n")

      edge_free_model(ctx)
      cat("✅ Model unloaded successfully\n\n")

    }, error = function(e) {
      cat("❌ Failed to load model with hash:", e$message, "\n\n")
    })
  }
}

# Example 4: Test compatibility of all found models
cat("4. Testing compatibility of all models...\n")
compatible_info <- edge_find_ollama_models(test_compatibility = TRUE)

if (!is.null(compatible_info) && length(compatible_info$models) > 0) {
  compatible_models <- sapply(compatible_info$models, function(m) m$compatible %||% FALSE)
  cat("Compatible models:", sum(compatible_models), "/", length(compatible_info$models), "\n")

  if (sum(compatible_models) > 0) {
    cat("\n✅ Recommended models for use:\n")
    for (model in compatible_info$models[compatible_models]) {
      cat(sprintf("  - %s (%s MB) - Hash: %s\n",
                  model$name, model$size_mb,
                  substr(model$sha256, 1, 12)))
    }
  }
}

cat("\n=== Integration Tips ===\n")
cat("• Use edge_find_ollama_models() to discover available models\n")
cat("• Use edge_load_ollama_model('hash') for convenient loading\n")
cat("• Start with test_compatibility=TRUE to find working models\n")
cat("• Use smaller n_ctx values for testing to save memory\n")
cat("• Models without .gguf extension will show a warning (normal)\n")

# Example 5: Advanced usage with error handling
example_advanced_usage <- function() {
  cat("\n5. Advanced Ollama usage pattern...\n")

  # Find models with compatibility testing
  ollama_models <- edge_find_ollama_models(test_compatibility = TRUE, max_size_gb = 5)

  if (is.null(ollama_models) || length(ollama_models$models) == 0) {
    return("No compatible Ollama models found")
  }

  # Filter for reasonably sized models (< 2GB for this example)
  small_models <- Filter(function(m) m$size_gb < 2 && (m$compatible %||% FALSE),
                        ollama_models$models)

  if (length(small_models) == 0) {
    return("No small compatible models found")
  }

  # Use the smallest compatible model
  model <- small_models[[which.min(sapply(small_models, function(m) m$size_gb))]]

  cat("Selected model:", model$name, "(", model$size_gb, "GB)\n")

  # Load and use
  ctx <- edge_load_ollama_model(substr(model$sha256, 1, 8), n_ctx = 2048)

  # Demonstrate streaming
  cat("Streaming response:\n> ")
  edge_stream_completion(ctx, "Explain what R programming is in simple terms:",
                        callback = function(data) {
                          if (!data$is_final) {
                            cat(data$token)
                            flush.console()
                          }
                        },
                        n_predict = 50)
  cat("\n")

  edge_free_model(ctx)
  return("Advanced example completed successfully")
}

# Run advanced example if models are available
if (exists("ollama_info") && !is.null(ollama_info) && length(ollama_info$models) > 0) {
  tryCatch({
    result <- example_advanced_usage()
    cat("Result:", result, "\n")
  }, error = function(e) {
    cat("Advanced example failed:", e$message, "\n")
  })
}

cat("\nOllama integration examples complete!\n")