# ==============================================================================
# Basic Usage Examples for edgemodelr
# ==============================================================================
# This file demonstrates the fundamental operations for getting started with
# edgemodelr: loading models, generating text, and proper cleanup.

library(edgemodelr)

# ------------------------------------------------------------------------------
# Example 1: Basic Model Loading and Text Generation
# ------------------------------------------------------------------------------

cat("Example 1: Basic Model Loading and Text Generation\n")
cat("=================================================\n\n")

# Try to find available models
ollama_info <- edge_find_ollama_models()
if (!is.null(ollama_info) && length(ollama_info$models) > 0) {
  # Use first Ollama model
  model_hash <- substr(ollama_info$models[[1]]$sha256, 1, 8)
  cat("Loading Ollama model...\n")
  ctx <- edge_load_ollama_model(model_hash, n_ctx = 1024, n_gpu_layers = 0)
} else {
  cat("❌ No models found. Please install Ollama and download a model:\n")
  cat("   ollama pull llama3.2:latest\n")
  quit()
}

# Generate a simple completion
prompt <- "The capital of France is"
cat("Prompt:", prompt, "\n")

result <- edge_completion(ctx, prompt, n_predict = 20, temperature = 0.3)
cat("Result:", result, "\n\n")

# Always clean up
edge_free_model(ctx)
cat("✅ Example 1 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 2: Different Temperature Settings
# ------------------------------------------------------------------------------

cat("Example 2: Different Temperature Settings\n")
cat("=========================================\n\n")

ctx <- edge_load_model(setup$available_models[1], n_ctx = 1024, n_gpu_layers = 0)

prompt <- "Once upon a time"

# Conservative generation (temperature = 0.1)
cat("Conservative (temp=0.1):\n")
result_conservative <- edge_completion(ctx, prompt, n_predict = 15, temperature = 0.1)
cat(result_conservative, "\n\n")

# Balanced generation (temperature = 0.7)
cat("Balanced (temp=0.7):\n")
result_balanced <- edge_completion(ctx, prompt, n_predict = 15, temperature = 0.7)
cat(result_balanced, "\n\n")

# Creative generation (temperature = 1.0)
cat("Creative (temp=1.0):\n")
result_creative <- edge_completion(ctx, prompt, n_predict = 15, temperature = 1.0)
cat(result_creative, "\n\n")

edge_free_model(ctx)
cat("✅ Example 2 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 3: Context Length Management
# ------------------------------------------------------------------------------

cat("Example 3: Context Length Management\n")
cat("====================================\n\n")

# Load model with small context for demonstration
ctx <- edge_load_model(setup$available_models[1], n_ctx = 256, n_gpu_layers = 0)

# Short context works fine
short_prompt <- "Hello world"
cat("Short prompt:", short_prompt, "\n")
result_short <- edge_completion(ctx, short_prompt, n_predict = 10, temperature = 0.5)
cat("Result:", result_short, "\n\n")

# Demonstrate context awareness
long_context <- paste(rep("This is a sentence. ", 10), collapse = "")
cat("Long context (", nchar(long_context), " characters):\n")
cat(substr(long_context, 1, 100), "...\n")

result_long <- edge_completion(ctx, long_context, n_predict = 10, temperature = 0.5)
cat("Result:", result_long, "\n\n")

edge_free_model(ctx)
cat("✅ Example 3 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 4: Error Handling and Validation
# ------------------------------------------------------------------------------

cat("Example 4: Error Handling and Validation\n")
cat("========================================\n\n")

# Demonstrate proper error handling
tryCatch({
  # Try to load a non-existent model
  ctx <- edge_load_model("nonexistent_model.gguf")
}, error = function(e) {
  cat("Expected error caught:", e$message, "\n")
})

# Load a valid model and demonstrate validation
ctx <- edge_load_model(setup$available_models[1], n_ctx = 512, n_gpu_layers = 0)

# Check if model is valid
if (is_valid_model(ctx)) {
  cat("✅ Model is valid and ready for inference\n")

  # Test with empty prompt (edge case)
  tryCatch({
    result <- edge_completion(ctx, "", n_predict = 5, temperature = 0.5)
    cat("Empty prompt result:", result, "\n")
  }, error = function(e) {
    cat("Empty prompt error:", e$message, "\n")
  })

  # Test with very long generation request
  tryCatch({
    result <- edge_completion(ctx, "Test", n_predict = 1000, temperature = 0.5)
    cat("Long generation successful (", nchar(result), " characters)\n")
  }, error = function(e) {
    cat("Long generation error:", e$message, "\n")
  })

} else {
  cat("❌ Model validation failed\n")
}

edge_free_model(ctx)
cat("✅ Example 4 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------

cat("🎉 Basic Usage Examples Complete!\n")
cat("=================================\n\n")
cat("Key takeaways:\n")
cat("• Use edge_find_ollama_models() to discover available models\n")
cat("• Use edge_load_model() to load GGUF files\n")
cat("• Generate text with edge_completion()\n")
cat("• Adjust temperature for different creativity levels\n")
cat("• Consider context length (n_ctx) for your use case\n")
cat("• Always call edge_free_model() when done\n")
cat("• Use proper error handling for robust applications\n\n")

cat("Next: See 02_ollama_integration.R for Ollama integration examples\n")