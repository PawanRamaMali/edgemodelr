# Modern 2024 Models Example
# This example showcases the newest small language models available in edgemodelr

library(edgemodelr)

# Function to showcase a model
showcase_model <- function(model_name, prompt = "Write a short poem about AI.") {
  cat("\n" , rep("=", 60), "\n")
  cat("Testing Model:", model_name, "\n")
  cat(rep("=", 60), "\n")
  
  tryCatch({
    # Quick setup for the model
    cat("Setting up", model_name, "...\n")
    setup <- edge_quick_setup(model_name)
    
    if (is.null(setup$context)) {
      cat("❌ Failed to load", model_name, "\n")
      return(NULL)
    }
    
    ctx <- setup$context
    cat("✅ Successfully loaded", model_name, "\n")
    cat("Model path:", setup$model_path, "\n\n")
    
    # Generate response
    cat("Prompt:", prompt, "\n")
    cat("Response:\n")
    response <- edge_completion(ctx, prompt, n_predict = 100)
    cat(response, "\n\n")
    
    # Clean up
    edge_free_model(ctx)
    cat("✅ Model cleaned up\n")
    
    return(response)
    
  }, error = function(e) {
    cat("❌ Error with", model_name, ":", e$message, "\n")
    return(NULL)
  })
}

# Main demonstration function
main <- function() {
  cat("🚀 Modern 2024 Models Showcase\n")
  cat("Demonstrating the latest small language models\n\n")
  
  # Show available models
  cat("📋 Available Models:\n")
  models <- edge_list_models()
  for (i in 1:nrow(models)) {
    cat(sprintf("  %s - %s (%s)\n", 
                models$name[i], 
                models$use_case[i], 
                models$size[i]))
  }
  
  # Showcase 2024 models
  modern_models <- c("llama3.2-1b", "qwen2.5-1.5b", "phi3.5-mini", "gemma2-2b")
  
  cat("\n🎯 Testing Modern 2024 Models:\n")
  
  results <- list()
  
  for (model in modern_models) {
    result <- showcase_model(model, "Explain quantum computing in simple terms.")
    results[[model]] <- result
    Sys.sleep(1)  # Brief pause between models
  }
  
  # Summary
  cat("\n📊 Summary:\n")
  for (model in names(results)) {
    status <- if (!is.null(results[[model]])) "✅ Success" else "❌ Failed"
    cat("  ", model, ":", status, "\n")
  }
  
  cat("\n💡 Tips:\n")
  cat("  • Use 'llama3.2-1b' for mobile/edge applications\n")
  cat("  • Use 'qwen2.5-1.5b' for multilingual tasks\n") 
  cat("  • Use 'phi3.5-mini' for reasoning tasks\n")
  cat("  • Use 'gemma2-2b' for balanced performance\n")
  
  return(results)
}

# Quick test function for a single model
test_llama32 <- function() {
  cat("🦙 Quick Llama 3.2 Test\n")
  return(showcase_model("llama3.2-1b", "Hello, how are you?"))
}

# Run the demonstration if called directly
if (interactive()) {
  cat("Run main() to test all 2024 models\n")
  cat("Run test_llama32() for a quick test\n")
} else {
  # Run when sourced
  main()
}