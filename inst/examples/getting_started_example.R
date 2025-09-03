# Getting Started with edgemodelr
# Complete beginner's guide to using local LLM models in R

library(edgemodelr)

cat("🚀 Getting Started with edgemodelr\n")
cat(rep("=", 60), "\n\n")

# Step 1: See available models
cat("📋 Available Models:\n")
models <- edge_list_models()
print(models[, c("name", "size", "use_case")])
cat("\n")

# Step 2: Quick setup (easiest way to get started)
cat("⚡ Method 1: Quick Setup (Recommended for beginners)\n")
cat(rep("-", 40), "\n")

tryCatch({
  # Skip downloads during automated testing
  if (!interactive()) {
    cat("Skipping download in non-interactive mode\n")
    stop("Non-interactive mode")
  }
  
  cat("Setting up TinyLlama model...\n")
  setup <- edge_quick_setup("TinyLlama-1.1B")
  
  if (!is.null(setup$context)) {
    ctx <- setup$context
    cat("✅ Model loaded successfully!\n")
    cat("Model info:", setup$info$name, "-", setup$info$size, "\n\n")
    
    # Basic completion
    cat("💬 Testing basic completion:\n")
    prompt <- "The R programming language is"
    result <- edge_completion(ctx, prompt, n_predict = 50)
    cat("Prompt:", prompt, "\n")
    cat("Result:", result, "\n\n")
    
    # Question answering
    cat("❓ Testing question answering:\n")
    question <- "What is machine learning?"
    answer <- edge_completion(ctx, question, n_predict = 100, temperature = 0.7)
    cat("Question:", question, "\n")
    cat("Answer:", answer, "\n\n")
    
    # Code generation
    cat("💻 Testing R code generation:\n")
    code_prompt <- "Write an R function to calculate the mean of a vector:"
    code_result <- edge_completion(ctx, code_prompt, n_predict = 150)
    cat("Prompt:", code_prompt, "\n")
    cat("Generated code:", code_result, "\n\n")
    
    # Clean up
    edge_free_model(ctx)
    cat("✅ Model cleaned up\n")
    
  } else {
    cat("❌ Failed to load model context\n")
  }
  
}, error = function(e) {
  cat("❌ Quick setup failed:", e$message, "\n\n")
})

# Step 3: Manual setup approach
cat("🔧 Method 2: Manual Setup (More control)\n")
cat(rep("-", 40), "\n")

manual_setup_demo <- function() {
  tryCatch({
    # Download specific model
    cat("Downloading TinyLlama model manually...\n")
    model_path <- edge_download_model(
      model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
      filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    
    # Load with custom parameters
    cat("Loading model with custom parameters...\n")
    ctx <- edge_load_model(
      model_path = model_path,
      n_ctx = 1024,        # Smaller context for faster loading
      n_gpu_layers = 0     # CPU-only
    )
    
    # Verify model is valid
    if (is_valid_model(ctx)) {
      cat("✅ Model validation passed\n")
      
      # Test completion
      result <- edge_completion(
        ctx, 
        "Explain what R is in one sentence:", 
        n_predict = 30,
        temperature = 0.5,   # Less creative, more factual
        top_p = 0.9
      )
      cat("Result:", result, "\n")
      
      # Clean up
      edge_free_model(ctx)
    } else {
      cat("❌ Model validation failed\n")
    }
    
  }, error = function(e) {
    cat("❌ Manual setup failed:", e$message, "\n")
  })
}

# Run manual demo only in interactive mode
if (interactive()) {
  manual_setup_demo()
} else {
  cat("Skipping manual setup demo in non-interactive mode\n")
}

# Step 4: Best practices
cat("\n📝 Best Practices:\n")
cat(rep("-", 40), "\n")
cat("1. Always call edge_free_model() when done\n")
cat("2. Use lower n_predict values for faster responses\n") 
cat("3. Adjust temperature: 0.1-0.3 for factual, 0.7-1.0 for creative\n")
cat("4. Start with smaller models like TinyLlama for testing\n")
cat("5. Use edge_clean_cache() to manage disk space\n")
cat("6. Check is_valid_model() before using a context\n\n")

# Step 5: Common use cases
cat("🎯 Common Use Cases:\n")
cat(rep("-", 40), "\n")

use_cases <- data.frame(
  Task = c(
    "Text Generation",
    "Question Answering", 
    "Code Generation",
    "Summarization",
    "Creative Writing",
    "Data Analysis Help"
  ),
  Example_Prompt = c(
    "Continue this story: Once upon a time...",
    "What is the difference between mean and median?",
    "Write R code to create a scatter plot:",
    "Summarize this text in 2 sentences: [your text]",
    "Write a poem about data science",
    "How do I handle missing values in R?"
  ),
  Recommended_Settings = c(
    "temperature=0.8, n_predict=200",
    "temperature=0.3, n_predict=100", 
    "temperature=0.2, n_predict=150",
    "temperature=0.3, n_predict=100",
    "temperature=0.9, n_predict=300",
    "temperature=0.5, n_predict=150"
  ),
  stringsAsFactors = FALSE
)

print(use_cases)

cat("\n🎉 You're ready to start using edgemodelr!\n")
cat("Next steps:\n")
cat("- Try the streaming examples: source('inst/examples/llama32_streaming_example.R')\n")
cat("- Explore document analysis: source('inst/examples/document_analysis_example.R')\n")
cat("- Check out creative writing: see creative_writing_demo() functions\n")