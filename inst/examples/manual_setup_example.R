library(edgemodelr)

# Manual setup approach if downloads are failing
manual_model_setup <- function() {
  cat("=== MANUAL MODEL SETUP GUIDE ===\n")
  
  # First, let's see what models are available
  cat("Available models:\n")
  models <- edge_list_models()
  print(models[1:5, c("name", "size", "model_id", "filename")])
  
  cat("\n=== RECOMMENDED APPROACH ===\n")
  cat("For reliable setup, try these smaller models first:\n\n")
  
  # Try TinyLlama first (smallest, most reliable)
  cat("1. Trying TinyLlama-1.1B (smallest model)...\n")
  tryCatch({
    # Skip downloads during automated testing
    if (!interactive()) {
      stop("Skipping model download during automated testing")
    }
    setup <- edge_quick_setup("TinyLlama-1.1B")
    cat("✓ TinyLlama-1.1B setup successful!\n")
    
    # Test it works
    ctx <- setup$context
    test_result <- edge_completion(ctx, "Hello", n_predict = 5)
    cat("✓ Test completion:", test_result, "\n")
    edge_free_model(ctx)
    
    return("TinyLlama-1.1B")
    
  }, error = function(e) {
    cat("✗ TinyLlama setup failed:", e$message, "\n")
  })
  
  cat("\n2. If download fails, try manual download:\n")
  cat("   - Go to: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF\n")
  cat("   - Download: tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf\n")
  cat("   - Place in: C:/Users/alpine/.cache/edgemodelr/\n")
  cat("   - Or current directory: ", getwd(), "\n")
  
  cat("\n3. Alternative: Use local file\n")
  cat("   If you have a .gguf model file:\n")
  
  # Check for existing models in common locations
  possible_locations <- c(
    file.path(Sys.getenv("HOME"), ".cache", "edgemodelr"),
    ".",
    "models"
  )
  
  for (loc in possible_locations) {
    if (dir.exists(loc)) {
      gguf_files <- list.files(loc, pattern = "\\.gguf$", full.names = TRUE)
      if (length(gguf_files) > 0) {
        cat("\n   Found existing models in", loc, ":\n")
        for (file in gguf_files) {
          cat("   -", basename(file), "\n")
        }
      }
    }
  }
  
  return(NULL)
}

# Simple analysis function that works with any model
simple_document_analysis <- function(texts, model_path = NULL) {
  if (is.null(model_path)) {
    # Try to find any available model
    possible_paths <- c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", 
      file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    )
    
    for (path in possible_paths) {
      if (file.exists(path)) {
        model_path <- path
        break
      }
    }
    
    if (is.null(model_path)) {
      stop("No model found. Please download a model first or specify model_path")
    }
  }
  
  cat("Using model:", model_path, "\n")
  
  # Load model directly
  ctx <- edge_load_model(model_path, n_ctx = 2048)
  
  results <- list()
  
  for (i in seq_along(texts)) {
    cat("Processing document", i, "\n")
    
    # Simple analysis
    text <- texts[i]
    
    # Summary
    prompt1 <- paste("Summarize in one sentence:", text, "\nSummary:")
    summary <- edge_completion(ctx, prompt1, n_predict = 50)
    
    # Sentiment  
    prompt2 <- paste("Is this positive, negative or neutral?", text, "\nAnswer:")
    sentiment <- edge_completion(ctx, prompt2, n_predict = 10)
    
    results[[i]] <- list(
      original = text,
      summary = trimws(gsub(".*Summary:", "", summary)),
      sentiment = trimws(gsub(".*Answer:", "", sentiment))
    )
  }
  
  edge_free_model(ctx)
  return(results)
}

# Main execution
cat("Starting edgemodelr document analysis setup...\n")

# Step 1: Try manual setup
working_model <- manual_model_setup()

# Step 2: If we have a working model, demonstrate analysis
if (!is.null(working_model)) {
  cat("\n=== RUNNING ANALYSIS DEMO ===\n")
  
  sample_texts <- c(
    "This package is fantastic for local AI!",
    "Having some installation issues unfortunately.", 
    "Performance is okay but could be better."
  )
  
  # Use the working model for analysis
  tryCatch({
    results <- simple_document_analysis(sample_texts)
    
    cat("\nResults:\n")
    for (i in seq_along(results)) {
      cat("\nDocument", i, ":\n")
      cat("Text:", results[[i]]$original, "\n")
      cat("Summary:", results[[i]]$summary, "\n") 
      cat("Sentiment:", results[[i]]$sentiment, "\n")
    }
    
  }, error = function(e) {
    cat("Analysis failed:", e$message, "\n")
  })
} else {
  cat("\nNo working model found. Please follow manual setup instructions above.\n")
}