library(edgemodelr)

# Document analysis function with better error handling and smaller model
analyze_documents <- function(texts, model_name = "TinyLlama-1.1B") {
  cat("Setting up model:", model_name, "\n")
  
  # Try to use existing model first, fallback to download if needed
  tryCatch({
    setup <- edge_quick_setup(model_name)
    ctx <- setup$context
  }, error = function(e) {
    cat("Setup failed, trying alternative approach...\n")
    
    # List available models and try a different one
    models <- edge_list_models()
    cat("Available models:\n")
    print(models[1:3, c("name", "size")])  # Show first 3 models
    
    # Try with TinyLlama (smaller, more reliable)
    setup <- edge_quick_setup("TinyLlama-1.1B")
    ctx <<- setup$context
  })
  
  results <- data.frame(
    document_id = seq_along(texts),
    original_text = texts,
    summary = character(length(texts)),
    sentiment = character(length(texts)),
    key_topics = character(length(texts)),
    stringsAsFactors = FALSE
  )
  
  for (i in seq_along(texts)) {
    cat("Processing document", i, "of", length(texts), "\n")
    
    # Generate summary - simpler prompt for smaller models
    summary_prompt <- paste("Text:", texts[i], "\n\nSummary (2-3 sentences):")
    summary_result <- edge_completion(ctx, summary_prompt, n_predict = 80, temperature = 0.7)
    # Extract just the summary part after the prompt
    summary_clean <- sub(paste0(".*", "Summary \\(2-3 sentences\\):"), "", summary_result)
    results$summary[i] <- trimws(summary_clean)
    
    # Analyze sentiment with simple prompt
    sentiment_prompt <- paste("Text:", texts[i], "\n\nSentiment (positive/negative/neutral):")
    sentiment_result <- edge_completion(ctx, sentiment_prompt, n_predict = 5, temperature = 0.3)
    # Extract sentiment
    sentiment_clean <- tolower(trimws(sub(".*Sentiment \\(positive/negative/neutral\\):", "", sentiment_result)))
    # Clean up to just get the sentiment word
    if (grepl("positive", sentiment_clean)) {
      results$sentiment[i] <- "positive"
    } else if (grepl("negative", sentiment_clean)) {
      results$sentiment[i] <- "negative"  
    } else {
      results$sentiment[i] <- "neutral"
    }
    
    # Extract key topics (simple approach)
    topics_prompt <- paste("Text:", texts[i], "\n\nMain topic:")
    topics_result <- edge_completion(ctx, topics_prompt, n_predict = 15, temperature = 0.5)
    topics_clean <- trimws(sub(".*Main topic:", "", topics_result))
    results$key_topics[i] <- topics_clean
    
    cat("  - Summary:", substr(results$summary[i], 1, 50), "...\n")
    cat("  - Sentiment:", results$sentiment[i], "\n")
    cat("  - Topic:", results$key_topics[i], "\n\n")
  }
  
  edge_free_model(ctx)
  return(results)
}

# Example usage with better error handling
main <- function() {
  sample_texts <- c(
    "This new R package is amazing! It makes local LLM inference so easy and the documentation is great.",
    "I'm having trouble with the installation. The documentation could be clearer and there are dependency issues.",
    "The performance is decent but could be improved for larger models. Overall it's a useful tool."
  )
  
  cat("Starting document analysis...\n")
  
  tryCatch({
    analysis <- analyze_documents(sample_texts)
    
    cat("\n=== ANALYSIS RESULTS ===\n")
    for (i in 1:nrow(analysis)) {
      cat("\nDocument", i, ":\n")
      cat("Original:", analysis$original_text[i], "\n")
      cat("Summary:", analysis$summary[i], "\n") 
      cat("Sentiment:", analysis$sentiment[i], "\n")
      cat("Topic:", analysis$key_topics[i], "\n")
      cat("-" %&gt;% rep(50) %&gt;% paste(collapse = ""), "\n")
    }
    
    return(analysis)
    
  }, error = function(e) {
    cat("Error in document analysis:", e$message, "\n")
    cat("\nTroubleshooting suggestions:\n")
    cat("1. Check internet connection for model download\n")
    cat("2. Try manually downloading a smaller model\n") 
    cat("3. Use edge_list_models() to see available options\n")
  })
}

# Run the analysis
if (interactive()) {
  result <- main()
}