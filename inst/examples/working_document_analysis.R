# Working Document Analysis with TinyLlama
# This uses the smaller model you already have downloaded

library(edgemodelr)

analyze_documents <- function(texts, model_name = "TinyLlama-1.1B") {
  cat("Setting up", model_name, "for document analysis...\n")
  
  # Use TinyLlama which you already have
  setup <- edge_quick_setup(model_name)
  ctx <- setup$context
  
  results <- data.frame(
    document_id = seq_along(texts),
    original_text = texts,
    summary = character(length(texts)),
    sentiment = character(length(texts)),
    stringsAsFactors = FALSE
  )
  
  for (i in seq_along(texts)) {
    cat("Processing document", i, "of", length(texts), "\n")
    
    # Generate summary with cleaner prompts for TinyLlama
    summary_prompt <- paste0(
      "<s>[INST] Summarize this text in 1-2 sentences: \"", 
      texts[i], 
      "\" [/INST]"
    )
    
    summary_raw <- edge_completion(ctx, summary_prompt, n_predict = 60, temperature = 0.7)
    # Clean up the summary (remove the prompt part)
    summary_clean <- gsub(".*\\[/INST\\]\\s*", "", summary_raw)
    summary_clean <- trimws(strsplit(summary_clean, "\n")[[1]][1]) # Take first line only
    results$summary[i] <- summary_clean
    
    # Analyze sentiment with simple prompt
    sentiment_prompt <- paste0(
      "<s>[INST] Is this text positive, negative, or neutral? Text: \"", 
      texts[i], 
      "\" Answer with just one word. [/INST]"
    )
    
    sentiment_raw <- edge_completion(ctx, sentiment_prompt, n_predict = 5, temperature = 0.3)
    sentiment_clean <- tolower(trimws(gsub(".*\\[/INST\\]\\s*", "", sentiment_raw)))
    
    # Extract just the sentiment word
    if (grepl("positive", sentiment_clean)) {
      results$sentiment[i] <- "positive"
    } else if (grepl("negative", sentiment_clean)) {
      results$sentiment[i] <- "negative"  
    } else {
      results$sentiment[i] <- "neutral"
    }
    
    # Show progress
    cat("  Summary:", results$summary[i], "\n")
    cat("  Sentiment:", results$sentiment[i], "\n\n")
  }
  
  edge_free_model(ctx)
  return(results)
}

# Your original example texts
sample_texts <- c(
  "This new R package is amazing! It makes local LLM inference so easy.",
  "I'm having trouble with the installation. The documentation could be clearer.",
  "The performance is decent but could be improved for larger models."
)

cat("=== DOCUMENT ANALYSIS DEMO ===\n\n")

# Run the analysis with TinyLlama (which you already have)
analysis <- analyze_documents(sample_texts)

# Display results nicely
cat("=== FINAL RESULTS ===\n")
for (i in 1:nrow(analysis)) {
  cat("\n--- Document", i, "---\n")
  cat("Original Text:\n", analysis$original_text[i], "\n\n")
  cat("Summary:\n", analysis$summary[i], "\n\n") 
  cat("Sentiment:", analysis$sentiment[i], "\n")
  cat(rep("-", 50), "\n")
}

# Also return as data frame
print("Analysis complete! Results:")
print(analysis[, c("document_id", "summary", "sentiment")])