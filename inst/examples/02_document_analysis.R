#' Professional Document Analysis with Modern Small Models
#' 
#' This example demonstrates comprehensive document analysis using the latest
#' small language models optimized for local inference.
#' 
#' Features:
#' - Multi-model support with automatic fallback
#' - Comprehensive text analysis (summary, sentiment, topics, complexity)
#' - Batch processing capabilities
#' - Professional output formatting
#' - Error handling and validation
#' 
#' @author edgemodelr team
#' @date 2024

library(edgemodelr)

#' Advanced document analyzer class
#' @description A comprehensive document analysis system with multiple analysis types
DocumentAnalyzer <- function() {
  
  #' Initialize analyzer with model selection
  #' @param preferred_models character vector of models to try in order
  #' @param fallback_model character fallback model if others fail
  initialize <- function(preferred_models = c("Llama-3.2-1B", "Phi-3.5-Mini", "TinyLlama-1.1B"),
                        fallback_model = "TinyLlama-1.1B") {
    
    cat("🚀 Initializing Document Analyzer...\n")
    
    # Try preferred models in order
    for (model_name in preferred_models) {
      cat(sprintf("   Trying %s...\n", model_name))
      
      tryCatch({
        setup <- edge_quick_setup(model_name)
        
        # Test the model works
        test_result <- edge_completion(setup$context, "Test", n_predict = 2)
        
        cat(sprintf("   ✅ Using %s\n", model_name))
        
        return(list(
          model_name = model_name,
          context = setup$context,
          model_info = setup
        ))
        
      }, error = function(e) {
        cat(sprintf("   ❌ %s failed: %s\n", model_name, e$message))
      })
    }
    
    # Fallback to most reliable model
    cat(sprintf("   🔄 Falling back to %s\n", fallback_model))
    setup <- edge_quick_setup(fallback_model)
    
    return(list(
      model_name = fallback_model,
      context = setup$context, 
      model_info = setup
    ))
  }
  
  #' Generate text summary
  #' @param text character input text
  #' @param ctx model context
  #' @param max_sentences integer maximum sentences in summary
  generate_summary <- function(text, ctx, max_sentences = 3) {
    
    # Optimize prompt based on text length
    if (nchar(text) < 500) {
      prompt <- sprintf("Summarize in %d sentences: %s\n\nSummary:", max_sentences, text)
    } else {
      prompt <- sprintf("Write a %d-sentence summary of this text:\n\n%s\n\nSummary:", 
                       max_sentences, substr(text, 1, 1000))
    }
    
    result <- edge_completion(ctx, prompt, n_predict = 80, temperature = 0.7)
    
    # Clean and extract summary
    summary <- gsub(".*Summary:\\s*", "", result)
    summary <- trimws(strsplit(summary, "\n")[[1]][1])
    
    return(summary)
  }
  
  #' Analyze sentiment
  #' @param text character input text
  #' @param ctx model context
  analyze_sentiment <- function(text, ctx) {
    
    prompt <- sprintf("Analyze sentiment (positive/negative/neutral): %s\n\nSentiment:", text)
    result <- edge_completion(ctx, prompt, n_predict = 10, temperature = 0.3)
    
    # Extract and standardize sentiment
    sentiment_raw <- tolower(trimws(gsub(".*Sentiment:\\s*", "", result)))
    
    if (grepl("positive|good|great|excellent|amazing", sentiment_raw)) {
      sentiment <- "positive"
      confidence <- "high"
    } else if (grepl("negative|bad|poor|terrible|awful", sentiment_raw)) {
      sentiment <- "negative" 
      confidence <- "high"
    } else if (grepl("neutral|mixed|moderate", sentiment_raw)) {
      sentiment <- "neutral"
      confidence <- "high"
    } else {
      # Fallback analysis
      sentiment <- "neutral"
      confidence <- "low"
    }
    
    return(list(sentiment = sentiment, confidence = confidence))
  }
  
  #' Extract key topics
  #' @param text character input text  
  #' @param ctx model context
  extract_topics <- function(text, ctx) {
    
    prompt <- sprintf("List 3 main topics in this text: %s\n\nTopics:", text)
    result <- edge_completion(ctx, prompt, n_predict = 50, temperature = 0.5)
    
    # Extract topics
    topics_raw <- gsub(".*Topics:\\s*", "", result)
    topics <- trimws(strsplit(topics_raw, "[,\n]")[[1]])
    topics <- topics[topics != "" & nchar(topics) > 2]
    topics <- topics[1:min(3, length(topics))] # Limit to 3 topics
    
    return(topics)
  }
  
  #' Assess text complexity
  #' @param text character input text
  assess_complexity <- function(text) {
    
    # Calculate basic metrics
    word_count <- length(strsplit(text, "\\s+")[[1]])
    sentence_count <- length(strsplit(text, "[.!?]+")[[1]])
    avg_sentence_length <- word_count / max(sentence_count, 1)
    
    # Complexity indicators
    complex_words <- length(grep("[A-Za-z]{8,}", strsplit(text, "\\s+")[[1]]))
    complex_ratio <- complex_words / word_count
    
    # Classify complexity
    if (avg_sentence_length > 20 || complex_ratio > 0.15) {
      complexity <- "high"
    } else if (avg_sentence_length > 15 || complex_ratio > 0.10) {
      complexity <- "medium"
    } else {
      complexity <- "low"
    }
    
    return(list(
      complexity = complexity,
      word_count = word_count,
      sentence_count = sentence_count,
      avg_sentence_length = round(avg_sentence_length, 1),
      complex_ratio = round(complex_ratio, 3)
    ))
  }
  
  #' Analyze multiple documents
  #' @param texts character vector of input texts
  #' @param model_config list model configuration
  analyze_documents <- function(texts, model_config = NULL) {
    
    if (is.null(model_config)) {
      model_config <- initialize()
    }
    
    ctx <- model_config$context
    
    cat(sprintf("📊 Analyzing %d documents with %s...\n\n", 
                length(texts), model_config$model_name))
    
    results <- data.frame(
      document_id = seq_along(texts),
      original_text = texts,
      summary = character(length(texts)),
      sentiment = character(length(texts)),
      sentiment_confidence = character(length(texts)),
      topics = character(length(texts)),
      complexity = character(length(texts)),
      word_count = integer(length(texts)),
      processing_time = numeric(length(texts)),
      stringsAsFactors = FALSE
    )
    
    # Process each document
    for (i in seq_along(texts)) {
      start_time <- Sys.time()
      
      cat(sprintf("Processing document %d/%d...\n", i, length(texts)))
      
      text <- texts[i]
      
      # Generate summary
      results$summary[i] <- generate_summary(text, ctx)
      
      # Analyze sentiment
      sentiment_result <- analyze_sentiment(text, ctx)
      results$sentiment[i] <- sentiment_result$sentiment
      results$sentiment_confidence[i] <- sentiment_result$confidence
      
      # Extract topics
      topics <- extract_topics(text, ctx)
      results$topics[i] <- paste(topics, collapse = ", ")
      
      # Assess complexity
      complexity_result <- assess_complexity(text)
      results$complexity[i] <- complexity_result$complexity
      results$word_count[i] <- complexity_result$word_count
      
      # Record processing time
      end_time <- Sys.time()
      results$processing_time[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      # Progress update
      cat(sprintf("   ✓ Summary: %s...\n", 
                  substr(results$summary[i], 1, 50)))
      cat(sprintf("   ✓ Sentiment: %s (%s confidence)\n", 
                  results$sentiment[i], results$sentiment_confidence[i]))
      cat(sprintf("   ✓ Topics: %s\n", results$topics[i]))
      cat(sprintf("   ✓ Complexity: %s (%d words)\n\n", 
                  results$complexity[i], results$word_count[i]))
    }
    
    # Cleanup
    edge_free_model(ctx)
    
    return(results)
  }
  
  # Return analysis functions
  list(
    initialize = initialize,
    analyze_documents = analyze_documents,
    generate_summary = generate_summary,
    analyze_sentiment = analyze_sentiment,
    extract_topics = extract_topics,
    assess_complexity = assess_complexity
  )
}

#' Example usage and demonstration
main <- function() {
  cat("=== ADVANCED DOCUMENT ANALYSIS DEMO ===\n\n")
  
  # Create analyzer instance
  analyzer <- DocumentAnalyzer()
  
  # Sample documents with different characteristics
  sample_documents <- c(
    # Positive, simple
    "This new R package for local AI inference is absolutely fantastic! The documentation is clear, the installation is straightforward, and the performance is impressive. I've been able to run language models on my laptop without any issues. The examples are helpful and the community seems very supportive. I highly recommend this tool for anyone interested in local AI development.",
    
    # Negative, medium complexity
    "I'm experiencing significant difficulties with the model download functionality. The documentation lacks crucial details about troubleshooting network issues, and the error messages are not informative. After spending hours trying different approaches, I still cannot get the larger models to download properly. The package shows promise but needs better error handling and more comprehensive setup guides.",
    
    # Neutral, complex  
    "The performance characteristics of quantized language models represent a fundamental trade-off between computational efficiency and output quality. While 4-bit quantization schemes like Q4_K_M provide substantial memory savings and inference speed improvements, they introduce quantization artifacts that may impact downstream task performance. Empirical evaluations suggest that for most practical applications, the degradation is minimal, particularly when using modern quantization-aware training methodologies.",
    
    # Technical, positive
    "The integration of GGUF format models with R through this package opens up exciting possibilities for statistical computing workflows. The ability to perform local inference without cloud dependencies addresses significant privacy and latency concerns in enterprise environments. The package's design follows R conventions well, making it accessible to the existing R ecosystem.",
    
    # Mixed sentiment
    "The package has both strengths and weaknesses. On the positive side, it makes local LLM inference accessible to R users and supports a good variety of models. However, the setup process can be challenging, especially for users without deep technical knowledge. The performance is adequate for small models but struggles with larger ones on resource-constrained systems."
  )
  
  # Analyze documents
  results <- analyzer$analyze_documents(sample_documents)
  
  # Display results
  cat("=== ANALYSIS RESULTS ===\n\n")
  
  for (i in 1:nrow(results)) {
    cat(sprintf("📄 DOCUMENT %d\n", i))
    cat(sprintf("📝 Original: %s...\n", substr(results$original_text[i], 1, 100)))
    cat(sprintf("📋 Summary: %s\n", results$summary[i]))
    cat(sprintf("💭 Sentiment: %s (%s confidence)\n", 
                results$sentiment[i], results$sentiment_confidence[i]))
    cat(sprintf("🏷️  Topics: %s\n", results$topics[i]))
    cat(sprintf("🎯 Complexity: %s (%d words)\n", 
                results$complexity[i], results$word_count[i]))
    cat(sprintf("⏱️  Processing: %.2f seconds\n", results$processing_time[i]))
    cat(sprintf("%s\n\n", paste(rep("-", 80), collapse = "")))
  }
  
  # Summary statistics
  cat("=== SUMMARY STATISTICS ===\n")
  cat(sprintf("Total documents processed: %d\n", nrow(results)))
  cat(sprintf("Average processing time: %.2f seconds\n", mean(results$processing_time)))
  cat(sprintf("Sentiment distribution:\n"))
  sentiment_counts <- table(results$sentiment)
  for (sentiment in names(sentiment_counts)) {
    cat(sprintf("   %s: %d documents\n", sentiment, sentiment_counts[sentiment]))
  }
  
  complexity_counts <- table(results$complexity)
  cat(sprintf("Complexity distribution:\n"))
  for (complexity in names(complexity_counts)) {
    cat(sprintf("   %s: %d documents\n", complexity, complexity_counts[complexity]))
  }
  
  return(results)
}

# Export results to CSV
#' @param results data.frame analysis results
#' @param filename character output filename
export_results <- function(results, filename = "document_analysis_results.csv") {
  write.csv(results, filename, row.names = FALSE)
  cat(sprintf("Results exported to %s\n", filename))
}

# Run demonstration if executed directly
if (sys.nframe() == 0) {
  analysis_results <- main()
  
  # Optional: export results
  # export_results(analysis_results)
}