# Text Analysis and NLP Tasks with edgemodelr
# Comprehensive examples for natural language processing using local LLMs

library(edgemodelr)

cat("📝 Text Analysis and NLP with edgemodelr\n")
cat(rep("=", 60), "\n\n")

# Initialize model for text analysis
setup_text_analyzer <- function() {
  cat("Setting up text analysis model...\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Demo mode: Using placeholder responses\n")
      return(NULL)
    }
    
    # Use a model good for text understanding
    setup <- edge_quick_setup("llama3.2-1b")
    
    if (!is.null(setup$context)) {
      cat("✅ Text analyzer ready!\n\n")
      return(setup$context)
    } else {
      cat("❌ Failed to load model\n")
      return(NULL)
    }
    
  }, error = function(e) {
    cat("❌ Setup failed:", e$message, "\n")
    return(NULL)
  })
}

# Text Summarization
summarize_text <- function(ctx, text, max_sentences = 3) {
  if (is.null(ctx)) return("Analyzer not available")
  
  prompt <- paste0(
    "Summarize the following text in exactly ", max_sentences, " sentences:\n\n",
    text, "\n\n",
    "Summary:"
  )
  
  result <- edge_completion(
    ctx,
    prompt,
    n_predict = 150,
    temperature = 0.3,  # Factual summaries
    top_p = 0.9
  )
  
  summary <- sub(".*Summary:", "", result)
  return(trimws(summary))
}

# Sentiment Analysis
analyze_sentiment <- function(ctx, text) {
  if (is.null(ctx)) return("Analyzer not available")
  
  prompt <- paste0(
    "Analyze the sentiment of this text. Classify as Positive, Negative, or Neutral, ",
    "and provide a confidence score (1-10) and brief explanation.\n\n",
    "Text: ", text, "\n\n",
    "Sentiment:"
  )
  
  result <- edge_completion(
    ctx,
    prompt,
    n_predict = 100,
    temperature = 0.2,
    top_p = 0.8
  )
  
  sentiment <- sub(".*Sentiment:", "", result)
  return(trimws(sentiment))
}

# Topic Extraction
extract_topics <- function(ctx, text, num_topics = 3) {
  if (is.null(ctx)) return("Analyzer not available")
  
  prompt <- paste0(
    "Extract the top ", num_topics, " main topics or themes from this text. ",
    "List them as brief phrases:\n\n",
    text, "\n\n",
    "Topics:"
  )
  
  result <- edge_completion(
    ctx,
    prompt,
    n_predict = 80,
    temperature = 0.4,
    top_p = 0.9
  )
  
  topics <- sub(".*Topics:", "", result)
  return(trimws(topics))
}

# Named Entity Recognition (basic)
extract_entities <- function(ctx, text) {
  if (is.null(ctx)) return("Analyzer not available")
  
  prompt <- paste0(
    "Extract and categorize named entities from this text. ",
    "Categories: People, Organizations, Locations, Dates, Numbers.\n\n",
    text, "\n\n",
    "Entities:"
  )
  
  result <- edge_completion(
    ctx,
    prompt,
    n_predict = 120,
    temperature = 0.2,
    top_p = 0.8
  )
  
  entities <- sub(".*Entities:", "", result)
  return(trimws(entities))
}

# Text Classification
classify_text <- function(ctx, text, categories) {
  if (is.null(ctx)) return("Analyzer not available")
  
  categories_str <- paste(categories, collapse = ", ")
  
  prompt <- paste0(
    "Classify this text into one of these categories: ", categories_str, "\n\n",
    "Text: ", text, "\n\n",
    "Choose the best category and explain why:\n",
    "Classification:"
  )
  
  result <- edge_completion(
    ctx,
    prompt,
    n_predict = 100,
    temperature = 0.3,
    top_p = 0.9
  )
  
  classification <- sub(".*Classification:", "", result)
  return(trimws(classification))
}

# Question Answering
answer_question <- function(ctx, context_text, question) {
  if (is.null(ctx)) return("Analyzer not available")
  
  prompt <- paste0(
    "Based on the following context, answer the question. ",
    "If the answer is not in the context, say 'Not available in context'.\n\n",
    "Context: ", context_text, "\n\n",
    "Question: ", question, "\n\n",
    "Answer:"
  )
  
  result <- edge_completion(
    ctx,
    prompt,
    n_predict = 150,
    temperature = 0.2,
    top_p = 0.9
  )
  
  answer <- sub(".*Answer:", "", result)
  return(trimws(answer))
}

# Comprehensive Text Analysis Function
analyze_text_comprehensive <- function(ctx, text) {
  if (is.null(ctx)) {
    return(list(
      summary = "Demo mode - full analysis would include summary",
      sentiment = "Demo mode - sentiment analysis result",
      topics = "Demo mode - extracted topics",
      entities = "Demo mode - named entities",
      word_count = nchar(text),
      analysis_complete = FALSE
    ))
  }
  
  cat("Running comprehensive text analysis...\n")
  
  analysis <- list()
  
  # Basic stats
  analysis$word_count <- length(strsplit(text, "\\s+")[[1]])
  analysis$char_count <- nchar(text)
  
  # AI-powered analysis
  cat("- Generating summary...\n")
  analysis$summary <- summarize_text(ctx, text, 2)
  
  cat("- Analyzing sentiment...\n")
  analysis$sentiment <- analyze_sentiment(ctx, text)
  
  cat("- Extracting topics...\n") 
  analysis$topics <- extract_topics(ctx, text, 3)
  
  cat("- Finding entities...\n")
  analysis$entities <- extract_entities(ctx, text)
  
  analysis$analysis_complete <- TRUE
  return(analysis)
}

# Demo with sample texts
demo_text_analysis <- function() {
  cat("🎬 Text Analysis Demo\n")
  cat(rep("-", 40), "\n\n")
  
  ctx <- setup_text_analyzer()
  
  # Sample texts for analysis
  sample_texts <- list(
    news_article = "Apple Inc. announced record quarterly earnings today, with CEO Tim Cook praising the strong performance of iPhone sales globally. The technology giant reported revenue of $123 billion, exceeding analyst expectations. Cook stated during the earnings call that the company's focus on innovation and customer satisfaction continues to drive growth. Apple shares rose 5% in after-hours trading following the announcement.",
    
    product_review = "I've been using this laptop for three months now and I'm really impressed with its performance. The battery life is exceptional, lasting me through entire work days without needing to charge. The display is crisp and vibrant, perfect for both work and entertainment. My only complaint is that it gets a bit warm during intensive tasks, but overall I'm very satisfied with this purchase.",
    
    research_abstract = "This study examines the impact of machine learning algorithms on financial prediction accuracy. We analyzed data from 500 companies over a 10-year period using various ML approaches including random forests, neural networks, and support vector machines. Results indicate that ensemble methods provide the most robust predictions, with accuracy improvements of 15-20% over traditional statistical methods. These findings have significant implications for algorithmic trading and risk management in financial institutions."
  )
  
  # Analyze each text
  for (text_name in names(sample_texts)) {
    text <- sample_texts[[text_name]]
    
    cat("\n", rep("=", 50), "\n")
    cat("Analyzing:", toupper(gsub("_", " ", text_name)), "\n")
    cat(rep("=", 50), "\n")
    cat("Text preview:", substr(text, 1, 100), "...\n\n")
    
    # Run comprehensive analysis
    analysis <- analyze_text_comprehensive(ctx, text)
    
    # Display results
    cat("📊 ANALYSIS RESULTS:\n")
    cat("- Word count:", analysis$word_count, "\n")
    cat("- Character count:", analysis$char_count, "\n\n")
    
    if (analysis$analysis_complete) {
      cat("📝 Summary:\n", analysis$summary, "\n\n")
      cat("😊 Sentiment:\n", analysis$sentiment, "\n\n") 
      cat("🏷️ Topics:\n", analysis$topics, "\n\n")
      cat("👤 Entities:\n", analysis$entities, "\n\n")
    }
    
    if (interactive()) Sys.sleep(2)  # Pause between analyses
  }
  
  # Demo specific NLP tasks
  cat("\n", rep("=", 50), "\n")
  cat("SPECIALIZED NLP TASKS DEMO\n")
  cat(rep("=", 50), "\n")
  
  if (!is.null(ctx)) {
    # Text classification demo
    cat("\n🏷️ Text Classification Demo:\n")
    tech_text <- "The new smartphone features advanced AI capabilities and 5G connectivity."
    categories <- c("Technology", "Sports", "Politics", "Entertainment", "Health")
    classification <- classify_text(ctx, tech_text, categories)
    cat("Text:", tech_text, "\n")
    cat("Categories:", paste(categories, collapse = ", "), "\n")
    cat("Result:", classification, "\n\n")
    
    # Question answering demo
    cat("❓ Question Answering Demo:\n")
    context <- "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 324 meters tall."
    question <- "How tall is the Eiffel Tower?"
    answer <- answer_question(ctx, context, question)
    cat("Context:", context, "\n")
    cat("Question:", question, "\n") 
    cat("Answer:", answer, "\n\n")
  }
  
  # Clean up
  if (!is.null(ctx)) {
    edge_free_model(ctx)
    cat("✅ Analysis complete, model cleaned up\n")
  }
}

# Batch text processing function
batch_process_texts <- function(texts, output_file = NULL) {
  cat("📋 Batch Processing Multiple Texts\n")
  cat(rep("-", 40), "\n\n")
  
  ctx <- setup_text_analyzer()
  if (is.null(ctx)) return(NULL)
  
  results <- list()
  
  for (i in seq_along(texts)) {
    cat("Processing text", i, "of", length(texts), "\n")
    
    analysis <- analyze_text_comprehensive(ctx, texts[[i]])
    analysis$text_id <- i
    analysis$original_text <- texts[[i]]
    
    results[[i]] <- analysis
    
    cat("✅ Text", i, "complete\n")
  }
  
  # Save results if requested
  if (!is.null(output_file)) {
    saveRDS(results, output_file)
    cat("📁 Results saved to:", output_file, "\n")
  }
  
  edge_free_model(ctx)
  return(results)
}

# Interactive text analyzer
interactive_text_analyzer <- function() {
  cat("🤖 Interactive Text Analyzer\n")
  cat("Commands:\n")
  cat("- 'analyze [text]' for comprehensive analysis\n")
  cat("- 'summarize [text]' for summarization\n")
  cat("- 'sentiment [text]' for sentiment analysis\n") 
  cat("- 'topics [text]' for topic extraction\n")
  cat("- 'classify [categories] [text]' for classification\n")
  cat("- 'quit' to exit\n\n")
  
  ctx <- setup_text_analyzer()
  if (is.null(ctx)) return(NULL)
  
  while (TRUE) {
    user_input <- if (interactive()) readline("📝 Enter command: ") else "quit"
    
    if (tolower(trimws(user_input)) %in% c("quit", "exit")) {
      break
    }
    
    if (startsWith(user_input, "analyze ")) {
      text <- substring(user_input, 9)
      analysis <- analyze_text_comprehensive(ctx, text)
      print(analysis)
      
    } else if (startsWith(user_input, "summarize ")) {
      text <- substring(user_input, 11)
      summary <- summarize_text(ctx, text)
      cat("Summary:", summary, "\n")
      
    } else if (startsWith(user_input, "sentiment ")) {
      text <- substring(user_input, 11)
      sentiment <- analyze_sentiment(ctx, text)
      cat("Sentiment:", sentiment, "\n")
      
    } else if (startsWith(user_input, "topics ")) {
      text <- substring(user_input, 8)
      topics <- extract_topics(ctx, text)
      cat("Topics:", topics, "\n")
      
    } else if (startsWith(user_input, "classify ")) {
      # Parse categories and text
      parts <- substring(user_input, 10)
      cat("Format: classify [category1,category2,...] [text]\n")
      
    } else {
      cat("Unknown command. Try 'analyze [your text]'\n")
    }
    
    cat("\n")
  }
  
  edge_free_model(ctx)
  cat("👋 Text analyzer session ended\n")
}

# Main execution
if (interactive()) {
  cat("Choose your option:\n")
  cat("1. demo_text_analysis() - Run comprehensive demo\n")
  cat("2. interactive_text_analyzer() - Interactive mode\n")
  cat("3. batch_process_texts(your_texts) - Process multiple texts\n\n")
  cat("Running demo automatically:\n")
  demo_text_analysis()
} else {
  demo_text_analysis()
}