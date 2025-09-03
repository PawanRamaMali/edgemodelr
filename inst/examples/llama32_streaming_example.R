# Llama 3.2 Streaming Example
# Demonstrates real-time streaming text generation with the latest Llama 3.2 model

library(edgemodelr)

# Simple streaming callback that prints tokens in real-time
streaming_callback <- function(data) {
  if (!data$is_final) {
    cat(data$token, sep = "")
    flush.console()  # Ensure immediate output
  }
  return(TRUE)  # Continue streaming
}

# Advanced streaming callback with token counting and timing
advanced_callback <- function() {
  start_time <- Sys.time()
  token_count <- 0
  
  return(function(data) {
    if (!data$is_final) {
      token_count <<- token_count + 1
      cat(data$token, sep = "")
      flush.console()
      
      # Show progress every 20 tokens
      if (token_count %% 20 == 0) {
        elapsed <- as.numeric(Sys.time() - start_time)
        tokens_per_sec <- token_count / elapsed
        cat(sprintf("\n[%d tokens, %.1f tok/s]", token_count, tokens_per_sec))
        flush.console()
      }
    }
    
    return(TRUE)
  })
}

# Demo function for streaming completion
demo_streaming_completion <- function() {
  cat("🚀 Llama 3.2 Streaming Completion Demo\n")
  cat(rep("=", 50), "\n\n")
  
  tryCatch({
    # Setup Llama 3.2 model
    cat("Loading Llama 3.2-1B model...\n")
    # Skip downloads during automated testing
    if (!interactive()) {
      stop("Skipping model download during automated testing")
    }
    setup <- edge_quick_setup("llama3.2-1b")
    
    if (is.null(setup$context)) {
      cat("❌ Failed to load Llama 3.2 model\n")
      return(NULL)
    }
    
    ctx <- setup$context
    cat("✅ Model loaded successfully\n\n")
    
    # Test prompts
    prompts <- c(
      "Write a short story about a robot learning to paint:",
      "Explain machine learning in simple terms:",
      "Create a recipe for chocolate chip cookies:",
      "Write a haiku about programming:"
    )
    
    for (i in seq_along(prompts)) {
      cat("\n", rep("-", 60), "\n")
      cat("Example", i, "- Streaming Response\n")
      cat(rep("-", 60), "\n")
      cat("Prompt:", prompts[i], "\n\n")
      cat("Response: ")
      
      # Stream the completion
      edge_stream_completion(
        ctx, 
        prompts[i],
        n_predict = 150,
        temperature = 0.7,
        callback = streaming_callback
      )
      
      cat("\n✅ Streaming complete\n")
      Sys.sleep(1)
    }
    
    # Clean up
    edge_free_model(ctx)
    cat("\n✅ Model cleaned up\n")
    
  }, error = function(e) {
    cat("❌ Error:", e$message, "\n")
  })
}

# Interactive streaming chat with Llama 3.2
demo_streaming_chat <- function() {
  cat("💬 Llama 3.2 Interactive Streaming Chat\n")
  cat(rep("=", 50), "\n\n")
  
  tryCatch({
    # Setup model
    cat("Loading Llama 3.2-1B for chat...\n")
    # Skip downloads during automated testing
    if (!interactive()) {
      stop("Skipping model download during automated testing")
    }
    setup <- edge_quick_setup("llama3.2-1b")
    
    if (is.null(setup$context)) {
      cat("❌ Failed to load model\n")
      return(NULL)
    }
    
    ctx <- setup$context
    cat("✅ Chat ready! Type 'quit' to exit\n\n")
    
    # Sample conversation for demo
    demo_messages <- c(
      "Hello! What can you help me with?",
      "Tell me about the latest trends in AI",
      "Can you write a Python function to calculate fibonacci numbers?",
      "quit"
    )
    
    for (message in demo_messages) {
      cat("User:", message, "\n")
      
      if (message == "quit") {
        cat("Goodbye! 👋\n")
        break
      }
      
      cat("Assistant: ")
      
      # Use advanced callback with stats
      callback <- advanced_callback()
      
      edge_stream_completion(
        ctx,
        paste("User:", message, "\nAssistant:"),
        n_predict = 200,
        temperature = 0.8,
        top_p = 0.9,
        callback = callback
      )
      
      cat("\n\n")
      Sys.sleep(2)  # Pause between demo messages
    }
    
    # Clean up
    edge_free_model(ctx)
    cat("✅ Chat session ended\n")
    
  }, error = function(e) {
    cat("❌ Chat error:", e$message, "\n")
  })
}

# Performance comparison: streaming vs non-streaming
compare_streaming_performance <- function() {
  cat("⚡ Streaming vs Non-Streaming Performance\n")
  cat(rep("=", 50), "\n\n")
  
  tryCatch({
    # Skip downloads during automated testing
    if (!interactive()) {
      stop("Skipping model download during automated testing")
    }
    setup <- edge_quick_setup("llama3.2-1b")
    
    if (is.null(setup$context)) {
      cat("❌ Failed to load model\n")
      return(NULL)
    }
    
    ctx <- setup$context
    prompt <- "Write a detailed explanation of how neural networks work:"
    
    # Non-streaming (blocking)
    cat("🔄 Non-streaming generation:\n")
    start_time <- Sys.time()
    result <- edge_completion(ctx, prompt, n_predict = 200)
    blocking_time <- as.numeric(Sys.time() - start_time)
    cat("Result:", substr(result, 1, 100), "...\n")
    cat("Time:", round(blocking_time, 2), "seconds\n\n")
    
    # Streaming
    cat("⚡ Streaming generation:\n")
    start_time <- Sys.time()
    
    # Callback that measures first token time
    first_token_time <- NULL
    token_count <- 0
    
    streaming_timer_callback <- function(data) {
      if (!data$is_final) {
        token_count <<- token_count + 1
        if (is.null(first_token_time)) {
          first_token_time <<- as.numeric(Sys.time() - start_time)
        }
        cat(data$token, sep = "")
        flush.console()
      }
      return(TRUE)
    }
    
    edge_stream_completion(
      ctx,
      prompt, 
      n_predict = 200,
      callback = streaming_timer_callback
    )
    
    total_streaming_time <- as.numeric(Sys.time() - start_time)
    
    cat("\n\n📊 Performance Comparison:\n")
    cat("  Non-streaming total time:", round(blocking_time, 2), "seconds\n")
    cat("  Streaming total time:    ", round(total_streaming_time, 2), "seconds\n")
    cat("  First token time:        ", round(first_token_time, 2), "seconds\n")
    cat("  Tokens generated:        ", token_count, "\n")
    
    if (!is.null(first_token_time)) {
      cat("  ⚡ Streaming advantage: User sees output", 
          round((blocking_time - first_token_time), 2), 
          "seconds earlier!\n")
    }
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Performance test error:", e$message, "\n")
  })
}

# Creative writing with streaming
creative_writing_demo <- function() {
  cat("✨ Creative Writing with Llama 3.2 Streaming\n")
  cat(rep("=", 50), "\n\n")
  
  tryCatch({
    # Skip downloads during automated testing
    if (!interactive()) {
      stop("Skipping model download during automated testing")
    }
    setup <- edge_quick_setup("llama3.2-1b")
    
    if (is.null(setup$context)) {
      cat("❌ Failed to load model\n")
      return(NULL)
    }
    
    ctx <- setup$context
    
    creative_prompts <- c(
      "In a world where colors have sounds, write about a painter who discovers they can hear their artwork:",
      "Write a story from the perspective of the last book in a library that's closing down:",
      "Describe a day in the life of someone who can taste emotions:"
    )
    
    for (i in seq_along(creative_prompts)) {
      cat("\n📖 Creative Story", i, "\n")
      cat(rep("-", 40), "\n")
      cat("Prompt:", creative_prompts[i], "\n\n")
      
      # Use creative parameters
      edge_stream_completion(
        ctx,
        creative_prompts[i],
        n_predict = 300,
        temperature = 0.9,  # High creativity
        top_p = 0.8,
        callback = streaming_callback
      )
      
      cat("\n\n✨ Story complete!\n")
      Sys.sleep(2)
    }
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Creative writing error:", e$message, "\n")
  })
}

# Main menu function
main <- function() {
  cat("🦙 Llama 3.2 Streaming Examples\n")
  cat("Choose your demo:\n\n")
  cat("1. demo_streaming_completion() - Basic streaming examples\n")
  cat("2. demo_streaming_chat() - Interactive chat demo\n") 
  cat("3. compare_streaming_performance() - Performance comparison\n")
  cat("4. creative_writing_demo() - Creative writing with streaming\n\n")
  cat("Or run all demos with: run_all_demos()\n")
}

# Run all demos
run_all_demos <- function() {
  cat("🚀 Running All Llama 3.2 Streaming Demos\n\n")
  
  demo_streaming_completion()
  Sys.sleep(2)
  
  demo_streaming_chat() 
  Sys.sleep(2)
  
  compare_streaming_performance()
  Sys.sleep(2)
  
  creative_writing_demo()
  
  cat("\n🎉 All demos complete!\n")
}

# Auto-run when sourced
if (interactive()) {
  main()
} else {
  # Run basic demo when sourced non-interactively
  demo_streaming_completion()
}