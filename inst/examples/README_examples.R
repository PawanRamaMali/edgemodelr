# edgemodelr: Complete Usage Examples
# This file contains examples that mirror the README documentation

# ============================================================================
# Installation and Setup Examples
# ============================================================================

library(edgemodelr)

cat("📋 edgemodelr - Complete Usage Guide\n")
cat(rep("=", 60), "\n\n")

# Example 1: Quick Start (Recommended)
quick_start_example <- function() {
  cat("🚀 Quick Start Example\n")
  cat(rep("-", 30), "\n\n")
  
  # This is the easiest way to get started
  tryCatch({
    if (!interactive()) {
      cat("Demo: Quick setup would download and load TinyLlama model\n")
      return(NULL)
    }
    
    # One-line setup: downloads model and creates context
    setup <- edge_quick_setup("TinyLlama-1.1B")
    
    if (!is.null(setup$context)) {
      ctx <- setup$context
      
      # Generate text
      response <- edge_completion(ctx, "The R programming language is")
      cat("Generated text:", response, "\n")
      
      # Always clean up
      edge_free_model(ctx)
      cat("✅ Quick start complete!\n\n")
    }
    
  }, error = function(e) {
    cat("❌ Quick start failed:", e$message, "\n\n")
  })
}

# Example 2: Manual Setup (More Control)
manual_setup_example <- function() {
  cat("🔧 Manual Setup Example\n")
  cat(rep("-", 30), "\n\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Demo: Manual setup steps\n")
      cat("1. List available models\n")
      cat("2. Download specific model\n") 
      cat("3. Load with custom parameters\n")
      return(NULL)
    }
    
    # Step 1: See what models are available
    models <- edge_list_models()
    cat("Available models:\n")
    print(models[1:3, c("name", "size", "use_case")])
    
    # Step 2: Download a specific model
    model_path <- edge_download_model(
      model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
      filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    
    # Step 3: Load with custom parameters
    ctx <- edge_load_model(
      model_path = model_path,
      n_ctx = 1024,        # Context window size
      n_gpu_layers = 0     # CPU-only (set > 0 for GPU)
    )
    
    # Verify model loaded correctly
    if (is_valid_model(ctx)) {
      cat("✅ Model loaded successfully\n")
      
      # Use the model
      result <- edge_completion(ctx, "Hello world!", n_predict = 30)
      cat("Result:", result, "\n")
      
      # Clean up
      edge_free_model(ctx)
    }
    
  }, error = function(e) {
    cat("❌ Manual setup failed:", e$message, "\n")
  })
  
  cat("\n")
}

# Example 3: Text Generation with Different Parameters
text_generation_example <- function() {
  cat("📝 Text Generation Example\n")
  cat(rep("-", 30), "\n\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Demo: Text generation with various parameters\n")
      return(NULL)
    }
    
    setup <- edge_quick_setup("TinyLlama-1.1B")
    if (is.null(setup$context)) return(NULL)
    
    ctx <- setup$context
    prompt <- "Artificial intelligence is"
    
    # Conservative/Factual generation (low temperature)
    cat("🧊 Conservative generation (temperature = 0.2):\n")
    conservative <- edge_completion(
      ctx, prompt, 
      n_predict = 50, 
      temperature = 0.2,  # Low creativity
      top_p = 0.9
    )
    cat(conservative, "\n\n")
    
    # Balanced generation (medium temperature)
    cat("⚖️ Balanced generation (temperature = 0.7):\n")
    balanced <- edge_completion(
      ctx, prompt,
      n_predict = 50,
      temperature = 0.7,  # Medium creativity
      top_p = 0.9
    )
    cat(balanced, "\n\n")
    
    # Creative generation (high temperature)
    cat("🎨 Creative generation (temperature = 1.0):\n")
    creative <- edge_completion(
      ctx, prompt,
      n_predict = 50,
      temperature = 1.0,  # High creativity
      top_p = 0.8
    )
    cat(creative, "\n\n")
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Text generation example failed:", e$message, "\n")
  })
}

# Example 4: Streaming Text Generation
streaming_example <- function() {
  cat("⚡ Streaming Text Generation Example\n")
  cat(rep("-", 30), "\n\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Demo: Streaming would show real-time token generation\n")
      return(NULL)
    }
    
    setup <- edge_quick_setup("TinyLlama-1.1B")
    if (is.null(setup$context)) return(NULL)
    
    ctx <- setup$context
    
    cat("🌊 Watch text generate in real-time:\n")
    cat("Prompt: 'Write a short story about a robot learning to cook:'\n\n")
    
    # Simple streaming callback
    streaming_callback <- function(data) {
      if (!data$is_final) {
        cat(data$token)
        utils::flush.console()
        return(TRUE)  # Continue
      } else {
        cat("\n✅ Streaming complete!\n")
        return(TRUE)
      }
    }
    
    # Stream the response
    result <- edge_stream_completion(
      ctx,
      "Write a short story about a robot learning to cook:",
      callback = streaming_callback,
      n_predict = 150,
      temperature = 0.8
    )
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Streaming example failed:", e$message, "\n")
  })
  
  cat("\n")
}

# Example 5: Interactive Chat
chat_example <- function() {
  cat("💬 Interactive Chat Example\n")
  cat(rep("-", 30), "\n\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Demo: Interactive chat session would start here\n")
      cat("Features:\n")
      cat("- Streaming responses\n")
      cat("- Conversation history\n")
      cat("- Custom system prompts\n")
      return(NULL)
    }
    
    setup <- edge_quick_setup("TinyLlama-1.1B")
    if (is.null(setup$context)) return(NULL)
    
    ctx <- setup$context
    
    cat("Starting interactive chat...\n")
    cat("(In a real session, this would be interactive)\n\n")
    
    # Demo conversation
    demo_conversation <- list(
      "Hello, who are you?",
      "Can you help me with R programming?",
      "What's the difference between a list and a vector in R?"
    )
    
    for (message in demo_conversation) {
      cat("👤 User:", message, "\n")
      cat("🤖 Assistant: ")
      
      # Simulate streaming response
      response <- edge_completion(
        ctx, 
        paste("Human:", message, "\nAssistant:"),
        n_predict = 80,
        temperature = 0.7
      )
      
      # Clean up response
      clean_response <- gsub(".*Assistant:", "", response)
      cat(trimws(clean_response), "\n\n")
      
      if (interactive()) Sys.sleep(1)  # Pause for demo
    }
    
    cat("💡 To start real interactive chat:\n")
    cat("edge_chat_stream(ctx, system_prompt = 'You are a helpful R assistant')\n\n")
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Chat example failed:", e$message, "\n")
  })
}

# Example 6: Practical Use Cases
practical_examples <- function() {
  cat("🎯 Practical Use Cases\n")
  cat(rep("-", 30), "\n\n")
  
  if (!interactive()) {
    cat("Demo: Practical applications would be shown here\n")
    return(NULL)
  }
  
  tryCatch({
    setup <- edge_quick_setup("TinyLlama-1.1B")
    if (is.null(setup$context)) return(NULL)
    
    ctx <- setup$context
    
    # Use Case 1: Code Documentation
    cat("📄 Use Case 1: Code Documentation\n")
    r_function <- "
    calculate_mean <- function(x, na.rm = TRUE) {
      if (!is.numeric(x)) stop('Input must be numeric')
      sum(x, na.rm = na.rm) / length(x[!is.na(x)])
    }"
    
    doc_prompt <- paste0(
      "Write documentation for this R function:\n",
      r_function, "\n\n",
      "Documentation:"
    )
    
    documentation <- edge_completion(ctx, doc_prompt, n_predict = 100, temperature = 0.3)
    cat("Generated documentation:\n", documentation, "\n\n")
    
    # Use Case 2: Data Analysis Explanation
    cat("📊 Use Case 2: Data Analysis Explanation\n")
    analysis_prompt <- "Explain what a correlation coefficient tells us about data in simple terms:"
    
    explanation <- edge_completion(ctx, analysis_prompt, n_predict = 100, temperature = 0.4)
    cat("Explanation:\n", explanation, "\n\n")
    
    # Use Case 3: Error Message Help
    cat("❌ Use Case 3: Error Message Help\n")
    error_prompt <- "I got this R error: 'object of type 'closure' is not subsettable'. What does this mean and how do I fix it?"
    
    help_response <- edge_completion(ctx, error_prompt, n_predict = 120, temperature = 0.3)
    cat("Help response:\n", help_response, "\n\n")
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Practical examples failed:", e$message, "\n")
  })
}

# Example 7: Best Practices
best_practices_example <- function() {
  cat("💡 Best Practices Example\n")
  cat(rep("-", 30), "\n\n")
  
  cat("📋 Performance Best Practices:\n")
  cat("1. Reuse model contexts instead of reloading\n")
  cat("2. Choose appropriate n_predict values\n")
  cat("3. Use lower temperature for factual tasks\n")
  cat("4. Always call edge_free_model() when done\n\n")
  
  # Demonstrate proper resource management
  cat("🔒 Resource Management Pattern:\n")
  cat("```r\n")
  cat("# Good practice: wrap in function with cleanup\n")
  cat("analyze_texts <- function(texts) {\n")
  cat("  ctx <- edge_load_model('model.gguf')\n") 
  cat("  on.exit(edge_free_model(ctx))  # Always cleanup\n")
  cat("  \n")
  cat("  results <- sapply(texts, function(text) {\n")
  cat("    edge_completion(ctx, paste('Analyze:', text))\n")
  cat("  })\n")
  cat("  \n") 
  cat("  return(results)\n")
  cat("}\n")
  cat("```\n\n")
  
  cat("🗂️ Cache Management:\n")
  cat("- Use edge_clean_cache() to manage disk space\n")
  cat("- Default cache: tools::R_user_dir('edgemodelr', 'cache')\n")
  cat("- Monitor cache size regularly\n\n")
  
  cat("⚙️ Model Selection Guide:\n")
  models_guide <- edge_list_models()[1:4, c("name", "size", "use_case")]
  print(models_guide)
  cat("\n")
}

# Main demonstration function
run_all_examples <- function() {
  cat("🎬 Running All Examples\n")
  cat(rep("=", 60), "\n\n")
  
  quick_start_example()
  Sys.sleep(1)
  
  manual_setup_example()
  Sys.sleep(1)
  
  text_generation_example()
  Sys.sleep(1)
  
  streaming_example()
  Sys.sleep(1)
  
  chat_example()
  Sys.sleep(1)
  
  practical_examples()
  Sys.sleep(1)
  
  best_practices_example()
  
  cat("🎉 All examples complete!\n")
  cat("\nNext steps:\n")
  cat("- Try the interactive examples\n")
  cat("- Explore other example files in inst/examples/\n")
  cat("- Check the package documentation: ?edgemodelr\n")
}

# Individual example menu
show_examples_menu <- function() {
  cat("Choose an example to run:\n\n")
  cat("1. quick_start_example() - Get started in 3 lines\n")
  cat("2. manual_setup_example() - Full control setup\n")
  cat("3. text_generation_example() - Different generation styles\n")
  cat("4. streaming_example() - Real-time text generation\n") 
  cat("5. chat_example() - Interactive conversation\n")
  cat("6. practical_examples() - Real-world use cases\n")
  cat("7. best_practices_example() - Performance tips\n")
  cat("8. run_all_examples() - Run everything\n\n")
  cat("Or explore other comprehensive examples:\n")
  cat("- source(system.file('examples/data_science_assistant_example.R', package='edgemodelr'))\n")
  cat("- source(system.file('examples/text_analysis_example.R', package='edgemodelr'))\n")
  cat("- source(system.file('examples/creative_writing_example.R', package='edgemodelr'))\n")
}

# Auto-run based on context
if (interactive()) {
  show_examples_menu()
  cat("\nRunning quick demo:\n")
  quick_start_example()
} else {
  # Run a basic demo when sourced
  cat("edgemodelr examples loaded. Available functions:\n")
  cat("- quick_start_example()\n")
  cat("- manual_setup_example()\n") 
  cat("- run_all_examples()\n")
  cat("- show_examples_menu()\n")
}