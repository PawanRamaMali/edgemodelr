# Advanced edgemodelr Usage Examples
# Performance optimization, multiple models, custom workflows

library(edgemodelr)

cat("🚀 Advanced edgemodelr Usage\n")
cat(rep("=", 60), "\n\n")

# Performance Optimization Techniques
demonstrate_performance_optimization <- function() {
  cat("⚡ Performance Optimization Techniques\n")
  cat(rep("-", 40), "\n\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Performance demo would run optimization tests here\n")
      return(NULL)
    }
    
    # Load a model for testing
    cat("Loading model for performance testing...\n")
    setup <- edge_quick_setup("TinyLlama-1.1B")
    
    if (is.null(setup$context)) {
      cat("❌ Model not available for performance testing\n")
      return(NULL)
    }
    
    ctx <- setup$context
    
    # Test 1: Context size impact
    cat("\n📏 Context Size Impact Test:\n")
    test_prompt <- "Explain machine learning in simple terms:"
    
    performance_results <- list()
    
    for (n_predict in c(20, 50, 100, 200)) {
      cat("Testing n_predict =", n_predict, "...")
      
      start_time <- Sys.time()
      result <- edge_completion(ctx, test_prompt, n_predict = n_predict, temperature = 0.5)
      end_time <- Sys.time()
      
      duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
      tokens_per_sec <- n_predict / duration
      
      performance_results[[paste0("n_predict_", n_predict)]] <- list(
        duration = duration,
        tokens_per_sec = tokens_per_sec,
        result_length = nchar(result)
      )
      
      cat(" ✅", round(duration, 2), "s (", round(tokens_per_sec, 1), " tok/s)\n")
    }
    
    # Test 2: Temperature impact on speed
    cat("\n🌡️ Temperature Impact Test:\n")
    for (temp in c(0.1, 0.5, 0.9)) {
      cat("Testing temperature =", temp, "...")
      
      start_time <- Sys.time()
      result <- edge_completion(ctx, test_prompt, n_predict = 50, temperature = temp)
      end_time <- Sys.time()
      
      duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
      cat(" ✅", round(duration, 2), "seconds\n")
    }
    
    # Performance recommendations
    cat("\n💡 Performance Recommendations:\n")
    cat("- Use smaller n_predict values for faster responses\n")
    cat("- Lower temperatures (0.1-0.3) are slightly faster\n")
    cat("- Reuse model contexts instead of reloading\n")
    cat("- Consider model size vs performance trade-offs\n\n")
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Performance test error:", e$message, "\n")
  })
}

# Multiple Model Management
demonstrate_multiple_models <- function() {
  cat("🏭 Multiple Model Management\n")
  cat(rep("-", 40), "\n\n")
  
  if (!interactive()) {
    cat("Multiple model demo would show concurrent model usage\n")
    return(NULL)
  }
  
  models <- list()
  
  tryCatch({
    # Load different models for different tasks
    cat("Loading specialized models...\n")
    
    # Model 1: General purpose (TinyLlama)
    cat("- Loading general purpose model...\n")
    setup1 <- edge_quick_setup("TinyLlama-1.1B")
    if (!is.null(setup1$context)) {
      models$general <- setup1$context
      cat("  ✅ General model ready\n")
    }
    
    # Model 2: Try a different model if available
    cat("- Attempting to load specialized model...\n")
    # Note: In practice, you'd load different models here
    # For demo, we'll show the concept
    cat("  ℹ️ Would load a second model for comparison\n")
    
    if (length(models) > 0) {
      # Demonstrate model comparison
      cat("\n⚖️ Model Comparison Demo:\n")
      test_prompts <- c(
        "What is artificial intelligence?",
        "Write a haiku about programming:",
        "Explain the difference between mean and median:"
      )
      
      for (prompt in test_prompts) {
        cat("\nPrompt:", prompt, "\n")
        
        if (!is.null(models$general)) {
          result_general <- edge_completion(
            models$general, 
            prompt, 
            n_predict = 80, 
            temperature = 0.7
          )
          cat("General Model:", substr(result_general, 1, 100), "...\n")
        }
        
        # Would compare with second model here
        cat("Specialized Model: [Would show different model's response]\n")
      }
      
      # Model selection recommendations
      cat("\n🎯 Model Selection Guide:\n")
      cat("- TinyLlama: Good for testing, lightweight tasks\n")
      cat("- Llama 3.2-1B: Better reasoning, modern training\n")
      cat("- Llama 3.2-3B: High quality, needs more resources\n")
      cat("- Choose based on: task complexity, speed needs, hardware\n\n")
      
      # Clean up
      for (model_name in names(models)) {
        if (!is.null(models[[model_name]])) {
          edge_free_model(models[[model_name]])
          cat("✅ Cleaned up", model_name, "model\n")
        }
      }
    }
    
  }, error = function(e) {
    cat("❌ Multiple model demo error:", e$message, "\n")
    
    # Clean up any loaded models
    for (model_name in names(models)) {
      if (!is.null(models[[model_name]])) {
        edge_free_model(models[[model_name]])
      }
    }
  })
}

# Custom Workflow Builder
create_custom_workflow <- function(workflow_type = "research_assistant") {
  cat("🔧 Custom Workflow:", workflow_type, "\n")
  cat(rep("-", 40), "\n\n")
  
  workflow_configs <- list(
    research_assistant = list(
      model = "llama3.2-1b",
      temperature = 0.3,
      n_predict = 200,
      system_prompt = "You are a thorough research assistant. Provide accurate, well-sourced information."
    ),
    creative_writer = list(
      model = "llama3.2-1b", 
      temperature = 0.8,
      n_predict = 300,
      system_prompt = "You are a creative writing assistant. Be imaginative and engaging."
    ),
    code_reviewer = list(
      model = "llama3.2-1b",
      temperature = 0.2,
      n_predict = 250,
      system_prompt = "You are an expert code reviewer. Focus on best practices, bugs, and improvements."
    )
  )
  
  if (!workflow_type %in% names(workflow_configs)) {
    cat("❌ Unknown workflow type. Available:", paste(names(workflow_configs), collapse = ", "), "\n")
    return(NULL)
  }
  
  config <- workflow_configs[[workflow_type]]
  
  tryCatch({
    if (!interactive()) {
      cat("Custom workflow demo for:", workflow_type, "\n")
      cat("- Model:", config$model, "\n")
      cat("- Temperature:", config$temperature, "\n")
      cat("- Max tokens:", config$n_predict, "\n")
      cat("- System prompt:", config$system_prompt, "\n")
      return(NULL)
    }
    
    cat("Setting up", workflow_type, "workflow...\n")
    setup <- edge_quick_setup(config$model)
    
    if (is.null(setup$context)) {
      cat("❌ Workflow setup failed\n")
      return(NULL)
    }
    
    ctx <- setup$context
    cat("✅ Workflow ready!\n\n")
    
    # Demonstrate workflow-specific tasks
    if (workflow_type == "research_assistant") {
      demo_tasks <- list(
        "Explain the current state of quantum computing research",
        "What are the main challenges in renewable energy adoption?",
        "Summarize the key findings about machine learning in healthcare"
      )
    } else if (workflow_type == "creative_writer") {
      demo_tasks <- list(
        "Write an opening paragraph for a mystery novel",
        "Create a dialogue between a robot and a human child",
        "Describe a magical forest in vivid detail"
      )
    } else if (workflow_type == "code_reviewer") {
      demo_tasks <- list(
        "Review this R function for potential improvements: function(x) { for(i in 1:length(x)) { if(x[i] > 0) return(i) } }",
        "What are best practices for error handling in R functions?",
        "Suggest optimizations for processing large datasets in R"
      )
    }
    
    cat("🎯 Demonstrating", workflow_type, "tasks:\n\n")
    
    for (i in seq_along(demo_tasks)) {
      task <- demo_tasks[[i]]
      cat("Task", i, ":", task, "\n")
      
      # Build prompt with system context
      full_prompt <- paste0(config$system_prompt, "\n\n", "User: ", task, "\n\nAssistant:")
      
      result <- edge_completion(
        ctx,
        full_prompt,
        n_predict = config$n_predict,
        temperature = config$temperature,
        top_p = 0.9
      )
      
      # Clean up response
      clean_result <- sub(".*Assistant:", "", result)
      clean_result <- trimws(clean_result)
      
      cat("Response:", substr(clean_result, 1, 200), "...\n\n")
      if (interactive()) Sys.sleep(1)
    }
    
    edge_free_model(ctx)
    cat("✅", workflow_type, "workflow demo complete\n")
    
  }, error = function(e) {
    cat("❌ Workflow error:", e$message, "\n")
  })
}

# Streaming Performance Analysis
analyze_streaming_performance <- function() {
  cat("📊 Streaming Performance Analysis\n")
  cat(rep("-", 40), "\n\n")
  
  if (!interactive()) {
    cat("Streaming analysis would measure token generation rates\n")
    return(NULL)
  }
  
  tryCatch({
    setup <- edge_quick_setup("TinyLlama-1.1B")
    if (is.null(setup$context)) return(NULL)
    
    ctx <- setup$context
    
    # Streaming performance measurement
    cat("Measuring streaming performance...\n")
    
    token_times <- c()
    token_count <- 0
    start_time <- Sys.time()
    first_token_time <- NULL
    
    performance_callback <- function(data) {
      current_time <- Sys.time()
      
      if (!data$is_final) {
        token_count <<- token_count + 1
        
        if (is.null(first_token_time)) {
          first_token_time <<- as.numeric(difftime(current_time, start_time, units = "secs"))
        }
        
        token_times <<- c(token_times, as.numeric(current_time))
        
        # Show progress
        if (token_count %% 10 == 0) {
          elapsed <- as.numeric(difftime(current_time, start_time, units = "secs"))
          rate <- token_count / elapsed
          cat("Tokens:", token_count, "Rate:", round(rate, 1), "tok/s\r")
          if (interactive()) utils::flush.console()
        }
      } else {
        cat("\n")
      }
      
      return(TRUE)
    }
    
    cat("Starting streaming generation...\n")
    result <- edge_stream_completion(
      ctx,
      "Write a detailed explanation of how neural networks learn:",
      callback = performance_callback,
      n_predict = 100,
      temperature = 0.7
    )
    
    total_time <- as.numeric(Sys.time() - start_time)
    
    # Calculate performance metrics
    cat("\n📈 Streaming Performance Results:\n")
    cat("- Total tokens generated:", token_count, "\n")
    cat("- Total time:", round(total_time, 2), "seconds\n")
    cat("- Average rate:", round(token_count / total_time, 2), "tokens/second\n")
    cat("- Time to first token:", round(first_token_time, 2), "seconds\n")
    cat("- Time per token (avg):", round(total_time / token_count * 1000, 1), "ms\n\n")
    
    # Performance insights
    cat("💡 Performance Insights:\n")
    if (first_token_time < 1.0) {
      cat("✅ Excellent first token latency\n")
    } else if (first_token_time < 3.0) {
      cat("⚠️ Moderate first token latency\n")
    } else {
      cat("🐌 High first token latency - consider smaller context or model\n")
    }
    
    avg_rate <- token_count / total_time
    if (avg_rate > 20) {
      cat("✅ Excellent generation speed\n")
    } else if (avg_rate > 10) {
      cat("⚠️ Moderate generation speed\n") 
    } else {
      cat("🐌 Low generation speed - check hardware/model size\n")
    }
    
    edge_free_model(ctx)
    
  }, error = function(e) {
    cat("❌ Streaming analysis error:", e$message, "\n")
  })
}

# Memory Management Best Practices
demonstrate_memory_management <- function() {
  cat("💾 Memory Management Best Practices\n")
  cat(rep("-", 40), "\n\n")
  
  cat("📋 Memory Management Guidelines:\n\n")
  
  cat("1. Model Loading:\n")
  cat("   ✅ DO: Load models once, reuse contexts\n")
  cat("   ❌ DON'T: Reload models frequently\n\n")
  
  cat("2. Context Management:\n") 
  cat("   ✅ DO: Call edge_free_model() when done\n")
  cat("   ❌ DON'T: Leave contexts hanging\n\n")
  
  cat("3. Multiple Models:\n")
  cat("   ✅ DO: Load only what you need\n") 
  cat("   ❌ DON'T: Keep unused models in memory\n\n")
  
  cat("4. Context Size:\n")
  cat("   ✅ DO: Choose appropriate n_ctx for task\n")
  cat("   ❌ DON'T: Use maximum context unnecessarily\n\n")
  
  # Demonstrate proper patterns
  cat("🎯 Proper Usage Patterns:\n\n")
  
  # Pattern 1: Single session
  cat("Pattern 1 - Single Session:\n")
  cat("```r\n")
  cat("ctx <- edge_load_model('model.gguf')\n")
  cat("result1 <- edge_completion(ctx, 'prompt1')\n") 
  cat("result2 <- edge_completion(ctx, 'prompt2')\n")
  cat("edge_free_model(ctx)  # Clean up\n")
  cat("```\n\n")
  
  # Pattern 2: Function wrapper
  cat("Pattern 2 - Function Wrapper:\n")
  cat("```r\n")
  cat("analyze_with_ai <- function(texts) {\n")
  cat("  ctx <- edge_load_model('model.gguf')\n")
  cat("  on.exit(edge_free_model(ctx))  # Always clean up\n")
  cat("  \n")
  cat("  results <- lapply(texts, function(text) {\n")
  cat("    edge_completion(ctx, paste('Analyze:', text))\n")
  cat("  })\n")
  cat("  \n")
  cat("  return(results)\n")
  cat("}\n")
  cat("```\n\n")
  
  # Pattern 3: Model manager
  cat("Pattern 3 - Model Manager Class:\n")
  cat("```r\n") 
  cat("model_manager <- list(\n")
  cat("  ctx = NULL,\n")
  cat("  load = function(path) { self$ctx <- edge_load_model(path) },\n")
  cat("  generate = function(prompt) { edge_completion(self$ctx, prompt) },\n")
  cat("  cleanup = function() { edge_free_model(self$ctx); self$ctx <- NULL }\n")
  cat(")\n")
  cat("```\n\n")
  
  cat("💡 Cache Management:\n")
  cat("- Use edge_clean_cache() regularly\n")
  cat("- Monitor cache size with tools::R_user_dir('edgemodelr', 'cache')\n")
  cat("- Set reasonable cache limits\n\n")
}

# Main demonstration
demo_advanced_usage <- function() {
  cat("🎬 Advanced Usage Demo\n")
  cat(rep("-", 40), "\n\n")
  
  # Run all advanced demos
  demonstrate_performance_optimization()
  if (interactive()) Sys.sleep(2)
  
  demonstrate_multiple_models()
  if (interactive()) Sys.sleep(2)
  
  create_custom_workflow("research_assistant")
  if (interactive()) Sys.sleep(2)
  
  analyze_streaming_performance()
  if (interactive()) Sys.sleep(2)
  
  demonstrate_memory_management()
  
  cat("✅ Advanced usage demo complete!\n")
}

# Main execution
if (interactive()) {
  cat("Choose advanced demo:\n")
  cat("1. demo_advanced_usage() - Complete advanced demo\n")
  cat("2. demonstrate_performance_optimization() - Performance testing\n")
  cat("3. demonstrate_multiple_models() - Multiple model management\n")
  cat("4. create_custom_workflow('type') - Custom workflows\n")
  cat("5. analyze_streaming_performance() - Streaming analysis\n")
  cat("6. demonstrate_memory_management() - Memory best practices\n\n")
  cat("Running complete demo:\n")
  demo_advanced_usage()
} else {
  demo_advanced_usage()
}