#' Comprehensive Model Benchmarking for Small Language Models
#'
#' This example provides systematic benchmarking and evaluation tools for
#' comparing different small language models on various tasks and metrics.
#'
#' Features:
#' - Performance benchmarking (speed, memory, accuracy)
#' - Task-specific evaluations (QA, summarization, coding)
#' - Resource usage monitoring
#' - Comparative analysis
#' - Export results to various formats
#' - Automated model recommendations
#'
#' @author edgemodelr team
#' @date 2024

library(edgemodelr)

#' Model Benchmarking System
#' @description Comprehensive benchmarking tools for model evaluation
ModelBenchmarker <- function() {
  
  #' Initialize benchmarking system
  initialize <- function() {
    cat("🏁 Initializing Model Benchmarking System...\n\n")
    
    # Define benchmark tasks
    tasks <- list(
      "question_answer" = list(
        name = "Question Answering",
        description = "Evaluate factual question answering capability"
      ),
      "summarization" = list(
        name = "Text Summarization", 
        description = "Evaluate text summarization quality and conciseness"
      ),
      "sentiment" = list(
        name = "Sentiment Analysis",
        description = "Evaluate sentiment classification accuracy"
      ),
      "creativity" = list(
        name = "Creative Writing",
        description = "Evaluate creative content generation ability"
      ),
      "coding" = list(
        name = "Code Generation",
        description = "Evaluate programming code generation capability"
      )
    )
    
    return(list(tasks = tasks, results = list()))
  }
  
  #' Get benchmark test cases
  #' @param task_type character type of benchmark task
  get_test_cases <- function(task_type) {
    
    test_cases <- switch(task_type,
      "question_answer" = list(
        list(prompt = "What is the capital of France?", 
             expected_keywords = c("Paris"), 
             max_tokens = 20),
        list(prompt = "How many days are in a leap year?", 
             expected_keywords = c("366"), 
             max_tokens = 20),
        list(prompt = "What programming language was developed by Guido van Rossum?", 
             expected_keywords = c("Python"), 
             max_tokens = 20)
      ),
      
      "summarization" = list(
        list(prompt = "Summarize in 2 sentences: Artificial intelligence has transformed many industries by automating tasks that previously required human intelligence. From healthcare diagnosis to financial analysis, AI systems can process vast amounts of data quickly and accurately. However, this transformation also raises concerns about job displacement and the need for workers to adapt to new technologies.", 
             expected_length = c(20, 60), 
             max_tokens = 80),
        list(prompt = "Summarize briefly: Climate change refers to long-term shifts in global temperatures and weather patterns. While some climate variations are natural, scientific evidence shows that human activities, particularly greenhouse gas emissions, are the primary driver of recent climate change.", 
             expected_length = c(15, 40), 
             max_tokens = 60)
      ),
      
      "sentiment" = list(
        list(prompt = "Analyze sentiment: I absolutely love this new product! It's amazing and works perfectly.", 
             expected_sentiment = "positive", 
             max_tokens = 10),
        list(prompt = "Analyze sentiment: This service is terrible and completely disappointing.", 
             expected_sentiment = "negative", 
             max_tokens = 10),
        list(prompt = "Analyze sentiment: The weather today is okay, nothing special.", 
             expected_sentiment = "neutral", 
             max_tokens = 10)
      ),
      
      "creativity" = list(
        list(prompt = "Write a creative opening line for a story about space exploration.", 
             min_length = 10, 
             max_tokens = 50),
        list(prompt = "Create an imaginative description of a futuristic city.", 
             min_length = 20, 
             max_tokens = 80)
      ),
      
      "coding" = list(
        list(prompt = "Write a Python function to calculate factorial of a number.", 
             expected_keywords = c("def", "factorial"), 
             max_tokens = 100),
        list(prompt = "Create an R function to find the mean of a vector.", 
             expected_keywords = c("function", "mean"), 
             max_tokens = 80)
      ),
      
      list()  # default empty
    )
    
    return(test_cases)
  }
  
  #' Benchmark a single model
  #' @param model_name character name of model to benchmark
  #' @param tasks character vector of tasks to run
  #' @param iterations integer number of iterations per task
  benchmark_model <- function(model_name, tasks = c("question_answer", "sentiment"), iterations = 1) {
    
    cat(sprintf("🧪 Benchmarking %s...\n", model_name))
    
    # Initialize model
    tryCatch({
      setup <- edge_quick_setup(model_name)
      ctx <- setup$context
    }, error = function(e) {
      cat(sprintf("❌ Failed to initialize %s: %s\n", model_name, e$message))
      return(NULL)
    })
    
    model_results <- list(
      model_name = model_name,
      benchmark_time = Sys.time(),
      task_results = list(),
      overall_metrics = list()
    )
    
    total_start_time <- Sys.time()
    
    # Run each task
    for (task_type in tasks) {
      cat(sprintf("   📋 Running %s task...\n", task_type))
      
      test_cases <- get_test_cases(task_type)
      task_results <- list()
      
      for (i in seq_along(test_cases)) {
        case <- test_cases[[i]]
        
        case_results <- list()
        response_times <- numeric()
        responses <- character()
        
        # Run multiple iterations
        for (iter in 1:iterations) {
          start_time <- Sys.time()
          
          tryCatch({
            response <- edge_completion(ctx, case$prompt, 
                                      n_predict = case$max_tokens, 
                                      temperature = 0.7)
            
            end_time <- Sys.time()
            response_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
            
            response_times <- c(response_times, response_time)
            responses <- c(responses, response)
            
          }, error = function(e) {
            cat(sprintf("      ⚠️ Error in iteration %d: %s\n", iter, e$message))
            response_times <<- c(response_times, NA)
            responses <<- c(responses, "")
          })
        }
        
        # Evaluate responses
        evaluation <- evaluate_responses(responses, case, task_type)
        
        case_results[[i]] <- list(
          prompt = case$prompt,
          responses = responses,
          response_times = response_times,
          avg_response_time = mean(response_times, na.rm = TRUE),
          evaluation = evaluation
        )
        
        cat(sprintf("      ✓ Case %d: %.2fs avg, Score: %.2f\n", 
                    i, mean(response_times, na.rm = TRUE), evaluation$score))
      }
      
      # Aggregate task results
      task_summary <- list(
        task_type = task_type,
        cases = case_results,
        avg_response_time = mean(sapply(case_results, function(x) x$avg_response_time), na.rm = TRUE),
        avg_score = mean(sapply(case_results, function(x) x$evaluation$score), na.rm = TRUE)
      )
      
      model_results$task_results[[task_type]] <- task_summary
      
      cat(sprintf("   ✅ %s completed: %.2fs avg, %.2f avg score\n", 
                  task_type, task_summary$avg_response_time, task_summary$avg_score))
    }
    
    total_end_time <- Sys.time()
    total_time <- as.numeric(difftime(total_end_time, total_start_time, units = "secs"))
    
    # Calculate overall metrics
    model_results$overall_metrics <- list(
      total_benchmark_time = total_time,
      avg_response_time = mean(sapply(model_results$task_results, function(x) x$avg_response_time)),
      avg_score = mean(sapply(model_results$task_results, function(x) x$avg_score)),
      tasks_completed = length(tasks),
      success_rate = calculate_success_rate(model_results$task_results)
    )
    
    # Cleanup
    edge_free_model(ctx)
    
    cat(sprintf("🏆 %s benchmark complete: %.2fs total, %.2f overall score\n\n", 
                model_name, total_time, model_results$overall_metrics$avg_score))
    
    return(model_results)
  }
  
  #' Evaluate response quality
  #' @param responses character vector of responses
  #' @param test_case list test case definition
  #' @param task_type character task type
  evaluate_responses <- function(responses, test_case, task_type) {
    
    valid_responses <- responses[responses != "" & !is.na(responses)]
    
    if (length(valid_responses) == 0) {
      return(list(score = 0, details = "No valid responses"))
    }
    
    # Use the best response for evaluation
    response <- valid_responses[1]
    
    score <- switch(task_type,
      "question_answer" = evaluate_qa_response(response, test_case),
      "summarization" = evaluate_summary_response(response, test_case),
      "sentiment" = evaluate_sentiment_response(response, test_case), 
      "creativity" = evaluate_creative_response(response, test_case),
      "coding" = evaluate_code_response(response, test_case),
      0.5  # default score
    )
    
    return(list(
      score = score,
      response_length = nchar(response),
      word_count = length(strsplit(response, "\\s+")[[1]]),
      details = sprintf("Evaluated %s task", task_type)
    ))
  }
  
  #' Evaluate QA response
  evaluate_qa_response <- function(response, test_case) {
    response_lower <- tolower(response)
    keywords_found <- sum(sapply(test_case$expected_keywords, 
                                function(kw) grepl(tolower(kw), response_lower)))
    score <- keywords_found / length(test_case$expected_keywords)
    return(min(score, 1.0))
  }
  
  #' Evaluate summary response
  evaluate_summary_response <- function(response, test_case) {
    word_count <- length(strsplit(response, "\\s+")[[1]])
    length_score <- if (word_count >= test_case$expected_length[1] && 
                       word_count <= test_case$expected_length[2]) 1.0 else 0.5
    
    # Basic quality check (has proper sentences)
    sentence_score <- if (grepl("\\.", response) && nchar(response) > 10) 1.0 else 0.5
    
    return(mean(c(length_score, sentence_score)))
  }
  
  #' Evaluate sentiment response
  evaluate_sentiment_response <- function(response, test_case) {
    response_lower <- tolower(response)
    if (grepl(test_case$expected_sentiment, response_lower)) {
      return(1.0)
    } else {
      return(0.0)
    }
  }
  
  #' Evaluate creative response
  evaluate_creative_response <- function(response, test_case) {
    word_count <- length(strsplit(response, "\\s+")[[1]])
    length_score <- if (word_count >= test_case$min_length) 1.0 else 0.5
    
    # Basic creativity indicators
    creativity_score <- if (grepl("[.!?]", response) && 
                           !grepl("^(I|The|This)", response)) 0.8 else 0.6
    
    return(mean(c(length_score, creativity_score)))
  }
  
  #' Evaluate code response
  evaluate_code_response <- function(response, test_case) {
    response_lower <- tolower(response)
    keywords_found <- sum(sapply(test_case$expected_keywords,
                                function(kw) grepl(tolower(kw), response_lower)))
    keyword_score <- keywords_found / length(test_case$expected_keywords)
    
    # Check for code-like structure
    structure_score <- if (grepl("[{}():]", response)) 1.0 else 0.5
    
    return(mean(c(keyword_score, structure_score)))
  }
  
  #' Calculate success rate
  calculate_success_rate <- function(task_results) {
    total_cases <- sum(sapply(task_results, function(x) length(x$cases)))
    successful_cases <- sum(sapply(task_results, function(x) {
      sum(sapply(x$cases, function(case) case$evaluation$score > 0.5))
    }))
    
    return(successful_cases / total_cases)
  }
  
  #' Compare multiple models
  #' @param model_names character vector of model names
  #' @param tasks character vector of tasks to run
  compare_models <- function(model_names, tasks = c("question_answer", "sentiment")) {
    
    cat("🏁 MULTI-MODEL COMPARISON\n")
    cat(sprintf("Models: %s\n", paste(model_names, collapse = ", ")))
    cat(sprintf("Tasks: %s\n\n", paste(tasks, collapse = ", ")))
    
    comparison_results <- list(
      comparison_time = Sys.time(),
      models = list(),
      summary = data.frame()
    )
    
    # Benchmark each model
    for (model_name in model_names) {
      model_result <- benchmark_model(model_name, tasks)
      
      if (!is.null(model_result)) {
        comparison_results$models[[model_name]] <- model_result
      }
    }
    
    # Create summary table
    if (length(comparison_results$models) > 0) {
      summary_data <- data.frame(
        Model = character(),
        Avg_Response_Time = numeric(),
        Avg_Score = numeric(),
        Success_Rate = numeric(),
        Tasks_Completed = integer(),
        stringsAsFactors = FALSE
      )
      
      for (model_name in names(comparison_results$models)) {
        result <- comparison_results$models[[model_name]]
        metrics <- result$overall_metrics
        
        summary_data <- rbind(summary_data, data.frame(
          Model = model_name,
          Avg_Response_Time = round(metrics$avg_response_time, 3),
          Avg_Score = round(metrics$avg_score, 3),
          Success_Rate = round(metrics$success_rate, 3),
          Tasks_Completed = metrics$tasks_completed,
          stringsAsFactors = FALSE
        ))
      }
      
      comparison_results$summary <- summary_data
    }
    
    return(comparison_results)
  }
  
  #' Display comparison results
  #' @param comparison_results list results from compare_models()
  display_comparison <- function(comparison_results) {
    
    cat("=== MODEL COMPARISON RESULTS ===\n\n")
    
    if (nrow(comparison_results$summary) == 0) {
      cat("No successful benchmarks to display.\n")
      return()
    }
    
    # Display summary table
    cat("📊 SUMMARY TABLE:\n")
    print(comparison_results$summary)
    cat("\n")
    
    # Find best performers
    summary <- comparison_results$summary
    
    if (nrow(summary) > 0) {
      fastest_model <- summary$Model[which.min(summary$Avg_Response_Time)]
      highest_score <- summary$Model[which.max(summary$Avg_Score)]
      highest_success <- summary$Model[which.max(summary$Success_Rate)]
      
      cat("🏆 PERFORMANCE HIGHLIGHTS:\n")
      cat(sprintf("   🚀 Fastest Response: %s (%.3fs avg)\n", 
                  fastest_model, min(summary$Avg_Response_Time)))
      cat(sprintf("   🎯 Highest Quality: %s (%.3f avg score)\n", 
                  highest_score, max(summary$Avg_Score)))
      cat(sprintf("   ✅ Most Reliable: %s (%.1f%% success rate)\n", 
                  highest_success, max(summary$Success_Rate) * 100))
      
      # Recommendations
      cat("\n💡 RECOMMENDATIONS:\n")
      
      if (fastest_model == highest_score) {
        cat(sprintf("   • %s offers the best overall balance of speed and quality\n", fastest_model))
      } else {
        cat(sprintf("   • Choose %s for speed-critical applications\n", fastest_model))
        cat(sprintf("   • Choose %s for quality-critical applications\n", highest_score))
      }
      
      if (max(summary$Success_Rate) < 0.8) {
        cat("   • Consider improving prompts or trying larger models for better reliability\n")
      }
      
      cat("\n")
    }
  }
  
  #' Export benchmark results
  #' @param results list benchmark results  
  #' @param output_dir character output directory
  export_results <- function(results, output_dir = "benchmark_results") {
    
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }
    
    timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    
    # Export summary table
    if ("summary" %in% names(results) && nrow(results$summary) > 0) {
      csv_file <- file.path(output_dir, sprintf("model_comparison_%s.csv", timestamp))
      write.csv(results$summary, csv_file, row.names = FALSE)
      cat(sprintf("📄 Summary exported: %s\n", csv_file))
    }
    
    # Export detailed results as JSON/RDS
    rds_file <- file.path(output_dir, sprintf("detailed_results_%s.rds", timestamp))
    saveRDS(results, rds_file)
    cat(sprintf("📊 Detailed results exported: %s\n", rds_file))
    
    cat(sprintf("Results exported to %s/\n", output_dir))
  }
  
  # Return benchmarker functions
  list(
    initialize = initialize,
    benchmark_model = benchmark_model,
    compare_models = compare_models,
    display_comparison = display_comparison,
    export_results = export_results,
    get_test_cases = get_test_cases
  )
}

#' Main demonstration
main <- function() {
  cat("=== MODEL BENCHMARKING DEMONSTRATION ===\n\n")
  
  # Create benchmarker
  benchmarker <- ModelBenchmarker()
  benchmarker$initialize()
  
  # Define models to test (starting with most reliable)
  test_models <- c("TinyLlama-1.1B", "Phi-3.5-Mini", "Llama-3.2-1B")
  
  # Select available models
  cat("🔍 Checking model availability...\n")
  available_models <- character()
  
  for (model in test_models) {
    tryCatch({
      # Quick availability check
      models_list <- edge_list_models()
      if (model %in% models_list$name) {
        available_models <- c(available_models, model)
        cat(sprintf("   ✅ %s available\n", model))
      } else {
        cat(sprintf("   ❌ %s not available\n", model))
      }
    }, error = function(e) {
      cat(sprintf("   ❌ %s check failed\n", model))
    })
  }
  
  if (length(available_models) == 0) {
    cat("⚠️ No models available for benchmarking\n")
    return(NULL)
  }
  
  # Run comparison with available models
  tasks_to_test <- c("question_answer", "sentiment")
  
  cat(sprintf("\n🚀 Running benchmark with %d models on %d tasks...\n\n", 
              length(available_models), length(tasks_to_test)))
  
  comparison_results <- benchmarker$compare_models(available_models, tasks_to_test)
  
  # Display results
  benchmarker$display_comparison(comparison_results)
  
  # Export results
  benchmarker$export_results(comparison_results)
  
  return(comparison_results)
}

#' Quick benchmark single model
#' @param model_name character model to benchmark
#' @param tasks character vector of tasks
quick_benchmark <- function(model_name = "TinyLlama-1.1B", 
                           tasks = c("question_answer", "sentiment")) {
  benchmarker <- ModelBenchmarker()
  benchmarker$initialize()
  
  result <- benchmarker$benchmark_model(model_name, tasks)
  
  if (!is.null(result)) {
    cat(sprintf("\n=== QUICK BENCHMARK: %s ===\n", model_name))
    cat(sprintf("Overall Score: %.2f\n", result$overall_metrics$avg_score))
    cat(sprintf("Avg Response Time: %.2fs\n", result$overall_metrics$avg_response_time))
    cat(sprintf("Success Rate: %.1f%%\n", result$overall_metrics$success_rate * 100))
  }
  
  return(result)
}

# Run demonstration if executed directly
if (sys.nframe() == 0) {
  benchmark_results <- main()
}