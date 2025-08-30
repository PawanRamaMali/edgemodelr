test_that("Integration tests work correctly", {
  
  test_that("Complete workflow integration test", {
    possible_paths <- c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    )
    
    model_path <- NULL
    for (path in possible_paths) {
      if (file.exists(path)) {
        model_path <- path
        break
      }
    }
    
    if (!is.null(model_path)) {
      # Complete workflow test
      
      # 1. Load model
      ctx <- edge_load_model(model_path, n_ctx = 512)
      expect_true(is_valid_model(ctx))
      
      # 2. Test multiple completions with the same context
      prompts <- c(
        "The weather today is",
        "In the future, technology will",
        "My favorite color is",
        "The best way to learn is"
      )
      
      results <- character(length(prompts))
      for (i in seq_along(prompts)) {
        results[i] <- edge_completion(ctx, prompts[i], n_predict = 8)
        expect_true(is.character(results[i]))
        expect_true(startsWith(results[i], prompts[i]))
      }
      
      # 3. Test that results are different (high probability)
      expect_true(length(unique(results)) > 1, 
                  info = "Results should generally be different")
      
      # 4. Free model
      edge_free_model(ctx)
      expect_false(is_valid_model(ctx))
      
      # 5. Verify we can't use the context after freeing
      expect_error(
        edge_completion(ctx, "Hello", n_predict = 5)
      )
      
    } else {
      skip("No test model available for integration tests")
    }
  })
  
  test_that("Model listing and information consistency", {
    # Test that model listing is consistent
    models1 <- edge_list_models()
    models2 <- edge_list_models()
    
    expect_equal(models1, models2, 
                 info = "Model listing should be consistent")
    
    # Check that listed models have reasonable properties
    for (i in 1:min(3, nrow(models1))) {
      model <- models1[i, ]
      
      # Model ID should be non-empty and reasonable format
      expect_true(nchar(model$model_id) > 0)
      expect_true(grepl("/", model$model_id), 
                  info = "Model ID should typically contain '/' for org/model format")
      
      # Filename should end with .gguf
      expect_true(endsWith(model$filename, ".gguf"))
      
      # Size should be informative
      expect_true(nchar(model$size) > 3)
      expect_true(grepl("MB|GB", model$size))
      
      # Use case should be informative
      expect_true(nchar(model$use_case) > 3)
    }
  })
  
  test_that("Package namespace and exports", {
    # Test that all expected functions are exported
    expected_functions <- c(
      "edge_load_model",
      "edge_completion", 
      "edge_free_model",
      "is_valid_model",
      "edge_list_models",
      "edge_download_model",
      "edge_quick_setup",
      "edge_stream_completion",
      "edge_chat_stream",
      "build_chat_prompt"
    )
    
    for (func_name in expected_functions) {
      expect_true(exists(func_name, mode = "function"),
                  info = paste("Function should be exported:", func_name))
    }
  })
  
  test_that("Package loads correctly", {
    # Test package loading and basic functionality
    expect_true(requireNamespace("edgemodelr", quietly = TRUE))
    
    # Test that we can access documentation (if available)
    # This is mainly to ensure the package structure is correct
    tryCatch({
      help_result <- utils::help("edge_load_model", package = "edgemodelr")
      # If help exists, it should have content
      if (length(help_result) > 0) {
        expect_true(length(help_result) > 0)
      }
    }, error = function(e) {
      # Help might not be available in all test environments
      message("Help documentation not available in test environment")
    })
  })
  
  test_that("Stress test with multiple operations", {
    possible_paths <- c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    )
    
    model_path <- NULL
    for (path in possible_paths) {
      if (file.exists(path)) {
        model_path <- path
        break
      }
    }
    
    if (!is.null(model_path)) {
      # Stress test: multiple load/use/free cycles
      
      for (cycle in 1:3) {
        # Load model
        ctx <- edge_load_model(model_path, n_ctx = 256)
        expect_true(is_valid_model(ctx))
        
        # Multiple completions
        for (completion in 1:5) {
          prompt <- paste("Test prompt", cycle, completion)
          result <- edge_completion(ctx, prompt, n_predict = 3)
          expect_true(is.character(result))
          expect_true(startsWith(result, prompt))
        }
        
        # Free model
        edge_free_model(ctx)
        expect_false(is_valid_model(ctx))
      }
      
    } else {
      skip("No test model available for stress tests")
    }
  })
  
  test_that("Performance and timing tests", {
    possible_paths <- c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    )
    
    model_path <- NULL
    for (path in possible_paths) {
      if (file.exists(path)) {
        model_path <- path
        break
      }
    }
    
    if (!is.null(model_path)) {
      # Test that operations complete in reasonable time
      
      # Model loading should complete within reasonable time (30 seconds)
      start_time <- Sys.time()
      ctx <- edge_load_model(model_path, n_ctx = 256)
      load_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      expect_true(is_valid_model(ctx))
      expect_true(load_time < 30, 
                  info = paste("Model loading took", round(load_time, 2), "seconds"))
      
      # Text completion should complete within reasonable time (10 seconds)
      start_time <- Sys.time()
      result <- edge_completion(ctx, "Quick test", n_predict = 5)
      completion_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      expect_true(is.character(result))
      expect_true(completion_time < 10,
                  info = paste("Text completion took", round(completion_time, 2), "seconds"))
      
      # Model freeing should be fast
      start_time <- Sys.time()
      edge_free_model(ctx)
      free_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      
      expect_true(free_time < 1,
                  info = paste("Model freeing took", round(free_time, 2), "seconds"))
      
    } else {
      skip("No test model available for performance tests")
    }
  })
  
  # Test comprehensive end-to-end scenarios
  test_that("End-to-end scenarios", {
    possible_paths <- c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    )
    
    model_path <- NULL
    for (path in possible_paths) {
      if (file.exists(path)) {
        model_path <- path
        break
      }
    }
    
    if (!is.null(model_path)) {
      # Scenario 1: Interactive conversation simulation
      ctx <- edge_load_model(model_path, n_ctx = 512)
      
      conversation_prompts <- c(
        "Hello! How are you today?",
        "What is your favorite programming language?",
        "Can you help me with a coding problem?",
        "What do you think about artificial intelligence?",
        "Thanks for the conversation!"
      )
      
      conversation_results <- list()
      for (i in seq_along(conversation_prompts)) {
        result <- edge_completion(ctx, conversation_prompts[i], n_predict = 15)
        conversation_results[[i]] <- result
        
        expect_true(is.character(result))
        expect_true(nchar(result) > nchar(conversation_prompts[i]))
        expect_true(startsWith(result, conversation_prompts[i]))
      }
      
      edge_free_model(ctx)
      
      # Scenario 2: Different model context sizes
      context_sizes <- c(128, 256, 512, 1024)
      
      for (ctx_size in context_sizes) {
        tryCatch({
          ctx_test <- edge_load_model(model_path, n_ctx = ctx_size)
          expect_true(is_valid_model(ctx_test))
          
          # Test with prompt appropriate for context size
          prompt_length <- min(50, ctx_size %/% 4)  # Conservative prompt length
          test_prompt <- paste(rep("test", prompt_length), collapse = " ")
          
          result <- edge_completion(ctx_test, test_prompt, n_predict = 5)
          expect_true(is.character(result))
          
          edge_free_model(ctx_test)
          
        }, error = function(e) {
          # Some context sizes might fail due to memory constraints
          expect_true(grepl("memory|context|allocation", e$message, ignore.case = TRUE))
        })
      }
      
      # Scenario 3: Different generation parameters
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      test_prompt <- "The weather is"
      
      # Test different n_predict values
      for (n_pred in c(1, 5, 10, 20, 50)) {
        result <- edge_completion(ctx, test_prompt, n_predict = n_pred)
        expect_true(is.character(result))
        expect_true(startsWith(result, test_prompt))
      }
      
      # Test different temperature values
      for (temp in c(0.1, 0.5, 0.8, 1.0, 1.5)) {
        result <- edge_completion(ctx, test_prompt, n_predict = 5, temperature = temp)
        expect_true(is.character(result))
        expect_true(startsWith(result, test_prompt))
      }
      
      # Test different top_p values
      for (top_p in c(0.1, 0.5, 0.9, 0.95)) {
        result <- edge_completion(ctx, test_prompt, n_predict = 5, top_p = top_p)
        expect_true(is.character(result))
        expect_true(startsWith(result, test_prompt))
      }
      
      edge_free_model(ctx)
      
    } else {
      skip("No test model available for end-to-end scenario tests")
    }
  })
  
  # Test concurrent operations and resource management
  test_that("Concurrent operations and resource management", {
    possible_paths <- c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    )
    
    model_path <- NULL
    for (path in possible_paths) {
      if (file.exists(path)) {
        model_path <- path
        break
      }
    }
    
    if (!is.null(model_path)) {
      # Test loading multiple models with different configurations
      contexts <- list()
      
      tryCatch({
        # Load multiple contexts with different settings
        contexts[[1]] <- edge_load_model(model_path, n_ctx = 128, n_gpu_layers = 0)
        contexts[[2]] <- edge_load_model(model_path, n_ctx = 256, n_gpu_layers = 0)
        contexts[[3]] <- edge_load_model(model_path, n_ctx = 512, n_gpu_layers = 0)
        
        # Verify all are valid
        for (i in 1:3) {
          expect_true(is_valid_model(contexts[[i]]))
        }
        
        # Use all contexts concurrently
        results <- list()
        for (i in 1:3) {
          results[[i]] <- edge_completion(contexts[[i]], 
                                        paste("Test prompt", i), 
                                        n_predict = 3)
          expect_true(is.character(results[[i]]))
        }
        
        # Clean up in reverse order
        for (i in 3:1) {
          edge_free_model(contexts[[i]])
          expect_false(is_valid_model(contexts[[i]]))
        }
        
      }, error = function(e) {
        # Clean up any successfully loaded contexts
        for (ctx in contexts) {
          if (!is.null(ctx) && is_valid_model(ctx)) {
            edge_free_model(ctx)
          }
        }
        
        # Multiple model loading might fail on systems with limited memory
        expect_true(grepl("memory|allocation|resource", e$message, ignore.case = TRUE))
      })
      
      # Test rapid load/free cycles for memory leak detection
      for (i in 1:10) {
        ctx <- edge_load_model(model_path, n_ctx = 64)
        expect_true(is_valid_model(ctx))
        
        # Quick test to ensure it works
        result <- edge_completion(ctx, "Hi", n_predict = 1)
        expect_true(is.character(result))
        
        edge_free_model(ctx)
        expect_false(is_valid_model(ctx))
      }
      
    } else {
      skip("No test model available for concurrent operations tests")
    }
  })
  
  # Test system integration and environment compatibility
  test_that("System integration and environment compatibility", {
    # Test that package works in current R environment
    expect_true(R.version$major >= "4", 
                info = "Package should work with R 4.0+")
    
    # Test platform compatibility
    platform <- Sys.info()["sysname"]
    expect_true(platform %in% c("Windows", "Linux", "Darwin"),
                info = paste("Unsupported platform:", platform))
    
    # Test that required system libraries are available (if any)
    # This is mostly informational
    arch <- Sys.info()["machine"]
    message(paste("Running on:", platform, arch))
    
    # Test memory availability
    if (platform == "Windows") {
      # On Windows, check available memory
      tryCatch({
        memory_info <- system("wmic OS get TotalVisibleMemorySize /value", intern = TRUE)
        memory_line <- memory_info[grep("TotalVisibleMemorySize", memory_info)]
        if (length(memory_line) > 0) {
          memory_kb <- as.numeric(gsub("TotalVisibleMemorySize=", "", memory_line))
          memory_gb <- memory_kb / (1024^2)
          
          if (memory_gb < 4) {
            message(paste("Low system memory detected:", round(memory_gb, 1), "GB"))
            message("Some tests may be skipped due to memory constraints")
          }
        }
      }, error = function(e) {
        message("Could not determine system memory")
      })
    }
    
    # Test that the package can handle the current working directory
    original_wd <- getwd()
    tryCatch({
      # Change to temp directory and back
      temp_dir <- tempdir()
      setwd(temp_dir)
      
      # Basic function should still work
      models <- edge_list_models()
      expect_true(is.data.frame(models))
      
    }, finally = {
      setwd(original_wd)
    })
  })
})