#!/usr/bin/env Rscript

# Comprehensive test runner for edgemodelr package
# This script loads the package directly and runs functional tests

cat("🧪 Running EXPANDED Comprehensive Tests for edgemodelr\n")
cat("======================================================\n\n")

# Load required libraries
suppressPackageStartupMessages({
  library(devtools)
  library(testthat)
})

# Load the package using devtools
cat("📦 Loading edgemodelr package...\n")
devtools::load_all(".", quiet = TRUE)

# Test counters
total_tests <- 0
passed_tests <- 0
failed_tests <- 0

# Helper function to run a test
run_test <- function(test_name, test_expr) {
  total_tests <<- total_tests + 1
  cat("🔸", test_name, "... ")
  
  tryCatch({
    test_expr
    cat("✅ PASS\n")
    passed_tests <<- passed_tests + 1
  }, error = function(e) {
    cat("❌ FAIL:", e$message, "\n")
    failed_tests <<- failed_tests + 1
  })
}

# ============================================================================
# Test 1: Basic Package Functions
# ============================================================================
cat("\n📋 Testing Basic Package Functions\n")
cat("-----------------------------------\n")

run_test("edge_list_models returns data frame", {
  models <- edge_list_models()
  stopifnot(is.data.frame(models))
  stopifnot(nrow(models) > 0)
  stopifnot("model_id" %in% colnames(models))
})

run_test("is_valid_model handles invalid inputs", {
  stopifnot(!is_valid_model(NULL))
  stopifnot(!is_valid_model("invalid"))
  stopifnot(!is_valid_model(123))
})

# ============================================================================
# Test 2: Model Loading and Management  
# ============================================================================
cat("\n🔄 Testing Model Loading and Management\n")
cat("--------------------------------------\n")

model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if (file.exists(model_path)) {
  cat("📁 Using model:", model_path, "\n")
} else {
  cat("📥 Model not found. Download with:\n")
  cat("   edge_download_model('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')\n\n")
}

if (file.exists(model_path)) {
  
  run_test("Model loads successfully", {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    stopifnot(!is.null(ctx))
    edge_free_model(ctx)
  })
  
  run_test("Model validation works", {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    stopifnot(is_valid_model(ctx))
    edge_free_model(ctx)
    stopifnot(!is_valid_model(ctx))  # Should be invalid after freeing
  })
  
  run_test("Different context sizes work", {
    # Test small context
    ctx1 <- edge_load_model(model_path, n_ctx = 128)
    stopifnot(is_valid_model(ctx1))
    edge_free_model(ctx1)
    
    # Test larger context
    ctx2 <- edge_load_model(model_path, n_ctx = 512)
    stopifnot(is_valid_model(ctx2))
    edge_free_model(ctx2)
  })
  
  run_test("Multiple models can be loaded", {
    ctx1 <- edge_load_model(model_path, n_ctx = 128)
    ctx2 <- edge_load_model(model_path, n_ctx = 128)
    
    stopifnot(is_valid_model(ctx1))
    stopifnot(is_valid_model(ctx2))
    
    edge_free_model(ctx1)
    edge_free_model(ctx2)
  })
  
  # ============================================================================
  # Test 3: Text Completion
  # ============================================================================
  cat("\n💬 Testing Text Completion\n")
  cat("-------------------------\n")
  
  # Load model for completion tests
  ctx <- edge_load_model(model_path, n_ctx = 512)
  
  run_test("Basic text completion works", {
    result <- edge_completion(ctx, "Hello", n_predict = 5)
    stopifnot(is.character(result))
    stopifnot(nchar(result) > nchar("Hello"))
    stopifnot(startsWith(result, "Hello"))
  })
  
  run_test("Different prompts work", {
    prompts <- c(
      "The capital of France is",
      "Once upon a time",
      "2 + 2 =",
      "The weather today"
    )
    
    for (prompt in prompts) {
      result <- edge_completion(ctx, prompt, n_predict = 8)
      stopifnot(is.character(result))
      stopifnot(startsWith(result, prompt))
    }
  })
  
  run_test("Different n_predict values work", {
    prompt <- "Hello world"
    
    result1 <- edge_completion(ctx, prompt, n_predict = 1)
    result2 <- edge_completion(ctx, prompt, n_predict = 5)
    result3 <- edge_completion(ctx, prompt, n_predict = 10)
    
    stopifnot(is.character(result1))
    stopifnot(is.character(result2))
    stopifnot(is.character(result3))
    stopifnot(nchar(result3) >= nchar(result2))
    stopifnot(nchar(result2) >= nchar(result1))
  })
  
  run_test("Empty prompt works", {
    result <- edge_completion(ctx, "", n_predict = 5)
    stopifnot(is.character(result))
    stopifnot(nchar(result) > 0)
  })
  
  run_test("Special characters work", {
    special_prompts <- c(
      "Hello! How are you?",
      "What is 2+2=",
      "The price is $100.",
      "Email: user@example.com"
    )
    
    for (prompt in special_prompts) {
      result <- edge_completion(ctx, prompt, n_predict = 5)
      stopifnot(is.character(result))
      stopifnot(startsWith(result, prompt))
    }
  })
  
  # Free the model
  edge_free_model(ctx)
  
  # ============================================================================
  # Test 4: Error Handling
  # ============================================================================
  cat("\n⚠️  Testing Error Handling\n")
  cat("--------------------------\n")
  
  run_test("Invalid model paths are rejected", {
    error_occurred <- FALSE
    tryCatch({
      edge_load_model("nonexistent.gguf")
    }, error = function(e) {
      error_occurred <<- TRUE
    })
    stopifnot(error_occurred)
  })
  
  run_test("Invalid contexts are rejected", {
    error_occurred <- FALSE
    tryCatch({
      edge_completion(NULL, "Hello", n_predict = 5)
    }, error = function(e) {
      error_occurred <<- TRUE
    })
    stopifnot(error_occurred)
  })
  
  run_test("Invalid n_predict values are rejected", {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    
    error_occurred <- FALSE
    tryCatch({
      edge_completion(ctx, "Hello", n_predict = -1)
    }, error = function(e) {
      error_occurred <<- TRUE
    })
    stopifnot(error_occurred)
    
    edge_free_model(ctx)
  })
  
  # ============================================================================
  # Test 5: Performance and Stress Tests
  # ============================================================================
  cat("\n🏃 Testing Performance and Stress\n")
  cat("---------------------------------\n")
  
  run_test("Multiple load/free cycles work", {
    for (i in 1:3) {
      ctx <- edge_load_model(model_path, n_ctx = 128)
      result <- edge_completion(ctx, paste("Test", i), n_predict = 3)
      stopifnot(is.character(result))
      edge_free_model(ctx)
    }
  })
  
  run_test("Model loading is reasonably fast", {
    start_time <- Sys.time()
    ctx <- edge_load_model(model_path, n_ctx = 256)
    load_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    stopifnot(load_time < 30)  # Should load within 30 seconds
    edge_free_model(ctx)
  })
  
  run_test("Text completion is reasonably fast", {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    
    start_time <- Sys.time()
    result <- edge_completion(ctx, "Quick test", n_predict = 5)
    completion_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    stopifnot(completion_time < 10)  # Should complete within 10 seconds
    stopifnot(is.character(result))
    
    edge_free_model(ctx)
  })
  
  # ============================================================================
  # Test 6: Edge Cases and Boundary Conditions
  # ============================================================================
  cat("\n🔬 Testing Edge Cases and Boundary Conditions\n")
  cat("---------------------------------------------\n")
  
  run_test("Minimum context size works", {
    ctx <- edge_load_model(model_path, n_ctx = 1)
    stopifnot(is_valid_model(ctx))
    result <- edge_completion(ctx, "Hi", n_predict = 1)
    stopifnot(is.character(result))
    stopifnot(startsWith(result, "Hi"))
    edge_free_model(ctx)
  })
  
  run_test("Empty prompt handling", {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    result <- edge_completion(ctx, "", n_predict = 3)
    stopifnot(is.character(result))
    edge_free_model(ctx)
  })
  
  run_test("Single character prompts", {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    result <- edge_completion(ctx, "A", n_predict = 2)
    stopifnot(is.character(result))
    stopifnot(startsWith(result, "A"))
    edge_free_model(ctx)
  })
  
  run_test("Special characters in prompts", {
    ctx <- edge_load_model(model_path, n_ctx = 256)
    special_prompts <- c("!@#$%", "123", "Hello!", "What?")
    for (prompt in special_prompts) {
      result <- edge_completion(ctx, prompt, n_predict = 2)
      stopifnot(is.character(result))
      stopifnot(startsWith(result, prompt))
    }
    edge_free_model(ctx)
  })
  
  run_test("Very long prompts", {
    ctx <- edge_load_model(model_path, n_ctx = 1024)
    long_prompt <- paste(rep("This is a test sentence.", 20), collapse = " ")
    result <- edge_completion(ctx, long_prompt, n_predict = 5)
    stopifnot(is.character(result))
    stopifnot(startsWith(result, long_prompt))
    edge_free_model(ctx)
  })
  
  # ============================================================================
  # Test 7: Stress and Memory Management
  # ============================================================================
  cat("\n💪 Testing Stress and Memory Management\n")
  cat("---------------------------------------\n")
  
  run_test("Rapid load/free cycles (20x)", {
    for (i in 1:20) {
      ctx <- edge_load_model(model_path, n_ctx = 128)
      stopifnot(is_valid_model(ctx))
      result <- edge_completion(ctx, paste("Cycle", i), n_predict = 1)
      stopifnot(is.character(result))
      edge_free_model(ctx)
      stopifnot(!is_valid_model(ctx))
    }
  })
  
  run_test("Multiple contexts simultaneously", {
    contexts <- list()
    for (i in 1:3) {
      ctx <- edge_load_model(model_path, n_ctx = 128)
      stopifnot(is_valid_model(ctx))
      contexts[[i]] <- ctx
    }
    
    for (i in 1:3) {
      result <- edge_completion(contexts[[i]], paste("Multi", i), n_predict = 2)
      stopifnot(is.character(result))
      stopifnot(startsWith(result, paste("Multi", i)))
    }
    
    for (i in 1:3) {
      edge_free_model(contexts[[i]])
    }
  })
  
  run_test("Large generation request", {
    ctx <- edge_load_model(model_path, n_ctx = 1024)
    result <- edge_completion(ctx, "Generate a story", n_predict = 100)
    stopifnot(is.character(result))
    stopifnot(startsWith(result, "Generate a story"))
    edge_free_model(ctx)
  })
  
  # ============================================================================
  # Test 8: Parameter Variations
  # ============================================================================
  cat("\n⚙️  Testing Parameter Variations\n")
  cat("--------------------------------\n")
  
  run_test("Different context sizes", {
    sizes <- c(64, 128, 256, 512, 1024)
    for (size in sizes) {
      ctx <- edge_load_model(model_path, n_ctx = size)
      stopifnot(is_valid_model(ctx))
      result <- edge_completion(ctx, "Size test", n_predict = 2)
      stopifnot(is.character(result))
      stopifnot(startsWith(result, "Size test"))
      edge_free_model(ctx)
    }
  })
  
  run_test("Different n_predict values", {
    ctx <- edge_load_model(model_path, n_ctx = 512)
    n_predict_values <- c(0, 1, 2, 5, 10, 20)
    for (n_pred in n_predict_values) {
      result <- edge_completion(ctx, "Predict test", n_predict = n_pred)
      stopifnot(is.character(result))
      if (n_pred == 0) {
        stopifnot(result == "Predict test")
      } else {
        stopifnot(startsWith(result, "Predict test"))
      }
    }
    edge_free_model(ctx)
  })
  
  run_test("Different GPU layer values", {
    gpu_values <- c(0, 5, 10, 22)
    for (gpu_layers in gpu_values) {
      ctx <- edge_load_model(model_path, n_ctx = 256, n_gpu_layers = gpu_layers)
      stopifnot(is_valid_model(ctx))
      result <- edge_completion(ctx, "GPU test", n_predict = 2)
      stopifnot(is.character(result))
      stopifnot(startsWith(result, "GPU test"))
      edge_free_model(ctx)
    }
  })
  
} else {
  cat("⚠️  No test model available. Skipping model-dependent tests.\n")
  cat("   To run full tests, ensure 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf' is present.\n")
}

# ============================================================================
# Test Results Summary
# ============================================================================
cat("\n📊 Test Results Summary\n")
cat("======================\n")
cat("Total Tests:  ", total_tests, "\n")
cat("Passed:       ", passed_tests, "\n")
cat("Failed:       ", failed_tests, "\n")
cat("Success Rate: ", round(passed_tests / total_tests * 100, 1), "%\n")

if (failed_tests == 0) {
  cat("\n🎉 All tests passed! The package is working correctly.\n")
} else {
  cat("\n⚠️  Some tests failed. Please review the failures above.\n")
}

# Return appropriate exit code
if (failed_tests > 0) {
  quit(status = 1)
} else {
  quit(status = 0)
}