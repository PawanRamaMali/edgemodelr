#' Modern Small Language Model Comparison for edgemodelr
#'
#' This example demonstrates the latest small quantized models available in 2024
#' that are efficient for local inference with edgemodelr package.
#'
#' @author edgemodelr team
#' @date 2024

library(edgemodelr)

#' Get information about modern small models
#' @return data.frame with model recommendations
get_modern_small_models <- function() {
  # Define modern small models with their characteristics
  models <- data.frame(
    name = c(
      "Llama-3.2-1B", "Llama-3.2-3B",
      "Phi-3.5-Mini", "Qwen2.5-1.5B", "Qwen2.5-3B",
      "Gemma-2-2B", "TinyLlama-1.1B"
    ),
    size_q4 = c(
      "~800MB", "~2GB", "~2.4GB", "~1GB", "~2GB",
      "~1.2GB", "~700MB"
    ),
    parameters = c(
      "1B", "3B", "3.8B", "1.5B", "3B",
      "2B", "1.1B"
    ),
    strengths = c(
      "Latest Meta model, mobile-optimized",
      "Best 3B model, excellent instruction following",
      "Exceptional quality for size, 7B performance",
      "Strong coding and math capabilities",
      "Advanced reasoning and structured output",
      "Google's efficient architecture",
      "Most lightweight, reliable baseline"
    ),
    use_cases = c(
      "Mobile, edge devices, basic tasks",
      "General purpose, content generation",
      "Quality-critical applications",
      "Code generation, mathematical tasks",
      "Complex reasoning, JSON generation",
      "Balanced performance tasks",
      "Resource-constrained environments"
    ),
    stringsAsFactors = FALSE
  )

  return(models)
}

#' Display model recommendations
show_model_recommendations <- function() {
  cat("=== MODERN SMALL LANGUAGE MODELS FOR LOCAL INFERENCE ===\n\n")

  models <- get_modern_small_models()

  for (i in 1:nrow(models)) {
    model <- models[i, ]
    cat(sprintf("📦 %s (%s parameters, %s quantized)\n",
                model$name, model$parameters, model$size_q4))
    cat(sprintf("   💪 Strengths: %s\n", model$strengths))
    cat(sprintf("   🎯 Use cases: %s\n\n", model$use_cases))
  }

  cat("📋 QUANTIZATION LEVELS:\n")
  cat("   Q2_K: Ultra-compressed (~50% size, some quality loss)\n")
  cat("   Q4_K_M: Recommended balance (good quality, 4x compression)\n")
  cat("   Q5_K_M: Higher quality (3x compression)\n")
  cat("   Q8_0: Near-original quality (2x compression)\n\n")

  cat("🔧 SETUP RECOMMENDATIONS:\n")
  cat("   • Start with TinyLlama-1.1B (most reliable, already available)\n")
  cat("   • Upgrade to Llama-3.2-1B for latest features\n")
  cat("   • Use Phi-3.5-Mini for quality-critical tasks\n")
  cat("   • Choose Qwen2.5 models for coding/math\n\n")
}

#' Test model availability and performance
#' @param model_names character vector of model names to test
test_model_availability <- function(model_names = c("TinyLlama-1.1B", "Phi-3.5-Mini")) {
  cat("=== TESTING MODEL AVAILABILITY ===\n\n")

  results <- data.frame(
    model = character(),
    status = character(),
    error = character(),
    stringsAsFactors = FALSE
  )

  for (model_name in model_names) {
    cat(sprintf("Testing %s...\n", model_name))

    tryCatch({
      setup <- edge_quick_setup(model_name)
      ctx <- setup$context

      # Quick test
      test_prompt <- "Hello"
      result <- edge_completion(ctx, test_prompt, n_predict = 5)

      edge_free_model(ctx)

      results <- rbind(results, data.frame(
        model = model_name,
        status = "✅ Available",
        error = "",
        stringsAsFactors = FALSE
      ))

      cat(sprintf("   ✅ %s is working\n", model_name))

    }, error = function(e) {
      results <<- rbind(results, data.frame(
        model = model_name,
        status = "❌ Failed",
        error = as.character(e$message),
        stringsAsFactors = FALSE
      ))

      cat(sprintf("   ❌ %s failed: %s\n", model_name, e$message))
    })
  }

  cat("\n")
  return(results)
}

#' Main demonstration function
main <- function() {
  # Show modern model recommendations
  show_model_recommendations()

  # List currently available models in edgemodelr
  cat("=== CURRENTLY AVAILABLE MODELS ===\n")
  available_models <- edge_list_models()
  print(available_models[1:10, c("name", "size", "use_case")])

  cat("\n=== TESTING MODEL AVAILABILITY ===\n")
  # Test a few models
  test_results <- test_model_availability()

  cat("\n=== RECOMMENDATIONS BASED ON TESTS ===\n")
  working_models <- test_results[test_results$status == "✅ Available", "model"]

  if (length(working_models) > 0) {
    cat("✅ Working models:\n")
    for (model in working_models) {
      cat(sprintf("   • %s\n", model))
    }
  } else {
    cat("❌ No models currently working. Try:\n")
    cat("   1. Check internet connection\n")
    cat("   2. Manually download TinyLlama model\n")
    cat("   3. Use edge_download_model() directly\n")
  }

  return(list(
    recommendations = get_modern_small_models(),
    availability = test_results
  ))
}

# Run the comparison if script is executed directly
if (sys.nframe() == 0) {
  result <- main()
}
