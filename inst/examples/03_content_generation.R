#' Advanced Content Generation with Small Language Models
#'
#' This example demonstrates various content generation tasks using modern
#' small language models, including creative writing, technical documentation,
#' code generation, and structured data creation.
#'
#' Features:
#' - Multiple content types (creative, technical, code, structured)
#' - Template-based generation
#' - Quality assessment
#' - Batch generation capabilities
#' - Export functionality
#'
#' @author edgemodelr team
#' @date 2024

library(edgemodelr)

#' Content Generation System
#' @description Professional content generation with multiple output formats
ContentGenerator <- function() {
  
  #' Initialize content generator
  #' @param model_name character model to use
  initialize <- function(model_name = "TinyLlama-1.1B") {
    cat(sprintf("🎨 Initializing Content Generator with %s...\n", model_name))
    
    setup <- edge_quick_setup(model_name)
    
    return(list(
      model_name = model_name,
      context = setup$context,
      setup_info = setup
    ))
  }
  
  #' Generate creative content
  #' @param prompt character creative prompt
  #' @param ctx model context  
  #' @param style character writing style
  #' @param length character target length
  generate_creative_content <- function(prompt, ctx, 
                                       style = "engaging", 
                                       length = "medium") {
    
    # Length mappings
    length_tokens <- switch(length,
      "short" = 100,
      "medium" = 200, 
      "long" = 400,
      200  # default
    )
    
    # Style instructions
    style_instruction <- switch(style,
      "engaging" = "Write in an engaging, conversational style that captures the reader's attention.",
      "professional" = "Write in a professional, clear, and informative tone.",
      "creative" = "Write creatively with vivid descriptions and imaginative language.",
      "technical" = "Write in a precise, technical style with accurate terminology.",
      "Write in an engaging style."  # default
    )
    
    full_prompt <- sprintf(
      "%s\n\n%s\n\nContent:\n%s",
      style_instruction,
      prompt,
      "Here is the requested content:"
    )
    
    result <- edge_completion(ctx, full_prompt, 
                             n_predict = length_tokens, 
                             temperature = 0.8)
    
    # Clean result
    content <- gsub(".*Content:\\s*", "", result)
    content <- gsub("Here is the requested content:\\s*", "", content)
    content <- trimws(content)
    
    return(list(
      content = content,
      style = style,
      length = length,
      word_count = length(strsplit(content, "\\s+")[[1]])
    ))
  }
  
  #' Generate technical documentation
  #' @param topic character documentation topic
  #' @param ctx model context
  #' @param format character output format
  generate_documentation <- function(topic, ctx, format = "guide") {
    
    format_instruction <- switch(format,
      "guide" = "Write a step-by-step guide with clear instructions and examples.",
      "reference" = "Write comprehensive reference documentation with detailed parameters.",
      "tutorial" = "Write a beginner-friendly tutorial with explanations and examples.",
      "api" = "Write API documentation with function signatures and usage examples.",
      "Write clear technical documentation."  # default
    )
    
    prompt <- sprintf(
      "%s\n\nTopic: %s\n\nDocumentation:",
      format_instruction,
      topic
    )
    
    result <- edge_completion(ctx, prompt, n_predict = 300, temperature = 0.6)
    
    # Clean and structure result
    doc_content <- gsub(".*Documentation:\\s*", "", result)
    doc_content <- trimws(doc_content)
    
    return(list(
      content = doc_content,
      topic = topic,
      format = format,
      sections = length(grep("^#|^\\*|^\\d\\.", strsplit(doc_content, "\n")[[1]]))
    ))
  }
  
  #' Generate code examples
  #' @param description character code description
  #' @param ctx model context
  #' @param language character programming language
  generate_code <- function(description, ctx, language = "R") {
    
    language_instruction <- switch(tolower(language),
      "r" = "Write R code with proper syntax, comments, and examples.",
      "python" = "Write Python code with proper syntax, comments, and examples.", 
      "javascript" = "Write JavaScript code with proper syntax, comments, and examples.",
      "sql" = "Write SQL code with proper syntax and comments.",
      "Write clean, well-commented code."  # default
    )
    
    prompt <- sprintf(
      "%s\n\nTask: %s\n\nCode:\n```%s",
      language_instruction,
      description,
      tolower(language)
    )
    
    result <- edge_completion(ctx, prompt, n_predict = 200, temperature = 0.4)
    
    # Extract code block
    code_pattern <- sprintf("```%s\\s*([\\s\\S]*?)```", tolower(language))
    code_match <- regmatches(result, regexpr(code_pattern, result, perl = TRUE))
    
    if (length(code_match) > 0) {
      code_content <- gsub(sprintf("```%s\\s*", tolower(language)), "", code_match)
      code_content <- gsub("```.*$", "", code_content)
    } else {
      # Fallback: extract everything after "Code:"
      code_content <- gsub(sprintf(".*Code:\\s*```%s\\s*", tolower(language)), "", result)
      code_content <- gsub("```.*$", "", code_content)
    }
    
    code_content <- trimws(code_content)
    
    return(list(
      code = code_content,
      description = description,
      language = language,
      lines = length(strsplit(code_content, "\n")[[1]])
    ))
  }
  
  #' Generate structured data
  #' @param schema character data schema description  
  #' @param ctx model context
  #' @param format character output format (json, csv, yaml)
  generate_structured_data <- function(schema, ctx, format = "json") {
    
    format_instruction <- switch(tolower(format),
      "json" = "Generate valid JSON data that follows the schema exactly.",
      "csv" = "Generate CSV data with proper headers and comma separation.",
      "yaml" = "Generate valid YAML data with proper indentation.",
      "xml" = "Generate valid XML data with proper tags and structure.",
      "Generate structured data."  # default
    )
    
    prompt <- sprintf(
      "%s\n\nSchema: %s\n\nGenerate 3-5 example records.\n\nData:",
      format_instruction,
      schema
    )
    
    result <- edge_completion(ctx, prompt, n_predict = 250, temperature = 0.5)
    
    # Clean result
    data_content <- gsub(".*Data:\\s*", "", result)
    data_content <- trimws(data_content)
    
    # Validate format (basic check)
    format_valid <- switch(tolower(format),
      "json" = grepl("^\\s*[\\[\\{]", data_content),
      "csv" = grepl(",", data_content),
      "yaml" = grepl("^\\s*-\\s+", data_content),
      "xml" = grepl("<.*>", data_content),
      TRUE  # default
    )
    
    return(list(
      data = data_content,
      schema = schema,
      format = format,
      valid = format_valid,
      size = nchar(data_content)
    ))
  }
  
  #' Batch content generation
  #' @param requests list of generation requests
  #' @param model_config list model configuration
  batch_generate <- function(requests, model_config = NULL) {
    
    if (is.null(model_config)) {
      model_config <- initialize()
    }
    
    ctx <- model_config$context
    
    cat(sprintf("🏭 Batch generating %d content pieces...\n\n", length(requests)))
    
    results <- list()
    
    for (i in seq_along(requests)) {
      req <- requests[[i]]
      
      cat(sprintf("Generating item %d/%d: %s\n", i, length(requests), req$type))
      
      start_time <- Sys.time()
      
      result <- switch(req$type,
        "creative" = generate_creative_content(req$prompt, ctx, req$style, req$length),
        "documentation" = generate_documentation(req$topic, ctx, req$format),
        "code" = generate_code(req$description, ctx, req$language),
        "structured" = generate_structured_data(req$schema, ctx, req$format),
        stop(sprintf("Unknown request type: %s", req$type))
      )
      
      end_time <- Sys.time()
      
      result$generation_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      result$request_id <- i
      result$request_type <- req$type
      
      results[[i]] <- result
      
      cat(sprintf("   ✓ Generated in %.2f seconds\n", result$generation_time))
    }
    
    edge_free_model(ctx)
    
    return(results)
  }
  
  # Return generator functions
  list(
    initialize = initialize,
    generate_creative_content = generate_creative_content,
    generate_documentation = generate_documentation, 
    generate_code = generate_code,
    generate_structured_data = generate_structured_data,
    batch_generate = batch_generate
  )
}

#' Assess content quality
#' @param content character generated content
#' @param content_type character type of content
assess_content_quality <- function(content, content_type = "general") {
  
  # Basic metrics
  word_count <- length(strsplit(content, "\\s+")[[1]])
  char_count <- nchar(content)
  sentence_count <- length(strsplit(content, "[.!?]+")[[1]])
  
  # Quality indicators
  avg_word_length <- mean(nchar(strsplit(content, "\\s+")[[1]]))
  readability_score <- 206.835 - (1.015 * word_count/sentence_count) - (84.6 * avg_word_length)
  
  # Content-specific checks
  quality_checks <- switch(content_type,
    "code" = list(
      has_comments = grepl("#|//|/\\*", content),
      has_functions = grepl("function|def|=>", content),
      proper_indentation = grepl("^\\s+", content)
    ),
    "documentation" = list(
      has_headings = grepl("^#|^##", content),
      has_examples = grepl("example|Example|EXAMPLE", content),
      has_steps = grepl("^\\d\\.|^\\*|^-", content)
    ),
    "structured" = list(
      valid_format = nchar(content) > 20,
      has_structure = grepl("[\\{\\[\\<]|,|:", content)
    ),
    list(  # general content
      coherent_length = word_count >= 20 && word_count <= 500,
      proper_grammar = !grepl("\\b\\w+\\s+\\1\\b", content)  # basic duplicate check
    )
  )
  
  # Overall quality score
  quality_score <- sum(unlist(quality_checks)) / length(quality_checks)
  
  return(list(
    word_count = word_count,
    char_count = char_count,
    sentence_count = sentence_count,
    readability_score = round(readability_score, 1),
    quality_score = round(quality_score, 2),
    quality_checks = quality_checks,
    assessment = ifelse(quality_score >= 0.7, "Good", 
                       ifelse(quality_score >= 0.4, "Fair", "Poor"))
  ))
}

#' Main demonstration
main <- function() {
  cat("=== ADVANCED CONTENT GENERATION DEMO ===\n\n")
  
  # Create generator
  generator <- ContentGenerator()
  
  # Define batch requests
  content_requests <- list(
    list(
      type = "creative",
      prompt = "Write about the future of local AI inference in personal computing",
      style = "engaging",
      length = "medium"
    ),
    list(
      type = "documentation", 
      topic = "How to set up edge inference for R users",
      format = "guide"
    ),
    list(
      type = "code",
      description = "Function to load and validate a language model with error handling",
      language = "R"
    ),
    list(
      type = "structured",
      schema = "User profile with name, email, preferences, and activity metrics",
      format = "json"
    ),
    list(
      type = "creative",
      prompt = "Describe the advantages of quantized neural networks",
      style = "professional", 
      length = "short"
    )
  )
  
  # Generate content
  results <- generator$batch_generate(content_requests)
  
  # Display and assess results
  cat("=== GENERATION RESULTS ===\n\n")
  
  for (i in seq_along(results)) {
    result <- results[[i]]
    
    cat(sprintf("🎯 CONTENT %d - %s\n", i, toupper(result$request_type)))
    
    # Get content based on type
    content <- switch(result$request_type,
      "creative" = result$content,
      "documentation" = result$content,
      "code" = result$code,
      "structured" = result$data,
      "Unknown content"
    )
    
    cat(sprintf("📝 Content Preview:\n%s...\n\n", substr(content, 1, 200)))
    
    # Assess quality
    quality <- assess_content_quality(content, result$request_type)
    
    cat(sprintf("📊 Quality Assessment:\n"))
    cat(sprintf("   Words: %d | Characters: %d | Sentences: %d\n", 
                quality$word_count, quality$char_count, quality$sentence_count))
    cat(sprintf("   Quality Score: %.2f | Assessment: %s\n", 
                quality$quality_score, quality$assessment))
    cat(sprintf("   Generation Time: %.2f seconds\n", result$generation_time))
    
    # Type-specific metrics
    if (result$request_type == "creative") {
      cat(sprintf("   Style: %s | Target Length: %s\n", result$style, result$length))
    } else if (result$request_type == "code") {
      cat(sprintf("   Language: %s | Lines: %d\n", result$language, result$lines))
    } else if (result$request_type == "structured") {
      cat(sprintf("   Format: %s | Valid: %s\n", result$format, result$valid))
    }
    
    cat(sprintf("%s\n\n", paste(rep("-", 80), collapse = "")))
  }
  
  # Summary statistics
  total_time <- sum(sapply(results, function(x) x$generation_time))
  avg_quality <- mean(sapply(results, function(r) {
    content <- switch(r$request_type,
      "creative" = r$content,
      "documentation" = r$content, 
      "code" = r$code,
      "structured" = r$data,
      ""
    )
    assess_content_quality(content, r$request_type)$quality_score
  }))
  
  cat("=== SUMMARY STATISTICS ===\n")
  cat(sprintf("Total items generated: %d\n", length(results)))
  cat(sprintf("Total generation time: %.2f seconds\n", total_time))
  cat(sprintf("Average quality score: %.2f\n", avg_quality))
  cat(sprintf("Content types: %s\n", 
              paste(unique(sapply(results, function(x) x$request_type)), collapse = ", ")))
  
  return(results)
}

#' Export generated content
#' @param results list generation results
#' @param output_dir character output directory
export_generated_content <- function(results, output_dir = "generated_content") {
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  cat(sprintf("📁 Exporting content to %s/\n", output_dir))
  
  for (i in seq_along(results)) {
    result <- results[[i]]
    
    # Get content and file extension
    content_info <- switch(result$request_type,
      "creative" = list(content = result$content, ext = "txt"),
      "documentation" = list(content = result$content, ext = "md"),
      "code" = list(content = result$code, ext = tolower(result$language)),
      "structured" = list(content = result$data, ext = tolower(result$format)),
      list(content = "Unknown content", ext = "txt")
    )
    
    filename <- sprintf("%s/content_%02d_%s.%s", 
                       output_dir, i, result$request_type, content_info$ext)
    
    writeLines(content_info$content, filename)
    cat(sprintf("   ✓ %s\n", basename(filename)))
  }
  
  cat(sprintf("Content exported to %d files\n", length(results)))
}

# Run demonstration if executed directly
if (sys.nframe() == 0) {
  generation_results <- main()
  
  # Optional: export content
  # export_generated_content(generation_results)
}