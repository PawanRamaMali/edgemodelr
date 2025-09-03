# Data Science Assistant Example
# Using edgemodelr as an AI assistant for data science tasks

library(edgemodelr)

cat("🔬 Data Science AI Assistant with edgemodelr\n")
cat(rep("=", 60), "\n\n")

# Initialize the assistant
initialize_assistant <- function() {
  cat("Initializing Data Science Assistant...\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Using demo mode for non-interactive session\n")
      return(NULL)
    }
    
    # Use a capable model for data science tasks
    setup <- edge_quick_setup("llama3.2-1b")  # Good for reasoning tasks
    
    if (!is.null(setup$context)) {
      cat("✅ Assistant ready! Model:", setup$info$name, "\n\n")
      return(setup$context)
    } else {
      cat("❌ Failed to load assistant model\n")
      return(NULL)
    }
    
  }, error = function(e) {
    cat("❌ Assistant initialization failed:", e$message, "\n")
    return(NULL)
  })
}

# R Code Help Function
get_r_help <- function(ctx, question, code_context = "") {
  if (is.null(ctx)) return("Assistant not available")
  
  prompt <- paste0(
    "You are an expert R programmer. Help with this question:\n",
    "Question: ", question, "\n",
    if (nchar(code_context) > 0) paste0("Context: ", code_context, "\n"),
    "Provide a clear, practical answer with R code examples.\n",
    "Answer:"
  )
  
  response <- edge_completion(
    ctx, 
    prompt, 
    n_predict = 200,
    temperature = 0.3,  # More factual, less creative
    top_p = 0.9
  )
  
  # Clean up the response
  answer <- sub(".*Answer:", "", response)
  return(trimws(answer))
}

# Statistical Analysis Helper
get_stats_help <- function(ctx, data_description, analysis_question) {
  if (is.null(ctx)) return("Assistant not available")
  
  prompt <- paste0(
    "You are a statistics expert. Help with this analysis:\n",
    "Data: ", data_description, "\n",
    "Question: ", analysis_question, "\n",
    "Provide specific R code and explain the statistical approach.\n",
    "Response:"
  )
  
  response <- edge_completion(
    ctx,
    prompt,
    n_predict = 250,
    temperature = 0.2,  # Very factual for statistics
    top_p = 0.85
  )
  
  answer <- sub(".*Response:", "", response)
  return(trimws(answer))
}

# Data Visualization Helper
get_visualization_help <- function(ctx, data_type, chart_request) {
  if (is.null(ctx)) return("Assistant not available")
  
  prompt <- paste0(
    "You are a data visualization expert using R and ggplot2.\n",
    "Data type: ", data_type, "\n",
    "Visualization request: ", chart_request, "\n",
    "Provide complete ggplot2 code with proper styling and labels.\n",
    "Code:"
  )
  
  response <- edge_completion(
    ctx,
    prompt,
    n_predict = 300,
    temperature = 0.4,
    top_p = 0.9
  )
  
  code <- sub(".*Code:", "", response)
  return(trimws(code))
}

# Code Review Helper
review_r_code <- function(ctx, code_snippet) {
  if (is.null(ctx)) return("Assistant not available")
  
  prompt <- paste0(
    "Review this R code for best practices, efficiency, and potential issues:\n\n",
    "```r\n",
    code_snippet, "\n",
    "```\n\n",
    "Provide:\n",
    "1. Issues found (if any)\n",
    "2. Suggestions for improvement\n",
    "3. Improved version (if needed)\n",
    "Review:"
  )
  
  response <- edge_completion(
    ctx,
    prompt,
    n_predict = 400,
    temperature = 0.3,
    top_p = 0.9
  )
  
  review <- sub(".*Review:", "", response)
  return(trimws(review))
}

# Demo function to showcase all capabilities
demo_data_science_assistant <- function() {
  cat("🎬 Data Science Assistant Demo\n")
  cat(rep("-", 40), "\n\n")
  
  # Initialize
  ctx <- initialize_assistant()
  if (is.null(ctx)) {
    cat("Demo unavailable - assistant not loaded\n")
    return(NULL)
  }
  
  # Demo 1: R Help
  cat("📝 Demo 1: R Programming Help\n")
  cat("Question: How do I handle missing values in a dataset?\n\n")
  
  r_help <- get_r_help(
    ctx, 
    "How do I handle missing values in a dataset?",
    "Working with a data.frame that has NA values in multiple columns"
  )
  cat("Assistant Response:\n", r_help, "\n\n")
  
  if (interactive()) Sys.sleep(1)
  
  # Demo 2: Statistical Analysis
  cat("📊 Demo 2: Statistical Analysis Help\n") 
  cat("Data: Customer satisfaction scores (1-10) from two different products\n")
  cat("Question: How do I test if there's a significant difference?\n\n")
  
  stats_help <- get_stats_help(
    ctx,
    "Customer satisfaction scores (1-10) from two different products",
    "How do I test if there's a significant difference between the two products?"
  )
  cat("Assistant Response:\n", stats_help, "\n\n")
  
  if (interactive()) Sys.sleep(1)
  
  # Demo 3: Visualization Help
  cat("📈 Demo 3: Data Visualization Help\n")
  cat("Data: Time series sales data with seasonal patterns\n")
  cat("Request: Create an elegant time series plot with trend line\n\n")
  
  viz_help <- get_visualization_help(
    ctx,
    "Time series sales data with seasonal patterns", 
    "Create an elegant time series plot with trend line and seasonal highlights"
  )
  cat("Assistant Response:\n", viz_help, "\n\n")
  
  if (interactive()) Sys.sleep(1)
  
  # Demo 4: Code Review
  cat("🔍 Demo 4: Code Review\n")
  cat("Reviewing a data processing function:\n\n")
  
  sample_code <- "
  process_data <- function(df) {
    for (i in 1:nrow(df)) {
      if (is.na(df[i,2])) {
        df[i,2] <- mean(df[,2], na.rm=TRUE)
      }
    }
    return(df)
  }"
  
  code_review <- review_r_code(ctx, sample_code)
  cat("Code to review:\n", sample_code, "\n\n")
  cat("Assistant Review:\n", code_review, "\n\n")
  
  # Clean up
  edge_free_model(ctx)
  cat("✅ Demo complete, model cleaned up\n")
}

# Interactive assistant function
interactive_data_science_assistant <- function() {
  cat("🤖 Interactive Data Science Assistant\n")
  cat("Type your R/statistics questions. Commands:\n")
  cat("- 'help [question]' for R programming help\n") 
  cat("- 'stats [description] [question]' for statistical analysis\n")
  cat("- 'plot [data_type] [request]' for visualization help\n")
  cat("- 'review [code]' for code review\n")
  cat("- 'quit' to exit\n\n")
  
  ctx <- initialize_assistant()
  if (is.null(ctx)) {
    cat("Interactive assistant unavailable\n")
    return(NULL)
  }
  
  while (TRUE) {
    user_input <- readline("🔬 Ask: ")
    
    if (tolower(trimws(user_input)) %in% c("quit", "exit", "bye")) {
      break
    }
    
    if (startsWith(user_input, "help ")) {
      question <- substring(user_input, 6)
      response <- get_r_help(ctx, question)
      cat("💡", response, "\n\n")
      
    } else if (startsWith(user_input, "stats ")) {
      parts <- strsplit(substring(user_input, 7), " ", fixed = TRUE)[[1]]
      if (length(parts) >= 2) {
        data_desc <- paste(parts[1:(length(parts)%/%2)], collapse = " ")
        question <- paste(parts[(length(parts)%/%2 + 1):length(parts)], collapse = " ")
        response <- get_stats_help(ctx, data_desc, question)
        cat("📊", response, "\n\n")
      } else {
        cat("Format: stats [data description] [question]\n\n")
      }
      
    } else if (startsWith(user_input, "plot ")) {
      parts <- strsplit(substring(user_input, 6), " ", fixed = TRUE)[[1]]
      if (length(parts) >= 2) {
        data_type <- parts[1]
        request <- paste(parts[2:length(parts)], collapse = " ")
        response <- get_visualization_help(ctx, data_type, request)
        cat("📈", response, "\n\n")
      } else {
        cat("Format: plot [data type] [visualization request]\n\n")
      }
      
    } else if (startsWith(user_input, "review ")) {
      code <- substring(user_input, 8)
      response <- review_r_code(ctx, code)
      cat("🔍", response, "\n\n")
      
    } else {
      # General question
      response <- get_r_help(ctx, user_input)
      cat("🤖", response, "\n\n")
    }
  }
  
  edge_free_model(ctx)
  cat("👋 Assistant session ended\n")
}

# Batch processing helper
batch_process_questions <- function(questions_file = NULL) {
  cat("📋 Batch Processing Data Science Questions\n")
  cat(rep("-", 40), "\n\n")
  
  # Sample questions if no file provided
  if (is.null(questions_file)) {
    questions <- list(
      list(type = "help", question = "How to merge two data frames in R?"),
      list(type = "stats", data = "Heights of students", question = "Test if height differs by gender"),
      list(type = "plot", data = "Correlation matrix", request = "Create a correlation heatmap"),
      list(type = "help", question = "Best way to handle outliers in regression analysis?")
    )
  } else {
    # Load questions from file (implement as needed)
    questions <- readRDS(questions_file)
  }
  
  ctx <- initialize_assistant()
  if (is.null(ctx)) return(NULL)
  
  results <- list()
  
  for (i in seq_along(questions)) {
    q <- questions[[i]]
    cat("Processing question", i, ":", q$question, "\n")
    
    if (q$type == "help") {
      response <- get_r_help(ctx, q$question)
    } else if (q$type == "stats") {
      response <- get_stats_help(ctx, q$data, q$question)
    } else if (q$type == "plot") {
      response <- get_visualization_help(ctx, q$data, q$request)
    }
    
    results[[i]] <- list(
      question = q,
      response = response,
      timestamp = Sys.time()
    )
    
    cat("✅ Completed\n\n")
  }
  
  edge_free_model(ctx)
  return(results)
}

# Main execution
if (interactive()) {
  cat("Choose your option:\n")
  cat("1. demo_data_science_assistant() - Run demo\n")
  cat("2. interactive_data_science_assistant() - Interactive session\n") 
  cat("3. batch_process_questions() - Process multiple questions\n\n")
  cat("Or run demo automatically:\n")
  demo_data_science_assistant()
} else {
  # Run demo in non-interactive mode
  demo_data_science_assistant()
}