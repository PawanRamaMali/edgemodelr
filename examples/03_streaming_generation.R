# ==============================================================================
# Streaming Text Generation Examples for edgemodelr
# ==============================================================================
# This file demonstrates real-time streaming text generation, interactive
# chat interfaces, and callback-based processing for responsive applications.

library(edgemodelr)

# ------------------------------------------------------------------------------
# Example 1: Basic Streaming Generation
# ------------------------------------------------------------------------------

cat("Example 1: Basic Streaming Generation\n")
cat("=====================================\n\n")

# Setup model
setup <- edge_setup()
if (is.null(setup) || length(setup$available_models) == 0) {
  # Try Ollama models if available
  ollama_info <- edge_find_ollama_models()
  if (!is.null(ollama_info) && length(ollama_info$models) > 0) {
    ctx <- edge_load_ollama_model(substr(ollama_info$models[[1]]$sha256, 1, 8),
                                  n_ctx = 1024, n_gpu_layers = 0)
  } else {
    cat("❌ No models found. Please install models first.\n")
    quit()
  }
} else {
  ctx <- edge_load_model(setup$available_models[1], n_ctx = 1024, n_gpu_layers = 0)
}

# Define a simple callback that prints tokens as they arrive
stream_callback <- function(token) {
  cat(token)
  flush.console()  # Ensure immediate output
  return(TRUE)     # Continue streaming
}

# Stream a story generation
cat("Generating story in real-time:\n")
cat("==============================\n")
prompt <- "Once upon a time in a magical forest"

edge_stream_completion(
  ctx = ctx,
  prompt = prompt,
  callback = stream_callback,
  n_predict = 100,
  temperature = 0.8
)

cat("\n\n✅ Example 1 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 2: Interactive Callback with Control
# ------------------------------------------------------------------------------

cat("Example 2: Interactive Callback with Control\n")
cat("============================================\n\n")

# More sophisticated callback that can stop generation
token_count <- 0
max_tokens <- 50

controlled_callback <- function(token) {
  token_count <<- token_count + 1

  # Add some formatting
  if (token == "\n") {
    cat("\n[", token_count, "] ", sep = "")
  } else {
    cat(token)
  }

  # Stop after max_tokens
  if (token_count >= max_tokens) {
    cat("\n[STOPPED: Reached token limit]")
    return(FALSE)  # Stop streaming
  }

  flush.console()
  return(TRUE)  # Continue streaming
}

cat("Controlled generation with token counting:\n")
prompt <- "Explain the benefits of R programming"

# Reset counter
token_count <- 0

edge_stream_completion(
  ctx = ctx,
  prompt = prompt,
  callback = controlled_callback,
  n_predict = 200,  # Request more than max_tokens to test control
  temperature = 0.6
)

cat("\n\n✅ Example 2 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 3: Collecting Streaming Output
# ------------------------------------------------------------------------------

cat("Example 3: Collecting Streaming Output\n")
cat("======================================\n\n")

# Callback that collects output for post-processing
collected_text <- character(0)

collecting_callback <- function(token) {
  collected_text <<- c(collected_text, token)
  cat(".")  # Show progress without printing content
  flush.console()
  return(TRUE)
}

cat("Collecting streaming output (dots show progress):\n")
prompt <- "List three advantages of machine learning"

edge_stream_completion(
  ctx = ctx,
  prompt = prompt,
  callback = collecting_callback,
  n_predict = 80,
  temperature = 0.4
)

# Process collected output
full_text <- paste(collected_text, collapse = "")
cat("\n\nCollected text:\n")
cat("===============\n")
cat(full_text)
cat("\n\nText statistics:\n")
cat("• Total tokens:", length(collected_text), "\n")
cat("• Total characters:", nchar(full_text), "\n")
cat("• Average token length:", round(nchar(full_text) / length(collected_text), 2), "\n")

cat("\n✅ Example 3 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 4: Streaming vs Non-Streaming Comparison
# ------------------------------------------------------------------------------

cat("Example 4: Streaming vs Non-Streaming Comparison\n")
cat("================================================\n\n")

prompt <- "Describe the water cycle in nature"

# Time the non-streaming approach
cat("Non-streaming generation:\n")
start_time <- Sys.time()
normal_result <- edge_completion(ctx, prompt, n_predict = 60, temperature = 0.5)
normal_time <- as.numeric(Sys.time() - start_time)

cat(normal_result)
cat(sprintf("\nCompleted in %.2f seconds\n\n", normal_time))

# Time the streaming approach
cat("Streaming generation:\n")
streaming_tokens <- character(0)
stream_start <- Sys.time()

timed_callback <- function(token) {
  streaming_tokens <<- c(streaming_tokens, token)
  cat(token)
  flush.console()
  return(TRUE)
}

edge_stream_completion(ctx, prompt, timed_callback, n_predict = 60, temperature = 0.5)
streaming_time <- as.numeric(Sys.time() - stream_start)

cat(sprintf("\nCompleted in %.2f seconds\n\n", streaming_time))

# Compare results
streaming_result <- paste(streaming_tokens, collapse = "")
cat("Comparison:\n")
cat("• Normal length:", nchar(normal_result), "characters\n")
cat("• Streaming length:", nchar(streaming_result), "characters\n")
cat("• Time difference:", round(streaming_time - normal_time, 2), "seconds\n")
cat("• Streaming provides real-time feedback!\n")

cat("\n✅ Example 4 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 5: Interactive Chat Interface
# ------------------------------------------------------------------------------

cat("Example 5: Interactive Chat Interface\n")
cat("=====================================\n\n")

# Simulate an interactive chat (in practice, you'd use readline or similar)
chat_history <- character(0)

# Function to simulate user input (replace with real input in practice)
simulate_user_inputs <- function() {
  return(c(
    "Hello! How are you?",
    "What can you help me with?",
    "Tell me a fun fact about R programming",
    "Thank you, goodbye!"
  ))
}

# Chat callback that formats responses nicely
chat_callback <- function(token) {
  cat(token)
  flush.console()
  return(TRUE)
}

cat("Starting simulated chat session:\n")
cat("================================\n\n")

user_inputs <- simulate_user_inputs()

for (i in seq_along(user_inputs)) {
  user_input <- user_inputs[i]

  # Display user input
  cat("User: ", user_input, "\n")
  cat("Assistant: ")

  # Create context-aware prompt
  if (length(chat_history) > 0) {
    # Include recent history for context
    recent_history <- tail(chat_history, 4)  # Last 2 exchanges
    context_prompt <- paste(c(recent_history, paste("User:", user_input), "Assistant:"), collapse = "\n")
  } else {
    context_prompt <- paste("User:", user_input, "\nAssistant:")
  }

  # Generate streaming response
  response_tokens <- character(0)
  response_callback <- function(token) {
    response_tokens <<- c(response_tokens, token)
    cat(token)
    flush.console()
    return(TRUE)
  }

  edge_stream_completion(
    ctx = ctx,
    prompt = context_prompt,
    callback = response_callback,
    n_predict = 50,
    temperature = 0.7
  )

  # Store in chat history
  response_text <- paste(response_tokens, collapse = "")
  chat_history <- c(chat_history,
                   paste("User:", user_input),
                   paste("Assistant:", response_text))

  cat("\n\n")
}

cat("✅ Example 5 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Example 6: Advanced Streaming Features
# ------------------------------------------------------------------------------

cat("Example 6: Advanced Streaming Features\n")
cat("======================================\n\n")

# Advanced callback with multiple features
advanced_callback <- function(token) {
  # Skip empty tokens
  if (nchar(token) == 0) return(TRUE)

  # Highlight certain words in the stream
  highlighted_token <- token
  if (grepl("important|key|crucial|essential", token, ignore.case = TRUE)) {
    highlighted_token <- paste0("**", token, "**")
  }

  # Add timestamp occasionally
  if (runif(1) < 0.1) {  # 10% chance
    cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")))
  }

  cat(highlighted_token)
  flush.console()

  # Continue streaming
  return(TRUE)
}

cat("Advanced streaming with highlighting and timestamps:\n")
prompt <- "Discuss the key principles of effective data visualization"

edge_stream_completion(
  ctx = ctx,
  prompt = prompt,
  callback = advanced_callback,
  n_predict = 100,
  temperature = 0.6
)

# Clean up
edge_free_model(ctx)

cat("\n\n✅ Example 6 completed successfully!\n\n")

# ------------------------------------------------------------------------------
# Summary and Best Practices
# ------------------------------------------------------------------------------

cat("🎉 Streaming Generation Examples Complete!\n")
cat("==========================================\n\n")

cat("Best practices for streaming generation:\n\n")

cat("🚀 Performance:\n")
cat("• Use streaming for better user experience in interactive apps\n")
cat("• Streaming provides immediate feedback vs waiting for completion\n")
cat("• Implement progress indicators using callback functions\n\n")

cat("🎛️  Callback Design:\n")
cat("• Return TRUE to continue, FALSE to stop generation early\n")
cat("• Use flush.console() to ensure immediate output display\n")
cat("• Implement token counting for precise control\n")
cat("• Collect tokens in callbacks for post-processing\n\n")

cat("💬 Interactive Applications:\n")
cat("• Maintain chat history for context-aware conversations\n")
cat("• Use edge_chat_stream() for built-in chat interface\n")
cat("• Implement proper error handling in callbacks\n")
cat("• Consider rate limiting for production applications\n\n")

cat("🛠️  Advanced Features:\n")
cat("• Add timestamps, formatting, or highlighting in callbacks\n")
cat("• Implement conditional stopping based on content\n")
cat("• Use streaming for real-time data processing pipelines\n")
cat("• Combine with async processing for scalable applications\n\n")

cat("⚠️  Important Notes:\n")
cat("• Streaming uses the same model context as regular generation\n")
cat("• Callback errors will stop the streaming process\n")
cat("• Always call edge_free_model() when done\n")
cat("• Test callbacks thoroughly before production use\n\n")

cat("Next: See 04_performance_optimization.R for performance tips\n")