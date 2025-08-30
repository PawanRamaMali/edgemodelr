#' Interactive Streaming Chat with Modern Small Models
#'
#' This example demonstrates real-time streaming conversation capabilities
#' using the latest small language models optimized for interactive chat.
#'
#' Features:
#' - Real-time streaming responses
#' - Conversation history management
#' - Multiple chat modes (assistant, creative, technical)
#' - Session persistence
#' - Performance monitoring
#' - Interactive CLI interface
#'
#' @author edgemodelr team
#' @date 2024

library(edgemodelr)

#' Streaming Chat System
#' @description Advanced chat system with streaming, context management, and multiple modes
StreamingChat <- function() {
  
  #' Initialize chat system
  #' @param model_name character model to use
  #' @param chat_mode character initial chat mode
  initialize <- function(model_name = "TinyLlama-1.1B", chat_mode = "assistant") {
    cat(sprintf("💬 Initializing Streaming Chat with %s...\n", model_name))
    cat(sprintf("🎭 Chat mode: %s\n\n", chat_mode))
    
    setup <- edge_quick_setup(model_name)
    
    # Initialize conversation history
    conversation_history <- list(
      messages = list(),
      tokens_used = 0,
      start_time = Sys.time()
    )
    
    return(list(
      model_name = model_name,
      context = setup$context,
      chat_mode = chat_mode,
      history = conversation_history,
      setup_info = setup
    ))
  }
  
  #' Get system prompt based on chat mode
  #' @param mode character chat mode
  get_system_prompt <- function(mode) {
    prompts <- list(
      "assistant" = "You are a helpful AI assistant. Provide clear, accurate, and concise responses.",
      "creative" = "You are a creative AI that helps with writing, brainstorming, and creative projects. Be imaginative and inspiring.",
      "technical" = "You are a technical AI expert. Provide detailed, accurate technical information with examples and best practices.",
      "friendly" = "You are a friendly conversational AI. Be warm, engaging, and personable in your responses.",
      "tutor" = "You are an educational AI tutor. Explain concepts clearly, ask questions to check understanding, and provide examples."
    )
    
    return(prompts[[mode]] %||% prompts[["assistant"]])
  }
  
  #' Build chat prompt with conversation history
  #' @param user_input character user's message
  #' @param history list conversation history
  #' @param mode character chat mode
  build_chat_prompt <- function(user_input, history, mode) {
    
    system_prompt <- get_system_prompt(mode)
    
    # Start with system prompt
    prompt_parts <- c(sprintf("<s>[INST] %s [/INST] I understand. I'm ready to help!</s>", system_prompt))
    
    # Add recent conversation history (last 3 exchanges to manage context)
    recent_messages <- tail(history$messages, 6)  # 3 exchanges = 6 messages
    
    for (i in seq_along(recent_messages)) {
      msg <- recent_messages[[i]]
      if (msg$role == "user") {
        prompt_parts <- c(prompt_parts, sprintf("<s>[INST] %s [/INST]", msg$content))
      } else {
        prompt_parts <- c(prompt_parts, sprintf(" %s</s>", msg$content))
      }
    }
    
    # Add current user input
    prompt_parts <- c(prompt_parts, sprintf("<s>[INST] %s [/INST]", user_input))
    
    return(paste(prompt_parts, collapse = ""))
  }
  
  #' Stream chat response with callback
  #' @param user_input character user's message
  #' @param chat_state list chat state
  #' @param stream_callback function callback for streaming chunks
  stream_chat_response <- function(user_input, chat_state, stream_callback = NULL) {
    
    start_time <- Sys.time()
    
    # Build prompt with context
    prompt <- build_chat_prompt(user_input, chat_state$history, chat_state$chat_mode)
    
    # Add user message to history
    chat_state$history$messages <- append(chat_state$history$messages, 
                                         list(list(role = "user", content = user_input, timestamp = start_time)))
    
    cat("🤖 Assistant: ")
    
    # Use streaming completion if available, otherwise simulate
    tryCatch({
      # Try streaming completion
      full_response <- ""
      
      # Create callback wrapper
      internal_callback <- function(chunk) {
        cat(chunk)  # Print to console
        flush.console()
        full_response <<- paste0(full_response, chunk)
        
        # Call user callback if provided
        if (!is.null(stream_callback)) {
          stream_callback(chunk, full_response)
        }
      }
      
      # Use streaming if available
      if (exists("edge_stream_completion")) {
        edge_stream_completion(chat_state$context, prompt, 
                              n_predict = 150, temperature = 0.7, 
                              callback = internal_callback)
      } else {
        # Fallback: simulate streaming with regular completion
        response <- edge_completion(chat_state$context, prompt, 
                                   n_predict = 150, temperature = 0.7)
        
        # Clean response (remove prompt echo)
        response <- gsub(".*\\[/INST\\]\\s*", "", response)
        response <- gsub("</s>.*$", "", response)
        response <- trimws(response)
        
        # Simulate streaming by printing character by character
        chars <- strsplit(response, "")[[1]]
        full_response <- ""
        
        for (char in chars) {
          cat(char)
          flush.console()
          full_response <- paste0(full_response, char)
          
          if (!is.null(stream_callback)) {
            stream_callback(char, full_response)
          }
          
          Sys.sleep(0.01)  # Small delay to simulate streaming
        }
      }
      
      cat("\n\n")
      
      end_time <- Sys.time()
      response_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
      
      # Add assistant response to history
      chat_state$history$messages <- append(chat_state$history$messages,
                                           list(list(role = "assistant", 
                                                    content = full_response,
                                                    timestamp = end_time,
                                                    response_time = response_time)))
      
      # Update token usage estimate
      estimated_tokens <- nchar(paste(user_input, full_response)) / 4
      chat_state$history$tokens_used <- chat_state$history$tokens_used + estimated_tokens
      
      return(list(
        response = full_response,
        response_time = response_time,
        tokens_used = estimated_tokens
      ))
      
    }, error = function(e) {
      cat(sprintf("\n❌ Error generating response: %s\n", e$message))
      return(NULL)
    })
  }
  
  #' Interactive chat loop
  #' @param chat_state list chat state
  interactive_chat <- function(chat_state) {
    
    cat("=== INTERACTIVE STREAMING CHAT ===\n")
    cat(sprintf("Model: %s | Mode: %s\n", chat_state$model_name, chat_state$chat_mode))
    cat("Commands: /quit to exit, /mode <mode> to change mode, /history to view conversation\n")
    cat("Available modes: assistant, creative, technical, friendly, tutor\n\n")
    
    while (TRUE) {
      # Get user input
      cat("👤 You: ")
      user_input <- readline()
      
      # Handle commands
      if (user_input == "/quit" || user_input == "exit" || user_input == "quit") {
        cat("👋 Goodbye! Thanks for chatting!\n")
        break
      }
      
      if (startsWith(user_input, "/mode ")) {
        new_mode <- trimws(sub("^/mode\\s+", "", user_input))
        valid_modes <- c("assistant", "creative", "technical", "friendly", "tutor")
        
        if (new_mode %in% valid_modes) {
          chat_state$chat_mode <- new_mode
          cat(sprintf("🎭 Chat mode changed to: %s\n\n", new_mode))
        } else {
          cat(sprintf("❌ Invalid mode. Valid modes: %s\n\n", paste(valid_modes, collapse = ", ")))
        }
        next
      }
      
      if (user_input == "/history") {
        show_conversation_history(chat_state$history)
        next
      }
      
      if (user_input == "/stats") {
        show_chat_statistics(chat_state)
        next
      }
      
      if (trimws(user_input) == "") {
        cat("Please enter a message or /quit to exit.\n\n")
        next
      }
      
      # Generate streaming response
      stream_chat_response(user_input, chat_state)
    }
    
    # Cleanup
    edge_free_model(chat_state$context)
    
    return(chat_state)
  }
  
  #' Show conversation history
  #' @param history list conversation history
  show_conversation_history <- function(history) {
    cat("\n=== CONVERSATION HISTORY ===\n")
    
    if (length(history$messages) == 0) {
      cat("No messages yet.\n\n")
      return()
    }
    
    for (i in seq_along(history$messages)) {
      msg <- history$messages[[i]]
      icon <- ifelse(msg$role == "user", "👤", "🤖")
      role <- ifelse(msg$role == "user", "You", "Assistant")
      
      cat(sprintf("%s %s: %s\n", icon, role, substr(msg$content, 1, 100)))
      if (nchar(msg$content) > 100) cat("...\n")
      
      if (!is.null(msg$response_time)) {
        cat(sprintf("   (Response time: %.2fs)\n", msg$response_time))
      }
      cat("\n")
    }
  }
  
  #' Show chat statistics
  #' @param chat_state list chat state
  show_chat_statistics <- function(chat_state) {
    history <- chat_state$history
    
    cat("\n=== CHAT STATISTICS ===\n")
    cat(sprintf("Model: %s\n", chat_state$model_name))
    cat(sprintf("Chat Mode: %s\n", chat_state$chat_mode))
    cat(sprintf("Session Duration: %.1f minutes\n", 
                as.numeric(difftime(Sys.time(), history$start_time, units = "mins"))))
    cat(sprintf("Total Messages: %d\n", length(history$messages)))
    
    user_messages <- sum(sapply(history$messages, function(x) x$role == "user"))
    assistant_messages <- sum(sapply(history$messages, function(x) x$role == "assistant"))
    
    cat(sprintf("User Messages: %d\n", user_messages))
    cat(sprintf("Assistant Messages: %d\n", assistant_messages))
    
    if (assistant_messages > 0) {
      response_times <- sapply(history$messages, function(x) {
        if (x$role == "assistant" && !is.null(x$response_time)) x$response_time else NA
      })
      response_times <- response_times[!is.na(response_times)]
      
      if (length(response_times) > 0) {
        cat(sprintf("Average Response Time: %.2f seconds\n", mean(response_times)))
        cat(sprintf("Fastest Response: %.2f seconds\n", min(response_times)))
        cat(sprintf("Slowest Response: %.2f seconds\n", max(response_times)))
      }
    }
    
    cat(sprintf("Estimated Tokens Used: %.0f\n", history$tokens_used))
    cat("\n")
  }
  
  #' Batch conversation simulation
  #' @param conversations list of conversation scenarios
  #' @param chat_state list chat state
  simulate_conversations <- function(conversations, chat_state) {
    cat("=== CONVERSATION SIMULATION ===\n\n")
    
    results <- list()
    
    for (i in seq_along(conversations)) {
      conv <- conversations[[i]]
      cat(sprintf("🎬 Scenario %d: %s\n", i, conv$title))
      
      scenario_results <- list(
        title = conv$title,
        exchanges = list(),
        total_time = 0
      )
      
      for (j in seq_along(conv$exchanges)) {
        user_input <- conv$exchanges[[j]]
        cat(sprintf("👤 User: %s\n", user_input))
        
        # Generate response
        response_info <- stream_chat_response(user_input, chat_state)
        
        if (!is.null(response_info)) {
          scenario_results$exchanges[[j]] <- list(
            user_input = user_input,
            assistant_response = response_info$response,
            response_time = response_info$response_time
          )
          scenario_results$total_time <- scenario_results$total_time + response_info$response_time
        }
        
        Sys.sleep(0.5)  # Brief pause between exchanges
      }
      
      results[[i]] <- scenario_results
      cat(sprintf("📊 Scenario completed in %.2f seconds\n\n", scenario_results$total_time))
    }
    
    return(results)
  }
  
  # Return chat functions
  list(
    initialize = initialize,
    stream_chat_response = stream_chat_response,
    interactive_chat = interactive_chat,
    simulate_conversations = simulate_conversations,
    show_conversation_history = show_conversation_history,
    show_chat_statistics = show_chat_statistics
  )
}

#' Demo conversations for testing
get_demo_conversations <- function() {
  list(
    list(
      title = "Technical Support",
      exchanges = c(
        "How do I install a new R package?",
        "What if I get permission errors?",
        "Thanks for the help!"
      )
    ),
    list(
      title = "Creative Writing",
      exchanges = c(
        "Help me write a short story about AI",
        "Make it more suspenseful",
        "Perfect! Can you suggest a title?"
      )
    ),
    list(
      title = "Educational Q&A",
      exchanges = c(
        "Explain how neural networks work",
        "What's the difference between supervised and unsupervised learning?",
        "Can you give an example of each?"
      )
    )
  )
}

#' Main demonstration
main <- function() {
  cat("=== STREAMING CHAT DEMONSTRATION ===\n\n")
  
  # Create chat system
  chat_system <- StreamingChat()
  
  # Initialize chat
  chat_state <- chat_system$initialize()
  
  # Option 1: Run interactive chat
  cat("Choose demo mode:\n")
  cat("1. Interactive chat (type messages)\n")
  cat("2. Automated conversation simulation\n")
  cat("3. Both\n\n")
  
  # For demo purposes, run simulation
  cat("Running automated simulation...\n\n")
  
  # Simulate conversations
  demo_conversations <- get_demo_conversations()
  simulation_results <- chat_system$simulate_conversations(demo_conversations, chat_state)
  
  # Show final statistics
  chat_system$show_chat_statistics(chat_state)
  
  cat("=== SIMULATION SUMMARY ===\n")
  for (i in seq_along(simulation_results)) {
    result <- simulation_results[[i]]
    cat(sprintf("📋 %s: %d exchanges in %.2f seconds\n", 
                result$title, length(result$exchanges), result$total_time))
  }
  
  # Interactive mode option
  cat("\n💬 Would you like to start interactive chat? (y/n): ")
  if (interactive()) {
    choice <- readline()
    if (tolower(choice) %in% c("y", "yes")) {
      chat_system$interactive_chat(chat_state)
    }
  } else {
    cat("n\n")
    cat("👋 Demo complete! Use interactive_chat() for live conversation.\n")
  }
  
  return(list(
    chat_state = chat_state,
    simulation_results = simulation_results
  ))
}

#' Start interactive chat session
#' @param model_name character model to use
#' @param mode character initial chat mode
start_chat <- function(model_name = "TinyLlama-1.1B", mode = "assistant") {
  chat_system <- StreamingChat()
  chat_state <- chat_system$initialize(model_name, mode)
  chat_system$interactive_chat(chat_state)
}

# Run demonstration if executed directly
if (sys.nframe() == 0) {
  demo_results <- main()
}