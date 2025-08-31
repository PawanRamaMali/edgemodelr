#' Advanced Shiny Streaming Chat Application
#'
#' This is a completely rewritten streaming chat application that provides
#' real-time token-by-token streaming with proper stop functionality.
#'
#' Features:
#' - Fast, real-time streaming updates
#' - Working stop streaming button
#' - Clean message formatting
#' - Model selection and management
#' - Proper error handling
#'
#' @author edgemodelr team
#' @date 2024

library(shiny)
library(shinyjs)
library(edgemodelr)

ui <- fluidPage(
  useShinyjs(),
  
  tags$head(
    tags$style(HTML("
      .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
      }
      
      .chat-container {
        border: 2px solid #ddd;
        border-radius: 10px;
        height: 500px;
        overflow-y: auto;
        padding: 15px;
        background-color: #fafafa;
        margin-bottom: 20px;
      }
      
      .message {
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 8px;
        clear: both;
      }
      
      .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-right: 20%;
      }
      
      .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
        margin-left: 20%;
      }
      
      .message-header {
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 14px;
      }
      
      .message-content {
        line-height: 1.4;
        white-space: pre-wrap;
      }
      
      .controls {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
      }
      
      .status-bar {
        background-color: #e8f5e8;
        border: 1px solid #c8e6c9;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        font-weight: bold;
      }
      
      .spinner {
        display: none;
        text-align: center;
        margin: 20px 0;
      }
      
      .spinner.active {
        display: block;
      }
      
      .btn-stop {
        background-color: #f44336 !important;
        border-color: #f44336 !important;
        color: white !important;
      }
      
      .btn-stop:hover {
        background-color: #d32f2f !important;
        border-color: #d32f2f !important;
      }
      
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    "))
  ),
  
  div(class = "main-container",
    titlePanel("edgemodelr Advanced Streaming Chat"),
    
    # Status bar
    div(id = "status-bar", class = "status-bar", "Loading models..."),
    
    # Model selection
    div(class = "controls",
      selectInput("model", "Select Model:", choices = NULL, width = "300px"),
      actionButton("load_model", "Load Model", class = "btn btn-info")
    ),
    
    # Chat display
    div(id = "chat-display", class = "chat-container",
      div(class = "message",
        div(class = "message-content", "No messages yet. Select a model and start chatting!")
      )
    ),
    
    # Spinner
    div(id = "spinner", class = "spinner",
      tags$div(style = "border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto;"),
      tags$p("Generating response...", style = "margin-top: 10px;")
    ),
    
    # Input controls
    div(class = "controls",
      textAreaInput("user_input", NULL, "", width = "100%", rows = 3, 
                   placeholder = "Type your message here..."),
      actionButton("send", "Send", class = "btn btn-primary"),
      actionButton("stop", "Stop", class = "btn btn-stop", style = "display: none;"),
      actionButton("clear", "Clear Chat", class = "btn btn-secondary")
    )
  ),
  
  # JavaScript for real-time updates
  tags$script(HTML("
    // Global variables for streaming
    var isStreaming = false;
    var streamingStopped = false;
    
    // Add custom message handler for streaming updates
    Shiny.addCustomMessageHandler('streamUpdate', function(data) {
      updateStreamingMessage(data.text);
    });
    
    // Add custom message handler for chat updates
    Shiny.addCustomMessageHandler('chatUpdate', function(data) {
      document.getElementById('chat-display').innerHTML = data.html;
      scrollToBottom();
    });
    
    // Add custom message handler for status updates
    Shiny.addCustomMessageHandler('statusUpdate', function(data) {
      document.getElementById('status-bar').innerHTML = data.message;
    });
    
    // Add custom message handler for UI state
    Shiny.addCustomMessageHandler('uiState', function(data) {
      var sendBtn = document.getElementById('send');
      var stopBtn = document.getElementById('stop');
      var spinner = document.getElementById('spinner');
      
      if (data.streaming) {
        sendBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        spinner.classList.add('active');
        isStreaming = true;
        // Reset cached element for new streaming session
        streamingMessageElement = null;
      } else {
        sendBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
        spinner.classList.remove('active');
        isStreaming = false;
        // Clear cached element
        streamingMessageElement = null;
      }
    });
    
    // Optimized function to update streaming message with minimal DOM queries
    var streamingMessageElement = null;
    
    function updateStreamingMessage(text) {
      // Cache the streaming element to avoid repeated DOM queries
      if (!streamingMessageElement) {
        var chatDisplay = document.getElementById('chat-display');
        var messages = chatDisplay.querySelectorAll('.message');
        var lastMessage = messages[messages.length - 1];
        
        // Find or create the streaming message element
        if (lastMessage && lastMessage.classList.contains('assistant-message')) {
          var contentDiv = lastMessage.querySelector('.message-content');
          if (contentDiv && (contentDiv.innerHTML.indexOf('...') !== -1 || isStreaming)) {
            streamingMessageElement = contentDiv;
          }
        }
        
        if (!streamingMessageElement) {
          // Create new assistant message for streaming
          var newMessage = document.createElement('div');
          newMessage.className = 'message assistant-message';
          newMessage.innerHTML = '<div class=\"message-header\">Assistant</div><div class=\"message-content\"></div>';
          chatDisplay.appendChild(newMessage);
          streamingMessageElement = newMessage.querySelector('.message-content');
        }
      }
      
      // Fast update - direct innerHTML assignment
      if (streamingMessageElement) {
        streamingMessageElement.innerHTML = text;
      }
      
      // Only scroll every few updates for performance
      if (Math.random() < 0.3) {  // 30% chance to scroll, reduces scroll overhead
        scrollToBottom();
      }
    }
    
    // Function to scroll to bottom
    function scrollToBottom() {
      var chatDisplay = document.getElementById('chat-display');
      chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }
  "))
)

server <- function(input, output, session) {
  # Reactive values
  values <- reactiveValues(
    history = character(0),
    ctx = NULL,
    is_generating = FALSE,
    current_response = ""
  )
  
  # Global stop flag that callback can access reliably
  session_env <- new.env()
  session_env$stop_streaming <- FALSE
  
  # Initialize app
  observe({
    tryCatch({
      models <- edge_list_models()
      if (nrow(models) > 0) {
        updateSelectInput(session, "model", choices = setNames(models$name, models$name))
        session$sendCustomMessage("statusUpdate", list(message = "Select a model to begin."))
      } else {
        session$sendCustomMessage("statusUpdate", list(message = "No models available."))
      }
    }, error = function(e) {
      session$sendCustomMessage("statusUpdate", list(message = paste("Error loading models:", e$message)))
    })
  })
  
  # Load model
  observeEvent(input$load_model, {
    req(input$model)
    
    if (values$is_generating) {
      showNotification("Please wait for current generation to complete.", type = "warning")
      return()
    }
    
    session$sendCustomMessage("statusUpdate", list(message = paste("Loading model:", input$model, "...")))
    
    # Free existing model
    if (!is.null(values$ctx)) {
      tryCatch(edge_free_model(values$ctx), error = function(e) {})
      values$ctx <- NULL
    }
    
    # Load new model
    tryCatch({
      setup <- edge_quick_setup(input$model)
      values$ctx <- setup$context
      
      if (!is.null(values$ctx)) {
        session$sendCustomMessage("statusUpdate", list(message = paste("Model", input$model, "loaded successfully.")))
      } else {
        session$sendCustomMessage("statusUpdate", list(message = "Model downloaded but not loaded. Check llama.cpp installation."))
      }
    }, error = function(e) {
      session$sendCustomMessage("statusUpdate", list(message = paste("Error loading model:", e$message)))
    })
  })
  
  # Send message
  observeEvent(input$send, {
    req(input$user_input, values$ctx)
    
    if (values$is_generating) {
      showNotification("Please wait for current generation to complete.", type = "warning")
      return()
    }
    
    if (trimws(input$user_input) == "") {
      showNotification("Please enter a message.", type = "warning")
      return()
    }
    
    # Add user message to history
    user_msg <- trimws(input$user_input)
    values$history <- c(values$history, paste("You:", user_msg))
    
    # Clear input
    updateTextAreaInput(session, "user_input", value = "")
    
    # Update UI state
    values$is_generating <- TRUE
    session_env$stop_streaming <- FALSE
    values$current_response <- ""
    session$sendCustomMessage("uiState", list(streaming = TRUE))
    session$sendCustomMessage("statusUpdate", list(message = "Generating response..."))
    
    # Update chat display with user message and prepare for streaming
    updateChatDisplay(values$history, "...")  # Add placeholder for assistant response
    
    # Build optimized prompt - keep it short for faster inference
    recent_history <- if (length(values$history) > 4) tail(values$history, 4) else values$history
    
    # Short system prompt to reduce processing time
    system_prompt <- "Be helpful and direct."
    
    # Minimal prompt formatting for speed
    if (length(recent_history) > 0) {
      prompt <- paste(c(system_prompt, recent_history, "Assistant:"), collapse = "\n")
    } else {
      prompt <- paste0(system_prompt, "\nAssistant:")
    }
    
    # Start streaming - store context locally
    local_ctx <- values$ctx
    
    tryCatch({
      # Start streaming with optimized parameters for speed
      result <- edge_stream_completion(
        ctx = local_ctx,
        prompt = prompt,
        n_predict = 100,  # Shorter responses for faster completion
        temperature = 0.7,  # Slightly lower for more focused responses
        callback = function(data) {
          # Check if stop was requested (simple environment check)
          if (session_env$stop_streaming) {
            return(FALSE)  # Stop streaming
          }
          
          if (!data$is_final && !is.null(data$token)) {
            # Update current response (minimal processing)
            values$current_response <- paste0(values$current_response, data$token)
            
            # Send real-time update to JavaScript (no text processing in callback for speed)
            session$sendCustomMessage("streamUpdate", list(text = values$current_response))
            
            return(TRUE)  # Continue streaming
          }
          
          return(TRUE)
        }
      )
      
      # Handle completion
      final_response <- values$current_response
      if (final_response == "" && !is.null(result$full_response)) {
        # Fallback if streaming didn't capture text
        final_response <- result$full_response
        
        # Remove the prompt part from the response
        if (length(recent_history) > 0) {
          # Find the last "Assistant:" and take everything after it
          parts <- strsplit(final_response, "Assistant:")[[1]]
          if (length(parts) > 1) {
            final_response <- parts[length(parts)]
          }
        }
        final_response <- trimws(final_response)
      }
      
      # Clean up any remaining formatting issues (do this only once at the end)
      final_response <- trimws(final_response)
      if (final_response != "") {
        # Remove dialogue markers and clean up
        final_response <- gsub("^(Robot:|Customer:|Scene \\d+:|\\(.*\\):?)", "", final_response)
        final_response <- gsub("\\n(Robot:|Customer:|Scene \\d+:|\\(.*\\):?)", "\n", final_response)
        final_response <- trimws(final_response)
      }
      
      # Add assistant response to history
      if (final_response != "") {
        values$history <- c(values$history, paste("Assistant:", final_response))
      }
      
      # Update final state
      values$is_generating <- FALSE
      values$current_response <- ""
      session$sendCustomMessage("uiState", list(streaming = FALSE))
      
      if (session_env$stop_streaming) {
        session$sendCustomMessage("statusUpdate", list(message = "Generation stopped."))
      } else {
        session$sendCustomMessage("statusUpdate", list(message = "Ready."))
      }
      
      # Update chat display with final history
      updateChatDisplay(values$history, "")
      
    }, error = function(e) {
      # Handle errors
      values$is_generating <- FALSE
      values$current_response <- ""
      session$sendCustomMessage("uiState", list(streaming = FALSE))
      session$sendCustomMessage("statusUpdate", list(message = paste("Error:", e$message)))
      showNotification(paste("Error:", e$message), type = "error")
    })
  })
  
  # Stop streaming
  observeEvent(input$stop, {
    if (values$is_generating) {
      session_env$stop_streaming <- TRUE
      session$sendCustomMessage("statusUpdate", list(message = "Stopping generation..."))
      showNotification("Stopping generation...", type = "message")
    }
  })
  
  # Clear chat
  observeEvent(input$clear, {
    if (values$is_generating) {
      showNotification("Cannot clear chat while generating.", type = "warning")
      return()
    }
    
    values$history <- character(0)
    values$current_response <- ""
    session_env$stop_streaming <- FALSE
    
    # Reset chat display
    session$sendCustomMessage("chatUpdate", list(
      html = '<div class="message"><div class="message-content">No messages yet. Select a model and start chatting!</div></div>'
    ))
    
    session$sendCustomMessage("statusUpdate", list(message = "Chat cleared."))
  })
  
  # Helper function to update chat display
  updateChatDisplay <- function(history, streaming_text = "") {
    if (length(history) == 0 && streaming_text == "") {
      html <- '<div class="message"><div class="message-content">No messages yet. Select a model and start chatting!</div></div>'
    } else {
      html <- ""
      
      # Process history
      for (msg in history) {
        if (startsWith(msg, "You:")) {
          content <- sub("^You:", "", msg)
          html <- paste0(html, '<div class="message user-message">',
                        '<div class="message-header">You</div>',
                        '<div class="message-content">', htmlEscape(content), '</div>',
                        '</div>')
        } else if (startsWith(msg, "Assistant:")) {
          content <- sub("^Assistant:", "", msg)
          html <- paste0(html, '<div class="message assistant-message">',
                        '<div class="message-header">Assistant</div>',
                        '<div class="message-content">', htmlEscape(content), '</div>',
                        '</div>')
        }
      }
      
      # Add streaming text if present
      if (streaming_text != "") {
        html <- paste0(html, '<div class="message assistant-message">',
                      '<div class="message-header">Assistant</div>',
                      '<div class="message-content">', htmlEscape(streaming_text), '</div>',
                      '</div>')
      }
    }
    
    session$sendCustomMessage("chatUpdate", list(html = html))
  }
  
  # Cleanup on session end
  session$onSessionEnded(function() {
    if (!is.null(values$ctx)) {
      tryCatch(edge_free_model(values$ctx), error = function(e) {})
    }
  })
}

# Helper function for HTML escaping
htmlEscape <- function(text) {
  text <- gsub("&", "&amp;", text)
  text <- gsub("<", "&lt;", text)
  text <- gsub(">", "&gt;", text)
  text <- gsub('"', "&quot;", text)
  text <- gsub("'", "&#x27;", text)
  return(text)
}

shinyApp(ui = ui, server = server)