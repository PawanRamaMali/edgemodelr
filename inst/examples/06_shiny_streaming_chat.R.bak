#' True Streaming Shiny Chat Application
#'
#' This implements real-time token streaming using a shared file approach
#' to work around Shiny's synchronous limitations with edge_stream_completion.
#'
#' @author edgemodelr team  
#' @date 2024

library(shiny)
library(shinyjs)
library(edgemodelr)

# HTML escaping function
html_escape <- function(text) {
  if (is.null(text) || text == "") return("")
  text <- gsub("&", "&amp;", text)
  text <- gsub("<", "&lt;", text) 
  text <- gsub(">", "&gt;", text)
  text <- gsub('"', "&quot;", text)
  return(text)
}

ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$style(HTML("
      .spinner {
        border: 3px solid #f3f3f3;
        border-top: 3px solid #007bff;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 10px;
      }
      
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      
      .chat-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 15px;
        height: 450px;
        overflow-y: auto;
        font-family: 'Courier New', monospace;
        line-height: 1.4;
      }
      
      .status-bar {
        background-color: #e9ecef;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-weight: 500;
      }
      
      .msg-user { color: #0066cc; font-weight: bold; }
      .msg-assistant { color: #28a745; font-weight: bold; }
      .msg-streaming { color: #6c757d; background-color: #f1f3f4; padding: 2px 4px; }
      
      .input-section {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 6px;
        border: 1px solid #dee2e6;
      }
    "))
  ),
  
  titlePanel("🚀 edgemodelr Real-Time Streaming Chat"),
  
  fluidRow(
    column(4,
      div(class = "input-section",
        h4("Chat Controls"),
        selectInput("model", "Model:", choices = NULL, width = "100%"),
        textAreaInput("user_message", "Your Message:", 
                     value = "", rows = 4, width = "100%",
                     placeholder = "Type your message here..."),
        div(
          actionButton("send_btn", "Send Message", 
                      class = "btn-primary", style = "width: 60%;"),
          actionButton("clear_btn", "Clear", 
                      class = "btn-outline-secondary", style = "width: 35%; margin-left: 5%;")
        )
      )
    ),
    
    column(8,
      div(class = "status-bar", 
          div(id = "status_spinner", style = "display: none;",
              div(class = "spinner"), "Processing..."
          ),
          textOutput("status_text", inline = TRUE)
      ),
      
      h4("💬 Conversation"),
      div(class = "chat-container", 
          uiOutput("chat_display"))
    )
  )
)

server <- function(input, output, session) {
  
  # Create temporary file for streaming communication
  stream_file <- tempfile(fileext = ".txt")
  file.create(stream_file)
  
  values <- reactiveValues(
    chat_history = character(0),
    model_ctx = NULL,
    status = "Initializing...",
    is_busy = FALSE,
    current_stream = "",
    streaming_active = FALSE,
    session_open = TRUE
  )
  
  # Load models on startup
  observe({
    tryCatch({
      models_df <- edge_list_models()
      if (nrow(models_df) > 0) {
        choices <- setNames(models_df$name, models_df$name)
        updateSelectInput(session, "model", choices = choices)
        values$status <- "Ready - select a model"
      } else {
        values$status <- "No models available"
      }
    }, error = function(e) {
      values$status <- paste("Error loading models:", e$message)
    })
  })
  
  # Status output
  output$status_text <- renderText({
    values$status
  })
  
  # Chat display
  output$chat_display <- renderUI({
    if (length(values$chat_history) == 0 && values$current_stream == "") {
      return(div(style = "color: #6c757d; font-style: italic; text-align: center; padding: 20px;",
                "No conversation yet. Select a model and start chatting!"))
    }
    
    # Build chat content
    chat_content <- list()
    
    # Add conversation history  
    for (msg in values$chat_history) {
      if (startsWith(msg, "User: ")) {
        content <- substring(msg, 7)
        chat_content[[length(chat_content) + 1]] <- 
          div(tags$span(class = "msg-user", "You: "), 
              html_escape(content), style = "margin-bottom: 8px;")
      } else if (startsWith(msg, "Assistant: ")) {
        content <- substring(msg, 12)
        chat_content[[length(chat_content) + 1]] <- 
          div(tags$span(class = "msg-assistant", "Assistant: "), 
              html_escape(content), style = "margin-bottom: 8px;")
      }
    }
    
    # Add current streaming content
    if (values$current_stream != "") {
      chat_content[[length(chat_content) + 1]] <- 
        div(tags$span(class = "msg-assistant", "Assistant: "),
            tags$span(class = "msg-streaming", html_escape(values$current_stream)),
            style = "margin-bottom: 8px;")
    }
    
    do.call(div, chat_content)
  })
  
  # Model loading
  observeEvent(input$model, {
    req(input$model)
    
    if (values$is_busy) {
      showNotification("Please wait for current operation to finish", type = "warning")
      return()
    }
    
    values$status <- paste("Loading model:", input$model)
    shinyjs::show("status_spinner")
    values$is_busy <- TRUE
    
    # Clean up previous model
    if (!is.null(values$model_ctx)) {
      try(edge_free_model(values$model_ctx), silent = TRUE)
      values$model_ctx <- NULL
    }
    
    tryCatch({
      setup_result <- edge_quick_setup(input$model)
      values$model_ctx <- setup_result$context
      values$status <- paste("✓ Model loaded:", input$model)
      showNotification("Model loaded successfully!", type = "message")
    }, error = function(e) {
      values$status <- paste("✗ Failed to load model:", e$message)
      showNotification(paste("Model loading failed:", e$message), type = "error")
    })
    
    values$is_busy <- FALSE
    shinyjs::hide("status_spinner")
  })
  
  # Streaming monitor - checks for new tokens every 50ms
  streaming_timer <- reactiveTimer(50)
  
  observe({
    streaming_timer()
    
    if (values$streaming_active && file.exists(stream_file)) {
      tryCatch({
        # Read current stream content
        current_content <- readLines(stream_file, warn = FALSE)
        if (length(current_content) > 0) {
          new_content <- paste(current_content, collapse = "")
          
          # Check if stream finished (marked with special ending)
          if (endsWith(new_content, "<<STREAM_END>>")) {
            new_content <- gsub("<<STREAM_END>>$", "", new_content)
            
            # Finalize stream
            if (new_content != "") {
              values$chat_history <- c(values$chat_history, paste("Assistant:", new_content))
            }
            values$current_stream <- ""
            values$streaming_active <- FALSE
            values$is_busy <- FALSE
            values$status <- "Ready for next message"
            shinyjs::hide("status_spinner")
            
            # Clear stream file
            writeLines("", stream_file)
          } else {
            # Update streaming display
            values$current_stream <- new_content
          }
        }
      }, error = function(e) {
        # Handle file read errors silently
      })
    }
  })
  
  # Send message handler
  observeEvent(input$send_btn, {
    req(input$user_message)
    req(values$model_ctx)
    
    if (values$is_busy || values$streaming_active) {
      showNotification("Please wait for response to complete", type = "warning")
      return()
    }
    
    user_msg <- trimws(input$user_message)
    if (user_msg == "") {
      showNotification("Please enter a message", type = "warning")
      return()
    }
    
    # Add user message to history
    values$chat_history <- c(values$chat_history, paste("User:", user_msg))
    updateTextAreaInput(session, "user_message", value = "")
    
    # Prepare for streaming
    values$is_busy <- TRUE
    values$streaming_active <- TRUE
    values$current_stream <- ""
    values$status <- "🤖 Generating response..."
    shinyjs::show("status_spinner")
    
    # Clear stream file
    writeLines("", stream_file)
    
    # Build conversation context
    recent_history <- tail(values$chat_history, 8)  # Last 4 exchanges
    conversation_prompt <- paste(recent_history, collapse = "\n")
    
    # Run streaming in background using system2 approach
    # This is a workaround for the synchronous nature of edge_stream_completion
    tryCatch({
      # Create streaming callback that writes to file
      streaming_callback <- function(data) {
        if (!values$session_open) return(FALSE)
        
        if (!data$is_final) {
          # Append token to file
          current <- ""
          if (file.exists(stream_file)) {
            current <- paste(readLines(stream_file, warn = FALSE), collapse = "")
          }
          writeLines(paste0(current, data$token), stream_file)
          
          return(TRUE)
        } else {
          # Mark end of stream
          current <- ""
          if (file.exists(stream_file)) {
            current <- paste(readLines(stream_file, warn = FALSE), collapse = "")
          }
          writeLines(paste0(current, "<<STREAM_END>>"), stream_file)
          return(TRUE)
        }
      }
      
      # Start streaming (this will block, but our timer will read the file)
      edge_stream_completion(
        ctx = values$model_ctx,
        prompt = conversation_prompt,
        n_predict = 200,
        temperature = 0.8,
        callback = streaming_callback
      )
      
    }, error = function(e) {
      values$status <- paste("Error:", e$message)
      values$streaming_active <- FALSE
      values$is_busy <- FALSE
      shinyjs::hide("status_spinner")
      showNotification(paste("Generation failed:", e$message), type = "error")
    })
  })
  
  # Clear chat
  observeEvent(input$clear_btn, {
    if (values$is_busy || values$streaming_active) {
      showNotification("Cannot clear during generation", type = "warning")
      return()
    }
    
    values$chat_history <- character(0)
    values$current_stream <- ""
    values$status <- if (is.null(values$model_ctx)) "Select a model" else "Chat cleared"
    writeLines("", stream_file)
  })
  
  # Session cleanup
  session$onSessionEnded(function() {
    values$session_open <- FALSE
    
    if (!is.null(values$model_ctx)) {
      try(edge_free_model(values$model_ctx), silent = TRUE)
    }
    
    # Clean up temp file
    if (file.exists(stream_file)) {
      unlink(stream_file)
    }
  })
}

shinyApp(ui, server)