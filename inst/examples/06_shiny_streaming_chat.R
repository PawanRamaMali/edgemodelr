#' Shiny Streaming Chat Application
#'
#' This example demonstrates a web-based streaming chat interface using Shiny
#' with real-time token streaming and proper spinner functionality.
#'
#' Features:
#' - Real-time streaming responses with proper callback handling
#' - Model selection from available models
#' - Loading spinner with proper show/hide functionality
#' - Bootstrap styling for professional appearance
#' - Error handling and status messages
#' - Conversation history display
#'
#' @author edgemodelr team
#' @date 2024

library(shiny)
library(shinyjs)
library(edgemodelr)

ui <- fluidPage(
  useShinyjs(),
  # Add Bootstrap CSS for better styling
  tags$head(
    tags$style(HTML("
      .spinner-border {
        width: 1.5rem;
        height: 1.5rem;
        border-width: 0.2em;
        animation: spinner-border .75s linear infinite;
      }
      
      @keyframes spinner-border {
        to { transform: rotate(360deg); }
      }
      
      .text-center {
        text-align: center !important;
      }
      
      .status-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin-bottom: 1rem;
      }
      
      .chat-output {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 1rem;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-family: monospace;
      }
    "))
  ),
  
  titlePanel("edgemodelr Streaming Chat"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      selectInput("model", 
                  "Select Model", 
                  choices = NULL,  # Will be populated on server start
                  selected = NULL),
      
      textAreaInput("user", 
                    "Your message", 
                    "", 
                    rows = 3,
                    placeholder = "Type your message here..."),
      
      actionButton("send", 
                   "Send", 
                   class = "btn-primary",
                   width = "100%"),
      
      br(), br(),
      
      actionButton("clear", 
                   "Clear Chat", 
                   class = "btn-warning",
                   width = "100%")
    ),
    
    mainPanel(
      width = 9,
      
      # Status section
      div(class = "status-box",
          h5("Status:"),
          textOutput("status")
      ),
      
      # Spinner
      div(
        id = "spinner",
        style = "display: none;",
        class = "text-center",
        br(),
        div(
          class = "spinner-border text-primary",
          role = "status",
          tags$span(class = "sr-only", "Loading...")
        ),
        br(),
        p("Generating response...")
      ),
      
      # Chat output
      h5("Conversation:"),
      div(class = "chat-output",
          verbatimTextOutput("chat", placeholder = TRUE)
      )
    )
  )
)

server <- function(input, output, session) {
  # Reactive values
  rv <- reactiveValues(
    history = character(),
    ctx = NULL,
    current_model = NULL,
    status = "Loading available models...",
    is_generating = FALSE
  )
  
  # Initialize model list on startup
  observe({
    tryCatch({
      models <- edge_list_models()
      if (nrow(models) > 0) {
        model_choices <- setNames(models$name, models$name)
        updateSelectInput(session, "model", choices = model_choices)
        rv$status <- "Select a model to begin."
      } else {
        rv$status <- "No models available. Please download models first."
      }
    }, error = function(e) {
      rv$status <- paste("Error loading model list:", e$message)
    })
  })
  
  # Render outputs
  output$chat <- renderText({
    if (length(rv$history) == 0) {
      "No messages yet. Select a model and start chatting!"
    } else {
      paste(rv$history, collapse = "\n\n")
    }
  })
  
  output$status <- renderText({ 
    rv$status 
  })
  
  # Model selection handler
  observeEvent(input$model, {
    req(input$model)
    
    if (rv$is_generating) {
      showNotification("Please wait for current generation to complete.", type = "warning")
      return()
    }
    
    rv$status <- paste("Loading model:", input$model, "...")
    shinyjs::show("spinner")
    
    # Free previous model if exists
    if (!is.null(rv$ctx)) {
      tryCatch({
        edge_free_model(rv$ctx)
      }, error = function(e) {
        # Ignore cleanup errors
      })
      rv$ctx <- NULL
    }
    
    # Load new model
    tryCatch({
      setup <- edge_quick_setup(input$model)
      rv$ctx <- setup$context
      rv$current_model <- input$model
      rv$status <- paste("Model", input$model, "ready. Type a message to begin.")
    }, error = function(e) {
      rv$status <- paste("Error loading model:", e$message)
      showNotification(paste("Failed to load model:", e$message), type = "error", duration = 5)
    })
    
    shinyjs::hide("spinner")
  }, ignoreNULL = TRUE, ignoreInit = FALSE)
  
  # Send message handler
  observeEvent(input$send, {
    req(input$user)
    req(rv$ctx)
    
    if (rv$is_generating) {
      showNotification("Please wait for current generation to complete.", type = "warning")
      return()
    }
    
    if (trimws(input$user) == "") {
      showNotification("Please enter a message.", type = "warning")
      return()
    }
    
    user_message <- trimws(input$user)
    rv$history <- c(rv$history, paste0("You: ", user_message))
    
    # Clear input and show spinner
    updateTextAreaInput(session, "user", value = "")
    shinyjs::show("spinner")
    rv$status <- "Generating response..."
    rv$is_generating <- TRUE
    
    # Force update the chat display
    output$chat <- renderText({
      paste(rv$history, collapse = "\n\n")
    })
    session$flushReact()
    
    # Prepare for streaming response
    assistant_response <- ""
    
    tryCatch({
      # Build prompt with conversation history
      prompt_parts <- rv$history[rv$history != paste0("You: ", user_message)]  # Exclude current message
      if (length(prompt_parts) > 6) {  # Keep last 6 messages for context
        prompt_parts <- tail(prompt_parts, 6)
      }
      prompt_parts <- c(prompt_parts, paste0("You: ", user_message))
      full_prompt <- paste(prompt_parts, collapse = "\n")
      
      # Stream completion with proper callback
      edge_stream_completion(rv$ctx,
        prompt = full_prompt,
        n_predict = 200,
        temperature = 0.8,
        callback = function(data) {
          # Handle the streaming data correctly
          if (!data$is_final) {
            # Accumulate the response
            assistant_response <<- paste0(assistant_response, data$token)
            
            # Update chat display with current response
            temp_history <- c(rv$history, paste0("Assistant: ", assistant_response))
            output$chat <- renderText({
              paste(temp_history, collapse = "\n\n")
            })
            session$flushReact()
            
            return(TRUE)  # Continue generation
          } else {
            # Final token - complete the response
            rv$history <<- c(rv$history, paste0("Assistant: ", assistant_response))
            
            # Update final display
            output$chat <- renderText({
              paste(rv$history, collapse = "\n\n")
            })
            
            # Hide spinner and update status
            shinyjs::hide("spinner")
            rv$status <- "Ready. Type your next message."
            rv$is_generating <- FALSE
            
            return(TRUE)  # End generation
          }
        }
      )
      
    }, error = function(e) {
      rv$status <- paste("Error generating response:", e$message)
      rv$is_generating <- FALSE
      shinyjs::hide("spinner")
      showNotification(paste("Generation failed:", e$message), type = "error", duration = 5)
    })
  })
  
  # Clear chat handler
  observeEvent(input$clear, {
    if (rv$is_generating) {
      showNotification("Cannot clear chat while generating. Please wait.", type = "warning")
      return()
    }
    
    rv$history <- character()
    rv$status <- if (is.null(rv$ctx)) "Select a model to begin." else "Chat cleared. Ready for new conversation."
    
    output$chat <- renderText({
      "No messages yet. Select a model and start chatting!"
    })
  })
  
  # Cleanup on session end
  session$onSessionEnded(function() {
    if (!is.null(rv$ctx)) {
      tryCatch({
        edge_free_model(rv$ctx)
      }, error = function(e) {
        # Ignore cleanup errors
      })
    }
  })
}

# Launch the app
shinyApp(ui, server)