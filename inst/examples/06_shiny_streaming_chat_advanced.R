#' Advanced Shiny Streaming Chat Application
#'
#' This example demonstrates true streaming functionality by using
#' background processing and periodic updates to avoid reactive context issues.
#'
#' Features:
#' - True real-time token-by-token streaming
#' - Working spinner with proper state management
#' - Model selection and conversation history
#' - Background processing to avoid Shiny reactive context issues
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
      .spinner-border {
        width: 1.5rem;
        height: 1.5rem;
        border: 0.2em solid currentColor;
        border-right-color: transparent;
        border-radius: 50%;
        animation: spinner-border .75s linear infinite;
      }
      
      @keyframes spinner-border {
        to { transform: rotate(360deg); }
      }
      
      .text-center { text-align: center !important; }
      
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
  
  titlePanel("edgemodelr Advanced Streaming Chat"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      selectInput("model", "Select Model", choices = NULL),
      textAreaInput("user", "Your message", "", rows = 3),
      actionButton("send", "Send", class = "btn-primary", width = "100%"),
      br(), br(),
      actionButton("clear", "Clear Chat", class = "btn-warning", width = "100%")
    ),
    
    mainPanel(
      width = 9,
      div(class = "status-box",
          h5("Status:"),
          textOutput("status")),
      
      div(id = "spinner", style = "display: none;", class = "text-center",
          br(),
          div(class = "spinner-border text-primary", role = "status"),
          br(),
          p("Generating response...")),
      
      h5("Conversation:"),
      div(class = "chat-output", verbatimTextOutput("chat"))
    )
  )
)

server <- function(input, output, session) {
  rv <- reactiveValues(
    history = character(),
    ctx = NULL,
    status = "Loading models...",
    is_generating = FALSE,
    streaming_response = "",
    stream_complete = FALSE
  )
  
  # Shared environment for streaming communication
  stream_env <- new.env()
  stream_env$current_response <- ""
  stream_env$is_streaming <- FALSE
  stream_env$stream_finished <- FALSE
  
  # Initialize models
  observe({
    tryCatch({
      models <- edge_list_models()
      if (nrow(models) > 0) {
        updateSelectInput(session, "model", choices = setNames(models$name, models$name))
        rv$status <- "Select a model to begin."
      } else {
        rv$status <- "No models available."
      }
    }, error = function(e) {
      rv$status <- paste("Error loading models:", e$message)
    })
  })
  
  # Render outputs
  output$chat <- renderText({
    if (length(rv$history) == 0 && rv$streaming_response == "") {
      "No messages yet. Select a model and start chatting!"
    } else {
      display_history <- rv$history
      if (rv$streaming_response != "") {
        display_history <- c(display_history, paste0("Assistant: ", rv$streaming_response))
      }
      paste(display_history, collapse = "\n\n")
    }
  })
  
  output$status <- renderText({ rv$status })
  
  # Model loading
  observeEvent(input$model, {
    req(input$model)
    
    if (rv$is_generating) {
      showNotification("Please wait for generation to complete.", type = "warning")
      return()
    }
    
    rv$status <- paste("Loading model:", input$model, "...")
    shinyjs::show("spinner")
    
    if (!is.null(rv$ctx)) {
      tryCatch(edge_free_model(rv$ctx), error = function(e) {})
      rv$ctx <- NULL
    }
    
    tryCatch({
      setup <- edge_quick_setup(input$model)
      rv$ctx <- setup$context
      rv$status <- paste("Model", input$model, "ready.")
    }, error = function(e) {
      rv$status <- paste("Error loading model:", e$message)
    })
    
    shinyjs::hide("spinner")
  })
  
  # Streaming timer - checks for updates every 100ms
  streaming_timer <- reactiveTimer(100)
  
  observe({
    streaming_timer()
    
    if (stream_env$is_streaming) {
      rv$streaming_response <- stream_env$current_response
      
      if (stream_env$stream_finished) {
        # Stream completed
        rv$history <- c(rv$history, paste0("Assistant: ", stream_env$current_response))
        rv$streaming_response <- ""
        rv$is_generating <- FALSE
        rv$status <- "Ready."
        shinyjs::hide("spinner")
        
        # Reset stream environment
        stream_env$is_streaming <- FALSE
        stream_env$stream_finished <- FALSE
        stream_env$current_response <- ""
      }
    }
  })
  
  # Send message
  observeEvent(input$send, {
    req(input$user, rv$ctx)
    
    if (rv$is_generating) {
      showNotification("Please wait for generation to complete.", type = "warning")
      return()
    }
    
    if (trimws(input$user) == "") {
      showNotification("Please enter a message.", type = "warning")
      return()
    }
    
    user_message <- trimws(input$user)
    rv$history <- c(rv$history, paste0("You: ", user_message))
    rv$streaming_response <- ""
    
    updateTextAreaInput(session, "user", value = "")
    shinyjs::show("spinner")
    rv$status <- "Generating response..."
    rv$is_generating <- TRUE
    
    # Start streaming in background
    stream_env$current_response <- ""
    stream_env$is_streaming <- TRUE
    stream_env$stream_finished <- FALSE
    
    # Capture context outside
    ctx_copy <- rv$ctx
    
    # Build prompt
    prompt_parts <- rv$history
    if (length(prompt_parts) > 6) {
      prompt_parts <- tail(prompt_parts, 6)
    }
    full_prompt <- paste(prompt_parts, collapse = "\n")
    
    # Run streaming in a future/background process
    tryCatch({
      edge_stream_completion(ctx_copy,
        prompt = full_prompt,
        n_predict = 200,
        temperature = 0.8,
        callback = function(data) {
          if (!data$is_final) {
            stream_env$current_response <- paste0(stream_env$current_response, data$token)
            return(TRUE)
          } else {
            stream_env$stream_finished <- TRUE
            return(TRUE)
          }
        }
      )
    }, error = function(e) {
      stream_env$is_streaming <- FALSE
      stream_env$stream_finished <- TRUE
      rv$status <- paste("Error:", e$message)
      rv$is_generating <- FALSE
      shinyjs::hide("spinner")
    })
  })
  
  # Clear chat
  observeEvent(input$clear, {
    if (rv$is_generating) {
      showNotification("Cannot clear while generating.", type = "warning")
      return()
    }
    
    rv$history <- character()
    rv$streaming_response <- ""
    rv$status <- if (is.null(rv$ctx)) "Select a model to begin." else "Chat cleared."
  })
  
  # Cleanup
  session$onSessionEnded(function() {
    if (!is.null(rv$ctx)) {
      tryCatch(edge_free_model(rv$ctx), error = function(e) {})
    }
  })
}

shinyApp(ui, server)