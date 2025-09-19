#' Modern Llama Chat Demo with bslib
#'
#' A clean, modern Shiny application using bslib components and spinners
#' to demonstrate edgemodelr's local LLM capabilities.
#'
#' @author edgemodelr demo
#' @date 2024

library(shiny)
library(bslib)
library(edgemodelr)

# Define the modern theme
theme <- bs_theme(
  version = 5,
  bg = "#ffffff",
  fg = "#333333",
  primary = "#007bff",
  secondary = "#6c757d",
  success = "#28a745",
  info = "#17a2b8",
  warning = "#ffc107",
  danger = "#dc3545",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter", wght = 600)
)

ui <- page_fillable(
  theme = theme,
  title = "🤖 Llama Chat Demo",

  # Custom CSS for enhanced styling
  tags$head(
    tags$style(HTML("
      .chat-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        min-height: 400px;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      }

      .message-bubble {
        margin: 10px 0;
        padding: 12px 16px;
        border-radius: 18px;
        max-width: 80%;
        word-wrap: break-word;
        animation: fadeIn 0.3s ease-in;
      }

      .user-bubble {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        margin-left: auto;
        text-align: right;
      }

      .assistant-bubble {
        background: linear-gradient(135deg, #28a745, #1e7e34);
        color: white;
        margin-right: auto;
      }

      .streaming-bubble {
        background: linear-gradient(135deg, #ffc107, #e0a800);
        color: #333;
        margin-right: auto;
        border: 2px dashed #dee2e6;
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .status-card {
        background: linear-gradient(135deg, #e9ecef, #f8f9fa);
        border-left: 4px solid #007bff;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
      }

      .spinner-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 20px;
      }

      .loading-text {
        font-weight: 500;
        color: #6c757d;
      }

      .input-section {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
      }

      .model-section {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
      }
    "))
  ),

  # Header
  card(
    class = "mb-4",
    card_header(
      class = "text-center",
      h2("🤖 Modern Llama Chat Demo", class = "mb-0"),
      p("Powered by edgemodelr & bslib", class = "text-muted mb-0")
    )
  ),

  layout_columns(
    col_widths = c(4, 8),

    # Sidebar - Model Controls
    card(
      class = "h-100",
      card_header("🔧 Model Controls"),
      card_body(
        # Model Selection Section
        div(class = "model-section",
          h5("Select Model", class = "mb-3"),
          selectInput(
            "model_choice",
            NULL,
            choices = NULL,
            width = "100%"
          ),
          div(
            id = "model_load_area",
            actionButton(
              "load_model_btn",
              "Load Model",
              class = "btn-primary w-100",
              icon = icon("download")
            )
          ),
          div(
            id = "model_spinner",
            class = "spinner-container",
            style = "display: none;",
            div(
              class = "spinner-border text-primary",
              role = "status",
              style = "width: 2rem; height: 2rem;"
            ),
            span("Loading model...", class = "loading-text")
          )
        ),

        # Status Section
        div(class = "status-card",
          h6("Status", class = "mb-2"),
          div(id = "status_display", "Initializing...")
        ),

        # Controls Section
        hr(),
        h5("Chat Controls", class = "mb-3"),
        div(
          actionButton(
            "clear_chat_btn",
            "Clear Chat",
            class = "btn-outline-secondary w-100 mb-2",
            icon = icon("trash")
          ),
          actionButton(
            "stop_generation_btn",
            "Stop Generation",
            class = "btn-outline-danger w-100",
            icon = icon("stop"),
            style = "display: none;"
          )
        )
      )
    ),

    # Main Chat Area
    card(
      class = "h-100",
      card_header("💬 Conversation"),
      card_body(
        # Chat Display
        div(
          id = "chat_display",
          class = "chat-container",
          div(
            class = "text-center text-muted",
            icon("comments", style = "font-size: 3rem; opacity: 0.3;"),
            br(), br(),
            "Load a model and start chatting!"
          )
        ),

        # Streaming Spinner
        div(
          id = "generation_spinner",
          class = "spinner-container",
          style = "display: none;",
          div(
            class = "spinner-grow text-success",
            role = "status",
            style = "width: 1.5rem; height: 1.5rem;"
          ),
          span("Generating response...", class = "loading-text")
        ),

        # Input Section
        div(class = "input-section",
          textAreaInput(
            "user_message",
            NULL,
            placeholder = "Type your message here...",
            rows = 3,
            width = "100%"
          ),
          div(
            class = "d-flex gap-2 mt-2",
            actionButton(
              "send_message_btn",
              "Send Message",
              class = "btn-success flex-grow-1",
              icon = icon("paper-plane")
            ),
            actionButton(
              "example_btn",
              "Try Example",
              class = "btn-outline-info",
              icon = icon("lightbulb")
            )
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {

  # Reactive values
  values <- reactiveValues(
    chat_history = list(),
    model_context = NULL,
    is_loading_model = FALSE,
    is_generating = FALSE,
    current_stream = "",
    model_loaded = FALSE,
    stream_file = NULL
  )

  # Streaming monitor - checks for new tokens every 50ms during generation
  streaming_timer <- reactiveTimer(50)

  observe({
    streaming_timer()

    if (values$is_generating && !is.null(values$stream_file) && file.exists(values$stream_file)) {
      tryCatch({
        # Read current stream content
        current_content <- readLines(values$stream_file, warn = FALSE)
        if (length(current_content) > 0) {
          new_content <- paste(current_content, collapse = "")

          # Check if stream finished (marked with special ending)
          if (endsWith(new_content, "<<STREAM_END>>")) {
            new_content <- gsub("<<STREAM_END>>$", "", new_content)

            # Finalize stream
            if (new_content != "") {
              values$chat_history <- append(values$chat_history, list(list(
                type = "assistant",
                content = new_content,
                timestamp = Sys.time()
              )))
            }

            values$current_stream <- ""
            values$is_generating <- FALSE
            updateStatus("✅ Response completed!")
            session$sendCustomMessage("showGenerationSpinner", list(show = FALSE))
            session$sendCustomMessage("toggleStopButton", list(show = FALSE))

            # Clear stream file
            if (!is.null(values$stream_file) && file.exists(values$stream_file)) {
              unlink(values$stream_file)
            }
            values$stream_file <- NULL

            # Update final chat display
            updateChatDisplay()
          } else {
            # Update streaming display in real-time
            values$current_stream <- new_content
            session$sendCustomMessage("updateStreaming", list(
              text = new_content
            ))
          }
        }
      }, error = function(e) {
        # Handle file read errors silently during streaming
      })
    }
  })

  # Initialize: Load available models
  observe({
    tryCatch({
      models_df <- edge_list_models()
      if (nrow(models_df) > 0) {
        choices <- setNames(models_df$name, models_df$name)
        updateSelectInput(session, "model_choice", choices = choices)
        updateStatus("✅ Models loaded. Select one to begin.")
      } else {
        updateStatus("⚠️ No models available. Please install a model first.")
      }
    }, error = function(e) {
      updateStatus(paste("❌ Error loading models:", e$message))
    })
  })

  # Helper function to update status
  updateStatus <- function(message) {
    session$sendCustomMessage("updateStatus", list(message = message))
  }

  # Load selected model
  observeEvent(input$load_model_btn, {
    req(input$model_choice)

    if (values$is_loading_model || values$is_generating) {
      showNotification("Please wait for current operation to complete.", type = "warning")
      return()
    }

    values$is_loading_model <- TRUE

    # Show loading spinner
    session$sendCustomMessage("showModelSpinner", list(show = TRUE))
    updateStatus("🔄 Loading model... This may take a moment.")

    # Free existing model if any
    if (!is.null(values$model_context)) {
      tryCatch(edge_free_model(values$model_context), error = function(e) {})
      values$model_context <- NULL
      values$model_loaded <- FALSE
    }

    # Load new model
    tryCatch({
      setup_result <- edge_quick_setup(input$model_choice)
      values$model_context <- setup_result$context
      values$model_loaded <- TRUE

      updateStatus(paste("✅ Model", input$model_choice, "loaded successfully!"))
      showNotification("Model loaded successfully!", type = "message")

    }, error = function(e) {
      updateStatus(paste("❌ Failed to load model:", e$message))
      showNotification(paste("Model loading failed:", e$message), type = "error")
      values$model_loaded <- FALSE
    })

    values$is_loading_model <- FALSE
    session$sendCustomMessage("showModelSpinner", list(show = FALSE))
  })

  # Send message
  observeEvent(input$send_message_btn, {
    req(input$user_message)

    if (!values$model_loaded) {
      showNotification("Please load a model first.", type = "warning")
      return()
    }

    if (values$is_generating) {
      showNotification("Please wait for current response to complete.", type = "warning")
      return()
    }

    user_text <- trimws(input$user_message)
    if (user_text == "") {
      showNotification("Please enter a message.", type = "warning")
      return()
    }

    # Add user message to chat
    values$chat_history <- append(values$chat_history, list(list(
      type = "user",
      content = user_text,
      timestamp = Sys.time()
    )))

    # Clear input and update chat display
    updateTextAreaInput(session, "user_message", value = "")
    updateChatDisplay()

    # Start generation
    values$is_generating <- TRUE
    values$current_stream <- ""
    updateStatus("🤖 Generating response...")
    session$sendCustomMessage("showGenerationSpinner", list(show = TRUE))
    session$sendCustomMessage("toggleStopButton", list(show = TRUE))

    # Build conversation context
    recent_messages <- tail(values$chat_history, 6)
    conversation <- paste(
      sapply(recent_messages, function(msg) {
        if (msg$type == "user") {
          paste("User:", msg$content)
        } else {
          paste("Assistant:", msg$content)
        }
      }),
      collapse = "\n"
    )

    # Create temporary file for streaming communication
    stream_file <- tempfile(fileext = ".txt")
    file.create(stream_file)
    values$stream_file <- stream_file

    # Generate response with file-based streaming (works better with Shiny)
    tryCatch({
      # Clear any existing content
      writeLines("", stream_file)

      # Run streaming in background using callback that writes to file
      streaming_callback <- function(data) {
        if (!values$is_generating) return(FALSE)

        if (!data$is_final && !is.null(data$token)) {
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

      # Start streaming
      edge_stream_completion(
        ctx = values$model_context,
        prompt = conversation,
        n_predict = 150,
        temperature = 0.8,
        callback = streaming_callback
      )

    }, error = function(e) {
      values$is_generating <- FALSE
      values$current_stream <- ""
      updateStatus(paste("❌ Generation error:", e$message))
      showNotification(paste("Error:", e$message), type = "error")

      # Clean up temp file
      if (file.exists(stream_file)) {
        unlink(stream_file)
      }
    })

    # Note: UI state changes are handled by the streaming timer monitor
  })

  # Stop generation
  observeEvent(input$stop_generation_btn, {
    if (values$is_generating) {
      values$is_generating <- FALSE
      updateStatus("⏹️ Generation stopped by user.")
      session$sendCustomMessage("showGenerationSpinner", list(show = FALSE))
      session$sendCustomMessage("toggleStopButton", list(show = FALSE))

      if (values$current_stream != "") {
        values$chat_history <- append(values$chat_history, list(list(
          type = "assistant",
          content = paste(values$current_stream, "(stopped)"),
          timestamp = Sys.time()
        )))
      }

      values$current_stream <- ""
      updateChatDisplay()
    }
  })

  # Try example message
  observeEvent(input$example_btn, {
    examples <- c(
      "What is the capital of France?",
      "Explain quantum computing in simple terms.",
      "Write a short poem about coding.",
      "What are the benefits of renewable energy?",
      "How does machine learning work?"
    )

    example <- sample(examples, 1)
    updateTextAreaInput(session, "user_message", value = example)
  })

  # Clear chat
  observeEvent(input$clear_chat_btn, {
    if (values$is_generating) {
      showNotification("Cannot clear chat during generation.", type = "warning")
      return()
    }

    values$chat_history <- list()
    values$current_stream <- ""
    updateChatDisplay()
    updateStatus("🗑️ Chat cleared.")
  })

  # Helper function to update chat display
  updateChatDisplay <- function() {
    session$sendCustomMessage("updateChat", list(
      messages = values$chat_history,
      streaming = values$current_stream
    ))
  }

  # Session cleanup
  session$onSessionEnded(function() {
    if (!is.null(values$model_context)) {
      tryCatch(edge_free_model(values$model_context), error = function(e) {})
    }
  })
}

# Add custom JavaScript handlers
js_code <- "
$(document).ready(function() {

  // Status update handler
  Shiny.addCustomMessageHandler('updateStatus', function(data) {
    $('#status_display').html(data.message);
  });

  // Model loading spinner handler
  Shiny.addCustomMessageHandler('showModelSpinner', function(data) {
    if (data.show) {
      $('#model_load_area').hide();
      $('#model_spinner').show();
    } else {
      $('#model_spinner').hide();
      $('#model_load_area').show();
    }
  });

  // Generation spinner handler
  Shiny.addCustomMessageHandler('showGenerationSpinner', function(data) {
    if (data.show) {
      $('#generation_spinner').show();
    } else {
      $('#generation_spinner').hide();
    }
  });

  // Stop button toggle handler
  Shiny.addCustomMessageHandler('toggleStopButton', function(data) {
    if (data.show) {
      $('#stop_generation_btn').show();
      $('#send_message_btn').prop('disabled', true);
    } else {
      $('#stop_generation_btn').hide();
      $('#send_message_btn').prop('disabled', false);
    }
  });

  // Chat update handler
  Shiny.addCustomMessageHandler('updateChat', function(data) {
    var chatHtml = '';

    if (data.messages.length === 0 && data.streaming === '') {
      chatHtml = '<div class=\"text-center text-muted\"><i class=\"fas fa-comments\" style=\"font-size: 3rem; opacity: 0.3;\"></i><br><br>Load a model and start chatting!</div>';
    } else {
      data.messages.forEach(function(msg) {
        var bubbleClass = msg.type === 'user' ? 'user-bubble' : 'assistant-bubble';
        var icon = msg.type === 'user' ? '👤' : '🤖';
        chatHtml += '<div class=\"message-bubble ' + bubbleClass + '\">' +
                   icon + ' ' + escapeHtml(msg.content) + '</div>';
      });

      if (data.streaming !== '') {
        chatHtml += '<div class=\"message-bubble streaming-bubble\">🤖 ' +
                   escapeHtml(data.streaming) + '<span class=\"spinner-border spinner-border-sm ms-2\"></span></div>';
      }
    }

    $('#chat_display').html(chatHtml);
    $('#chat_display').scrollTop($('#chat_display')[0].scrollHeight);
  });

  // Streaming update handler
  Shiny.addCustomMessageHandler('updateStreaming', function(data) {
    var streamingBubble = $('.streaming-bubble');
    if (streamingBubble.length > 0) {
      streamingBubble.html('🤖 ' + escapeHtml(data.text) + '<span class=\"spinner-border spinner-border-sm ms-2\"></span>');
      $('#chat_display').scrollTop($('#chat_display')[0].scrollHeight);
    }
  });

  // HTML escape function
  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

});
"

# Add the JavaScript to the UI
ui <- tagList(
  ui,
  tags$script(HTML(js_code))
)

# Run the app
shinyApp(ui = ui, server = server)