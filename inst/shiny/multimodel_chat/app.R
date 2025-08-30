#' Shiny demo: Streaming chat with model selection
#'
#' This Shiny app allows users to select from available models and
#' chat with streaming responses. A Bootstrap spinner is shown
#' while the model is generating a response.
#'
#' Run with:
#'   shiny::runApp(system.file("shiny/multimodel_chat", package = "edgemodelr"))

library(shiny)
library(shinyjs)
library(edgemodelr)

ui <- fluidPage(
  useShinyjs(),
  titlePanel("edgemodelr Streaming Chat"),
  sidebarLayout(
    sidebarPanel(
      selectInput("model", "Select Model", choices = edge_list_models()$model_name),
      textAreaInput("user", "Your message", "", rows = 3),
      actionButton("send", "Send")
    ),
    mainPanel(
      tags$div(
        id = "spinner",
        style = "display:none;",
        class = "text-center",
        tags$div(class = "spinner-border", role = "status",
                 tags$span(class = "sr-only", "Loading...")
        )
      ),
      verbatimTextOutput("chat")
    )
  )
)

server <- function(input, output, session) {
  rv <- reactiveValues(history = character())

  output$chat <- renderText({
    paste(rv$history, collapse = "\n\n")
  })

  observeEvent(input$send, {
    req(input$user, input$model)

    rv$history <- c(rv$history, paste0("You: ", input$user))
    output$chat <- renderText({ paste(rv$history, collapse = "\n\n") })
    shinyjs::reset("user")
    shinyjs::show("spinner")

    setup <- edge_quick_setup(input$model)
    ctx <- setup$context
    response <- ""

    edge_stream_completion(ctx,
      prompt = paste(rv$history, collapse = "\n"),
      callback = function(data) {
        if (!data$is_final) {
          response <<- paste0(response, data$token)
          output$chat <- renderText({
            paste(c(rv$history, paste0("Assistant: ", response)), collapse = "\n\n")
          })
          session$flushReact()
          TRUE
        } else {
          rv$history <<- c(rv$history, paste0("Assistant: ", response))
          output$chat <- renderText({ paste(rv$history, collapse = "\n\n") })
          shinyjs::hide("spinner")
          edge_free_model(ctx)
          TRUE
        }
      }
    )
  })
}

shinyApp(ui, server)
