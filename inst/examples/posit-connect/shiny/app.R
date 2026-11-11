# Shiny chat app that uses the edgemodelr model API (../api) on Posit Connect.
#
# The app never loads a model. Each question becomes a job on the API:
# POST /v1/jobs returns at once, and the app polls GET /v1/jobs/<id> every
# half second, showing the queue position and then the answer as it is
# written. No request stays open while the model works, so the app stays
# responsive however busy the model is.
#
# Environment variables (Vars panel on Connect):
#   EDGEMODELR_API_URL  URL of the API content, e.g.
#                       https://connect.example.com/content/<guid>
#   EDGEMODELR_API_KEY  the API's key, if it has one (sent as X-API-Key)
#   CONNECT_API_KEY     set by Connect itself; sent so the app can reach an
#                       API whose access is restricted

library(shiny)
library(bslib)
library(httr2)

`%||%` <- function(x, y) if (is.null(x)) y else x

api_url <- sub("/+$", "", Sys.getenv("EDGEMODELR_API_URL", "http://127.0.0.1:8080"))
api_key <- Sys.getenv("EDGEMODELR_API_KEY")
connect_key <- Sys.getenv("CONNECT_API_KEY")

api_request <- function(path) {
  req <- request(paste0(api_url, path))
  req <- req_timeout(req, 30)
  req <- req_error(req, is_error = function(resp) FALSE)
  if (nzchar(connect_key)) req <- req_headers(req, Authorization = paste("Key", connect_key))
  if (nzchar(api_key)) req <- req_headers(req, `X-API-Key` = api_key)
  req
}

start_job <- function(messages) {
  resp <- req_perform(req_body_json(api_request("/v1/jobs"),
    list(messages = messages, max_tokens = 300, temperature = 0.7)))
  list(status = resp_status(resp), body = resp_body_json(resp))
}

get_job <- function(id) {
  resp_body_json(req_perform(api_request(paste0("/v1/jobs/", id))))
}

cancel_job <- function(id) {
  req_perform(req_method(api_request(paste0("/v1/jobs/", id)), "DELETE"))
}

system_prompt <- "You are a helpful assistant. Answer briefly."

bubble <- function(role, text, note = NULL) {
  div(class = paste("msg", role),
      div(class = "who", if (role == "user") "You" else "Assistant"),
      div(class = "text", text),
      if (!is.null(note)) div(class = "note", note))
}

ui <- page_fillable(
  title = "Local model chat",
  tags$style(HTML("
    .msg { padding: .6rem .8rem; border-radius: .6rem; margin-bottom: .5rem; white-space: pre-wrap; }
    .msg.user { background: var(--bs-primary-bg-subtle); margin-left: 15%; }
    .msg.assistant { background: var(--bs-tertiary-bg); margin-right: 15%; }
    .who { font-size: .75rem; opacity: .7; }
    .note { font-size: .8rem; opacity: .7; margin-top: .3rem; }
  ")),
  card(
    card_header("Local model chat"),
    div(style = "overflow-y: auto; flex: 1;", uiOutput("messages")),
    card_footer(
      layout_columns(
        col_widths = c(8, 2, 2),
        textAreaInput("prompt", NULL, placeholder = "Ask something", rows = 2, width = "100%"),
        actionButton("send", "Send", class = "btn-primary"),
        actionButton("stop", "Stop")
      )
    )
  )
)

server <- function(input, output, session) {
  history <- reactiveVal(list(list(role = "system", content = system_prompt)))
  job <- reactiveVal(NULL)
  polling <- reactiveVal(FALSE)

  observeEvent(input$send, {
    text <- trimws(input$prompt %||% "")
    if (!nzchar(text) || !is.null(job())) return()
    messages <- c(history(), list(list(role = "user", content = text)))
    history(messages)
    updateTextAreaInput(session, "prompt", value = "")
    out <- tryCatch(start_job(messages), error = function(e) {
      list(status = 0L, body = list(error = list(message = conditionMessage(e))))
    })
    if (identical(out$status, 202L)) {
      job(list(id = out$body$id, status = "queued", text = "", position = NA))
      polling(TRUE)
    } else {
      showNotification(paste("The model API is not available:", out$body$error$message %||% out$status),
                       type = "error")
    }
  })

  observeEvent(input$stop, {
    j <- job()
    if (!is.null(j)) try(cancel_job(j$id), silent = TRUE)
  })

  # A user who closes the page frees the slot for the others.
  session$onSessionEnded(function() {
    j <- isolate(job())
    if (!is.null(j)) try(cancel_job(j$id), silent = TRUE)
  })

  observe({
    req(polling())
    invalidateLater(500)
    j <- isolate(job())
    if (is.null(j)) return()
    g <- tryCatch(get_job(j$id), error = function(e) NULL)
    if (is.null(g)) return()
    if (identical(g$status, "done")) {
      content <- g$response$choices[[1]]$message$content %||% ""
      history(c(isolate(history()), list(list(role = "assistant", content = content))))
      job(NULL)
      polling(FALSE)
    } else if (g$status %in% c("error", "cancelled")) {
      if (identical(g$status, "error")) showNotification(g$error$message %||% "The job failed", type = "error")
      job(NULL)
      polling(FALSE)
    } else {
      job(list(id = j$id, status = g$status, text = g$text %||% "",
               position = g$queue_position %||% NA))
    }
  })

  output$messages <- renderUI({
    turns <- Filter(function(m) m$role != "system", history())
    shown <- lapply(turns, function(m) bubble(m$role, m$content))
    j <- job()
    if (!is.null(j)) {
      note <- if (identical(j$status, "queued") && !is.na(j$position)) {
        sprintf("Waiting for the model: %d ahead in the queue", as.integer(j$position) - 1L)
      } else if (!nzchar(j$text)) {
        "Thinking..."
      }
      shown <- c(shown, list(bubble("assistant", j$text, note)))
    }
    tagList(shown)
  })
}

shinyApp(ui, server)
