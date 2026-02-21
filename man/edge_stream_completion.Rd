\name{edge_stream_completion}
\alias{edge_stream_completion}
\title{Stream text completion with real-time token generation}
\usage{
edge_stream_completion(
  ctx,
  prompt,
  callback,
  n_predict = 128L,
  temperature = 0.8,
  top_p = 0.95,
  timeout_seconds = NULL
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{prompt}{Input text prompt}

\item{callback}{Function called for each generated token. Receives list with token info.}

\item{n_predict}{Maximum tokens to generate (default: 128)}

\item{temperature}{Sampling temperature (default: 0.8)}

\item{top_p}{Top-p sampling parameter (default: 0.95)}

\item{timeout_seconds}{Optional timeout in seconds for inference}
}
\value{
List with full response and generation statistics
}
\description{
Stream text completion with real-time token generation
}
\examples{
\dontrun{
# Requires a downloaded model (not run in checks)
model_path <- "model.gguf"
if (file.exists(model_path)) {
  ctx <- edge_load_model(model_path)

  # Basic streaming with token display
  result <- edge_stream_completion(ctx, "Hello, how are you?",
    callback = function(data) {
      if (!data$is_final) {
        cat(data$token)
        flush.console()
      } else {
        cat("\n[Done: ", data$total_tokens, " tokens]\n")
      }
      return(TRUE)  # Continue generation
    })

  edge_free_model(ctx)
}
}
}
