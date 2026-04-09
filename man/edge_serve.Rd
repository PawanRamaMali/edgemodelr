\name{edge_serve}
\alias{edge_serve}
\title{Serve a model as a local OpenAI-compatible API}
\usage{
edge_serve(
  model_path,
  host = "127.0.0.1",
  port = 8080L,
  n_ctx = 2048L,
  n_gpu_layers = 0L,
  embeddings = FALSE,
  api_key = NULL
)
}
\arguments{
\item{model_path}{Path to a .gguf model file}

\item{host}{Host to bind to (default: "127.0.0.1" for local only)}

\item{port}{Port number (default: 8080)}

\item{n_ctx}{Context size (default: 2048)}

\item{n_gpu_layers}{GPU layers (default: 0, use -1 for full GPU offload)}

\item{embeddings}{Enable embeddings endpoint (default: FALSE)}

\item{api_key}{Optional API key for authentication}
}
\description{
Starts a local Plumber API server that exposes the loaded model through
endpoints compatible with the OpenAI API format. Requires the \pkg{plumber}
package.
}
\details{
Endpoints served:
\itemize{
  \item \code{POST /v1/completions} -- Text completion
  \item \code{POST /v1/chat/completions} -- Chat completion
  \item \code{POST /v1/embeddings} -- Text embeddings (if enabled)
  \item \code{GET /v1/models} -- List loaded model
  \item \code{GET /health} -- Health check
}
}
\examples{
\dontrun{
edge_serve("model.gguf", port = 8080)

# From curl:
# curl http://localhost:8080/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
}
}
