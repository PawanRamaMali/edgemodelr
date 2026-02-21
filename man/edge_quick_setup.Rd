\name{edge_quick_setup}
\alias{edge_quick_setup}
\title{Quick setup for a popular model}
\usage{
edge_quick_setup(
  model_name,
  cache_dir = NULL,
  verify_checksum = TRUE,
  expected_sha256 = NULL,
  trust_first_use = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{model_name}{Name of the model from edge_list_models()}

\item{cache_dir}{Directory to store downloaded models}

\item{verify_checksum}{Verify SHA-256 checksum if available (default: TRUE)}

\item{expected_sha256}{Optional expected SHA-256 hash for the model file}

\item{trust_first_use}{Store a local hash if no known hash exists (default: FALSE)}

\item{verbose}{Whether to print setup progress messages}
}
\value{
List with model path and context (if llama.cpp is available)
}
\description{
Quick setup for a popular model
}
\examples{
\dontrun{
# Quick setup with TinyLlama (downloads model, not run in checks)
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

if (!is.null(ctx)) {
  response <- edge_completion(ctx, "Hello!")
  cat("Response:", response, "\n")
  edge_free_model(ctx)
}
}
}
