\name{edge_quick_setup}
\alias{edge_quick_setup}
\title{Quick setup for a popular model}
\usage{
edge_quick_setup(model_name, cache_dir = NULL, verbose = TRUE)
}
\arguments{
\item{model_name}{Name of the model from edge_list_models()}

\item{cache_dir}{Directory to store downloaded models}

\item{verbose}{Whether to print setup progress messages}
}
\value{
List with model path and context (if llama.cpp is available)
}
\description{
Quick setup for a popular model
}
\examples{
\donttest{
# Quick setup with TinyLlama
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

if (!is.null(ctx)) {
  response <- edge_completion(ctx, "Hello!")
  cat("Response:", response, "\n")
  edge_free_model(ctx)
}
}
}