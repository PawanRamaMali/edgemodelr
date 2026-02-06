\name{edge_completion}
\alias{edge_completion}
\title{Generate text completion using loaded model}
\usage{
edge_completion(ctx, prompt, n_predict = 128L, temperature = 0.8, top_p = 0.95)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{prompt}{Input text prompt}

\item{n_predict}{Maximum tokens to generate (default: 128)}

\item{temperature}{Sampling temperature (default: 0.8)}

\item{top_p}{Top-p sampling parameter (default: 0.95)}
}
\value{
Generated text as character string
}
\description{
Generate text completion using loaded model
}
\examples{
\dontrun{
# Requires a downloaded model (not run in checks)
model_path <- "model.gguf"
if (file.exists(model_path)) {
  ctx <- edge_load_model(model_path)
  result <- edge_completion(ctx, "The capital of France is", n_predict = 50)
  cat(result)
  edge_free_model(ctx)
}
}
}