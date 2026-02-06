\name{edge_free_model}
\alias{edge_free_model}
\title{Free model context and release memory}
\usage{
edge_free_model(ctx)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}
}
\value{
NULL (invisibly)
}
\description{
Free model context and release memory
}
\examples{
\dontrun{
# Requires a downloaded model (not run in checks)
model_path <- "model.gguf"
if (file.exists(model_path)) {
  ctx <- edge_load_model(model_path)
  # ... use model ...
  edge_free_model(ctx)  # Clean up
}
}
}