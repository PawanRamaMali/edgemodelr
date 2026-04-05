\name{edge_model_n_embd}
\alias{edge_model_n_embd}
\title{Get the embedding dimension of a loaded model}
\usage{
edge_model_n_embd(ctx)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}
}
\value{
Integer giving the embedding vector dimension
}
\description{
Returns the size of the embedding vectors produced by the loaded model.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")
edge_model_n_embd(ctx)  # e.g., 4096
edge_free_model(ctx)
}
}
\seealso{
\code{\link{edge_embeddings}}
}
