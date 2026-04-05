\name{edge_search}
\alias{edge_search}
\title{Search an embedding index for relevant chunks}
\usage{
edge_search(index, ctx, query, top_k = 5L)
}
\arguments{
\item{index}{An \code{edge_index} object from \code{\link{edge_index_documents}}}

\item{ctx}{Model context (same model used to build the index)}

\item{query}{Query text string}

\item{top_k}{Number of results to return (default: 5)}
}
\value{
A data frame with columns: \code{chunk}, \code{score}, \code{source}, \code{index}
}
\description{
Finds the most similar text chunks to a query using cosine similarity
over the embedding index.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")
index <- edge_index_documents("./docs/", ctx)
results <- edge_search(index, ctx, "quarterly revenue")
print(results)
}
}
\seealso{
\code{\link{edge_index_documents}}, \code{\link{edge_ask}}
}
