\name{edge_similarity}
\alias{edge_similarity}
\title{Compute cosine similarity between two embedding vectors}
\usage{
edge_similarity(a, b)
}
\arguments{
\item{a}{Numeric vector (embedding)}

\item{b}{Numeric vector (embedding)}
}
\value{
Cosine similarity score between -1 and 1
}
\description{
Computes the cosine similarity between two embedding vectors.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")
embs <- edge_embeddings(ctx, c("happy cat", "joyful kitten", "stock market"))

edge_similarity(embs[1,], embs[2,])  # high (similar meaning)
edge_similarity(embs[1,], embs[3,])  # low (different topics)
}
}
\seealso{
\code{\link{edge_embeddings}}, \code{\link{edge_similarity_matrix}}
}
