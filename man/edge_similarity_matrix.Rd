\name{edge_similarity_matrix}
\alias{edge_similarity_matrix}
\title{Compute a similarity matrix for a set of embeddings}
\usage{
edge_similarity_matrix(embeddings)
}
\arguments{
\item{embeddings}{A numeric matrix where each row is an embedding vector
  (as returned by \code{\link{edge_embeddings}})}
}
\value{
A symmetric numeric matrix of pairwise cosine similarities
}
\description{
Efficiently computes all pairwise cosine similarities between embedding
vectors using normalized matrix multiplication.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")
embs <- edge_embeddings(ctx, c("cat", "kitten", "car", "automobile"))
sim_mat <- edge_similarity_matrix(embs)
# sim_mat[1,2] high (cat~kitten), sim_mat[1,3] low (cat~car)
}
}
\seealso{
\code{\link{edge_embeddings}}, \code{\link{edge_similarity}}
}
