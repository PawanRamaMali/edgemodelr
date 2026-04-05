\name{edge_embeddings}
\alias{edge_embeddings}
\title{Extract text embeddings from a model}
\usage{
edge_embeddings(ctx, texts, normalize = TRUE)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{texts}{Character vector of texts to embed}

\item{normalize}{Whether to L2-normalize the embeddings (default: TRUE).
  Normalized embeddings allow cosine similarity to be computed as a simple dot product.}
}
\value{
A numeric matrix with dimensions (n_texts x n_embd), where each row
is the embedding vector for the corresponding input text.
}
\description{
Computes dense vector embeddings for one or more text inputs using the
loaded model. These embeddings can be used for semantic search, clustering,
similarity comparison, and as input to downstream models.
}
\details{
For best results, use a model designed for embeddings (e.g., nomic-embed-text).
However, generative models can also produce useful embeddings from their
hidden states.

The embedding dimension depends on the model architecture. Use
\code{edge_model_n_embd()} to query the dimension.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")

# Single embedding
emb <- edge_embeddings(ctx, "The cat sat on the mat")
dim(emb)  # 1 x n_embd

# Multiple embeddings
texts <- c("cats are great", "dogs are loyal", "the stock market crashed")
embs <- edge_embeddings(ctx, texts)

# Compute similarity
edge_similarity(embs[1,], embs[2,])  # high (both about pets)
edge_similarity(embs[1,], embs[3,])  # low (different topics)

edge_free_model(ctx)
}
}
\seealso{
\code{\link{edge_similarity}}, \code{\link{edge_similarity_matrix}}, \code{\link{edge_model_n_embd}}
}
