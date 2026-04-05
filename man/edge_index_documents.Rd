\name{edge_index_documents}
\alias{edge_index_documents}
\title{Build an embedding index from text documents}
\usage{
edge_index_documents(
  source,
  ctx,
  chunk_size = 500L,
  chunk_overlap = 50L,
  file_pattern = "*.txt",
  normalize = TRUE,
  progress = TRUE
)
}
\arguments{
\item{source}{Either a directory path containing text files, or a character
  vector of text chunks to index directly.}

\item{ctx}{Model context from edge_load_model() (used for embedding)}

\item{chunk_size}{Approximate number of characters per chunk (default: 500)}

\item{chunk_overlap}{Number of characters of overlap between chunks (default: 50)}

\item{file_pattern}{Glob pattern for files to read (default: "*.txt")}

\item{normalize}{Normalize embeddings (default: TRUE)}

\item{progress}{Show progress messages (default: TRUE)}
}
\value{
An \code{edge_index} object containing chunks, embeddings, and source metadata.
}
\description{
Reads text files from a directory (or accepts text directly), splits into
chunks, computes embeddings, and returns an index object for semantic search
and retrieval-augmented generation.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")

# Index a directory
index <- edge_index_documents("./reports/", ctx)

# Or index text directly
index <- edge_index_documents(
  c("Revenue grew 15\%", "New product launched"), ctx)

# Search the index
results <- edge_search(index, ctx, "revenue growth")
}
}
\seealso{
\code{\link{edge_search}}, \code{\link{edge_ask}}
}
