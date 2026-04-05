\name{edge_ask}
\alias{edge_ask}
\title{Ask a question using retrieval-augmented generation}
\usage{
edge_ask(
  ctx,
  question,
  index,
  top_k = 3L,
  n_predict = 256L,
  temperature = 0.3,
  system_prompt = NULL,
  return_context = FALSE
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{question}{The question to answer}

\item{index}{An \code{edge_index} object from \code{\link{edge_index_documents}}}

\item{top_k}{Number of context chunks to retrieve (default: 3)}

\item{n_predict}{Maximum tokens for the answer (default: 256)}

\item{temperature}{Sampling temperature (default: 0.3)}

\item{system_prompt}{Optional system-level instruction for the model}

\item{return_context}{If TRUE, return both answer and retrieved context (default: FALSE)}
}
\value{
If \code{return_context = FALSE}: a character string with the answer.
If \code{return_context = TRUE}: a list with \code{answer}, \code{context}, and \code{prompt}.
}
\description{
Combines semantic search with text generation: retrieves relevant context
from the index, then generates an answer grounded in that context.
This is the core RAG (Retrieval-Augmented Generation) function.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")
index <- edge_index_documents("./reports/", ctx)

answer <- edge_ask(ctx, "What were Q3 revenues?", index)
cat(answer)

# With context for debugging
result <- edge_ask(ctx, "What were Q3 revenues?", index,
  return_context = TRUE)
cat(result$answer)
print(result$context)
}
}
\seealso{
\code{\link{edge_index_documents}}, \code{\link{edge_search}}
}
