\name{edge_classify}
\alias{edge_classify}
\title{Classify text into predefined categories}
\usage{
edge_classify(
  ctx,
  text,
  categories,
  instruction = NULL,
  temperature = 0.1
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{text}{Text to classify (character string or vector for batch)}

\item{categories}{Character vector of allowed categories}

\item{instruction}{Optional classification instruction}

\item{temperature}{Sampling temperature (default: 0.1)}
}
\value{
Character string (single text) or character vector (batch) with the predicted category.
Output is guaranteed to be one of the specified categories.
}
\description{
Classify text into predefined categories using grammar-constrained generation.
The grammar constraint ensures the model can only output one of the provided categories.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")

# Single classification
result <- edge_classify(ctx,
  "I love this product!",
  categories = c("positive", "negative", "neutral"))

# Batch classification
texts <- c("Great product!", "Terrible experience", "It was okay")
results <- edge_classify(ctx, texts,
  categories = c("positive", "negative", "neutral"))

edge_free_model(ctx)
}
}
\seealso{
\code{\link{edge_extract}}, \code{\link{edge_grammar_completion}}
}
