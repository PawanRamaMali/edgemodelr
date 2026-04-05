\name{edge_extract}
\alias{edge_extract}
\title{Extract structured data from text}
\usage{
edge_extract(
  ctx,
  text,
  schema,
  instruction = NULL,
  n_predict = 512L,
  temperature = 0.2
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{text}{The input text to analyze}

\item{schema}{A named list defining the extraction schema (see \code{\link{edge_json_grammar}})}

\item{instruction}{Optional instruction to guide extraction (default: auto-generated)}

\item{n_predict}{Maximum tokens to generate (default: 512)}

\item{temperature}{Sampling temperature (default: 0.2, very low for factual extraction)}
}
\value{
A named list with the extracted fields, or the raw JSON string if parsing fails.
Requires the \pkg{jsonlite} package for automatic parsing.
}
\description{
High-level function that combines prompt construction with grammar-constrained
generation to extract structured data from text. Returns a parsed R list.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")

result <- edge_extract(ctx,
  "I absolutely love this product! Best purchase ever.",
  schema = list(
    sentiment = c("positive", "negative", "neutral"),
    confidence = "number"
  ))

edge_free_model(ctx)
}
}
\seealso{
\code{\link{edge_json_grammar}}, \code{\link{edge_grammar_completion}}, \code{\link{edge_classify}}
}
