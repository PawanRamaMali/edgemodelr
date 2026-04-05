\name{edge_extract_batch}
\alias{edge_extract_batch}
\title{Extract structured data from multiple texts}
\usage{
edge_extract_batch(
  ctx,
  texts,
  schema,
  instruction = NULL,
  n_predict = 512L,
  temperature = 0.2,
  progress = TRUE
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{texts}{Character vector of texts to process}

\item{schema}{A named list defining the extraction schema (see \code{\link{edge_json_grammar}})}

\item{instruction}{Optional instruction to guide extraction}

\item{n_predict}{Maximum tokens to generate per text (default: 512)}

\item{temperature}{Sampling temperature (default: 0.2)}

\item{progress}{Show progress messages (default: TRUE)}
}
\value{
A data frame with one row per text and columns for each schema field
}
\description{
Batch version of \code{\link{edge_extract}} that processes a vector of texts
and returns a data frame.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")
reviews <- c("Love it!", "Worst purchase ever", "It's okay")
results <- edge_extract_batch(ctx, reviews,
  schema = list(
    sentiment = c("positive", "negative", "neutral"),
    confidence = "number"
  ))
edge_free_model(ctx)
}
}
\seealso{
\code{\link{edge_extract}}, \code{\link{edge_map}}
}
