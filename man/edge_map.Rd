\name{edge_map}
\alias{edge_map}
\title{Apply a prompt template to a vector of texts}
\usage{
edge_map(
  ctx,
  texts,
  prompt_template,
  n_predict = 128L,
  temperature = 0.7,
  top_p = 0.95,
  grammar = NULL,
  progress = TRUE
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{texts}{Character vector of input texts}

\item{prompt_template}{A string with \code{\{text\}} as a placeholder, or a function
  that takes a single text and returns a prompt string.}

\item{n_predict}{Maximum tokens to generate per text (default: 128)}

\item{temperature}{Sampling temperature (default: 0.7)}

\item{top_p}{Nucleus sampling threshold (default: 0.95)}

\item{grammar}{Optional GBNF grammar string to constrain output}

\item{progress}{Show progress messages (default: TRUE)}
}
\value{
Character vector of completions, same length as \code{texts}
}
\description{
Maps a prompt template over a character vector, generating completions for
each element. This is the primary function for batch LLM operations on data frames.
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")
df <- data.frame(review = c("Great product!", "Terrible quality"))
df$summary <- edge_map(ctx, df$review, "Summarize in 5 words: {text}")
edge_free_model(ctx)
}
}
\seealso{
\code{\link{edge_classify}}, \code{\link{edge_extract_batch}}
}
