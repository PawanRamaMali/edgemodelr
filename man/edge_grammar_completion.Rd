\name{edge_grammar_completion}
\alias{edge_grammar_completion}
\title{Generate text constrained by a GBNF grammar}
\usage{
edge_grammar_completion(
  ctx,
  prompt,
  grammar,
  grammar_root = "root",
  n_predict = 512L,
  temperature = 0.3,
  top_p = 0.95
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{prompt}{The input prompt}

\item{grammar}{A GBNF grammar string defining allowed output structure}

\item{grammar_root}{The root rule name in the grammar (default: "root")}

\item{n_predict}{Maximum tokens to generate (default: 512)}

\item{temperature}{Sampling temperature (default: 0.3, lower for structured output)}

\item{top_p}{Nucleus sampling threshold (default: 0.95)}
}
\value{
Character string containing only the generated text (not the prompt)
}
\description{
Uses llama.cpp's grammar-constrained sampling to force output to conform
to a GBNF grammar specification. This ensures structured, parseable output.
}
\details{
GBNF (GGML BNF) is a format for defining formal grammars that constrain
model output. This is useful for generating JSON, XML, or any structured format.

Common GBNF patterns:
\itemize{
  \item JSON object: Use \code{edge_json_grammar()} for convenience
  \item Enum/choices: \code{'root ::= "yes" | "no" | "maybe"'}
  \item Number: \code{'root ::= [0-9]+'}
}
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")

# Force yes/no output
grammar <- 'root ::= "yes" | "no"'
result <- edge_grammar_completion(ctx, "Is the sky blue? Answer:", grammar)

# Force JSON output
json_grammar <- edge_json_grammar(list(
  sentiment = c("positive", "negative", "neutral"),
  confidence = "number"
))
result <- edge_grammar_completion(ctx,
  "Analyze sentiment: 'I love this product'\nJSON:",
  json_grammar)

edge_free_model(ctx)
}
}
\seealso{
\code{\link{edge_json_grammar}}, \code{\link{edge_extract}}, \code{\link{edge_classify}}
}
