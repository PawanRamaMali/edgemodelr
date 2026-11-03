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
  top_p = 0.95,
  jump_forward = TRUE
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{prompt}{The input prompt}

\item{grammar}{A GBNF grammar string defining allowed output structure}

\item{grammar_root}{The root rule name in the grammar (default: "root")}

\item{n_predict}{Maximum tokens to generate (default: 512)}

\item{temperature}{Sampling temperature (default: 0.3, lower for structured output). 0 gives greedy decoding: the same prompt always gives the same text.}

\item{top_p}{Nucleus sampling threshold (default: 0.95)}

\item{jump_forward}{If TRUE (default), text that the grammar allows in only
one way (JSON keys, separators, the rest of an enum value once it is
unambiguous) is added without sampling, and its tokens are decoded in one
batch, in fewer decode steps. The forced text is tokenized the standard way,
which can differ from how the model would split it, so the output can
differ slightly from \code{jump_forward = FALSE}. It always matches the
grammar. Set FALSE to sample every token.}
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

Jump-forward leaves the last token of each forced stretch for the model to
sample, since that token may merge with the text after it (a model writes
an opening brace and quote as one token). It saves the most with long
JSON keys and \code{edge_json_grammar(compact = TRUE)}, where the layout
between fields is fixed and so is forced too: with four long keys a 0.6B
model wrote the object about 3x faster. With short keys the saving is small.
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
