\name{edge_json_grammar}
\alias{edge_json_grammar}
\title{Generate a GBNF grammar for JSON output from a schema}
\usage{
edge_json_grammar(schema, compact = FALSE)
}
\arguments{
\item{schema}{A named list where each element defines a field. Values can be:
  \itemize{
    \item \code{"string"} - any string value
    \item \code{"number"} - a numeric value (integer or decimal)
    \item \code{"integer"} - an integer value
    \item \code{"boolean"} - true or false
    \item A character vector - enum of allowed values, or \code{I("x")}
      for a single allowed value
  }}

\item{compact}{If TRUE, the layout is fixed to \code{{"key": value, "key": value}},
with one space after each colon and comma. Every key and separator is then
forced, and \code{edge_grammar_completion()} adds them without sampling.
If FALSE (default), the model may put no space, one space or a newline and
indentation around each key and separator.}
}
\value{
A GBNF grammar string suitable for use with \code{\link{edge_grammar_completion}}
}
\description{
Converts a simple R list schema into a GBNF grammar string that constrains
model output to valid JSON matching the schema.
}
\examples{
# Schema with enum and free-text fields
grammar <- edge_json_grammar(list(
  sentiment = c("positive", "negative", "neutral"),
  confidence = "number",
  explanation = "string"
))
cat(grammar)
}
\seealso{
\code{\link{edge_grammar_completion}}, \code{\link{edge_extract}}
}
