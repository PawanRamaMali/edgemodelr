\name{edge_chat_completion}
\alias{edge_chat_completion}
\title{Generate a chat completion using the model's native template}
\usage{
edge_chat_completion(
  ctx,
  messages,
  n_predict = 256L,
  temperature = 0.7,
  top_p = 0.95
)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{messages}{A list of message objects, each with \code{role} ("system",
  "user", or "assistant") and \code{content} (character string).}

\item{n_predict}{Maximum tokens to generate (default: 256)}

\item{temperature}{Sampling temperature (default: 0.7)}

\item{top_p}{Nucleus sampling threshold (default: 0.95)}
}
\value{
Character string containing only the assistant's response text.
}
\description{
Formats messages using the model's built-in chat template (read from GGUF
metadata), generates a completion, and returns only the assistant's response.
This is the recommended way to do multi-turn chat, as it uses the correct
template format for the specific model (e.g., ChatML for Qwen, Llama format
for Llama models).
}
\examples{
\dontrun{
ctx <- edge_load_model("model.gguf")

answer <- edge_chat_completion(ctx, list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "What is R?")
))
cat(answer)

edge_free_model(ctx)
}
}
\seealso{
\code{\link{build_chat_prompt}}, \code{\link{edge_completion}}
}
