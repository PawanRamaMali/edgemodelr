\name{edge_chat_stream}
\alias{edge_chat_stream}
\title{Interactive chat session with streaming responses}
\usage{
edge_chat_stream(ctx, system_prompt = NULL, max_history = 10, n_predict = 200L,
                 temperature = 0.8, verbose = TRUE)
}
\arguments{
\item{ctx}{Model context from edge_load_model()}

\item{system_prompt}{Optional system prompt to set context}

\item{max_history}{Maximum conversation turns to keep in context (default: 10)}

\item{n_predict}{Maximum tokens per response (default: 200)}

\item{temperature}{Sampling temperature (default: 0.8)}

\item{verbose}{Whether to print responses to console (default: TRUE)}
}
\value{
NULL (runs interactively)
}
\description{
Interactive chat session with streaming responses
}
\examples{
\donttest{
setup <- edge_quick_setup("TinyLlama-1.1B")
ctx <- setup$context

if (!is.null(ctx)) {
  # Start interactive chat with streaming
  # edge_chat_stream(ctx, 
  #   system_prompt = "You are a helpful R programming assistant.")
  
  edge_free_model(ctx)
}
}
}