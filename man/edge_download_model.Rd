\name{edge_download_model}
\alias{edge_download_model}
\title{Download a GGUF model from Hugging Face}
\usage{
edge_download_model(model_id, filename, cache_dir = NULL,
                    force_download = FALSE, verbose = TRUE)
}
\arguments{
\item{model_id}{Hugging Face model identifier (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")}

\item{filename}{Specific GGUF file to download}

\item{cache_dir}{Directory to store downloaded models (default: "~/.cache/edgemodelr")}

\item{force_download}{Force re-download even if file exists}

\item{verbose}{Whether to print download progress messages}
}
\value{
Path to the downloaded model file
}
\description{
Download a GGUF model from Hugging Face
}
\examples{
\donttest{
# Download TinyLlama model
#model_path <- edge_download_model(
#  model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
#  filename = "tinyllama-1.1b-chat-v1.0.q4_k_m.gguf"
#)

# Use the downloaded model (example only - requires actual model)
if (FALSE && file.exists(model_path)) {
  ctx <- edge_load_model(model_path)
  response <- edge_completion(ctx, "Hello, how are you?")
  edge_free_model(ctx)
}
}
}
