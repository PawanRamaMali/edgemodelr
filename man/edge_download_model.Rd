\name{edge_download_model}
\alias{edge_download_model}
\title{Download a GGUF model from Hugging Face}
\usage{
edge_download_model(
  model_id,
  filename,
  cache_dir = NULL,
  force_download = FALSE,
  verify_checksum = TRUE,
  expected_sha256 = NULL,
  trust_first_use = FALSE,
  verbose = TRUE
)
}
\arguments{
\item{model_id}{Hugging Face model identifier (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")}

\item{filename}{Specific GGUF file to download}

\item{cache_dir}{Directory to store downloaded models (default: "~/.cache/edgemodelr")}

\item{force_download}{Force re-download even if file exists}

\item{verify_checksum}{Verify SHA-256 checksum if available (default: TRUE)}

\item{expected_sha256}{Optional expected SHA-256 hash for the model file}

\item{trust_first_use}{Store a local hash if no known hash exists (default: FALSE)}

\item{verbose}{Whether to print download progress messages}
}
\value{
Path to the downloaded model file
}
\description{
Download a GGUF model from Hugging Face
}
\examples{
\dontrun{
# Download TinyLlama model (large file, not run in checks)
model_path <- edge_download_model(
  model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
  filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)

# Use the downloaded model
if (file.exists(model_path)) {
  ctx <- edge_load_model(model_path)
  response <- edge_completion(ctx, "Hello, how are you?")
  edge_free_model(ctx)
}
}
}
