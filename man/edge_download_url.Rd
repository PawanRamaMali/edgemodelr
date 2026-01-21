\name{edge_download_url}
\alias{edge_download_url}
\title{Download a model from a direct URL}
\usage{
edge_download_url(url, filename, cache_dir = NULL,
                  force_download = FALSE, verbose = TRUE)
}
\arguments{
\item{url}{Direct download URL for the model}

\item{filename}{Local filename to save as}

\item{cache_dir}{Directory to store downloaded models (default: user cache directory)}

\item{force_download}{Force re-download even if file exists}

\item{verbose}{Whether to print progress messages}
}
\value{
Path to the downloaded model file
}
\description{
Downloads a GGUF model file from any URL. Supports resume and validates GGUF format.
This function is useful for downloading models from GPT4All CDN or other direct sources
that don't require authentication.
}
\examples{
\donttest{
# Download from GPT4All CDN
model_path <- edge_download_url(
  url = "https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf",
  filename = "mistral-7b.gguf"
)

# Use the downloaded model
if (file.exists(model_path)) {
  ctx <- edge_load_model(model_path)
  response <- edge_completion(ctx, "Hello!")
  edge_free_model(ctx)
}
}
}
