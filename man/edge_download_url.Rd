\name{edge_download_url}
\alias{edge_download_url}
\title{Download a model from a direct URL}
\usage{
edge_download_url(
  url,
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
\item{url}{Direct download URL for the model}

\item{filename}{Local filename to save as}

\item{cache_dir}{Directory to store downloaded models (default: user cache directory)}

\item{force_download}{Force re-download even if file exists}

\item{verify_checksum}{Verify SHA-256 checksum if available (default: TRUE)}

\item{expected_sha256}{Optional expected SHA-256 hash for the file}

\item{trust_first_use}{Store a local hash if no known hash exists (default: FALSE)}

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
\dontrun{
# Download from GPT4All CDN (large file, not run in checks)
model_path <- edge_download_url(
  url = "https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf",
  filename = "mistral-7b.gguf"
)
}
}
