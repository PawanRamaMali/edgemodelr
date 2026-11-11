# edgemodelr model API for Posit Connect.
#
# Deploy this folder as a plumber API. Connect runs this file and serves the
# router it returns. Settings come from environment variables (the Vars
# panel on Connect):
#
#   EDGEMODELR_MODEL       path of a .gguf file on the server, or
#   EDGEMODELR_MODEL_REPO  Hugging Face repo and
#   EDGEMODELR_MODEL_FILE  file to download on first start
#   EDGEMODELR_CACHE_DIR   where downloads go; use a persistent path, since
#                          each deploy starts in a fresh folder
#   EDGEMODELR_SLOTS       requests generated at once (default 8)
#   EDGEMODELR_CTX         context per request in tokens (default 2048)
#   EDGEMODELR_THREADS     CPU threads (default: the container's CPU limit)
#   EDGEMODELR_API_KEY     optional key, sent by clients as X-API-Key

library(edgemodelr)

env <- function(name, default = "") {
  value <- Sys.getenv(name)
  if (nzchar(value)) value else default
}

model <- env("EDGEMODELR_MODEL")
if (!nzchar(model) || !file.exists(model)) {
  model <- edge_download_model(
    env("EDGEMODELR_MODEL_REPO", "unsloth/gemma-3-1b-it-GGUF"),
    env("EDGEMODELR_MODEL_FILE", "gemma-3-1b-it-Q4_K_M.gguf"),
    cache_dir = env("EDGEMODELR_CACHE_DIR", tools::R_user_dir("edgemodelr", "cache")),
    trust_first_use = TRUE
  )
}

edge_api(
  model,
  n_slots = as.integer(env("EDGEMODELR_SLOTS", "8")),
  n_ctx = as.integer(env("EDGEMODELR_CTX", "2048")),
  max_queue = 512L,
  api_key = env("EDGEMODELR_API_KEY")
)
