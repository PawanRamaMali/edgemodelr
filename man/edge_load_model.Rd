\name{edge_load_model}
\alias{edge_load_model}
\title{Load a local GGUF model for inference}
\usage{
edge_load_model(model_path, n_ctx = 2048L, n_gpu_layers = 0L)
}
\arguments{
\item{model_path}{Path to a .gguf model file}

\item{n_ctx}{Maximum context length (default: 2048)}

\item{n_gpu_layers}{Number of layers to offload to GPU (default: 0, CPU-only)}
}
\value{
External pointer to the loaded model context
}
\description{
Load a local GGUF model for inference
}
\examples{
\donttest{
# Load a TinyLlama model
model_path <- "~/models/TinyLlama-1.1B-Chat.Q4_K_M.gguf"
if (file.exists(model_path)) {
  ctx <- edge_load_model(model_path, n_ctx = 2048)
  
  # Generate completion
  result <- edge_completion(ctx, "Explain R data.frame:", n_predict = 100)
  cat(result)
  
  # Free model when done
  edge_free_model(ctx)
}
}
}