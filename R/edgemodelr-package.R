#' @details
#' The edgemodelr package provides R bindings for Inference for Local Language Models 
#' using llama.cpp and GGUF model files. This enables completely private, 
#' on-device text generation without requiring cloud APIs or internet connectivity.
#' 
#' @section Main Functions:
#' \describe{
#'   \item{\code{\link{edge_load_model}}}{Load a GGUF model file}
#'   \item{\code{\link{edge_completion}}}{Generate text completions}
#'   \item{\code{\link{edge_stream_completion}}}{Stream text generation in real-time}
#'   \item{\code{\link{edge_chat_stream}}}{Interactive chat interface}
#'   \item{\code{\link{edge_quick_setup}}}{One-line model download and setup}
#'   \item{\code{\link{edge_free_model}}}{Release model memory}
#' }
#' 
#' @section Model Management:
#' \describe{
#'   \item{\code{\link{edge_list_models}}}{List available pre-configured models}
#'   \item{\code{\link{edge_download_model}}}{Download models from Hugging Face}
#' }
#' 
#' @section Getting Started:
#' 
#' Basic usage workflow:
#' \enumerate{
#'   \item Download a model: \code{setup <- edge_quick_setup("TinyLlama-1.1B")}
#'   \item Generate text: \code{edge_completion(setup$context, "Hello")}
#'   \item Clean up: \code{edge_free_model(setup$context)}
#' }
#' 
#' For interactive chat:
#' \preformatted{
#' setup <- edge_quick_setup("TinyLlama-1.1B")
#' edge_chat_stream(setup$context)
#' }
#' 
#' @section Examples:
#' 
#' See comprehensive examples in the package:
#' \itemize{
#'   \item \code{system.file("examples/getting_started_example.R", package = "edgemodelr")}
#'   \item \code{system.file("examples/data_science_assistant_example.R", package = "edgemodelr")}
#'   \item \code{system.file("examples/text_analysis_example.R", package = "edgemodelr")}
#'   \item \code{system.file("examples/creative_writing_example.R", package = "edgemodelr")}
#'   \item \code{system.file("examples/advanced_usage_example.R", package = "edgemodelr")}
#' }
#' 
#' Run examples:
#' \preformatted{
#' # Getting started guide
#' source(system.file("examples/getting_started_example.R", package = "edgemodelr"))
#' 
#' # Data science assistant
#' source(system.file("examples/data_science_assistant_example.R", package = "edgemodelr"))
#' }
#' 
#' @section System Requirements:
#' 
#' \itemize{
#'   \item C++17 compatible compiler
#'   \item Sufficient RAM for model size (1GB+ for small models, 8GB+ for 7B models)
#'   \item GGUF model files (downloaded automatically or manually)
#' }
#' 
#' @section Privacy and Security:
#' 
#' This package processes all data locally on your machine. No data is sent to 
#' external servers, ensuring complete privacy and control over your text 
#' generation workflows.
#' 
#' @author Pawan Rama Mali \email{prm@@outlook.in}
#' 
#' @seealso 
#' \itemize{
#'   \item Package repository: \url{https://github.com/PawanRamaMali/edgemodelr}
#'   \item llama.cpp project: \url{https://github.com/ggml-org/llama.cpp}
#'   \item GGUF format: \url{https://github.com/ggml-org/ggml}
#' }
#'
#' @keywords internal
"_PACKAGE"

## usethis namespace: start
#' @useDynLib edgemodelr, .registration = TRUE
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL