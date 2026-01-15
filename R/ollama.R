#' Find and load Ollama models
#'
#' Utility functions to discover and work with locally stored Ollama models.
#' Ollama stores models as SHA-256 named blobs which are GGUF files that can
#' be used directly with edgemodelr.
#'
#' @param ollama_dir Optional path to Ollama models directory. If NULL, will auto-detect.
#' @param test_compatibility If TRUE, test if each model can be loaded successfully
#' @param max_size_gb Maximum model size in GB to consider (default: 10)
#' @return List with ollama_path and discovered models information
#'
#' @examples
#' \donttest{
#' # Find Ollama models
#' ollama_info <- edge_find_ollama_models()
#'
#' if (!is.null(ollama_info) && length(ollama_info$models) > 0) {
#'   # Use first compatible model
#'   model_path <- ollama_info$models[[1]]$path
#'   ctx <- edge_load_model(model_path)
#'   result <- edge_completion(ctx, "Hello", n_predict = 10)
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_find_ollama_models <- function(ollama_dir = NULL, test_compatibility = FALSE, max_size_gb = 10) {
  # Possible Ollama model directories
  possible_paths <- c(
    "/var/snap/ollama/common/models/blobs",    # Linux snap
    "/var/lib/ollama/.ollama/models/blobs",    # Linux binary (.deb package)
    "~/.ollama/models/blobs",                  # Standard (may fail on Windows OneDrive)
    file.path(Sys.getenv("HOME"), ".ollama", "models", "blobs"), # HOME variable
    file.path(Sys.getenv("USERPROFILE"), ".ollama", "models", "blobs"), # Windows USERPROFILE (more reliable)
    file.path(Sys.getenv("APPDATA"), "Ollama", "models", "blobs"), # Windows AppData capital
    file.path(Sys.getenv("APPDATA"), ".ollama", "models", "blobs"), # Windows AppData dot
    file.path(Sys.getenv("LOCALAPPDATA"), "Ollama", "models", "blobs"), # Windows LocalAppData capital
    file.path(Sys.getenv("LOCALAPPDATA"), ".ollama", "models", "blobs") # Windows LocalAppData dot
  )

  if (!is.null(ollama_dir)) {
    possible_paths <- c(ollama_dir, possible_paths)
  }

  # Find existing directory
  ollama_path <- NULL
  for (path in possible_paths) {
    expanded <- path.expand(path)
    if (dir.exists(expanded)) {
      ollama_path <- expanded
      break
    }
  }

  if (is.null(ollama_path)) {
    message("No Ollama models directory found. Is Ollama installed?")
    return(NULL)
  }

  message("Found Ollama models in: ", ollama_path)

  # Find blob files (SHA-256 named files)
  all_files <- list.files(ollama_path, pattern = "^sha256-", full.names = TRUE)

  if (length(all_files) == 0) {
    message("No Ollama model blobs found")
    return(list(ollama_path = ollama_path, models = list()))
  }

  message("Found ", length(all_files), " potential model blobs")

  models <- list()
  max_size_bytes <- max_size_gb * 1024^3

  for (file_path in all_files) {
    file_info <- file.info(file_path)

    # Skip very small files (likely not models) or very large ones
    if (file_info$size < 1024^2 || file_info$size > max_size_bytes) next

    # Check if it looks like a GGUF file
    is_gguf <- FALSE
    tryCatch({
      con <- file(file_path, "rb")
      on.exit(close(con))
      header <- readBin(con, "raw", n = 4)
      if (length(header) == 4 && rawToChar(header) == "GGUF") {
        is_gguf <- TRUE
      }
    }, error = function(e) {
      # Skip files we can't read
    })

    if (!is_gguf) next

    size_mb <- round(file_info$size / 1024^2, 1)
    sha256_hash <- gsub("^sha256-", "", basename(file_path))

    model_info <- list(
      name = paste0("ollama_", size_mb, "mb_", substr(sha256_hash, 1, 8)),
      path = file_path,
      size_mb = size_mb,
      size_gb = round(size_mb / 1024, 2),
      sha256 = sha256_hash,
      source = "ollama",
      compatible = NULL  # Will be tested if requested
    )

    # Test compatibility if requested
    if (test_compatibility) {
      model_info$compatible <- test_ollama_model_compatibility(file_path)
      if (model_info$compatible) {
        message("[+] ", model_info$name, " (", size_mb, "MB) - Compatible")
      } else {
        message("[!] ", model_info$name, " (", size_mb, "MB) - Not compatible")
        next  # Skip incompatible models
      }
    } else {
      message("Found: ", model_info$name, " (", size_mb, "MB)")
    }

    models[[length(models) + 1]] <- model_info
  }

  return(list(
    ollama_path = ollama_path,
    models = models,
    total_found = length(all_files),
    gguf_models = length(models)
  ))
}

#' Test if an Ollama model is compatible with edgemodelr
#'
#' @param model_path Path to the Ollama blob file
#' @return TRUE if model loads successfully, FALSE otherwise
#' @keywords internal
test_ollama_model_compatibility <- function(model_path) {
  tryCatch({
    # Try to load with minimal resources
    suppressWarnings({
      ctx <- edge_load_model(model_path, n_ctx = 256, n_gpu_layers = 0)
      if (is_valid_model(ctx)) {
        # Quick test to make sure inference works
        edge_completion(ctx, "Hi", n_predict = 1, temperature = 0.1)
        edge_free_model(ctx)
        return(TRUE)
      }
    })
    return(FALSE)
  }, error = function(e) {
    return(FALSE)
  })
}

#' Load an Ollama model by partial SHA-256 hash
#'
#' Find and load an Ollama model using a partial SHA-256 hash instead of the full path.
#' This is more convenient than typing out the full blob path.
#'
#' @param partial_hash First few characters of the SHA-256 hash
#' @param n_ctx Maximum context length (default: 2048)
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0)
#' @return Model context if successful, throws error if not found or incompatible
#'
#' @examples
#' \donttest{
#' # Load model using first 8 characters of SHA hash
#' # ctx <- edge_load_ollama_model("b112e727")
#' # result <- edge_completion(ctx, "Hello", n_predict = 10)
#' # edge_free_model(ctx)
#' }
#' @export
edge_load_ollama_model <- function(partial_hash, n_ctx = 2048L, n_gpu_layers = 0L) {
  ollama_info <- edge_find_ollama_models()

  if (is.null(ollama_info) || length(ollama_info$models) == 0) {
    stop("No Ollama models found. Make sure Ollama is installed and has downloaded models.")
  }

  # Find matching model by partial hash
  matches <- sapply(ollama_info$models, function(model) {
    startsWith(model$sha256, partial_hash)
  })

  matching_models <- ollama_info$models[matches]

  if (length(matching_models) == 0) {
    available_hashes <- sapply(ollama_info$models, function(m) substr(m$sha256, 1, 8))
    stop("No Ollama model found matching hash: ", partial_hash,
         "\nAvailable models: ", paste(available_hashes, collapse = ", "))
  }

  if (length(matching_models) > 1) {
    model_info <- sapply(matching_models, function(m) {
      paste0(substr(m$sha256, 1, 12), " (", m$size_mb, "MB)")
    })
    stop("Multiple models match hash: ", partial_hash,
         "\nMatching models: ", paste(model_info, collapse = ", "),
         "\nUse a longer hash to disambiguate.")
  }

  model <- matching_models[[1]]
  message("Loading Ollama model: ", model$name, " (", model$size_mb, "MB)")

  # Load the model
  return(edge_load_model(model$path, n_ctx = n_ctx, n_gpu_layers = n_gpu_layers))
}