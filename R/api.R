#' Load a local GGUF model for inference
#'
#' @param model_path Path to a .gguf model file
#' @param n_ctx Maximum context length (default: 2048)
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0, CPU-only)
#' @param n_threads Number of CPU threads for inference (default: NULL = use all hardware threads).
#'   Set to a lower value to leave cores free for other tasks.
#' @param flash_attn Enable flash attention for faster inference (default: TRUE).
#'   Reduces memory usage and improves speed. Set to FALSE for maximum compatibility.
#' @return External pointer to the loaded model context
#'
#' @examples
#' \dontrun{
#' # Quick setup with automatic model download (downloads ~700MB)
#' setup <- edge_quick_setup("TinyLlama-1.1B")
#' if (!is.null(setup$context)) {
#'   ctx <- setup$context
#'
#'   # Generate completion
#'   result <- edge_completion(ctx, "Explain R data.frame:", n_predict = 100)
#'   cat(result)
#'
#'   # Free model when done
#'   edge_free_model(ctx)
#' }
#'
#' # Manual model loading with threading control
#' model_path <- "path/to/your/model.gguf"
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path, n_ctx = 2048, n_threads = 4, flash_attn = TRUE)
#'   # ... use model ...
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_load_model <- function(model_path, n_ctx = 2048L, n_gpu_layers = 0L, n_threads = NULL, flash_attn = TRUE) {
  if (!file.exists(model_path)) {
    stop("Model file does not exist: ", model_path, "\n",
         "Try these options:\n",
         "  1. Download a model: edge_download_model('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')\n",
         "  2. Quick setup: edge_quick_setup('TinyLlama-1.1B')\n",
         "  3. List models: edge_list_models()")
  }
  
  # Check if it's a directory instead of a file
  if (dir.exists(model_path)) {
    stop("Path is a directory, not a file: ", model_path)
  }
  
  if (!grepl("\\.gguf$", model_path, ignore.case = TRUE)) {
    warning("Model file should have .gguf extension for optimal compatibility")
  }
  
  # Validate and optimize parameters
  if (!is.numeric(n_ctx) || n_ctx <= 0) {
    stop("n_ctx must be a positive integer")
  }
  if (!is.numeric(n_gpu_layers) || n_gpu_layers < 0) {
    stop("n_gpu_layers must be a non-negative integer")
  }
  # Validate n_threads: NULL means auto-detect (passed as 0L)
  if (!is.null(n_threads)) {
    if (!is.numeric(n_threads) || n_threads < 1) {
      stop("n_threads must be a positive integer or NULL for auto-detect")
    }
  }
  if (!is.logical(flash_attn) || length(flash_attn) != 1) {
    stop("flash_attn must be TRUE or FALSE")
  }

  # Adaptive context size optimization based on model size
  model_size_mb <- file.info(model_path)$size / (1024^2)

  # Optional memory budget check (best-effort)
  available_ram_gb <- getOption("edgemodelr.available_ram_gb", NA_real_)
  if (is.numeric(available_ram_gb) && !is.na(available_ram_gb) && available_ram_gb > 0) {
    available_mb <- available_ram_gb * 1024
    if (model_size_mb > available_mb * 0.8) {
      warning("Model size (", round(model_size_mb, 1), " MB) is close to or exceeds available RAM (",
              round(available_mb, 1), " MB). Inference may be unstable or slow.")
    }
  }

  # Auto-optimize context for small models (< 2GB)
  if (model_size_mb < 2000 && n_ctx == 2048) {
    # Small models work best with smaller contexts for faster inference
    optimal_ctx <- if (model_size_mb < 1000) 1024 else 1536
    n_ctx <- optimal_ctx
    message("Optimized context size to ", n_ctx, " for small model (", round(model_size_mb, 1), "MB). ",
            "Use n_ctx parameter to override.")
  }

  # Clamp to reasonable range
  n_ctx <- max(512, min(n_ctx, 32768))

  # Warn about large contexts
  if (n_ctx > 8192) {
    message("Large context size (", n_ctx, ") may impact performance. Consider using smaller values for faster inference.")
  }

  # Try to load the model using the raw Rcpp function
  ctx <- tryCatch({
    edge_load_model_internal(normalizePath(model_path),
                             as.integer(n_ctx),
                             as.integer(n_gpu_layers),
                             as.integer(if (is.null(n_threads)) 0L else n_threads),
                             as.logical(flash_attn))
  }, error = function(e) {
    # Provide more context about what went wrong
    if (grepl("llama_load_model_from_file", e$message)) {
      stop("Model found but llama.cpp not available for loading.\n",
           "Install llama.cpp system-wide, then:\n",
           "  devtools::load_all()  # Rebuild package\n",
           "  ctx <- edge_load_model('", basename(model_path), "')")
    }
    stop(e$message)
  })

  # Touch file to update LRU metadata (best-effort)
  try(Sys.setFileTime(model_path, Sys.time()), silent = TRUE)

  ctx
}

#' Generate text completion using loaded model
#'
#' @param ctx Model context from edge_load_model()
#' @param prompt Input text prompt
#' @param n_predict Maximum tokens to generate (default: 128)
#' @param temperature Sampling temperature (default: 0.8)
#' @param top_p Top-p sampling parameter (default: 0.95)
#' @param timeout_seconds Optional timeout in seconds for inference
#' @return Generated text as character string
#' 
#' @examples
#' \dontrun{
#' # Basic completion example
#' setup <- edge_quick_setup("TinyLlama-1.1B")
#' if (!is.null(setup$context)) {
#'   ctx <- setup$context
#'   
#'   # Simple completion
#'   result <- edge_completion(ctx, "The capital of France is", n_predict = 50)
#'   cat("Result:", result, "\n")
#'   
#'   # Completion with custom parameters
#'   creative_result <- edge_completion(
#'     ctx, 
#'     "Write a short poem about data science:",
#'     n_predict = 100,
#'     temperature = 0.9,
#'     top_p = 0.8
#'   )
#'   cat("Creative result:", creative_result, "\n")
#'   
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_completion <- function(ctx, prompt, n_predict = 128L, temperature = 0.8, top_p = 0.95,
                            timeout_seconds = NULL) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(prompt) || length(prompt) != 1L) {
    stop("Prompt must be a single character string")
  }

  # Optimize parameters for better performance and quality
  n_predict <- max(1, min(n_predict, 4096))  # Clamp to reasonable range
  temperature <- max(0.0, min(temperature, 2.0))  # Clamp temperature
  top_p <- max(0.1, min(top_p, 1.0))  # Clamp top_p

  if (!is.null(timeout_seconds)) {
    if (!is.numeric(timeout_seconds) || length(timeout_seconds) != 1 || timeout_seconds <= 0) {
      stop("timeout_seconds must be a positive number of seconds")
    }
    old_limits <- setTimeLimit(elapsed = timeout_seconds, transient = TRUE)
    on.exit({
      setTimeLimit(cpu = Inf, elapsed = Inf, transient = TRUE)
    }, add = TRUE)
  }

  edge_completion_internal(ctx,
                         prompt,
                         as.integer(n_predict),
                         as.numeric(temperature),
                         as.numeric(top_p))
}

#' Free model context and release memory
#'
#' @param ctx Model context from edge_load_model()
#' @return NULL (invisibly)
#' 
#' @examples
#' \dontrun{
#' # Proper cleanup after model usage
#' setup <- edge_quick_setup("TinyLlama-1.1B")
#' if (!is.null(setup$context)) {
#'   ctx <- setup$context
#'   
#'   # Use model for various tasks
#'   result <- edge_completion(ctx, "Hello", n_predict = 20)
#'   cat(result)
#'   
#'   # Always clean up when done
#'   edge_free_model(ctx)
#' }
#' 
#' # Safe cleanup - handles invalid contexts gracefully
#' edge_free_model(NULL)  # Safe, no error
#' edge_free_model("invalid")  # Safe, no error
#' }
#' @export
edge_free_model <- function(ctx) {
  # Handle invalid contexts gracefully without warnings
  if (is.null(ctx)) {
    return(invisible(NULL))
  }
  if (!inherits(ctx, "externalptr")) {
    return(invisible(NULL))
  }
  
  invisible(edge_free_model_internal(ctx))
}

#' Check if model context is valid
#'
#' @param ctx Model context to check
#' @return Logical indicating if context is valid
#' @export
is_valid_model <- function(ctx) {
  tryCatch({
    is_valid_model_internal(ctx)
  }, error = function(e) FALSE)
}

#' Download a GGUF model from Hugging Face
#'
#' @param model_id Hugging Face model identifier (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
#' @param filename Specific GGUF file to download
#' @param cache_dir Directory to store downloaded models (default: user cache directory via tools::R_user_dir())
#' @param force_download Force re-download even if file exists
#' @param verify_checksum Verify SHA-256 checksum if available (default: TRUE)
#' @param expected_sha256 Optional expected SHA-256 hash for the model file
#' @param trust_first_use Store a local hash if no known hash exists (default: FALSE)
#' @return Path to the downloaded model file
#'
#' @examples
#' \dontrun{
#' # Download TinyLlama model
#' model_path <- edge_download_model(
#'   model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
#'   filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
#' )
#'
#' # Use the downloaded model
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path)
#'   response <- edge_completion(ctx, "Hello, how are you?")
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_download_model <- function(model_id, filename, cache_dir = NULL, force_download = FALSE,
                                verify_checksum = TRUE, expected_sha256 = NULL,
                                trust_first_use = FALSE, verbose = TRUE) {
  # Parameter validation
  if (is.null(model_id) || !is.character(model_id) || length(model_id) != 1) {
    stop("model_id must be a string")
  }
  if (nchar(model_id) == 0) {
    stop("model_id cannot be empty")
  }
  if (is.null(filename) || !is.character(filename) || length(filename) != 1) {
    stop("filename must be a string")
  }
  if (nchar(filename) == 0) {
    stop("filename cannot be empty")
  }

  # Set default cache directory using R_user_dir (CRAN compliant)
  if (is.null(cache_dir)) {
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
  }

  # Create cache directory if it doesn't exist (with user consent)
  if (!dir.exists(cache_dir)) {
    # Ask for user consent in interactive sessions
    if (interactive()) {
      consent <- utils::askYesNo(
        paste("edgemodelr needs to create a cache directory to store downloaded models.\n",
              "Location:", cache_dir, "\n",
              "This will help avoid re-downloading models.\n",
              "Create cache directory?"),
        default = TRUE
      )

      if (is.na(consent) || !consent) {
        stop("User declined to create cache directory. ",
             "Download cancelled. You can specify a custom cache_dir parameter.")
      }
    }

    dir.create(cache_dir, recursive = TRUE)
    if (verbose) message("Created cache directory: ", cache_dir)
  }

  if (!.is_writable_dir(cache_dir)) {
    stop("Cache directory is not writable: ", cache_dir, "\n",
         "Specify a writable cache_dir or adjust permissions.")
  }

  # Construct local file path
  local_path <- file.path(cache_dir, basename(filename))

  # Check if file already exists and is valid
  if (file.exists(local_path) && !force_download) {
    # Verify it's a valid GGUF file (starts with "GGUF" magic bytes)
    if (.is_valid_gguf_file(local_path)) {
      .validate_model_checksum(local_path,
                               model_id = model_id,
                               filename = filename,
                               cache_dir = cache_dir,
                               expected_sha256 = expected_sha256,
                               verify_checksum = verify_checksum,
                               trust_first_use = trust_first_use,
                               verbose = verbose)
      if (verbose) message("Model already exists: ", local_path)
      return(local_path)
    } else {
      if (verbose) message("Existing file is incomplete or invalid, re-downloading...")
      file.remove(local_path)
    }
  }

  # Construct Hugging Face URL
  base_url <- "https://huggingface.co"
  download_url <- file.path(base_url, model_id, "resolve", "main", filename)

  if (verbose) {
    message("Downloading model...")
    message("From: ", download_url)
    message("To: ", local_path)
  }

  # Download using robust method with retry support
  download_success <- .robust_download(download_url, local_path, verbose = verbose)

  if (download_success && file.exists(local_path)) {
    # Verify the downloaded file is valid
    if (.is_valid_gguf_file(local_path)) {
      .validate_model_checksum(local_path,
                               model_id = model_id,
                               filename = filename,
                               cache_dir = cache_dir,
                               expected_sha256 = expected_sha256,
                               verify_checksum = verify_checksum,
                               trust_first_use = trust_first_use,
                               verbose = verbose)
      if (verbose) {
        file_size_mb <- round(file.info(local_path)$size / (1024^2), 1)
        message("Download completed successfully!")
        message("Model size: ", file_size_mb, "MB")
      }
      try(Sys.setFileTime(local_path, Sys.time()), silent = TRUE)
      return(local_path)
    } else {
      # Downloaded file is not valid GGUF
      file.remove(local_path)
      stop("Downloaded file is not a valid GGUF model. ",
           "This may indicate the model requires authentication.\n",
           "For gated models (like Gemma), you need to:\n",
           "1. Create a Hugging Face account at https://huggingface.co\n",
           "2. Accept the model's license agreement\n",
           "3. Create an access token at https://huggingface.co/settings/tokens\n",
           "4. Set HF_TOKEN environment variable: Sys.setenv(HF_TOKEN='your_token')")
    }
  } else {
    # Clean up partial download
    if (file.exists(local_path)) {
      file.remove(local_path)
    }

    stop("Failed to download model: All download methods failed\n",
         "Possible solutions:\n",
         "1. Check your internet connection\n",
         "2. Try downloading manually with curl:\n",
         "   curl -L -C - -o '", local_path, "' '", download_url, "'\n",
         "3. For gated models, set HF_TOKEN environment variable\n",
         "4. Or use Ollama: ollama pull <model_name>")
  }
}

#' Check if a file is a valid GGUF file
#' @param path Path to the file
#' @return TRUE if valid GGUF, FALSE otherwise
#' @keywords internal
.is_valid_gguf_file <- function(path) {
  if (!file.exists(path)) return(FALSE)

  file_size <- file.info(path)$size
  # GGUF files should be at least 1MB for any real model
  if (file_size < 1024 * 1024) return(FALSE)

  # Check GGUF magic bytes: "GGUF" at start of file
  tryCatch({
    con <- file(path, "rb")
    on.exit(close(con))
    magic <- readBin(con, "raw", n = 4)
    return(identical(magic, charToRaw("GGUF")))
  }, error = function(e) {
    return(FALSE)
  })
}

#' Compute SHA-256 hash for a file (best-effort, no new hard dependencies)
#' @param path Path to the file
#' @return Lowercase hex string or NA if computation not available
#' @keywords internal
.compute_sha256 <- function(path) {
  if (!file.exists(path)) return(NA_character_)

  # Preferred: openssl package if available
  if (requireNamespace("openssl", quietly = TRUE)) {
    con <- file(path, "rb")
    on.exit(close(con), add = TRUE)
    return(tolower(paste(as.character(openssl::sha256(con)), collapse = "")))
  }

  # Fallback: system utilities
  if (nzchar(Sys.which("sha256sum"))) {
    out <- tryCatch(system2("sha256sum", args = shQuote(path), stdout = TRUE, stderr = TRUE), error = function(e) "")
    if (length(out) > 0 && grepl("^[0-9a-fA-F]{64}", out[1])) {
      return(tolower(strsplit(out[1], "\\s+")[[1]][1]))
    }
  }
  if (nzchar(Sys.which("shasum"))) {
    out <- tryCatch(system2("shasum", args = c("-a", "256", shQuote(path)), stdout = TRUE, stderr = TRUE), error = function(e) "")
    if (length(out) > 0 && grepl("^[0-9a-fA-F]{64}", out[1])) {
      return(tolower(strsplit(out[1], "\\s+")[[1]][1]))
    }
  }
  if (nzchar(Sys.which("certutil"))) {
    out <- tryCatch(system2("certutil", args = c("-hashfile", shQuote(path), "SHA256"), stdout = TRUE, stderr = TRUE), error = function(e) "")
    hash_line <- out[2]
    if (length(hash_line) > 0 && grepl("^[0-9a-fA-F]{64}$", gsub("\\s+", "", hash_line))) {
      return(tolower(gsub("\\s+", "", hash_line)))
    }
  }

  NA_character_
}

#' Check directory writeability
#' @keywords internal
.is_writable_dir <- function(path) {
  if (!dir.exists(path)) return(FALSE)
  test_file <- file.path(path, paste0(".edgemodelr_write_test_", Sys.getpid()))
  ok <- tryCatch({
    writeLines("test", test_file)
    TRUE
  }, error = function(e) FALSE)
  if (file.exists(test_file)) file.remove(test_file)
  ok
}

#' Known hashes cache helpers (Trust-On-First-Use optional)
#' @keywords internal
.known_hashes_path <- function(cache_dir) {
  file.path(cache_dir, "known_hashes.csv")
}

.get_known_hashes <- function(cache_dir) {
  path <- .known_hashes_path(cache_dir)
  if (!file.exists(path)) {
    return(data.frame(model_id = character(), filename = character(), sha256 = character(), stringsAsFactors = FALSE))
  }
  tryCatch({
    read.csv(path, stringsAsFactors = FALSE)
  }, error = function(e) {
    data.frame(model_id = character(), filename = character(), sha256 = character(), stringsAsFactors = FALSE)
  })
}

.save_known_hashes <- function(df, cache_dir) {
  path <- .known_hashes_path(cache_dir)
  tryCatch({
    write.csv(df, path, row.names = FALSE)
  }, error = function(e) {
    NULL
  })
}

.validate_model_checksum <- function(path, model_id = NULL, filename = NULL, cache_dir = NULL,
                                    expected_sha256 = NULL, verify_checksum = TRUE, trust_first_use = FALSE,
                                    verbose = TRUE) {
  if (!verify_checksum) return(TRUE)

  if (is.null(cache_dir)) {
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
  }

  known <- .get_known_hashes(cache_dir)
  if (is.null(expected_sha256) && nrow(known) > 0) {
    hit <- known
    if (!is.null(model_id)) hit <- hit[hit$model_id == model_id, , drop = FALSE]
    if (!is.null(filename)) hit <- hit[hit$filename == filename, , drop = FALSE]
    if (nrow(hit) > 0) {
      expected_sha256 <- hit$sha256[1]
    }
  }

  if (is.null(expected_sha256) || !nzchar(expected_sha256)) {
    if (verbose) {
      message("Checksum: no known SHA-256 for this model; skipping verification. ",
              "Set expected_sha256 or enable trust-first-use to store a local hash.")
    }
    if (trust_first_use) {
      computed <- .compute_sha256(path)
      if (!is.na(computed)) {
        known <- rbind(known, data.frame(model_id = model_id, filename = filename, sha256 = computed,
                                         stringsAsFactors = FALSE))
        .save_known_hashes(known, cache_dir)
        if (verbose) message("Checksum stored locally (trust-first-use).")
      } else if (verbose) {
        message("Checksum: unable to compute SHA-256 on this system.")
      }
    }
    return(TRUE)
  }

  computed <- .compute_sha256(path)
  if (is.na(computed)) {
    stop("Checksum verification requested but SHA-256 could not be computed on this system.\n",
         "Install the 'openssl' R package or ensure sha256sum/shasum/certutil is available.")
  }

  if (!identical(tolower(expected_sha256), tolower(computed))) {
    stop("Checksum mismatch for downloaded model. ",
         "Expected: ", expected_sha256, " but got: ", computed)
  }

  TRUE
}

#' Robust file download with retry and resume support
#' @param url URL to download
#' @param destfile Destination file path
#' @param verbose Print progress messages
#' @param max_retries Maximum number of retry attempts
#' @return TRUE if successful, FALSE otherwise
#' @keywords internal
.robust_download <- function(url, destfile, verbose = TRUE, max_retries = 3) {
  # Get HF token if available
  hf_token <- Sys.getenv("HF_TOKEN", unset = "")
  if (nchar(hf_token) == 0) {
    hf_token <- Sys.getenv("HUGGING_FACE_HUB_TOKEN", unset = "")
  }

  # Try curl first (most reliable for large files with redirects)
  curl_path <- Sys.which("curl")
  if (nchar(curl_path) > 0) {
    if (verbose) message("Using curl for download...")
    success <- .download_with_curl(url, destfile, hf_token, verbose, max_retries)
    if (success) return(TRUE)
  }

  # Try wget as fallback

  wget_path <- Sys.which("wget")
  if (nchar(wget_path) > 0) {
    if (verbose) message("Trying wget as fallback...")
    success <- .download_with_wget(url, destfile, hf_token, verbose, max_retries)
    if (success) return(TRUE)
  }

  # Try R's download.file with libcurl method as last resort
  if (verbose) message("Trying R download.file with libcurl...")
  success <- .download_with_r(url, destfile, hf_token, verbose, max_retries)
  if (success) return(TRUE)

  return(FALSE)
}

#' Download using curl command
#' @keywords internal
.download_with_curl <- function(url, destfile, hf_token = "", verbose = TRUE, max_retries = 3) {
  for (attempt in 1:max_retries) {
    if (attempt > 1 && verbose) {
      message("Retry attempt ", attempt, " of ", max_retries, "...")
    }

    # Build curl command with proper options for large files
    curl_args <- c(
      "-L",                          # Follow redirects
      "-C", "-",                     # Resume if partial file exists
      "--connect-timeout", "30",     # Connection timeout
      "--max-time", "7200",          # Max 2 hours for large files
      "--retry", "3",                # Curl's own retry
      "--retry-delay", "5",          # Delay between retries
      "-#"                           # Progress bar
    )

    # Add authentication header if token available
    if (nchar(hf_token) > 0) {
      curl_args <- c(curl_args, "-H", paste0("Authorization: Bearer ", hf_token))
    }

    curl_args <- c(curl_args, "-o", destfile, url)

    # Run curl
    result <- tryCatch({
      system2("curl", args = curl_args, stdout = if (verbose) "" else FALSE,
              stderr = if (verbose) "" else FALSE)
    }, error = function(e) {
      return(1)
    })

    # Check if download succeeded and file is valid
    if (result == 0 && file.exists(destfile) && .is_valid_gguf_file(destfile)) {
      return(TRUE)
    }

    # Wait before retry
    if (attempt < max_retries) {
      Sys.sleep(2)
    }
  }

  return(FALSE)
}

#' Download using wget command
#' @keywords internal
.download_with_wget <- function(url, destfile, hf_token = "", verbose = TRUE, max_retries = 3) {
  for (attempt in 1:max_retries) {
    if (attempt > 1 && verbose) {
      message("Retry attempt ", attempt, " of ", max_retries, "...")
    }

    # Build wget command
    wget_args <- c(
      "-c",                          # Continue partial downloads
      "--timeout=30",                # Timeout
      "--tries=3",                   # Wget's own retry
      "-O", destfile
    )

    # Add authentication if token available
    if (nchar(hf_token) > 0) {
      wget_args <- c(wget_args, paste0("--header=Authorization: Bearer ", hf_token))
    }

    wget_args <- c(wget_args, url)

    # Run wget
    result <- tryCatch({
      system2("wget", args = wget_args, stdout = if (verbose) "" else FALSE,
              stderr = if (verbose) "" else FALSE)
    }, error = function(e) {
      return(1)
    })

    if (result == 0 && file.exists(destfile) && .is_valid_gguf_file(destfile)) {
      return(TRUE)
    }

    if (attempt < max_retries) {
      Sys.sleep(2)
    }
  }

  return(FALSE)
}

#' Download using R's download.file with libcurl (last resort, no auth)
#' @keywords internal
.download_with_r <- function(url, destfile, hf_token = "", verbose = TRUE, max_retries = 3) {
  # Prefer the curl R package when a token is needed, as download.file(method="libcurl")
  # does not support custom request headers.
  if (nchar(hf_token) > 0 && requireNamespace("curl", quietly = TRUE)) {
    if (verbose) message("Using curl R package for authenticated download...")
    for (attempt in 1:max_retries) {
      if (attempt > 1 && verbose) message("Retry attempt ", attempt, " of ", max_retries, "...")
      result <- tryCatch({
        h <- curl::new_handle()
        curl::handle_setheaders(h, "Authorization" = paste("Bearer", hf_token))
        curl::curl_download(url, destfile, handle = h, quiet = !verbose)
        TRUE
      }, error = function(e) {
        if (verbose) message("curl download error: ", e$message)
        FALSE
      })
      if (result && file.exists(destfile) && .is_valid_gguf_file(destfile)) return(TRUE)
      if (attempt < max_retries) Sys.sleep(2)
    }
    return(FALSE)
  }

  # Unauthenticated fallback via download.file / libcurl
  for (attempt in 1:max_retries) {
    if (attempt > 1 && verbose) {
      message("Retry attempt ", attempt, " of ", max_retries, "...")
    }

    result <- tryCatch({
      old_timeout <- getOption("timeout")
      options(timeout = 7200)  # 2 hours for large files
      on.exit(options(timeout = old_timeout), add = TRUE)

      utils::download.file(
        url,
        destfile,
        mode = "wb",
        method = "libcurl",
        quiet = !verbose,
        cacheOK = FALSE
      )
      0  # Success
    }, error = function(e) {
      if (verbose) message("Download error: ", e$message)
      1  # Failure
    }, warning = function(w) {
      if (verbose) message("Download warning: ", w$message)
      if (file.exists(destfile) && file.info(destfile)$size > 1024 * 1024) return(0)
      return(1)
    })

    if (result == 0 && file.exists(destfile) && .is_valid_gguf_file(destfile)) {
      return(TRUE)
    }

    if (attempt < max_retries) {
      Sys.sleep(2)
    }
  }

  return(FALSE)
}

#' Resolve a model name to the canonical entry in edge_list_models()
#' @param model_name User-provided model name
#' @param models Optional models data.frame
#' @return Canonical model name or NA if not found
#' @keywords internal
edge_resolve_model_name <- function(model_name, models = NULL) {
  if (is.null(models)) {
    models <- edge_list_models()
  }
  if (!is.character(model_name) || length(model_name) != 1) return(NA_character_)

  if (model_name %in% models$name) return(model_name)

  lower <- tolower(model_name)
  name_lower <- tolower(models$name)
  if (lower %in% name_lower) {
    return(models$name[match(lower, name_lower)])
  }

  aliases <- c(
    "llama-3.2-3b" = "llama3.2-3b",
    "llama3.2-3b-instruct" = "llama3.2-3b",
    "llama-3.2-3b-instruct" = "llama3.2-3b",
    "llama-3-8b" = "llama3-8b",
    "llama3-8b-instruct" = "llama3-8b",
    "llama-3-8b-instruct" = "llama3-8b",
    "phi-3-mini" = "phi3-mini",
    "phi-3-mini-4k" = "phi3-mini",
    "phi3-mini-4k" = "phi3-mini",
    "tinyllama" = "TinyLlama-1.1B",
    "tinyllama-1.1b" = "TinyLlama-1.1B",
    "tinyllama-chat" = "TinyLlama-1.1B",
    "tinyllama-openorca" = "TinyLlama-OpenOrca",
    "tiny-llama" = "TinyLlama-1.1B",
    "mistral" = "mistral-7b",
    "mistral-7b-instruct" = "mistral-7b",
    "orca2" = "orca2-13b",
    "orca-2" = "orca2-13b",
    "wizard" = "wizardlm-13b",
    "wizardlm" = "wizardlm-13b",
    "hermes" = "hermes-13b"
  )
  if (lower %in% names(aliases)) {
    canonical <- aliases[[lower]]
    # Verify the canonical name actually exists in models
    if (canonical %in% models$name) return(canonical)
    # Fall through to partial match if canonical was stale
  }

  # Partial substring match (last resort)
  partial <- which(startsWith(tolower(models$name), lower) | startsWith(lower, tolower(models$name)))
  if (length(partial) == 1L) {
    return(models$name[partial])
  }

  NA_character_
}

#' List popular pre-configured models
#'
#' Models are sourced from GPT4All CDN (direct download, no auth required) and Hugging Face.
#' GPT4All models are recommended for offline use as they don't require any accounts.
#'
#' @return Data frame with model information
#' @export
edge_list_models <- function() {
  models <- data.frame(
    name = c(
      # Small models (testing/mobile)
      "TinyLlama-1.1B", "TinyLlama-OpenOrca",
      # Medium models (2-5GB)
      "llama3-8b", "mistral-7b", "llama3.2-3b", "phi3-mini",
      # Large models (7-10GB) - Direct download from GPT4All CDN
      "orca2-13b", "wizardlm-13b", "hermes-13b", "starcoder"
    ),
    size = c(
      "~700MB", "~700MB",
      "~4.7GB", "~4.1GB", "~2GB", "~2.4GB",
      "~7.4GB", "~7.4GB", "~7.4GB", "~9GB"
    ),
    download_url = c(
      # HuggingFace models
      "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf",
      # GPT4All CDN - direct download, no auth required
      "https://gpt4all.io/models/gguf/Meta-Llama-3-8B-Instruct.Q4_0.gguf",
      "https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf",
      "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
      "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
      # Large models - GPT4All CDN (no auth, direct download)
      "https://gpt4all.io/models/gguf/orca-2-13b.Q4_0.gguf",
      "https://gpt4all.io/models/gguf/wizardlm-13b-v1.2.Q4_0.gguf",
      "https://gpt4all.io/models/gguf/nous-hermes-llama2-13b.Q4_0.gguf",
      "https://gpt4all.io/models/gguf/starcoder-newbpe-q4_0.gguf"
    ),
    filename = c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "tinyllama-1.1b-1t-openorca.Q4_K_M.gguf",
      "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
      "mistral-7b-instruct-v0.1.Q4_0.gguf",
      "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
      "Phi-3-mini-4k-instruct-q4.gguf",
      "orca-2-13b.Q4_0.gguf",
      "wizardlm-13b-v1.2.Q4_0.gguf",
      "nous-hermes-llama2-13b.Q4_0.gguf",
      "starcoder-newbpe-q4_0.gguf"
    ),
    use_case = c(
      "Testing", "Better Chat",
      "General (8B)", "General (7B)", "Mobile (3B)", "Reasoning",
      "13B Chat", "13B Instruct", "13B Chat", "Code (15B)"
    ),
    source = c(
      "huggingface", "huggingface",
      "gpt4all", "gpt4all", "huggingface", "huggingface",
      "gpt4all", "gpt4all", "gpt4all", "gpt4all"
    ),
    requires_auth = c(
      FALSE, FALSE,
      FALSE, FALSE, FALSE, FALSE,
      FALSE, FALSE, FALSE, FALSE
    ),
    stringsAsFactors = FALSE
  )

  return(models)
}

#' Quick setup for a popular model
#'
#' Downloads a model from the built-in model list and loads it for inference.
#' GPT4All models are downloaded directly without authentication.
#'
#' @param model_name Name of the model from edge_list_models()
#' @param cache_dir Directory to store downloaded models (default: user cache directory via tools::R_user_dir())
#' @param verify_checksum Verify SHA-256 checksum if available (default: TRUE)
#' @param expected_sha256 Optional expected SHA-256 hash for the model file
#' @param trust_first_use Store a local hash if no known hash exists (default: FALSE)
#' @param verbose Whether to print status messages (default: TRUE)
#' @return List with model path and context (if llama.cpp is available)
#'
#' @examples
#' \dontrun{
#' # Quick setup with a 7B model (downloads ~4GB, not run in checks)
#' setup <- edge_quick_setup("mistral-7b")
#' ctx <- setup$context
#'
#' # Or use a larger 13B model
#' setup <- edge_quick_setup("orca2-13b")  # 7.4GB, direct download
#'
#' if (!is.null(ctx)) {
#'   response <- edge_completion(ctx, "Hello!")
#'   cat("Response:", response, "\n")
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_quick_setup <- function(model_name, cache_dir = NULL, verify_checksum = TRUE,
                             expected_sha256 = NULL, trust_first_use = FALSE, verbose = TRUE) {
  # Parameter validation
  if (is.null(model_name)) {
    model_name <- ""
  }
  if (!is.character(model_name) || length(model_name) != 1) {
    stop("model_name must be a string")
  }
  if (nchar(model_name) == 0) {
    stop("model_name cannot be empty")
  }

  models <- edge_list_models()
  resolved_name <- edge_resolve_model_name(model_name, models)
  if (is.na(resolved_name)) {
    stop("Model '", model_name, "' not found. Available models:\n",
         paste(models$name, collapse = ", "))
  }
  model_info <- models[models$name == resolved_name, , drop = FALSE]
  if (nrow(model_info) == 0) {
    stop("Model '", model_name, "' not found. Available models:\n",
         paste(models$name, collapse = ", "))
  }
  model_info <- model_info[1, , drop = FALSE]

  if (verbose) {
    message("Setting up ", model_name, " (", model_info$size, ")...")
    if (model_info$source == "gpt4all") {
      message("Source: GPT4All CDN (direct download, no auth required)")
    } else if (model_info$source == "huggingface") {
      message("Source: Hugging Face (some models may require HF_TOKEN depending on license)")
    }
  }

  # Download model using direct URL
  model_path <- edge_download_url(
    url = model_info$download_url,
    filename = model_info$filename,
    cache_dir = cache_dir,
    verify_checksum = verify_checksum,
    expected_sha256 = expected_sha256,
    trust_first_use = trust_first_use,
    verbose = verbose
  )

  # Try to load model (will show helpful error if llama.cpp not available)
  ctx <- tryCatch({
    edge_load_model(model_path)
  }, error = function(e) {
    warning("Model downloaded but llama.cpp not available for inference.\n",
            "Model path: ", model_path, "\n",
            "Install llama.cpp system-wide to use for inference.")
    NULL
  })

  return(list(
    model_path = model_path,
    context = ctx,
    info = model_info
  ))
}

#' Download a model from a direct URL
#'
#' Downloads a GGUF model file from any URL. Supports resume and validates GGUF format.
#'
#' @param url Direct download URL for the model
#' @param filename Local filename to save as
#' @param cache_dir Directory to store downloaded models (default: user cache directory)
#' @param force_download Force re-download even if file exists
#' @param verify_checksum Verify SHA-256 checksum if available (default: TRUE)
#' @param expected_sha256 Optional expected SHA-256 hash for the file
#' @param trust_first_use Store a local hash if no known hash exists (default: FALSE)
#' @param verbose Whether to print progress messages
#' @return Path to the downloaded model file
#'
#' @examples
#' \dontrun{
#' # Download from GPT4All CDN (large file, not run in checks)
#' model_path <- edge_download_url(
#'   url = "https://gpt4all.io/models/gguf/mistral-7b-instruct-v0.1.Q4_0.gguf",
#'   filename = "mistral-7b.gguf"
#' )
#' }
#' @export
edge_download_url <- function(url, filename, cache_dir = NULL, force_download = FALSE,
                              verify_checksum = TRUE, expected_sha256 = NULL,
                              trust_first_use = FALSE, verbose = TRUE) {
  # Validate inputs

if (is.null(url) || !is.character(url) || nchar(url) == 0) {
    stop("url must be a non-empty string")
  }
  if (is.null(filename) || !is.character(filename) || nchar(filename) == 0) {
    stop("filename must be a non-empty string")
  }

  # Set default cache directory
  if (is.null(cache_dir)) {
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
  }

  # Create cache directory if needed
  if (!dir.exists(cache_dir)) {
    if (interactive()) {
      consent <- utils::askYesNo(
        paste("Create cache directory for models?\n", "Location:", cache_dir),
        default = TRUE
      )
      if (is.na(consent) || !consent) {
        stop("Cache directory required. Specify cache_dir parameter.")
      }
    }
    dir.create(cache_dir, recursive = TRUE)
    if (verbose) message("Created cache directory: ", cache_dir)
  }

  if (!.is_writable_dir(cache_dir)) {
    stop("Cache directory is not writable: ", cache_dir, "\n",
         "Specify a writable cache_dir or adjust permissions.")
  }

  local_path <- file.path(cache_dir, basename(filename))

  # Check if valid file exists
  if (file.exists(local_path) && !force_download) {
    if (.is_valid_gguf_file(local_path)) {
      .validate_model_checksum(local_path,
                               model_id = NULL,
                               filename = filename,
                               cache_dir = cache_dir,
                               expected_sha256 = expected_sha256,
                               verify_checksum = verify_checksum,
                               trust_first_use = trust_first_use,
                               verbose = verbose)
      if (verbose) message("Model already exists: ", local_path)
      return(local_path)
    } else {
      if (verbose) message("Existing file invalid, re-downloading...")
      file.remove(local_path)
    }
  }

  if (verbose) {
    message("Downloading model...")
    message("From: ", url)
    message("To: ", local_path)
  }

  # Download using robust method
  success <- .robust_download(url, local_path, verbose = verbose)

  if (success && file.exists(local_path) && .is_valid_gguf_file(local_path)) {
    .validate_model_checksum(local_path,
                             model_id = NULL,
                             filename = filename,
                             cache_dir = cache_dir,
                             expected_sha256 = expected_sha256,
                             verify_checksum = verify_checksum,
                             trust_first_use = trust_first_use,
                             verbose = verbose)
    if (verbose) {
      file_size_gb <- round(file.info(local_path)$size / (1024^3), 2)
      message("Download completed! Size: ", file_size_gb, " GB")
    }
    try(Sys.setFileTime(local_path, Sys.time()), silent = TRUE)
    return(local_path)
  } else {
    if (file.exists(local_path)) file.remove(local_path)
    stop("Download failed. Check your internet connection and try again.")
  }
}

#' Get optimized configuration for small language models
#'
#' Returns recommended parameters for loading and using small models (1B-3B parameters)
#' to maximize inference speed on resource-constrained devices.
#'
#' @param model_size_mb Model file size in MB (if known). If NULL, uses conservative defaults.
#' @param available_ram_gb Available system RAM in GB. If NULL, uses conservative defaults.
#' @param target Device target: "mobile", "laptop", "desktop", or "server" (default: "laptop")
#' @return List with optimized parameters for edge_load_model() and edge_completion()
#'
#' @examples
#' # Get optimized config for a 700MB model on a laptop
#' config <- edge_small_model_config(model_size_mb = 700, available_ram_gb = 8)
#'
#' # Use the config to load a model
#' \dontrun{
#' model_path <- "path/to/tinyllama.gguf"
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(
#'     model_path,
#'     n_ctx = config$n_ctx,
#'     n_gpu_layers = config$n_gpu_layers
#'   )
#'
#'   result <- edge_completion(
#'     ctx,
#'     prompt = "Hello",
#'     n_predict = config$recommended_n_predict,
#'     temperature = config$recommended_temperature
#'   )
#'
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_small_model_config <- function(model_size_mb = NULL, available_ram_gb = NULL, target = "laptop") {
  # Default configurations based on target device
  configs <- list(
    mobile = list(
      n_ctx = 512,
      n_gpu_layers = 0,
      recommended_n_predict = 50,
      recommended_temperature = 0.7,
      description = "Optimized for mobile devices with limited RAM"
    ),
    laptop = list(
      n_ctx = 1024,
      n_gpu_layers = 0,
      recommended_n_predict = 100,
      recommended_temperature = 0.8,
      description = "Balanced performance for laptops"
    ),
    desktop = list(
      n_ctx = 2048,
      n_gpu_layers = 0,
      recommended_n_predict = 150,
      recommended_temperature = 0.8,
      description = "Higher quality for desktop systems"
    ),
    server = list(
      n_ctx = 4096,
      n_gpu_layers = 0,
      recommended_n_predict = 200,
      recommended_temperature = 0.8,
      description = "Maximum quality for server deployments"
    )
  )

  # Get base config
  if (!target %in% names(configs)) {
    warning("Unknown target '", target, "'. Using 'laptop' defaults.")
    target <- "laptop"
  }

  config <- configs[[target]]

  # Adjust based on model size if provided
  if (!is.null(model_size_mb)) {
    if (model_size_mb < 1000) {
      # Very small models (< 1GB): can use larger context
      config$n_ctx <- min(config$n_ctx * 1.5, 2048)
      config$recommended_n_predict <- min(config$recommended_n_predict * 1.2, 200)
    } else if (model_size_mb > 2000) {
      # Larger models (> 2GB): reduce context to save memory
      config$n_ctx <- max(config$n_ctx * 0.75, 512)
      config$recommended_n_predict <- max(config$recommended_n_predict * 0.8, 50)
    }
  }

  # Adjust based on available RAM if provided
  if (!is.null(available_ram_gb)) {
    if (available_ram_gb < 4) {
      config$n_ctx <- min(config$n_ctx, 512)
      config$recommended_n_predict <- min(config$recommended_n_predict, 50)
    } else if (available_ram_gb > 16) {
      config$n_ctx <- min(config$n_ctx * 1.5, 4096)
      config$recommended_n_predict <- min(config$recommended_n_predict * 1.5, 300)
    }
  }

  # Add performance tips
  config$tips <- c(
    paste0("Recommended context size: ", config$n_ctx, " tokens"),
    paste0("Recommended generation length: ", config$recommended_n_predict, " tokens"),
    "For faster inference, use temperature=0.0 (greedy decoding)",
    "For better quality, increase temperature to 0.8-1.0",
    "Small models work best with concise, clear prompts"
  )

  return(config)
}

#' Stream text completion with real-time token generation
#'
#' @param ctx Model context from edge_load_model()
#' @param prompt Input text prompt
#' @param callback Function called for each generated token. Receives list with token info.
#' @param n_predict Maximum tokens to generate (default: 128)
#' @param temperature Sampling temperature (default: 0.8)
#' @param top_p Top-p sampling parameter (default: 0.95)
#' @return List with full response and generation statistics
#' 
#' @examples
#' \dontrun{
#' model_path <- file.path(tempdir(), "model.gguf")
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path)
#'   
#'   # Basic streaming with token display
#'   result <- edge_stream_completion(ctx, "Hello, how are you?", 
#'     callback = function(data) {
#'       if (!data$is_final) {
#'         cat(data$token)
#'         utils::flush.console()
#'       } else {
#'         cat("\n[Done: ", data$total_tokens, " tokens]\n")
#'       }
#'       return(TRUE)  # Continue generation
#'     })
#'   
#'   edge_free_model(ctx)
#' }
#' }
#' @param timeout_seconds Optional timeout in seconds for inference
#' @export
edge_stream_completion <- function(ctx, prompt, callback, n_predict = 128L, temperature = 0.8, top_p = 0.95,
                                   timeout_seconds = NULL) {
  if (!is.function(callback)) {
    stop("Callback must be a function")
  }

  if (!is.character(prompt) || length(prompt) != 1L) {
    stop("Prompt must be a single character string")
  }

  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  
  if (!is.null(timeout_seconds)) {
    if (!is.numeric(timeout_seconds) || length(timeout_seconds) != 1 || timeout_seconds <= 0) {
      stop("timeout_seconds must be a positive number of seconds")
    }
  }

  start_time <- Sys.time()
  safe_callback <- function(data) {
    if (!is.null(timeout_seconds)) {
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
      if (elapsed > timeout_seconds) {
        message("Timeout reached, stopping generation.")
        return(FALSE)
      }
    }

    result <- tryCatch({
      callback(data)
    }, error = function(e) {
      message("Callback error: ", e$message)
      return(FALSE)
    })

    if (!is.logical(result) || length(result) != 1 || is.na(result)) {
      message("Callback must return TRUE/FALSE; stopping generation.")
      return(FALSE)
    }

    result
  }

  edge_completion_stream_internal(ctx, prompt, safe_callback,
                                 as.integer(n_predict),
                                 as.numeric(temperature),
                                 as.numeric(top_p))
}

#' Interactive chat session with streaming responses
#'
#' @param ctx Model context from edge_load_model()
#' @param system_prompt Optional system prompt to set context
#' @param max_history Maximum conversation turns to keep in context (default: 10)
#' @param n_predict Maximum tokens per response (default: 200)
#' @param temperature Sampling temperature (default: 0.8)
#' @param verbose Whether to print status messages (default: TRUE)
#' @return NULL (runs interactively)
#' 
#' @examples
#' \dontrun{
#' setup <- edge_quick_setup("TinyLlama-1.1B")
#' ctx <- setup$context
#' 
#' if (!is.null(ctx)) {
#'   # Start interactive chat with streaming
#'   # edge_chat_stream(ctx, 
#'   #   system_prompt = "You are a helpful R programming assistant.")
#'   
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_chat_stream <- function(ctx, system_prompt = NULL, max_history = 10, n_predict = 200L, temperature = 0.8, verbose = TRUE) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  
  conversation_history <- list()
  
  # Add system prompt if provided
  if (!is.null(system_prompt)) {
    conversation_history <- append(conversation_history, 
                                 list(list(role = "system", content = system_prompt)))
    }
  
  if (verbose) {
    message("Chat started! Type 'quit', 'exit', or 'bye' to end.")
    message("Responses will stream in real-time.")
  }
  if (verbose) cat("\n")
  
  while (TRUE) {
    user_input <- readline("You: ")
    
    # Check for exit commands
    if (tolower(trimws(user_input)) %in% c("quit", "exit", "bye", "")) {
      if (verbose) message("Chat ended!")
      break
    }
    
    # Add user message to history
    conversation_history <- append(conversation_history, 
                                 list(list(role = "user", content = user_input)))
    
    # Build prompt from conversation history
    prompt <- build_chat_prompt(conversation_history)
    
    # Stream the response
    if (verbose) {
      cat("Assistant: ")
      utils::flush.console()
    }
    
    current_response <- ""
    
    result <- edge_stream_completion(ctx, prompt, 
      callback = function(data) {
        if (!data$is_final) {
          if (verbose) {
            cat(data$token)
            utils::flush.console()
          }
          return(TRUE)  # Continue generation
        } else {
          if (verbose) cat("\n\n")
          current_response <<- data$full_response
          return(TRUE)
        }
      },
      n_predict = n_predict, 
      temperature = temperature)
    
    # Use the response captured during streaming
    assistant_response <- current_response
    
    # Add assistant response to history
    conversation_history <- append(conversation_history, 
                                 list(list(role = "assistant", content = assistant_response)))
    
    # Trim history if too long
    if (length(conversation_history) > max_history * 2) {
      # Keep system prompt (if exists) and most recent exchanges
      system_msgs <- conversation_history[sapply(conversation_history, function(x) x$role == "system")]
      recent_msgs <- tail(conversation_history[sapply(conversation_history, function(x) x$role != "system")], 
                         max_history * 2 - length(system_msgs))
      conversation_history <- c(system_msgs, recent_msgs)
    }
  }
}

#' Build chat prompt from conversation history
#' @param history List of conversation turns with role and content
#' @return Formatted prompt string
#' @export
build_chat_prompt <- function(history) {
  if (length(history) == 0) {
    return("")
  }

  if (!is.list(history) || !all(sapply(history, function(x) is.list(x) && !is.null(x$role) && !is.null(x$content)))) {
    stop("history must be a list of turns each with $role and $content fields")
  }

  prompt_parts <- c()

  for (turn in history) {
    if (turn$role == "system") {
      prompt_parts <- c(prompt_parts, paste("System:", turn$content))
    } else if (turn$role == "user") {
      prompt_parts <- c(prompt_parts, paste("Human:", turn$content))
    } else if (turn$role == "assistant") {
      prompt_parts <- c(prompt_parts, paste("Assistant:", turn$content))
    }
  }
  
  # Add the start of assistant response
  prompt <- paste(c(prompt_parts, "Assistant:"), collapse = "\n")
  return(prompt)
}

#' Clean up cache directory and manage storage
#'
#' Remove outdated model files from the cache directory to comply with CRAN
#' policies about actively managing cached content and keeping sizes small.
#'
#' @param cache_dir Cache directory path (default: user cache directory)
#' @param max_age_days Maximum age of files to keep in days (default: option edgemodelr.cache_max_age_days or 30)
#' @param max_size_mb Maximum total cache size in MB (default: option edgemodelr.cache_max_size_mb or 5000)
#' @param use_lru If TRUE, evict least-recently-used files when size exceeds limit
#' @param ask Whether to ask for user confirmation before deletion (only in interactive sessions)
#' @param verbose Whether to print status messages (default: TRUE)
#' @return Invisible list of deleted files
#' @examples
#' \dontrun{
#' # Clean cache files older than 30 days
#' edge_clean_cache()
#' 
#' # Clean cache with custom settings
#' edge_clean_cache(max_age_days = 7, max_size_mb = 100)
#' }
#' @export
edge_clean_cache <- function(cache_dir = NULL,
                             max_age_days = getOption("edgemodelr.cache_max_age_days", 30),
                             max_size_mb = getOption("edgemodelr.cache_max_size_mb", 5000),
                             use_lru = TRUE,
                             ask = TRUE,
                             verbose = TRUE) {
  if (is.null(cache_dir)) {
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
  }
  
  if (!dir.exists(cache_dir)) {
    if (verbose) message("Cache directory does not exist: ", cache_dir)
    return(invisible(character(0)))
  }
  
  files <- list.files(cache_dir, full.names = TRUE, recursive = TRUE)
  if (length(files) == 0) {
    if (verbose) message("Cache directory is empty")
    return(invisible(character(0)))
  }
  
  file_info <- file.info(files)
  file_info$path <- files
  file_info$age_days <- as.numeric(difftime(Sys.time(), file_info$mtime, units = "days"))
  file_info$size_mb <- file_info$size / (1024^2)
  
  # Files to delete by age
  old_files <- file_info[file_info$age_days > max_age_days, ]
  
  # Files to delete by size (oldest first if total exceeds limit)
  total_size <- sum(file_info$size_mb, na.rm = TRUE)
  size_files <- character(0)
  if (total_size > max_size_mb) {
    file_info_sorted <- if (use_lru) file_info[order(file_info$mtime), ] else file_info
    cumsum_size <- cumsum(file_info_sorted$size_mb)
    excess_files <- file_info_sorted[cumsum_size <= (total_size - max_size_mb), ]
    if (nrow(excess_files) > 0) {
      size_files <- excess_files$path
    }
  }
  
  files_to_delete <- unique(c(old_files$path, size_files))
  
  if (length(files_to_delete) == 0) {
    if (verbose) message("No files need cleaning")
    return(invisible(character(0)))
  }
  
  # Show what will be deleted
  total_delete_size <- sum(file_info[file_info$path %in% files_to_delete, "size_mb"], na.rm = TRUE)
  if (verbose) {
    message("Found ", length(files_to_delete), " files to delete (", 
            round(total_delete_size, 1), " MB)")
  }
  
  # Ask for confirmation only when requested and in an interactive session
  if (ask && interactive()) {
    consent <- utils::askYesNo(
      paste("Delete", length(files_to_delete), "cached files?"),
      default = TRUE
    )
    
    if (is.na(consent) || !consent) {
      if (verbose) message("Cleanup cancelled by user")
      return(invisible(character(0)))
    }
  }
  
  # Delete files
  deleted_files <- character(0)
  for (file in files_to_delete) {
    if (file.exists(file)) {
      success <- file.remove(file)
      if (success) {
        deleted_files <- c(deleted_files, file)
      }
    }
  }
  
  if (length(deleted_files) > 0) {
    if (verbose) message("Deleted ", length(deleted_files), " files from cache")
  }
  
  invisible(deleted_files)
}

#' Cache size information
#' @param cache_dir Cache directory path
#' @return List with total_size_mb and file_count
#' @export
edge_cache_info <- function(cache_dir = NULL) {
  if (is.null(cache_dir)) {
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
  }
  if (!dir.exists(cache_dir)) {
    return(list(cache_dir = cache_dir, total_size_mb = 0, file_count = 0))
  }
  files <- list.files(cache_dir, full.names = TRUE, recursive = TRUE)
  if (length(files) == 0) {
    return(list(cache_dir = cache_dir, total_size_mb = 0, file_count = 0))
  }
  size_mb <- sum(file.info(files)$size / (1024^2), na.rm = TRUE)
  list(cache_dir = cache_dir, total_size_mb = round(size_mb, 1), file_count = length(files))
}

#' Control llama.cpp logging verbosity
#'
#' Enable or disable verbose output from the underlying llama.cpp library.
#' By default, all output except errors is suppressed to comply with CRAN policies.
#'
#' @param enabled Logical. If TRUE, enables verbose llama.cpp output. If FALSE (default), 
#'   suppresses all output except errors.
#' @return Invisible NULL
#' @examples
#' # Enable verbose output (not recommended for normal use)
#' edge_set_verbose(TRUE)
#' 
#' # Disable verbose output (default, recommended)
#' edge_set_verbose(FALSE)
#' @export
edge_set_verbose <- function(enabled = FALSE) {
  set_llama_logging(enabled)
  invisible(NULL)
}

#' Performance benchmarking for model inference
#'
#' Test inference speed and throughput with the current model to measure
#' the effectiveness of optimizations.
#'
#' @param ctx Model context from edge_load_model()
#' @param prompt Test prompt to use for benchmarking (default: standard test)
#' @param n_predict Number of tokens to generate for the test
#' @param iterations Number of test iterations to average results
#' @param track_memory If TRUE, attempt to report peak memory usage (best-effort)
#' @return List with performance metrics
#'
#' @examples
#' \dontrun{
#' setup <- edge_quick_setup("TinyLlama-1.1B")
#' if (!is.null(setup$context)) {
#'   ctx <- setup$context
#'   perf <- edge_benchmark(ctx)
#'   print(perf)
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_benchmark <- function(ctx, prompt = "The quick brown fox", n_predict = 50, iterations = 3,
                           track_memory = FALSE) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context")
  }

  times <- numeric(iterations)
  tokens_per_sec <- numeric(iterations)
  peak_memory_mb <- NA_real_

  message("Running performance benchmark with ", iterations, " iterations...")

  for (i in 1:iterations) {
    start_time <- Sys.time()
    if (track_memory && exists("memory.size", where = baseenv(), inherits = TRUE)) {
      peak_memory_mb <- max(peak_memory_mb, utils::memory.size(), na.rm = TRUE)
    }
    result <- edge_completion(ctx, prompt, n_predict = n_predict, temperature = 0.0)
    end_time <- Sys.time()

    elapsed <- as.numeric(end_time - start_time, units = "secs")
    times[i] <- elapsed
    tokens_per_sec[i] <- n_predict / elapsed

    message("Iteration ", i, ": ", round(elapsed, 3), "s (", round(tokens_per_sec[i], 1), " tokens/sec)")
  }

  list(
    avg_time_per_token = mean(times) / n_predict,
    avg_tokens_per_second = mean(tokens_per_sec),
    min_tokens_per_second = min(tokens_per_sec),
    max_tokens_per_second = max(tokens_per_sec),
    total_time = sum(times),
    iterations = iterations,
    tokens_per_iteration = n_predict,
    peak_memory_mb = if (track_memory) peak_memory_mb else NA_real_
  )
}

#' Query SIMD optimization status
#'
#' Reports which SIMD (Single Instruction Multiple Data) features were enabled
#' at compile time. This helps verify that the package is using CPU-optimized
#' code paths for faster inference.
#'
#' @return List with:
#' \describe{
#'   \item{architecture}{CPU architecture (e.g., "x86_64", "aarch64")}
#'   \item{compiler_features}{Character vector of compiler-detected SIMD features}
#'   \item{ggml_features}{Character vector of GGML-level optimization flags}
#'   \item{is_generic}{Logical; TRUE if compiled with generic (scalar) fallback}
#' }
#'
#' @examples
#' info <- edge_simd_info()
#' cat("Architecture:", info$architecture, "\n")
#' cat("SIMD features:", paste(info$compiler_features, collapse = ", "), "\n")
#' if (info$is_generic) {
#'   cat("Running in generic mode. Reinstall with EDGEMODELR_SIMD=AVX2 for better performance.\n")
#' }
#' @export
edge_simd_info <- function() {
  edge_simd_info_internal()
}

#' Find and prepare GGUF models for use with edgemodelr
#'
#' This function finds compatible GGUF model files from various sources including
#' Ollama installations, custom directories, or any folder containing GGUF files.
#' It tests each model for compatibility with edgemodelr and creates organized
#' copies or links for easy access.
#'
#' @param source_dirs Vector of directories to search for GGUF files. If NULL,
#'   automatically searches common locations including Ollama installation.
#' @param target_dir Directory where to create links/copies of compatible models.
#'   If NULL, creates a "local_models" directory in the current working directory.
#' @param create_links Logical. If TRUE (default), creates symbolic links to save disk space.
#'   If FALSE, copies the files (uses more disk space but more compatible).
#' @param model_pattern Optional pattern to filter model files by name.
#' @param test_compatibility Logical. If TRUE (default), tests each GGUF file for
#'   compatibility with edgemodelr before including it.
#' @param min_size_mb Minimum file size in MB to consider (default: 50MB).
#'   Helps filter out config files and focus on actual models.
#' @param verbose Logical. Whether to print detailed progress information.
#' @return List containing information about compatible models, including paths and metadata
#'
#' @details
#' This function performs the following steps:
#' \enumerate{
#'   \item Searches specified directories (or auto-detects common locations)
#'   \item Identifies GGUF format files above the minimum size threshold
#'   \item Optionally tests each file for compatibility with edgemodelr
#'   \item Creates organized symbolic links or copies in the target directory
#'   \item Returns detailed information about working models
#' }
#'
#' The function automatically searches these locations if no source_dirs specified:
#' \itemize{
#'   \item Ollama models directory (~/.ollama/models or %USERPROFILE%/.ollama/models)
#'   \item Current working directory
#'   \item ~/models directory (if exists)
#'   \item Common model storage locations
#' }
#'
#' @examples
#' \dontrun{
#' # Basic usage - auto-detect and test all GGUF models
#' models_info <- edge_find_gguf_models()
#' if (!is.null(models_info) && length(models_info$models) > 0) {
#'   # Load the first compatible model
#'   ctx <- edge_load_model(models_info$models[[1]]$path)
#'   result <- edge_completion(ctx, "Hello", n_predict = 20)
#'   edge_free_model(ctx)
#' }
#'
#' # Search specific directories
#' models_info <- edge_find_gguf_models(source_dirs = c("~/Downloads", "~/models"))
#'
#' # Skip compatibility testing (faster but less reliable)
#' models_info <- edge_find_gguf_models(test_compatibility = FALSE)
#'
#' # Copy files instead of creating links
#' models_info <- edge_find_gguf_models(create_links = FALSE)
#'
#' # Filter for specific models
#' models_info <- edge_find_gguf_models(model_pattern = "llama")
#' }
#' @export
edge_find_gguf_models <- function(source_dirs = NULL, target_dir = NULL, create_links = TRUE,
                                  model_pattern = NULL, test_compatibility = TRUE,
                                  min_size_mb = 50, verbose = TRUE) {

  # Set default target directory
  if (is.null(target_dir)) {
    target_dir <- file.path(getwd(), "local_models")
  }
  target_dir <- path.expand(target_dir)

  if (verbose) {
    message("[*] Searching for GGUF model files...")
  }

  # Determine source directories to search
  if (is.null(source_dirs)) {
    source_dirs <- c()

    # Add Ollama directory if it exists
    ollama_paths <- c(
      path.expand("~/.ollama/models/blobs"),
      file.path(Sys.getenv("USERPROFILE"), ".ollama", "models", "blobs"),
      file.path(Sys.getenv("USERPROFILE"), "OneDrive", ".ollama", "models", "blobs"),
      file.path(Sys.getenv("APPDATA"), "Ollama", "models", "blobs"),
      file.path(Sys.getenv("APPDATA"), ".ollama", "models", "blobs"),
      file.path(Sys.getenv("LOCALAPPDATA"), "Ollama", "models", "blobs"),
      file.path(Sys.getenv("LOCALAPPDATA"), ".ollama", "models", "blobs"),
      "/usr/share/ollama/models/blobs"
    )
    for (path in ollama_paths) {
      expanded <- path.expand(path)
      if (dir.exists(expanded)) {
        source_dirs <- c(source_dirs, expanded)
        if (verbose) message("[*] Found Ollama models: ", path)
        break
      }
    }

    # Add common model directories
    common_dirs <- c(
      getwd(),
      path.expand("~/models"),
      path.expand("~/Downloads"),
      file.path(getwd(), "models"),
      file.path(getwd(), "local_models")
    )

    for (dir in common_dirs) {
      if (dir.exists(dir)) {
        source_dirs <- c(source_dirs, dir)
      }
    }

    if (length(source_dirs) == 0) {
      if (verbose) message("[!] No source directories found")
      return(NULL)
    }
  } else {
    # Expand user-provided directories
    source_dirs <- path.expand(source_dirs)
    source_dirs <- source_dirs[dir.exists(source_dirs)]

    if (length(source_dirs) == 0) {
      if (verbose) message("[!] None of the specified directories exist")
      return(NULL)
    }
  }

  if (verbose) {
    message("[*] Searching ", length(source_dirs), " directories for GGUF files")
  }

  # Find all potential GGUF files
  all_files <- c()
  for (dir in source_dirs) {
    # Look for .gguf files
    gguf_files <- list.files(dir, pattern = "\\.gguf$", full.names = TRUE, recursive = TRUE)

    # Also check files without extension (like Ollama blobs)
    all_dir_files <- list.files(dir, full.names = TRUE, recursive = FALSE)

    for (file in all_dir_files) {
      if (file.exists(file) && !dir.exists(file) && file.size(file) > min_size_mb * 1024 * 1024) {
        # Check if it's a GGUF file by reading header
        tryCatch({
          con <- file(file, "rb")
          header <- readBin(con, "raw", 4)
          close(con)

          if (length(header) == 4 && identical(header, charToRaw("GGUF"))) {
            gguf_files <- c(gguf_files, file)
          }
        }, error = function(e) {
          # Skip files that can't be read
        })
      }
    }

    all_files <- c(all_files, gguf_files)
  }

  # Remove duplicates and filter by size
  all_files <- unique(all_files)
  all_files <- all_files[file.size(all_files) >= min_size_mb * 1024 * 1024]

  if (length(all_files) == 0) {
    if (verbose) {
      message("[!] No GGUF files found above ", min_size_mb, "MB")
      message("[*] Try: edge_quick_setup('TinyLlama-1.1B') to download a compatible model")
    }
    return(NULL)
  }

  # Apply pattern filter if specified
  if (!is.null(model_pattern)) {
    pattern_matches <- grepl(model_pattern, basename(all_files), ignore.case = TRUE)
    all_files <- all_files[pattern_matches]

    if (length(all_files) == 0) {
      if (verbose) message("[!] No files match pattern: ", model_pattern)
      return(NULL)
    }
  }

  if (verbose) {
    message("[*] Found ", length(all_files), " GGUF files to process")
  }

  # Create target directory
  if (!dir.exists(target_dir)) {
    dir.create(target_dir, recursive = TRUE)
    if (verbose) message("[*] Created target directory: ", target_dir)
  }

  # Process each file
  compatible_models <- list()

  for (i in seq_along(all_files)) {
    file_path <- all_files[i]
    file_name <- basename(file_path)
    size_mb <- round(file.size(file_path) / (1024^2), 1)

    if (verbose) {
      message("[*] Processing ", i, "/", length(all_files), ": ", file_name, " (", size_mb, "MB)")
    }

    # Generate a clean model name
    if (grepl("\\.gguf$", file_name)) {
      model_name <- gsub("\\.gguf$", "", file_name)
    } else {
      # For Ollama blobs, create descriptive name
      model_name <- paste0("model_", size_mb, "mb_", substr(file_name, 8, 15))
    }

    model_name <- gsub("[^a-zA-Z0-9._-]", "_", model_name)  # Clean name

    # Test compatibility if requested
    is_compatible <- TRUE
    test_result <- ""

    if (test_compatibility) {
      if (verbose) message("  [*] Testing compatibility...")

      tryCatch({
        # Try to load the model
        temp_ctx <- edge_load_model(file_path, n_ctx = 256, n_gpu_layers = 0)

        if (is_valid_model(temp_ctx)) {
          # Quick test
          test_result <- edge_completion(temp_ctx, "Hi", n_predict = 3, temperature = 0.1)
          edge_free_model(temp_ctx)

          if (verbose) message("  [+] Compatible! Test: '", test_result, "'")
        } else {
          is_compatible <- FALSE
          if (verbose) message("  [!] Model validation failed")
        }
      }, error = function(e) {
        is_compatible <- FALSE
        if (verbose) message("  [!] Compatibility test failed: ", e$message)
      })
    } else {
      if (verbose) message("  [*] Skipping compatibility test")
    }

    if (is_compatible) {
      # Create target file
      target_file <- file.path(target_dir, paste0(model_name, ".gguf"))

      # Ensure unique filename
      counter <- 1
      original_target <- target_file
      while (file.exists(target_file)) {
        target_file <- file.path(target_dir, paste0(model_name, "_", counter, ".gguf"))
        counter <- counter + 1
      }

      # Copy or link the file
      success <- FALSE
      if (create_links && .Platform$OS.type != "windows") {
        tryCatch({
          file.symlink(file_path, target_file)
          success <- TRUE
          if (verbose) message("  [*] Created symbolic link")
        }, error = function(e) {
          if (verbose) message("  [!] Symlink failed, copying...")
        })
      }

      if (!success) {
        success <- file.copy(file_path, target_file, overwrite = TRUE)
        if (success && verbose) message("  [*] Copied file")
      }

      if (success) {
        compatible_models[[model_name]] <- list(
          name = model_name,
          path = target_file,
          source = file_path,
          size_mb = size_mb,
          compatible = is_compatible,
          test_result = test_result,
          linked = create_links && .Platform$OS.type != "windows"
        )
      }
    }
  }

  if (length(compatible_models) == 0) {
    if (verbose) {
      message("[!] No compatible models found")
      if (test_compatibility) {
        message("[*] Try with test_compatibility = FALSE to include untested models")
      }
      message("[*] Or use: edge_quick_setup('TinyLlama-1.1B') for a guaranteed working model")
    }
    return(NULL)
  }

  if (verbose) {
    message("[+] Found ", length(compatible_models), " compatible models!")
    message("[*] Usage example:")
    message("   ctx <- edge_load_model('", compatible_models[[1]]$path, "')")
    message("   result <- edge_completion(ctx, 'Your prompt here', n_predict = 50)")
    message("   edge_free_model(ctx)")
  }

  return(list(
    target_dir = target_dir,
    models = compatible_models,
    total_models = length(compatible_models),
    method = ifelse(test_compatibility, "tested_compatibility", "format_only"),
    search_dirs = source_dirs,
    disk_space_saved = ifelse(create_links, sum(sapply(compatible_models, function(x) x$size_mb)), 0)
  ))
}

