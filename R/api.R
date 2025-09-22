#' Load a local GGUF model for inference
#'
#' @param model_path Path to a .gguf model file
#' @param n_ctx Maximum context length (default: 2048)
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0, CPU-only)
#' @return External pointer to the loaded model context
#' 
#' @examples
#' \donttest{
#' # Quick setup with automatic model download
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
#' # Manual model loading from downloaded file
#' model_path <- "path/to/your/model.gguf"
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path, n_ctx = 2048, n_gpu_layers = 0)
#'   # ... use model ...
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_load_model <- function(model_path, n_ctx = 2048L, n_gpu_layers = 0L) {
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

  # Optimize context size for better performance
  n_ctx <- max(512, min(n_ctx, 32768))  # Clamp to reasonable range

  # Inform user about optimizations
  if (n_ctx > 8192) {
    message("Large context size (", n_ctx, ") may impact performance. Consider using smaller values for faster inference.")
  }

  # Try to load the model using the raw Rcpp function
  tryCatch({
    edge_load_model_internal(normalizePath(model_path), 
                           as.integer(n_ctx),
                           as.integer(n_gpu_layers))
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
}

#' Generate text completion using loaded model
#'
#' @param ctx Model context from edge_load_model()
#' @param prompt Input text prompt
#' @param n_predict Maximum tokens to generate (default: 128)
#' @param temperature Sampling temperature (default: 0.8)
#' @param top_p Top-p sampling parameter (default: 0.95)
#' @return Generated text as character string
#' 
#' @examples
#' \donttest{
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
edge_completion <- function(ctx, prompt, n_predict = 128L, temperature = 0.8, top_p = 0.95) {
  if (!is.character(prompt) || length(prompt) != 1L) {
    stop("Prompt must be a single character string")
  }

  # Optimize parameters for better performance and quality
  n_predict <- max(1, min(n_predict, 4096))  # Clamp to reasonable range
  temperature <- max(0.0, min(temperature, 2.0))  # Clamp temperature
  top_p <- max(0.1, min(top_p, 1.0))  # Clamp top_p

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
#' \donttest{
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
#' @return Path to the downloaded model file
#' 
#' @examples
#' \donttest{
#' # Download TinyLlama model
#' model_path <- edge_download_model(
#'   model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
#'   filename = "tinyllama-1.1b-chat-v1.0.q4_k_m.gguf"
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
edge_download_model <- function(model_id, filename, cache_dir = NULL, force_download = FALSE, verbose = TRUE) {
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
  
  # Construct local file path
  local_path <- file.path(cache_dir, basename(filename))
  
  # Check if file already exists
  if (file.exists(local_path) && !force_download) {
    if (verbose) message("Model already exists: ", local_path)
    return(local_path)
  }
  
  # Construct Hugging Face URL
  base_url <- "https://huggingface.co"
  download_url <- file.path(base_url, model_id, "resolve", "main", filename)
  
  if (verbose) {
    message("Downloading model...")
    message("From: ", download_url)
    message("To: ", local_path)
  }
  
  # Download the file
  tryCatch({
    # Try different download methods
    download_success <- FALSE
    methods_to_try <- c("auto", "curl", "wget", "wininet")
    
    for (method in methods_to_try) {
      tryCatch({
        utils::download.file(download_url, local_path, mode = "wb", method = method, quiet = FALSE)
        if (file.exists(local_path) && file.info(local_path)$size > 1000) {  # At least 1KB
          download_success <- TRUE
          break
        }
      }, error = function(e) {
        # Silently try next method
      })
    }
    
    if (download_success) {
      if (verbose) {
        message("Download completed successfully!")
        message("Model size: ", round(file.info(local_path)$size / (1024^2), 1), "MB")
      }
      return(local_path)
    } else {
      stop("All download methods failed")
    }
    
  }, error = function(e) {
    # Clean up partial download
    if (file.exists(local_path)) {
      file.remove(local_path)
    }
    
    stop("Failed to download model: ", e$message, "\n",
         "Possible solutions:\n",
         "1. Check your internet connection\n",
         "2. Try downloading manually:\n",
         "   curl -L -o '", local_path, "' '", download_url, "'\n",
         "3. Or use a different model from edge_list_models()")
  })
}

#' List popular pre-configured models
#'
#' @return Data frame with model information
#' @export
edge_list_models <- function() {
  models <- data.frame(
    name = c(
      "TinyLlama-1.1B", "TinyLlama-OpenOrca", 
      "llama3.2-1b", "llama3.2-3b",
      "phi3-mini", "qwen2.5-1.5b", "gemma2-2b"
    ),
    size = c(
      "~700MB", "~700MB",
      "~800MB", "~2GB", 
      "~2.4GB", "~1GB", "~1.6GB"
    ),
    model_id = c(
      "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
      "TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF",
      "bartowski/Llama-3.2-1B-Instruct-GGUF",
      "bartowski/Llama-3.2-3B-Instruct-GGUF", 
      "microsoft/Phi-3-mini-4k-instruct-gguf",
      "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
      "google/gemma-2-2b-it-gguf"
    ),
    filename = c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "tinyllama-1.1b-1t-openorca.Q4_K_M.gguf",
      "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
      "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
      "Phi-3-mini-4k-instruct-q4.gguf",
      "qwen2.5-1.5b-instruct-q4_k_m.gguf", 
      "gemma-2-2b-it-q4_k_m.gguf"
    ),
    use_case = c(
      "Testing", "Better Chat", 
      "2024 Mobile", "2024 General", 
      "2024 Reasoning", "2024 Multilingual", "2024 Gemma"
    ),
    stringsAsFactors = FALSE
  )
  
  return(models)
}

#' Quick setup for a popular model
#'
#' @param model_name Name of the model from edge_list_models()
#' @param cache_dir Directory to store downloaded models (default: user cache directory via tools::R_user_dir())
#' @param verbose Whether to print status messages (default: TRUE)
#' @return List with model path and context (if llama.cpp is available)
#' 
#' @examples
#' \donttest{
#' # Quick setup with 2024 models
#' setup <- edge_quick_setup("llama3.2-1b")  # Modern small model
#' ctx <- setup$context
#' 
#' if (!is.null(ctx)) {
#'   response <- edge_completion(ctx, "Hello!")
#'   cat("Response:", response, "\n")
#'   edge_free_model(ctx)
#' }
#' }
#' @export  
edge_quick_setup <- function(model_name, cache_dir = NULL, verbose = TRUE) {
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
  model_info <- models[models$name == model_name, ]
  
  if (nrow(model_info) == 0) {
    stop("Model '", model_name, "' not found. Available models:\n", 
         paste(models$name, collapse = ", "))
  }
  
  if (verbose) message("Setting up ", model_name, "...")
  
  # Download model
  model_path <- edge_download_model(
    model_id = model_info$model_id,
    filename = model_info$filename,
    cache_dir = cache_dir,
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
#' \donttest{
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
#' @export
edge_stream_completion <- function(ctx, prompt, callback, n_predict = 128L, temperature = 0.8, top_p = 0.95) {
  if (!is.function(callback)) {
    stop("Callback must be a function")
  }
  
  if (!is.character(prompt) || length(prompt) != 1L) {
    stop("Prompt must be a single character string")
  }
  
  edge_completion_stream_internal(ctx, prompt, callback, 
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
#' \donttest{
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
    
    # Extract assistant response (remove the prompt part)
    assistant_response <- sub(prompt, "", result$full_response, fixed = TRUE)
    
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
#' @param max_age_days Maximum age of files to keep in days (default: 30)
#' @param max_size_mb Maximum total cache size in MB (default: 500)
#' @param interactive Whether to ask for user confirmation before deletion
#' @return Invisible list of deleted files
#' @examples
#' \donttest{
#' # Clean cache files older than 30 days
#' edge_clean_cache()
#' 
#' # Clean cache with custom settings
#' edge_clean_cache(max_age_days = 7, max_size_mb = 100)
#' }
#' @export
edge_clean_cache <- function(cache_dir = NULL, max_age_days = 30, max_size_mb = 500, interactive = TRUE, verbose = TRUE) {
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
    file_info_sorted <- file_info[order(file_info$mtime), ]
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
  
  # Ask for confirmation in interactive mode
  if (interactive && interactive()) {
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
#' @return List with performance metrics
#'
#' @examples
#' \donttest{
#' setup <- edge_quick_setup("TinyLlama-1.1B")
#' if (!is.null(setup$context)) {
#'   ctx <- setup$context
#'   perf <- edge_benchmark(ctx)
#'   print(perf)
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_benchmark <- function(ctx, prompt = "The quick brown fox", n_predict = 50, iterations = 3) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context")
  }

  times <- numeric(iterations)
  tokens_per_sec <- numeric(iterations)

  message("Running performance benchmark with ", iterations, " iterations...")

  for (i in 1:iterations) {
    start_time <- Sys.time()
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
    tokens_per_iteration = n_predict
  )
}