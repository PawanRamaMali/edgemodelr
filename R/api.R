#' Load a local GGUF model for inference
#'
#' @param model_path Path to a .gguf model file
#' @param n_ctx Maximum context length (default: 2048)
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0, CPU-only)
#' @return External pointer to the loaded model context
#' 
#' @examples
#' \donttest{
#' # Load a TinyLlama model
#' model_path <- "~/models/TinyLlama-1.1B-Chat.Q4_K_M.gguf"
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path, n_ctx = 2048)
#'   
#'   # Generate completion
#'   result <- edge_completion(ctx, "Explain R data.frame:", n_predict = 100)
#'   cat(result)
#'   
#'   # Free model when done
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_load_model <- function(model_path, n_ctx = 2048L, n_gpu_layers = 0L) {
  if (!file.exists(model_path)) {
    stop("Model file not found: ", model_path, "\n",
         "Try these options:\n",
         "  1. Download a model: edge_download_model('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf')\n",
         "  2. Quick setup: edge_quick_setup('TinyLlama-1.1B')\n",
         "  3. List models: edge_list_models()")
  }
  
  if (!grepl("\\.gguf$", model_path, ignore.case = TRUE)) {
    warning("Model file should have .gguf extension for optimal compatibility")
  }
  
  # Try to load the model using the raw Rcpp function
  tryCatch({
    .Call(`_edgemodelr_edge_load_model`,
          normalizePath(model_path), 
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
#' model_path <- "model.gguf"
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path)
#'   result <- edge_completion(ctx, "The capital of France is", n_predict = 50)
#'   cat(result)
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_completion <- function(ctx, prompt, n_predict = 128L, temperature = 0.8, top_p = 0.95) {
  if (!is.character(prompt) || length(prompt) != 1L) {
    stop("Prompt must be a single character string")
  }
  
  .Call(`_edgemodelr_edge_completion`,
        ctx, 
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
#' model_path <- "model.gguf"
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path)
#'   # ... use model ...
#'   edge_free_model(ctx)  # Clean up
#' }
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
  
  invisible(.Call(`_edgemodelr_edge_free_model`, ctx))
}

#' Check if model context is valid
#'
#' @param ctx Model context to check
#' @return Logical indicating if context is valid
#' @export
is_valid_model <- function(ctx) {
  tryCatch({
    .Call(`_edgemodelr_is_valid_model`, ctx)
  }, error = function(e) FALSE)
}

#' Download a GGUF model from Hugging Face
#'
#' @param model_id Hugging Face model identifier (e.g., "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
#' @param filename Specific GGUF file to download
#' @param cache_dir Directory to store downloaded models (default: "~/.cache/edgemodelr")
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
edge_download_model <- function(model_id, filename, cache_dir = NULL, force_download = FALSE) {
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
  
  # Set default cache directory
  if (is.null(cache_dir)) {
    cache_dir <- file.path(path.expand("~"), ".cache", "edgemodelr")
  }
  
  # Create cache directory if it doesn't exist
  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, recursive = TRUE)
    message("Created cache directory: ", cache_dir)
  }
  
  # Construct local file path
  local_path <- file.path(cache_dir, basename(filename))
  
  # Check if file already exists
  if (file.exists(local_path) && !force_download) {
    message("Model already exists: ", local_path)
    return(local_path)
  }
  
  # Construct Hugging Face URL
  base_url <- "https://huggingface.co"
  download_url <- file.path(base_url, model_id, "resolve", "main", filename)
  
  message("Downloading model...")
  message("From: ", download_url)
  message("To: ", local_path)
  
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
      message("Download completed successfully!")
      message("Model size: ", round(file.info(local_path)$size / (1024^2), 1), "MB")
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
#' @param cache_dir Directory to store downloaded models
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
edge_quick_setup <- function(model_name, cache_dir = NULL) {
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
  
  message("Setting up ", model_name, "...")
  
  # Download model
  model_path <- edge_download_model(
    model_id = model_info$model_id,
    filename = model_info$filename,
    cache_dir = cache_dir
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
#' model_path <- "model.gguf"
#' if (file.exists(model_path)) {
#'   ctx <- edge_load_model(model_path)
#'   
#'   # Basic streaming with token display
#'   result <- edge_stream_completion(ctx, "Hello, how are you?", 
#'     callback = function(data) {
#'       if (!data$is_final) {
#'         cat(data$token)
#'         flush.console()
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
  if (!is.character(prompt) || length(prompt) != 1L) {
    stop("Prompt must be a single character string")
  }
  
  if (!is.function(callback)) {
    stop("Callback must be a function")
  }
  
  .Call(`_edgemodelr_edge_completion_stream`,
        ctx, prompt, callback, 
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
edge_chat_stream <- function(ctx, system_prompt = NULL, max_history = 10, n_predict = 200L, temperature = 0.8) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  
  conversation_history <- list()
  
  # Add system prompt if provided
  if (!is.null(system_prompt)) {
    conversation_history <- append(conversation_history, 
                                 list(list(role = "system", content = system_prompt)))
    message("System prompt set.")
  }
  
  message("Chat started! Type 'quit', 'exit', or 'bye' to end.")
  message("Responses will stream in real-time.")
  cat("\n")
  
  while (TRUE) {
    user_input <- readline("You: ")
    
    # Check for exit commands
    if (tolower(trimws(user_input)) %in% c("quit", "exit", "bye", "")) {
      message("Chat ended!")
      break
    }
    
    # Add user message to history
    conversation_history <- append(conversation_history, 
                                 list(list(role = "user", content = user_input)))
    
    # Build prompt from conversation history
    prompt <- build_chat_prompt(conversation_history)
    
    # Stream the response
    cat("Assistant: ")
    utils::flush.console()
    
    current_response <- ""
    
    result <- edge_stream_completion(ctx, prompt, 
      callback = function(data) {
        if (!data$is_final) {
          cat(data$token)
          utils::flush.console()
          return(TRUE)  # Continue generation
        } else {
          cat("\n\n")
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