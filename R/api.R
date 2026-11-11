#' Load a local GGUF model for inference
#'
#' @param model_path Path to a .gguf model file
#' @param n_ctx Maximum context length (default: 2048)
#' @param n_gpu_layers Number of layers to offload to GPU (default: 0, CPU-only).
#'   Use -1 to offload all layers (requires CUDA backend via edge_install_cuda()).
#' @param n_threads Number of CPU threads for inference (default: NULL = one per
#'   physical core). Extra hyperthreads slow generation down, often by half.
#'   Set to a lower value to leave cores free for other tasks.
#' @param flash_attn Enable flash attention for faster inference (default: TRUE).
#'   Reduces memory usage and improves speed. Set to FALSE for maximum compatibility.
#' @param embeddings Enable embedding extraction mode (default: FALSE).
#'   Must be TRUE to use \code{\link{edge_embeddings}} with this context.
#'   A model loaded with \code{embeddings = TRUE} can still be used for
#'   text generation.
#' @param prompt_cache Keep the KV cache between calls and reuse the part of a new prompt
#'   that matches the start of the previous one, so only the rest is processed
#'   (default: FALSE). Much faster when prompts share a long fixed beginning,
#'   such as a system prompt. Each completion still depends only on its own
#'   prompt.
#' @param lora Path(s) to LoRA adapters in GGUF format to apply on top of the
#'   model (default: NULL, none). The adapters must be made for this base
#'   model. A PEFT adapter can be converted with llama.cpp's
#'   \code{convert_lora_to_gguf.py}.
#' @param lora_scale Scale for each adapter, recycled to the number of
#'   adapters (default: 1). 0 turns an adapter off. Change it later with
#'   \code{\link{edge_lora_scale}}.
#' @param n_parallel How many prompts \code{\link{edge_map}} and
#'   \code{\link{edge_extract_batch}} generate at once (default: 1). Each
#'   step then decodes one token for every prompt in a single batch, which on
#'   CPU costs little more than one token, so batches of texts finish several
#'   times faster. The prompts share the context, so make \code{n_ctx} large
#'   enough for all of them with their answers; fewer run at once when they
#'   do not fit.
#' @param cache_type_k,cache_type_v Number format of the keys and values in
#'   the KV cache (default: "f16"). One of "f16", "bf16", "q8_0", "q5_1",
#'   "q5_0", "q4_1", "q4_0" or "iq4_nl". "q8_0" for both halves the memory
#'   the cache takes, with almost no change in output, so a larger
#'   \code{n_ctx} fits; the 4-bit types save more and cost some quality.
#'   A quantized \code{cache_type_v} needs \code{flash_attn = TRUE}.
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
edge_load_model <- function(model_path, n_ctx = 2048L, n_gpu_layers = 0L, n_threads = NULL, flash_attn = TRUE, embeddings = FALSE,
                            prompt_cache = FALSE, lora = NULL, lora_scale = 1,
                            n_parallel = 1L, cache_type_k = "f16", cache_type_v = "f16") {
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
  if (!is.numeric(n_gpu_layers) || (n_gpu_layers < 0 && n_gpu_layers != -1)) {
    stop("n_gpu_layers must be a non-negative integer, or -1 to offload all layers to GPU")
  }
  # Validate n_threads: NULL means one thread per physical core
  if (!is.null(n_threads)) {
    if (!is.numeric(n_threads) || n_threads < 1) {
      stop("n_threads must be a positive integer or NULL for auto-detect")
    }
  }
  if (!is.logical(flash_attn) || length(flash_attn) != 1) {
    stop("flash_attn must be TRUE or FALSE")
  }
  if (is.null(lora)) lora <- character()
  if (!is.character(lora)) stop("lora must be a character vector of adapter paths")
  missing_lora <- lora[!file.exists(lora)]
  if (length(missing_lora)) {
    stop("LoRA adapter file does not exist: ", paste(missing_lora, collapse = ", "))
  }
  if (!is.numeric(lora_scale) || length(lora_scale) < 1 || anyNA(lora_scale)) {
    stop("lora_scale must be a numeric vector")
  }
  lora_scale <- rep_len(as.numeric(lora_scale), length(lora))
  if (!is.numeric(n_parallel) || length(n_parallel) != 1L || is.na(n_parallel) ||
      n_parallel < 1 || n_parallel > 64 || n_parallel != round(n_parallel)) {
    stop("n_parallel must be a whole number from 1 to 64")
  }
  cache_types <- c("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl")
  for (arg in c("cache_type_k", "cache_type_v")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1L || !value %in% cache_types) {
      stop(arg, " must be one of ", paste0('"', cache_types, '"', collapse = ", "))
    }
  }
  if (!cache_type_v %in% c("f16", "bf16") && !isTRUE(flash_attn)) {
    stop("a quantized cache_type_v needs flash_attn = TRUE")
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

  # Clamp to reasonable range (allow as low as 128 for short tasks)
  n_ctx <- max(128, min(n_ctx, 32768))

  # Warn about large contexts
  if (n_ctx > 8192) {
    message("Large context size (", n_ctx, ") may impact performance. Consider using smaller values for faster inference.")
  }

  # Try to load the model using the raw Rcpp function
  # -1 means "all layers on GPU"; translate to a large number for llama.cpp
  n_gpu_layers_actual <- if (n_gpu_layers == -1) .Machine$integer.max else as.integer(n_gpu_layers)

  ctx <- tryCatch({
    edge_load_model_internal(normalizePath(model_path),
                             as.integer(n_ctx),
                             n_gpu_layers_actual,
                             as.integer(if (is.null(n_threads)) .default_threads() else n_threads),
                             as.logical(flash_attn),
                             as.logical(embeddings),
                             isTRUE(prompt_cache),
                             normalizePath(lora),
                             lora_scale,
                             as.integer(n_parallel),
                             cache_type_k,
                             cache_type_v)
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

  # Batches use every hardware thread, which in a container with a CPU
  # limit is far more than it may use.
  quota <- .cpu_quota()
  if (!is.na(quota)) {
    n <- if (is.null(n_threads)) .default_threads() else as.integer(n_threads)
    n <- max(1L, n)
    edge_set_threads_internal(ctx, n, max(n, as.integer(floor(quota))))
  }

  # Touch file to update LRU metadata (best-effort)
  try(Sys.setFileTime(model_path, Sys.time()), silent = TRUE)

  ctx
}

#' Get or change the scale of loaded LoRA adapters
#'
#' Adapters are loaded with \code{edge_load_model(lora = ...)}. Changing a
#' scale takes effect on the next call and needs no reload, so one base model
#' can switch between adapters, or turn them off with 0, from call to call.
#'
#' @param ctx Model context from \code{\link{edge_load_model}}
#' @param scale NULL to only read the scales, or one number per loaded
#'   adapter, in the order they were given to \code{edge_load_model()}
#' @return The adapter scales, a numeric vector (invisibly when setting)
#'
#' @details Changing a scale clears the context's KV cache, so with
#'   \code{prompt_cache = TRUE} the next prompt is processed in full.
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("base.gguf", lora = c("terms.gguf", "labels.gguf"),
#'                        lora_scale = c(1, 0))
#' edge_completion(ctx, "Map 'Once daily' to a CDISC term:")
#' edge_lora_scale(ctx, c(0, 1))   # switch to the second adapter
#' edge_lora_scale(ctx)            # 0 1
#' edge_free_model(ctx)
#' }
#' @export
edge_lora_scale <- function(ctx, scale = NULL) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (is.null(scale)) return(edge_lora_scales_internal(ctx))
  if (!is.numeric(scale) || anyNA(scale)) stop("scale must be a numeric vector")
  invisible(edge_lora_scales_internal(ctx, as.numeric(scale)))
}

#' Save or load the prompt cache
#'
#' With \code{prompt_cache = TRUE} a context keeps the processed tokens of the
#' last prompt and reuses the part a new prompt shares with it.
#' \code{edge_save_state()} writes that cache to a file and
#' \code{edge_load_state()} reads it back, in the same or a later R session,
#' so a long system prompt or document is processed once and not again every
#' session.
#'
#' @param ctx Model context from \code{\link{edge_load_model}}, loaded with
#'   \code{prompt_cache = TRUE}
#' @param path File to write or read
#' @return The number of tokens saved or loaded, invisibly
#'
#' @details The file holds the KV cache, so it is large: about
#'   \code{n_tokens * n_layers * 2 * n_kv_heads * head_size * 2} bytes, for
#'   example 22 MB for 1000 tokens of a 1B model. It can only be loaded into a
#'   context of the same model, with the same LoRA adapters at the same
#'   scales; anything else is an error. The saved tokens must fit in the
#'   context's \code{n_ctx}.
#'
#'   To cache just a fixed beginning, run it once with a single token to
#'   generate, then save: \code{edge_completion(ctx, prefix, n_predict = 1)}.
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf", prompt_cache = TRUE)
#' edge_completion(ctx, long_document, n_predict = 1)
#' edge_save_state(ctx, "document.kv")
#'
#' # Later, in a new session:
#' ctx <- edge_load_model("model.gguf", prompt_cache = TRUE)
#' edge_load_state(ctx, "document.kv")
#' edge_completion(ctx, paste0(long_document, "\nSummary:"))  # skips the document
#' }
#' @export
edge_save_state <- function(ctx, path) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(path) || length(path) != 1L || is.na(path) || !nzchar(path)) {
    stop("path must be a single file path")
  }
  invisible(edge_save_state_internal(ctx, path.expand(path)))
}

#' @rdname edge_save_state
#' @export
edge_load_state <- function(ctx, path) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(path) || length(path) != 1L || is.na(path) || !file.exists(path)) {
    stop("path must be an existing state file")
  }
  invisible(edge_load_state_internal(ctx, path.expand(path)))
}

# Validates the sampling arguments shared by the generation functions and
# returns them as the list the C++ code reads. With seed = NULL the seed is
# drawn from R's random number generator, so set.seed() makes sampled text
# reproducible. Greedy decoding (temperature 0) uses no seed and draws none.
.sampling_options <- function(temperature, top_p, top_k = 0L, min_p = 0,
                              repeat_penalty = 1, seed = NULL, stop = NULL) {
  num1 <- function(x, name) {
    if (!is.numeric(x) || length(x) != 1L || is.na(x)) {
      stop(name, " must be a single number", call. = FALSE)
    }
    as.numeric(x)
  }
  temperature <- num1(temperature, "temperature")
  top_p <- num1(top_p, "top_p")
  top_k <- num1(top_k, "top_k")
  min_p <- num1(min_p, "min_p")
  repeat_penalty <- num1(repeat_penalty, "repeat_penalty")
  if (top_k < 0 || top_k != round(top_k)) {
    stop("top_k must be a whole number, 0 or more (0 turns it off)", call. = FALSE)
  }
  if (min_p < 0 || min_p >= 1) {
    stop("min_p must be at least 0 and less than 1 (0 turns it off)", call. = FALSE)
  }
  if (repeat_penalty <= 0) {
    stop("repeat_penalty must be positive (1 turns it off)", call. = FALSE)
  }
  if (!is.null(stop)) {
    if (!is.character(stop) || anyNA(stop) || any(!nzchar(stop))) {
      stop("stop must be a character vector of non-empty strings", call. = FALSE)
    }
  }
  if (is.null(seed)) {
    seed <- if (temperature > 0) sample.int(.Machine$integer.max, 1L) else 0
  } else {
    seed <- num1(seed, "seed")
    if (seed < 0 || seed >= 2^32 - 1 || seed != round(seed)) {
      stop("seed must be a whole number from 0 to 4294967294", call. = FALSE)
    }
  }
  list(temperature = temperature, top_p = top_p, top_k = as.integer(top_k),
       min_p = min_p, repeat_penalty = repeat_penalty, seed = as.numeric(seed),
       stop = if (is.null(stop)) NULL else as.character(stop))
}

#' Generate text completion using loaded model
#'
#' @param ctx Model context from edge_load_model()
#' @param prompt Input text prompt
#' @param n_predict Maximum tokens to generate (default: 128)
#' @param temperature Sampling temperature (default: 0.8). 0 gives greedy
#'   decoding: the same prompt always gives the same text.
#' @param top_p Top-p sampling parameter (default: 0.95)
#' @param timeout_seconds Optional timeout in seconds for inference
#' @param top_k Sample only from the k most likely tokens (default: 0, off).
#'   40 is a common choice.
#' @param min_p Drop tokens less likely than this fraction of the most likely
#'   one (default: 0, off). 0.05 is a common choice.
#' @param repeat_penalty Penalty for tokens used in the last 64 generated
#'   tokens (default: 1, off). Values from 1.1 to 1.3 stop small models
#'   from repeating themselves.
#' @param seed Seed for sampling (default: NULL, drawn from R's random number
#'   generator, so \code{set.seed()} makes the text reproducible). Ignored
#'   when \code{temperature = 0}.
#' @param stop Optional character vector of stop strings. Generation ends
#'   where the first one appears; the stop string itself is not returned.
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
#'   # Reproducible sampling, less repetition, stop at the first blank line
#'   edge_completion(ctx, "List three colours:", seed = 42,
#'                   repeat_penalty = 1.2, stop = "\n\n")
#'
#'   edge_free_model(ctx)
#' }
#' }
#' @export
edge_completion <- function(ctx, prompt, n_predict = 128L, temperature = 0.8, top_p = 0.95,
                            timeout_seconds = NULL, top_k = 0L, min_p = 0,
                            repeat_penalty = 1, seed = NULL, stop = NULL) {
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

  if (is.null(timeout_seconds)) {
    timeout_seconds <- 0
  } else if (!is.numeric(timeout_seconds) || length(timeout_seconds) != 1 ||
               is.na(timeout_seconds) || timeout_seconds <= 0) {
    stop("timeout_seconds must be a positive number of seconds")
  }

  sampling <- .sampling_options(temperature, top_p, top_k, min_p,
                                repeat_penalty, seed, stop)
  # The timeout is checked in the generation loop, once per token.
  edge_completion_internal(ctx, prompt, as.integer(n_predict), sampling,
                           as.numeric(timeout_seconds))
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
edge_download_model <- function(model_id, filename = NULL, cache_dir = NULL, force_download = FALSE,
                                verify_checksum = TRUE, expected_sha256 = NULL,
                                trust_first_use = FALSE, verbose = TRUE) {
  # Parameter validation
  if (is.null(model_id) || !is.character(model_id) || length(model_id) != 1) {
    stop("model_id must be a string")
  }
  if (nchar(model_id) == 0) {
    stop("model_id cannot be empty")
  }

  # Check if model_id is a friendly name from edge_list_models()
  resolved <- edge_resolve_model_name(model_id)
  if (!is.na(resolved)) {
    models <- edge_list_models()
    row <- models[models$name == resolved, ]
    if (verbose) message("Resolved '", model_id, "' to: ", resolved,
                          " (", row$size, ")")
    # Use the pre-configured URL and filename
    if (is.null(filename)) filename <- row$filename
    return(edge_download_url(row$download_url, filename = filename,
                              cache_dir = cache_dir, force_download = force_download,
                              verify_checksum = verify_checksum,
                              expected_sha256 = expected_sha256,
                              trust_first_use = trust_first_use,
                              verbose = verbose))
  }

  # Standard HuggingFace repo_id/filename path
  if (is.null(filename) || !is.character(filename) || length(filename) != 1 || nchar(filename) == 0) {
    stop("filename is required when model_id is a HuggingFace repo ID.\n",
         "Example: edge_download_model('unsloth/Qwen3-0.6B-GGUF', 'Qwen3-0.6B-Q4_K_M.gguf')\n",
         "Or use a friendly name: edge_download_model('Qwen3-0.6B')")
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
    if (!is.null(model_id)) hit <- hit[hit$model_id %in% model_id, , drop = FALSE]
    if (!is.null(filename)) hit <- hit[hit$filename %in% filename, , drop = FALSE]
    if (nrow(hit) > 0) {
      expected_sha256 <- hit$sha256[1]
    }
  }

  if (is.null(expected_sha256) || is.na(expected_sha256) || !nzchar(expected_sha256)) {
    if (verbose) {
      message("Checksum: no known SHA-256 for this model; skipping verification. ",
              "Set expected_sha256 or enable trust-first-use to store a local hash.")
    }
    if (trust_first_use) {
      computed <- .compute_sha256(path)
      if (!is.na(computed)) {
        # model_id is NULL for downloads by URL; a NULL column would make
        # the new row the wrong width for rbind().
        known <- rbind(known, data.frame(model_id = if (is.null(model_id)) NA_character_ else model_id,
                                         filename = if (is.null(filename)) NA_character_ else filename,
                                         sha256 = computed,
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

  # Try httr package if available (handles corporate proxies and SSL better)
  if (requireNamespace("httr", quietly = TRUE)) {
    if (verbose) message("Trying httr (handles corporate proxies/SSL)...")
    success <- .download_with_httr(url, destfile, hf_token, verbose)
    if (success) return(TRUE)
  }

  # Try R's download.file with libcurl method as last resort
  if (verbose) message("Trying R download.file with libcurl...")
  success <- .download_with_r(url, destfile, hf_token, verbose, max_retries)
  if (success) return(TRUE)

  return(FALSE)
}

#' Download using httr package (best for corporate networks)
#' @keywords internal
.download_with_httr <- function(url, destfile, hf_token = "", verbose = TRUE) {
  tryCatch({
    headers <- list()
    if (nchar(hf_token) > 0) {
      headers[["Authorization"]] <- paste("Bearer", hf_token)
    }
    resp <- httr::GET(url,
                       httr::write_disk(destfile, overwrite = TRUE),
                       if (verbose) httr::progress("down") else NULL,
                       do.call(httr::add_headers, headers),
                       httr::config(followlocation = TRUE))
    httr::status_code(resp) == 200 && file.exists(destfile) &&
      file.info(destfile)$size > 1024
  }, error = function(e) {
    if (verbose) message("httr download failed: ", e$message)
    if (file.exists(destfile)) try(file.remove(destfile), silent = TRUE)
    FALSE
  })
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
    "hermes" = "hermes-13b",
    # Qwen3 aliases
    "qwen3" = "Qwen3-0.6B",
    "qwen3-0.6b" = "Qwen3-0.6B",
    "qwen3-1.7b" = "Qwen3-1.7B",
    "qwen3-4b" = "Qwen3-4B",
    "qwen3-8b" = "Qwen3-8B",
    "qwen-3" = "Qwen3-0.6B",
    "qwen-3-0.6b" = "Qwen3-0.6B",
    "qwen-3-1.7b" = "Qwen3-1.7B",
    "qwen-3-4b" = "Qwen3-4B",
    "qwen-3-8b" = "Qwen3-8B"
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
      # Qwen3 models (new, excellent quality)
      "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
      # Medium models (2-5GB)
      "llama3-8b", "mistral-7b", "llama3.2-3b", "phi3-mini",
      # Large models (7-10GB) - Direct download from GPT4All CDN
      "orca2-13b", "wizardlm-13b", "hermes-13b", "starcoder"
    ),
    size = c(
      "~700MB", "~700MB",
      "~397MB", "~1.1GB", "~2.5GB", "~5.2GB",
      "~4.7GB", "~4.1GB", "~2GB", "~2.4GB",
      "~7.4GB", "~7.4GB", "~7.4GB", "~9GB"
    ),
    download_url = c(
      # HuggingFace models
      "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf",
      # Qwen3 GGUF from unsloth (no auth required)
      "https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf",
      "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
      "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
      "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
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
      "Qwen3-0.6B-Q4_K_M.gguf",
      "Qwen3-1.7B-Q4_K_M.gguf",
      "Qwen3-4B-Q4_K_M.gguf",
      "Qwen3-8B-Q4_K_M.gguf",
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
      "Fast chat (0.6B)", "Balanced (1.7B)", "Quality chat (4B)", "Best quality (8B)",
      "General (8B)", "General (7B)", "Mobile (3B)", "Reasoning",
      "13B Chat", "13B Instruct", "13B Chat", "Code (15B)"
    ),
    source = c(
      "huggingface", "huggingface",
      "huggingface", "huggingface", "huggingface", "huggingface",
      "gpt4all", "gpt4all", "huggingface", "huggingface",
      "gpt4all", "gpt4all", "gpt4all", "gpt4all"
    ),
    requires_auth = c(
      FALSE, FALSE,
      FALSE, FALSE, FALSE, FALSE,
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
#' @param temperature Sampling temperature (default: 0.8). 0 gives greedy
#'   decoding: the same prompt always gives the same text.
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
#' @inheritParams edge_completion
#' @details With \code{stop} strings, text that could be the start of one is
#'   held back until the next token decides it, so a callback can receive
#'   several tokens' text at once and never sees the stop string.
#' @export
edge_stream_completion <- function(ctx, prompt, callback, n_predict = 128L, temperature = 0.8, top_p = 0.95,
                                   timeout_seconds = NULL, top_k = 0L, min_p = 0,
                                   repeat_penalty = 1, seed = NULL, stop = NULL) {
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

  sampling <- .sampling_options(temperature, top_p, top_k, min_p,
                                repeat_penalty, seed, stop)
  edge_completion_stream_internal(ctx, prompt, safe_callback,
                                  as.integer(n_predict), sampling)
}

#' Interactive chat session with streaming responses
#'
#' @param ctx Model context from edge_load_model()
#' @param system_prompt Optional system prompt to set context
#' @param max_history Maximum conversation turns to keep in context (default: 10)
#' @param n_predict Maximum tokens per response (default: 200)
#' @param temperature Sampling temperature (default: 0.8)
#' @param verbose Whether to print status messages (default: TRUE)
#' @inheritParams edge_completion
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
edge_chat_stream <- function(ctx, system_prompt = NULL, max_history = 10, n_predict = 200L,
                             temperature = 0.8, verbose = TRUE,
                             top_p = 0.95, top_k = 0L, min_p = 0, repeat_penalty = 1,
                             seed = NULL, stop = NULL) {
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
      temperature = temperature, top_p = top_p, top_k = top_k,
      min_p = min_p, repeat_penalty = repeat_penalty, seed = seed,
      stop = stop)

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
build_chat_prompt <- function(history, ctx = NULL) {
  if (length(history) == 0) {
    return("")
  }

  if (!is.list(history) || !all(sapply(history, function(x) is.list(x) && !is.null(x$role) && !is.null(x$content)))) {
    stop("history must be a list of turns each with $role and $content fields")
  }

  # If a model context is provided, use the model's native chat template
  if (!is.null(ctx) && is_valid_model(ctx)) {
    return(edge_chat_apply_template_internal(ctx, history, TRUE))
  }

  # Fallback: generic ChatML format (works with most models)
  prompt_parts <- character()
  for (turn in history) {
    prompt_parts <- c(prompt_parts,
      paste0("<|im_start|>", turn$role, "\n", turn$content, "<|im_end|>"))
  }
  prompt <- paste(c(prompt_parts, "<|im_start|>assistant\n"), collapse = "\n")
  return(prompt)
}

#' Generate a chat completion using the model's native template
#'
#' Formats messages using the model's built-in chat template (read from GGUF
#' metadata), generates a completion, and returns only the assistant's response.
#' This is the recommended way to do multi-turn chat, as it uses the correct
#' template format for the specific model (e.g., ChatML for Qwen, Llama format
#' for Llama models).
#'
#' @param ctx Model context from edge_load_model()
#' @param messages A list of message objects, each with \code{role} ("system",
#'   "user", or "assistant") and \code{content} (character string).
#' @param n_predict Maximum tokens to generate (default: 256)
#' @param temperature Sampling temperature (default: 0.7)
#' @param top_p Nucleus sampling threshold (default: 0.95)
#' @param tools Optional list of tools from \code{\link{edge_tool}} that the
#'   model may call. See Details.
#' @param tool_choice With \code{tools}: "auto" (default) lets the model
#'   either call a tool or answer in text, "required" makes it call one of
#'   the tools, a tool's name makes it call that tool, and "none" makes it
#'   answer in text. With "auto", a call identical to the last one, whose
#'   result is already in \code{messages}, is not returned; the model
#'   answers in text instead.
#' @inheritParams edge_completion
#' @return Without \code{tools}, a character string containing only the
#'   assistant's response text. With \code{tools}, a list with
#'   \code{content} (the text answer, or NULL when the model called a tool),
#'   \code{tool_calls} (a list of calls, each with \code{name} and
#'   \code{arguments}, a named list) and \code{message} (the assistant
#'   message to append to \code{messages} before the next call).
#'
#' @details
#' With \code{tools}, the tools are described in the system message and
#' generation is constrained by a grammar, so a tool call is always valid
#' JSON naming a known tool, with every argument present and of the right
#' type. The model writes a call as
#' \code{{"tool": "name", "arguments": {...}}}. Run the tool yourself, append
#' \code{result$message} and a message with \code{role = "tool"}, the tool's
#' \code{name} and its result as \code{content}, then call again so the model
#' can use the result. Tool results are passed to the model as user
#' messages, since not every chat template has a tool role.
#'
#' The tool call is always generated greedily; \code{temperature} and the
#' other sampling arguments apply to the text answer.
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#'
#' # Simple chat
#' answer <- edge_chat_completion(ctx, list(
#'   list(role = "user", content = "What is R?")
#' ))
#' cat(answer)
#'
#' # Multi-turn with system prompt
#' answer <- edge_chat_completion(ctx, list(
#'   list(role = "system", content = "You are a helpful R tutor."),
#'   list(role = "user", content = "What is a data.frame?"),
#'   list(role = "assistant", content = "A data.frame is a table..."),
#'   list(role = "user", content = "How do I filter rows?")
#' ))
#'
#' # Tool calling
#' weather <- edge_tool("get_weather", "Get the current weather for a city",
#'                      list(city = "string", unit = c("celsius", "fahrenheit")))
#' msgs <- list(list(role = "user", content = "Is it warm in Paris?"))
#' res <- edge_chat_completion(ctx, msgs, tools = list(weather), temperature = 0)
#' if (length(res$tool_calls) > 0) {
#'   call <- res$tool_calls[[1]]
#'   call$arguments$city  # "Paris"
#'   msgs <- c(msgs, list(res$message,
#'                        list(role = "tool", name = call$name, content = "24 C, sunny")))
#'   res <- edge_chat_completion(ctx, msgs, tools = list(weather), temperature = 0)
#'   cat(res$content)
#' }
#'
#' edge_free_model(ctx)
#' }
#' @seealso \code{\link{build_chat_prompt}}, \code{\link{edge_completion}},
#'   \code{\link{edge_tool}}
#' @export
edge_chat_completion <- function(ctx, messages, n_predict = 256L,
                                  temperature = 0.7, top_p = 0.95, top_k = 0L,
                                  min_p = 0, repeat_penalty = 1, seed = NULL,
                                  stop = NULL, tools = NULL,
                                  tool_choice = "auto") {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.list(messages) || length(messages) == 0L) {
    stop("messages must be a non-empty list of message objects with $role and $content")
  }

  n_predict <- max(1L, min(as.integer(n_predict), 4096L))
  temperature <- max(0.0, min(temperature, 2.0))
  top_p <- max(0.1, min(top_p, 1.0))
  sampling <- .sampling_options(temperature, top_p, top_k, min_p,
                                repeat_penalty, seed, stop)

  if (is.null(tools)) {
    # Format prompt using model's native chat template
    prompt <- build_chat_prompt(messages, ctx = ctx)
    return(edge_completion_internal(ctx, prompt, as.integer(n_predict), sampling))
  }

  tools <- .check_tools(tools)
  forced <- .check_tool_choice(tool_choice, tools)
  if (identical(forced, character())) {
    plain <- build_chat_prompt(.messages_with_tools(messages, NULL, NULL), ctx = ctx)
    text <- edge_completion_internal(ctx, plain, as.integer(n_predict), sampling)
    return(list(content = text, tool_calls = list(),
                message = list(role = "assistant", content = text)))
  }
  prompt <- build_chat_prompt(.messages_with_tools(messages, tools, forced), ctx = ctx)
  grammar <- .tool_grammar(tools, forced)
  # The call itself is chosen greedily: sampling there only adds wrong tools
  # and arguments. The user's sampling applies to the text answer.
  call_sampling <- .sampling_options(0, top_p)
  raw <- edge_completion_grammar_internal(ctx, prompt, grammar, "root",
                                          as.integer(n_predict), call_sampling, TRUE)
  reply <- .parse_tool_reply(raw, tools)
  if (!isTRUE(reply$none) &&
      !(is.null(forced) && .repeats_last_call(reply, messages, tools))) {
    return(reply)
  }

  # No tool needed, or the model asked again for the result it already has:
  # answer in text, from the conversation without the tool instructions, so
  # the model does not write JSON again.
  plain <- build_chat_prompt(.messages_with_tools(messages, NULL, NULL), ctx = ctx)
  text <- edge_completion_internal(ctx, plain, as.integer(n_predict), sampling)
  list(content = text, tool_calls = list(),
       message = list(role = "assistant", content = text))
}

# A field type is an enum when it is a character vector of allowed values.
# I("x") marks a single allowed value, which would otherwise be a type name.
.is_enum <- function(spec) {
  is.character(spec) && (length(spec) > 1L || inherits(spec, "AsIs"))
}

#' Define a tool the model can call
#'
#' A tool is a name, a description and typed arguments. Pass a list of tools
#' to \code{\link{edge_chat_completion}} and the model can call them; the
#' arguments always come back with the types given here.
#'
#' @param name Tool name: letters, digits and underscores, starting with a
#'   letter.
#' @param description What the tool does, in one sentence. The model reads
#'   it to decide when to call the tool.
#' @param parameters A named list of arguments, in the format of
#'   \code{\link{edge_json_grammar}}: "string", "number", "integer",
#'   "boolean" or a character vector of allowed values (use \code{I("x")}
#'   for a single allowed value). An argument can also be
#'   \code{list(type = ..., description = ...)} to tell the model what it
#'   means. Every argument is required. Default: no arguments.
#' @return An \code{edge_tool} object
#'
#' @examples
#' weather <- edge_tool("get_weather", "Get the current weather for a city",
#'   list(city = list(type = "string", description = "City name"),
#'        unit = c("celsius", "fahrenheit")))
#' weather
#' @seealso \code{\link{edge_chat_completion}}
#' @export
edge_tool <- function(name, description, parameters = list()) {
  ident <- "^[A-Za-z][A-Za-z0-9_]*$"
  if (!is.character(name) || length(name) != 1L || !grepl(ident, name)) {
    stop("name must be a single name of letters, digits and underscores, starting with a letter")
  }
  if (!is.character(description) || length(description) != 1L || is.na(description)) {
    stop("description must be a single character string")
  }
  if (!is.list(parameters) || (length(parameters) > 0L && is.null(names(parameters)))) {
    stop("parameters must be a named list")
  }
  arg_names <- names(parameters)
  if (length(parameters) > 0L &&
      (any(!grepl(ident, arg_names)) || anyDuplicated(arg_names))) {
    stop("argument names must be unique and made of letters, digits and underscores")
  }
  types <- list()
  notes <- character()
  for (i in seq_along(parameters)) {
    spec <- parameters[[i]]
    note <- ""
    if (is.list(spec)) {
      note <- if (is.null(spec$description)) "" else as.character(spec$description)
      spec <- spec$type
    }
    enum <- .is_enum(spec)
    if (!is.character(spec) || length(spec) == 0L || anyNA(spec) ||
        (!enum && !tolower(spec) %in% c("string", "number", "integer", "boolean"))) {
      stop("argument '", arg_names[i], "' must have type 'string', 'number', ",
           "'integer', 'boolean' or a character vector of allowed values")
    }
    if (enum && any(grepl('["\\\n]', spec))) {
      stop("allowed values of '", arg_names[i], "' must not contain quotes, ",
           "backslashes or line breaks")
    }
    types[[arg_names[i]]] <- if (enum) spec else tolower(spec)
    notes <- c(notes, note)
  }
  structure(list(name = name, description = description,
                 parameters = types, notes = notes),
            class = "edge_tool")
}

#' @export
print.edge_tool <- function(x, ...) {
  cat(.describe_tool(x), "\n", sep = "")
  invisible(x)
}

.check_tools <- function(tools) {
  if (inherits(tools, "edge_tool")) tools <- list(tools)
  if (!is.list(tools) || length(tools) == 0L ||
      !all(vapply(tools, inherits, logical(1), "edge_tool"))) {
    stop("tools must be a list of tools made with edge_tool()")
  }
  tool_names <- vapply(tools, function(t) t$name, character(1))
  if (anyDuplicated(tool_names)) stop("tool names must be unique")
  names(tools) <- tool_names
  tools
}

# NULL lets the model choose between a call and a text answer, character()
# means no call; otherwise the names of the tools it must choose from.
.check_tool_choice <- function(tool_choice, tools) {
  if (!is.character(tool_choice) || length(tool_choice) != 1L || is.na(tool_choice)) {
    stop("tool_choice must be \"auto\", \"required\", \"none\" or the name of a tool")
  }
  if (tool_choice == "auto") return(NULL)
  if (tool_choice == "none") return(character())
  if (tool_choice == "required") return(names(tools))
  if (tool_choice %in% names(tools)) return(tool_choice)
  stop("tool_choice must be \"auto\", \"required\", \"none\" or the name of a tool")
}

# One tool as the model sees it: name(arg: type, ...): description.
.describe_tool <- function(tool) {
  args <- vapply(seq_along(tool$parameters), function(i) {
    type <- tool$parameters[[i]]
    type <- if (.is_enum(type)) {
      paste0("one of ", paste0('"', type, '"', collapse = ", "))
    } else {
      type
    }
    note <- tool$notes[i]
    paste0(names(tool$parameters)[i], ": ", type,
           if (nzchar(note)) paste0(" (", note, ")") else "")
  }, character(1))
  paste0(tool$name, "(", paste(args, collapse = ", "), "): ", tool$description)
}

# Describes the tools in the system message and turns tool results into user
# messages, since the chat templates llama.cpp applies have no tool role.
.messages_with_tools <- function(messages, tools, forced) {
  out <- lapply(messages, function(m) {
    if (!is.list(m) || is.null(m$role) || is.null(m$content)) {
      stop("messages must be a list of turns each with $role and $content fields")
    }
    if (identical(m$role, "tool")) {
      who <- if (is.null(m$name)) "the tool" else m$name
      list(role = "user", content = paste0(
        "Result of ", who, ": ", m$content,
        "\nUse this result to answer my question."))
    } else {
      list(role = m$role, content = as.character(m$content))
    }
  })
  if (is.null(tools)) return(out)

  # A placeholder rather than a real tool in the example: a concrete name
  # there made small models call that tool for every request.
  usable <- if (is.null(forced)) tools else tools[forced]
  how <- if (length(usable) == 1L && !is.null(forced)) {
    paste0('{"tool": "', usable[[1]]$name, '", "arguments": {...}}')
  } else {
    '{"tool": "<tool name>", "arguments": {...}}'
  }
  text <- paste0(
    "You can call these tools:\n",
    paste0("- ", vapply(usable, .describe_tool, character(1)), collapse = "\n"),
    "\n\n",
    if (is.null(forced)) {
      paste0("Reply with only a JSON object, either\n", how,
             " to call the tool that fits the request, or\n",
             '{"tool": "none"} when no tool fits it, or when a tool result ',
             "above already answers it.")
    } else {
      paste0("Reply with only a JSON object that calls a tool: ", how)
    })

  if (identical(out[[1]]$role, "system")) {
    out[[1]]$content <- paste0(out[[1]]$content, "\n\n", text)
  } else {
    out <- c(list(list(role = "system", content = text)), out)
  }
  out
}

# A call is {"tool": "name", "arguments": {...}} in the compact layout, so
# jump-forward writes the keys and the rest of a tool name once it is
# unambiguous. Without forced tools the model may write {"tool": "none"}
# instead, and the text answer is then generated without a grammar. Letting
# the model start a text answer here made small models write the call inside
# prose or a code block, where it could not be read.
.tool_grammar <- function(tools, forced) {
  usable <- if (is.null(forced)) tools else tools[forced]
  alts <- character()
  defs <- character()
  needed <- character()
  for (i in seq_along(usable)) {
    t <- usable[[i]]
    prefix <- paste0("t", i, "-")
    obj <- .json_object_rule(t$parameters, prefix = prefix, compact = TRUE)
    defs <- c(defs,
              paste0(prefix, "call ::= ", '"\\"', t$name, '\\", \\"arguments\\": " ', obj$rule),
              obj$defs)
    needed <- c(needed, obj$needed)
    alts <- c(alts, paste0(prefix, "call"))
  }
  if (is.null(forced)) alts <- c(alts, '"\\"none\\""')
  call <- paste0('call ::= "{\\"tool\\": " (', paste(alts, collapse = " | "), ') "}"')
  root <- "root ::= call"
  shared <- c(
    'ws ::= ""',
    if ("string" %in% needed) 'string ::= "\\"" [^"\\\\\\n]* "\\""',
    if ("number" %in% needed) 'number ::= "-"? [0-9]+ ("." [0-9]+)?',
    if ("integer" %in% needed) 'integer ::= "-"? [0-9]+',
    if ("boolean" %in% needed) 'boolean ::= "true" | "false"'
  )
  paste0(paste(c(root, call, defs, shared), collapse = "\n"), "\n")
}

# Turns generated text into the edge_chat_completion() result. The grammar
# fixes the layout of a call, so the fields can be read in order with
# regular expressions; string values cannot contain quotes or backslashes.
.parse_tool_reply <- function(raw, tools) {
  if (identical(raw, '{"tool": "none"}')) return(list(none = TRUE))
  message <- list(role = "assistant", content = raw)
  head <- regmatches(raw, regexec('^\\{"tool": "([A-Za-z][A-Za-z0-9_]*)", "arguments": \\{', raw))[[1]]
  if (length(head) == 0L) {
    return(list(content = raw, tool_calls = list(), message = message))
  }
  tool <- tools[[head[2]]]
  rest <- substring(raw, nchar(head[1]))
  args <- list()
  ok <- TRUE
  for (arg in names(tool$parameters)) {
    m <- regmatches(rest, regexec(
      paste0('"', arg, '": ("[^"]*"|-?[0-9]+(\\.[0-9]+)?|true|false)'), rest))[[1]]
    if (length(m) == 0L) {
      ok <- FALSE
      break
    }
    v <- m[2]
    type <- tool$parameters[[arg]]
    args[[arg]] <- if (startsWith(v, '"')) {
      substr(v, 2L, nchar(v) - 1L)
    } else if (v %in% c("true", "false")) {
      v == "true"
    } else if (identical(type, "integer")) {
      as.integer(v)
    } else {
      as.numeric(v)
    }
  }
  if (!ok || !endsWith(raw, "}}")) {
    warning("The tool call was cut off; increase n_predict. Returning the text as content.",
            call. = FALSE)
    return(list(content = raw, tool_calls = list(), message = message))
  }
  list(content = NULL,
       tool_calls = list(list(name = tool$name, arguments = args)),
       message = message)
}

# TRUE when reply makes the same call as the last assistant call in messages
# and that call's result is already in the conversation. Small models often
# repeat a call instead of using its result.
.repeats_last_call <- function(reply, messages, tools) {
  if (length(reply$tool_calls) == 0L) return(FALSE)
  seen_result <- FALSE
  for (m in rev(messages)) {
    if (!is.list(m)) next
    if (identical(m$role, "tool")) {
      seen_result <- TRUE
    } else if (identical(m$role, "assistant")) {
      if (!seen_result) return(FALSE)
      last <- suppressWarnings(.parse_tool_reply(as.character(m$content), tools))
      if (length(last$tool_calls) == 0L) return(FALSE)
      return(identical(last$tool_calls[[1]], reply$tool_calls[[1]]))
    }
  }
  FALSE
}

#' Clean up cache directory and manage storage
#'
#' Remove outdated model files from the cache directory to comply with CRAN
#' policies about actively managing cached content and keeping sizes small.
#' The CUDA backend in the `cuda` subdirectory and the local checksum store
#' are never removed.
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

  files <- list.files(cache_dir, recursive = TRUE)
  # Leave the CUDA backend installed by edge_install_cuda() and the local
  # checksum store alone: they are not cached models.
  keep <- grepl("^cuda/", files) | basename(files) == "known_hashes.csv"
  files <- file.path(cache_dir, files[!keep])
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
    # Evict files until the total is under the limit: a file goes if the
    # files evicted before it do not yet free enough space.
    freed_before <- cumsum(file_info_sorted$size_mb) - file_info_sorted$size_mb
    excess_files <- file_info_sorted[freed_before < (total_size - max_size_mb), ]
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
#'   \item{kernels}{The quantized kernels in use: "avx2" when a generic
#'     build found AVX2 on this CPU and switched to its AVX2 kernels,
#'     "generic", or "native" for a build made with \code{EDGEMODELR_SIMD}.
#'     Set the environment variable \code{EDGEMODELR_KERNELS=generic}
#'     before loading the package to keep the generic kernels.}
#' }
#'
#' @examples
#' info <- edge_simd_info()
#' cat("Architecture:", info$architecture, "\n")
#' cat("SIMD features:", paste(info$compiler_features, collapse = ", "), "\n")
#' cat("Kernels:", info$kernels, "\n")
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


# ── CUDA / GPU backend support ─────────────────────────────────────────────────

#' Return the default path for the installed CUDA backend DLL
#' @keywords internal
edge_cuda_default_path <- function() {
  cuda_dir <- file.path(tools::R_user_dir("edgemodelr", "cache"), "cuda")
  if (!dir.exists(cuda_dir)) return(NULL)
  if (.Platform$OS.type == "windows") {
    dlls <- list.files(cuda_dir, pattern = "^ggml-cuda.*\\.dll$", full.names = TRUE)
  } else {
    dlls <- list.files(cuda_dir, pattern = "^(lib)?ggml-cuda.*\\.so(\\..*)?$", full.names = TRUE)
  }
  if (length(dlls) > 0) dlls[which.max(file.info(dlls)$mtime)] else NULL
}

#' Install the CUDA backend for GPU-accelerated inference
#'
#' Downloads a pre-built GGML CUDA backend shared library and stores it in the
#' edgemodelr cache. After installation, restart your R session (or call
#' `edge_reload_cuda()`) so the GPU backend is picked up during model loading.
#'
#' @param cuda_version CUDA version string. One of `"12.4"` or `"13.4"` (default).
#'   Use `"13.4"` for Blackwell GPUs (RTX 50 series, GeForce sm_120).
#' @param force Re-download even if already installed.
#' @param llama_build llama.cpp build number to pull the backend from
#'   (default: `"b11960"`, the build of llama.cpp v0.8.0 bundled in this
#'   package).
#' @return Invisibly returns the path to the installed CUDA backend DLL.
#' @examples
#' \dontrun{
#' edge_install_cuda()                  # GPU inference enabled after restart
#' ctx <- edge_load_model("model.gguf", n_gpu_layers = 35)
#' }
#' @export
edge_install_cuda <- function(cuda_version = "13.4",
                              force = FALSE,
                              llama_build = "b11960") {
  if (!.Platform$OS.type == "windows") {
    stop("edge_install_cuda() currently supports Windows only. ",
         "On Linux/macOS, build with EDGEMODELR_CUDA=1 at install time.")
  }

  cuda_dir <- file.path(tools::R_user_dir("edgemodelr", "cache"), "cuda")
  dir.create(cuda_dir, recursive = TRUE, showWarnings = FALSE)

  dll_name <- sprintf("ggml-cuda-%s.dll", cuda_version)
  dest <- file.path(cuda_dir, dll_name)

  if (file.exists(dest) && !force) {
    message("CUDA backend already installed at:\n  ", dest)
    message("Use force = TRUE to re-download.")
    invisible(edge_activate_cuda(dest))
    return(invisible(dest))
  }

  # Download from llama.cpp GitHub Releases
  zip_name <- sprintf("llama-%s-bin-win-cuda-%s-x64.zip", llama_build, cuda_version)
  release_url <- sprintf(
    "https://github.com/ggml-org/llama.cpp/releases/download/%s/%s",
    llama_build, zip_name
  )

  tmp_zip <- tempfile(fileext = ".zip")
  on.exit(unlink(tmp_zip), add = TRUE)

  message("Downloading llama.cpp CUDA backend (", cuda_version, ") from GitHub Releases...")
  message("This is a ~200MB download. Please wait...")

  # R's default 60 second timeout covers the whole transfer, which is too
  # short for this download on most connections.
  old_timeout <- options(timeout = max(3600, getOption("timeout")))
  on.exit(options(old_timeout), add = TRUE)

  tryCatch({
    utils::download.file(release_url, tmp_zip, mode = "wb", quiet = FALSE)
  }, error = function(e) {
    stop("Failed to download CUDA backend: ", conditionMessage(e),
         "\nURL: ", release_url)
  })

  # Extract ggml-cuda*.dll plus ggml shared libraries it depends on
  message("Extracting CUDA backend DLL and dependencies...")
  zip_contents <- utils::unzip(tmp_zip, list = TRUE)
  all_dlls <- zip_contents$Name[grepl("[.]dll$", zip_contents$Name)]

  cuda_dlls <- all_dlls[grepl("ggml-cuda", all_dlls, fixed = TRUE)]
  if (length(cuda_dlls) == 0) {
    stop("Could not find ggml-cuda DLL in the downloaded archive.\n",
         "Contents: ", paste(head(zip_contents$Name, 20), collapse = ", "))
  }

  # Extract the best match (prefer exact ggml-cuda-{version}.dll)
  best <- cuda_dlls[grepl(sprintf("ggml-cuda-%s", cuda_version), cuda_dlls)]
  if (length(best) == 0) best <- cuda_dlls[1]

  # Also extract ggml shared runtime DLLs that ggml-cuda.dll links against
  # (ggml-base.dll and ggml.dll from the same release are required dependencies)
  companion_dlls <- all_dlls[grepl("^(ggml-base|ggml)[.]dll$", basename(all_dlls))]
  to_extract <- unique(c(best[1], companion_dlls))
  utils::unzip(tmp_zip, files = to_extract, exdir = cuda_dir, junkpaths = TRUE)
  if (length(companion_dlls) > 0) {
    message("Also extracted companion DLLs: ",
            paste(basename(companion_dlls), collapse = ", "))
  }

  # Rename ggml-cuda to our canonical versioned name if needed
  extracted <- file.path(cuda_dir, basename(best[1]))
  if (!identical(extracted, dest)) {
    file.rename(extracted, dest)
  }

  message("CUDA backend installed at:\n  ", dest)
  message("\nTo activate GPU support, restart your R session or call:")
  message("  edge_reload_cuda()")

  edge_activate_cuda(dest)
  invisible(dest)
}

#' Install CUDA runtime libraries required for GPU inference
#'
#' The CUDA backend (`ggml-cuda-XX.dll`) requires two sets of runtime DLLs
#' that are **not** included with the NVIDIA display driver:
#' \itemize{
#'   \item \strong{nvcudart_hybrid64.dll}: CUDA hybrid runtime, already
#'         present on your system in the Windows DriverStore (installed with
#'         the GPU driver). This function copies it to the edgemodelr cache.
#'   \item \strong{cublas64_13.dll / cublasLt64_13.dll}: cuBLAS linear-algebra
#'         library (~400 MB download from NVIDIA's official redistrib server).
#' }
#'
#' After running this function, call \code{edge_reload_cuda()} and load a model
#' with \code{n_gpu_layers = -1} to run on your GPU.
#'
#' @param cuda_version CUDA major version string, e.g. \code{"13.4"} (default).
#'   Must match the version used with \code{\link{edge_install_cuda}()}.
#' @param force Reinstall even if the runtime DLLs are already present.
#' @return Invisibly returns the cuda cache directory path.
#' @seealso \code{\link{edge_install_cuda}}, \code{\link{edge_reload_cuda}}
#' @examples
#' \dontrun{
#' edge_install_cuda()          # install ggml-cuda GPU backend DLL (~140 MB)
#' edge_install_cuda_toolkit()  # install CUDA runtime DLLs (~400 MB, one-time)
#' edge_reload_cuda()           # activate in this R session
#' ctx <- edge_load_model("model.gguf", n_gpu_layers = -1)
#' result <- edge_completion(ctx, "Hello", n_predict = 20)
#' }
#' @export
edge_install_cuda_toolkit <- function(cuda_version = "13.4", force = FALSE) {
  if (.Platform$OS.type != "windows") {
    stop("edge_install_cuda_toolkit() is only needed on Windows. ",
         "On Linux/macOS, install the CUDA Toolkit via your package manager.")
  }

  cuda_dir <- file.path(tools::R_user_dir("edgemodelr", "cache"), "cuda")
  dir.create(cuda_dir, recursive = TRUE, showWarnings = FALSE)

  # Check if already installed
  cublas_present <- length(list.files(cuda_dir, pattern = "^cublas64.*\\.dll$",
                                      ignore.case = TRUE)) > 0
  cudart_present <- length(list.files(cuda_dir, pattern = "^nvcudart_hybrid64\\.dll$",
                                      ignore.case = TRUE)) > 0
  if (!force && cublas_present && cudart_present) {
    message("CUDA runtime already installed in:\n  ", cuda_dir)
    message("Use force = TRUE to reinstall.")
    return(invisible(cuda_dir))
  }

  # --- Step 1: nvcudart_hybrid64.dll from Windows DriverStore (no download) ---
  if (force || !cudart_present) {
    driver_store <- file.path(Sys.getenv("SystemRoot", "C:\\Windows"),
                              "System32", "DriverStore", "FileRepository")
    hybrid_dlls <- list.files(driver_store, pattern = "^nvcudart_hybrid64\\.dll$",
                               recursive = TRUE, full.names = TRUE)
    if (length(hybrid_dlls) > 0) {
      dest <- file.path(cuda_dir, "nvcudart_hybrid64.dll")
      file.copy(hybrid_dlls[1], dest, overwrite = TRUE)
      message("Copied nvcudart_hybrid64.dll from Windows DriverStore.")
    } else {
      message("Note: nvcudart_hybrid64.dll not found in DriverStore (unexpected).")
      message("      It is normally installed with the NVIDIA GPU driver.")
    }
  }

  # --- Step 2: cublas64_13.dll from NVIDIA redistrib (~400 MB) ---
  if (force || !cublas_present) {
    major_minor <- paste(strsplit(cuda_version, "\\.")[[1]][1:2], collapse = ".")
    redistrib_base <- "https://developer.download.nvidia.com/compute/cuda/redist"
    manifest_url   <- sprintf("%s/redistrib_%s.0.json", redistrib_base, major_minor)

    message("Fetching NVIDIA redistrib manifest for CUDA ", cuda_version, "...")
    manifest_text <- tryCatch({
      con <- url(manifest_url)
      on.exit(close(con), add = TRUE)
      paste(readLines(con, warn = FALSE), collapse = "\n")
    }, error = function(e) {
      stop("Could not fetch NVIDIA redistrib manifest for CUDA ", cuda_version,
           ".\nURL: ", manifest_url,
           "\nMake sure you are connected to the internet and CUDA ",
           cuda_version, " is publicly released.")
    })

    # Extract relative_path for libcublas/windows-x86_64
    pattern <- '"relative_path":\\s*"(libcublas/windows-x86_64/[^"]+\\.zip)"'
    m <- regmatches(manifest_text, regexpr(pattern, manifest_text, perl = TRUE))
    if (length(m) == 0) {
      stop("Could not find libcublas/windows-x86_64 in the CUDA ", cuda_version,
           " redistrib manifest.")
    }
    rel_path <- sub(pattern, "\\1", m, perl = TRUE)
    cublas_url <- paste0(redistrib_base, "/", rel_path)

    tmp_zip <- tempfile(fileext = ".zip")
    on.exit(unlink(tmp_zip), add = TRUE)

    message("Downloading cuBLAS for CUDA ", cuda_version, "...")
    message("URL: ", cublas_url)
    message("(This is a ~400 MB one-time download. Please wait...)")
    old_timeout <- getOption("timeout")
    options(timeout = 3600)   # 1-hour cap for large files
    on.exit(options(timeout = old_timeout), add = TRUE)
    tryCatch({
      utils::download.file(cublas_url, tmp_zip, mode = "wb", quiet = FALSE)
    }, error = function(e) {
      stop("Download failed: ", conditionMessage(e))
    })

    # Extract only cublas64_XX.dll and cublasLt64_XX.dll
    zip_contents <- utils::unzip(tmp_zip, list = TRUE)
    cublas_dlls  <- zip_contents$Name[
      grepl("^(cublas64|cublasLt64).*\\.dll$", basename(zip_contents$Name),
            ignore.case = TRUE)
    ]
    if (length(cublas_dlls) == 0) {
      stop("No cublas DLLs found in the downloaded archive.")
    }
    message("Extracting cuBLAS DLLs...")
    utils::unzip(tmp_zip, files = cublas_dlls, exdir = cuda_dir, junkpaths = TRUE)
    message("Installed: ", paste(basename(cublas_dlls), collapse = ", "))
  }

  installed <- list.files(cuda_dir, pattern = "\\.(dll|DLL)$", full.names = FALSE)
  message("\nAll files in cuda dir:\n", paste0("  ", installed, collapse = "\n"))
  message("\nDone! Call edge_reload_cuda() to activate GPU inference.")

  invisible(cuda_dir)
}

#' Activate an installed CUDA backend without restarting R
#'
#' This loads the CUDA backend DLL into the current session. It must be called
#' before the first `edge_load_model()` call in the session, otherwise a session
#' restart is required.
#'
#' @param path Path to the CUDA DLL. Defaults to the standard install location.
#' @return Invisibly returns `TRUE` if activation succeeded.
#' @export
edge_reload_cuda <- function(path = NULL) {
  if (is.null(path)) {
    path <- edge_cuda_default_path()
    if (is.null(path)) {
      stop("No CUDA backend found. Run edge_install_cuda() first.")
    }
  }
  edge_activate_cuda(path)
}

#' @keywords internal
edge_activate_cuda <- function(path) {
  if (!file.exists(path)) {
    warning("CUDA backend not found at: ", path)
    return(invisible(FALSE))
  }
  # Add the cuda directory to PATH so Windows DLL loader can find cudart/cublas
  # dependencies when ggml-cuda-XX.dll is loaded.
  cuda_dir <- normalizePath(dirname(path), winslash = "\\", mustWork = FALSE)
  current_path <- Sys.getenv("PATH")
  if (!grepl(cuda_dir, current_path, fixed = TRUE)) {
    path_sep <- if (.Platform$OS.type == "windows") ";" else ":"
    Sys.setenv(PATH = paste(cuda_dir, current_path, sep = path_sep))
  }
  ok <- edge_use_cuda_backend_internal(path)
  if (ok) {
    message("CUDA backend registered: ", path)
    message("GPU layers will be used when n_gpu_layers > 0 in edge_load_model().")
  }
  invisible(ok)
}

#' Check whether a CUDA backend is installed and active
#'
#' @return A list with `installed` (logical), `active` (logical), and `path`
#'   (character or NA).
#' @export
edge_cuda_info <- function() {
  installed_path <- edge_cuda_default_path()
  active_path    <- edge_cuda_backend_path_internal()
  backend_loaded <- edge_cuda_backend_loaded_internal()

  list(
    installed = !is.null(installed_path),
    active    = backend_loaded,
    path      = if (nchar(active_path) > 0) active_path
                else if (!is.null(installed_path)) installed_path
                else NA_character_
  )
}

# ============================================================================
# Structured Output / Grammar-Constrained Generation
# ============================================================================

#' Generate text constrained by a GBNF grammar
#'
#' Uses llama.cpp's grammar-constrained sampling to force output to conform
#' to a GBNF grammar specification. This ensures structured, parseable output.
#'
#' @param ctx Model context from edge_load_model()
#' @param prompt The input prompt
#' @param grammar A GBNF grammar string defining allowed output structure
#' @param grammar_root The root rule name in the grammar (default: "root")
#' @param n_predict Maximum tokens to generate (default: 512)
#' @param temperature Sampling temperature (default: 0.3, lower for structured
#'   output). 0 gives greedy decoding: the same prompt always gives the same text.
#' @param top_p Nucleus sampling threshold (default: 0.95)
#' @param jump_forward If TRUE (default), text that the grammar allows in only
#'   one way (JSON keys, separators, the rest of an enum value once it is
#'   unambiguous) is added without sampling, and its tokens are decoded in one
#'   batch, in fewer decode steps. The forced text is tokenized the standard way,
#'   which can differ from how the model would split it, so the output can
#'   differ slightly from \code{jump_forward = FALSE}. It always matches the
#'   grammar. Set FALSE to sample every token.
#' @inheritParams edge_completion
#' @return Character string containing only the generated text (not the prompt)
#'
#' @details
#' GBNF (GGML BNF) is a format for defining formal grammars that constrain
#' model output. This is useful for generating JSON, XML, or any structured format.
#'
#' Jump-forward leaves the last token of each forced stretch for the model to
#' sample, since that token may merge with the text after it (a model writes
#' an opening brace and quote as one token). It saves the most with long
#' JSON keys and \code{edge_json_grammar(compact = TRUE)}, where the layout
#' between fields is fixed and so is forced too: with four long keys a 0.6B
#' model wrote the object about 3x faster. With short keys the saving is small.
#'
#' Common GBNF patterns:
#' \itemize{
#'   \item JSON object: Use \code{edge_json_grammar()} for convenience
#'   \item Enum/choices: \code{'root ::= "yes" | "no" | "maybe"'}
#'   \item Number: \code{'root ::= [0-9]+'}
#' }
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#'
#' # Force yes/no output
#' grammar <- 'root ::= "yes" | "no"'
#' result <- edge_grammar_completion(ctx, "Is the sky blue? Answer:", grammar)
#'
#' # Force JSON output
#' json_grammar <- edge_json_grammar(list(
#'   sentiment = c("positive", "negative", "neutral"),
#'   confidence = "number"
#' ))
#' result <- edge_grammar_completion(ctx,
#'   "Analyze sentiment: 'I love this product'\nJSON:",
#'   json_grammar)
#'
#' edge_free_model(ctx)
#' }
#' @seealso \code{\link{edge_json_grammar}}, \code{\link{edge_extract}}, \code{\link{edge_classify}}
#' @export
edge_grammar_completion <- function(ctx, prompt, grammar, grammar_root = "root",
                                     n_predict = 512L, temperature = 0.3, top_p = 0.95,
                                     jump_forward = TRUE, top_k = 0L, min_p = 0,
                                     repeat_penalty = 1, seed = NULL, stop = NULL) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(prompt) || length(prompt) != 1L) {
    stop("Prompt must be a single character string")
  }
  if (!is.character(grammar) || length(grammar) != 1L || nchar(grammar) == 0L) {
    stop("grammar must be a non-empty character string containing a GBNF grammar")
  }

  n_predict <- max(1L, min(as.integer(n_predict), 4096L))
  temperature <- max(0.0, min(temperature, 2.0))
  top_p <- max(0.1, min(top_p, 1.0))

  sampling <- .sampling_options(temperature, top_p, top_k, min_p,
                                repeat_penalty, seed, stop)
  edge_completion_grammar_internal(
    ctx, prompt, grammar, grammar_root,
    as.integer(n_predict), sampling, isTRUE(jump_forward)
  )
}

# The GBNF for one JSON object with the fields of a schema (see
# edge_json_grammar()). Returns the object's rule body, the field rules it
# uses (named prefix + "field-" + i) and the shared rules they need
# ("string", "number", ...). The caller adds the shared rules and ws once.
.json_object_rule <- function(schema, prefix = "", compact = FALSE) {
  field_rules <- character()
  field_entries <- character()
  needed <- character()
  for (i in seq_along(schema)) {
    field_name <- names(schema)[i]
    field_spec <- schema[[i]]
    rule_name <- paste0(prefix, "field-", i)

    if (is.character(field_spec) && length(field_spec) == 1L && !.is_enum(field_spec)) {
      type <- tolower(field_spec)
      if (!type %in% c("string", "number", "integer", "boolean")) {
        stop("Unknown type '", type, "' for field '", field_name,
             "'. Use 'string', 'number', 'integer', 'boolean', or a character vector for enum.")
      }
      field_rules <- c(field_rules, paste0(rule_name, " ::= ", type))
      needed <- c(needed, type)
    } else if (.is_enum(field_spec)) {
      # Enum values
      enum_options <- paste0('"\\\"', field_spec, '\\\""', collapse = " | ")
      field_rules <- c(field_rules, paste0(rule_name, " ::= ", enum_options))
    } else {
      stop("Invalid schema for field '", field_name,
           "'. Use a string type name or character vector for enum.")
    }

    # The field entry: "key": value
    sep <- if (isTRUE(compact)) '\\": " ' else '\\"" ws ":" ws '
    field_entries <- c(field_entries, paste0('"\\"', field_name, sep, rule_name))
  }

  fields <- paste(field_entries,
                  collapse = if (isTRUE(compact)) ' ", " ' else ' ws "," ws ')
  rule <- if (length(field_entries) == 0L) {
    '"{}"'
  } else {
    paste0('"{" ws ', fields, ' ws "}"')
  }
  list(rule = rule, defs = field_rules, needed = unique(needed))
}

#' Generate a GBNF grammar for JSON output from a schema
#'
#' Converts a simple R list schema into a GBNF grammar string that constrains
#' model output to valid JSON matching the schema.
#'
#' @param schema A named list where each element defines a field. Values can be:
#'   \itemize{
#'     \item \code{"string"} - any string value
#'     \item \code{"number"} - a numeric value (integer or decimal)
#'     \item \code{"integer"} - an integer value
#'     \item \code{"boolean"} - true or false
#'     \item A character vector - enum of allowed values (e.g., \code{c("positive", "negative")}),
#'       or \code{I("x")} for a single allowed value
#'   }
#' @param compact If TRUE, the layout is fixed to \code{{"key": value, "key": value}},
#'   with one space after each colon and comma. Every key and separator is then
#'   forced, and \code{edge_grammar_completion()} adds them without sampling.
#'   If FALSE (default), the model may put no space, one space or a newline and
#'   indentation around each key and separator.
#' @return A GBNF grammar string
#'
#' @examples
#' \dontrun{
#' # Schema with enum and free-text fields
#' grammar <- edge_json_grammar(list(
#'   sentiment = c("positive", "negative", "neutral"),
#'   confidence = "number",
#'   explanation = "string"
#' ))
#' cat(grammar)
#'
#' # Use with edge_grammar_completion
#' ctx <- edge_load_model("model.gguf")
#' result <- edge_grammar_completion(ctx,
#'   "Analyze: 'Great product!'\nJSON:", grammar)
#' parsed <- jsonlite::fromJSON(result)
#' }
#' @export
edge_json_grammar <- function(schema, compact = FALSE) {
  if (!is.list(schema) || is.null(names(schema))) {
    stop("schema must be a named list")
  }

  # Nothing, one space, or one newline and indentation, as in llama.cpp's
  # JSON grammars. With unlimited whitespace, greedy decoding (temperature
  # 0) loops on newlines until n_predict runs out.
  ws <- if (isTRUE(compact)) 'ws ::= ""\n' else 'ws ::= "" | " " | "\\n" [ \\t]{0,20}\n'
  string_rule <- 'string ::= "\\"" [^"\\\\]* "\\""\n'
  number_rule <- 'number ::= "-"? [0-9]+ ("." [0-9]+)?\n'
  integer_rule <- 'integer ::= "-"? [0-9]+\n'
  boolean_rule <- 'boolean ::= "true" | "false"\n'

  obj <- .json_object_rule(schema, compact = compact)
  root_rule <- paste0("root ::= ", obj$rule, "\n")

  # Assemble grammar
  grammar_parts <- root_rule
  grammar_parts <- c(grammar_parts, paste0(obj$defs, "\n"))
  grammar_parts <- c(grammar_parts, ws)
  if ("string" %in% obj$needed) grammar_parts <- c(grammar_parts, string_rule)
  if ("number" %in% obj$needed) grammar_parts <- c(grammar_parts, number_rule)
  if ("integer" %in% obj$needed) grammar_parts <- c(grammar_parts, integer_rule)
  if ("boolean" %in% obj$needed) grammar_parts <- c(grammar_parts, boolean_rule)

  paste(grammar_parts, collapse = "")
}

#' Extract structured data from text
#'
#' High-level function that combines prompt construction with grammar-constrained
#' generation to extract structured data from text. Returns a parsed R list.
#'
#' @param ctx Model context from edge_load_model()
#' @param text The input text to analyze
#' @param schema A named list defining the extraction schema (see \code{\link{edge_json_grammar}})
#' @param instruction Optional instruction to guide extraction (default: auto-generated)
#' @param n_predict Maximum tokens to generate (default: 512)
#' @param temperature Sampling temperature (default: 0.2, very low for factual extraction)
#' @param draft If TRUE, first let the model answer in free text, then run the
#'   grammar-constrained pass with that draft in the prompt (default: FALSE).
#'   It adds a second generation pass, so it takes two to four times as long.
#' @return A named list with the extracted fields, or the raw JSON string if parsing fails.
#'   Requires the \pkg{jsonlite} package for automatic parsing.
#'
#' @details
#' The grammar guarantees well-formed JSON, which is where most of its accuracy
#' gain comes from. Forcing a very small model straight into a strict format
#' can, however, cost answer accuracy. With \code{draft = TRUE} the model first
#' reasons about the fields without constraints and the grammar is applied only
#' to the final answer. Published results for this "draft, then constrain"
#' pattern show gains for 4B-14B models (e.g. micro-F1 0.644 to 0.795 for
#' Qwen3-14B on GenSIE 2026 zero-shot extraction). It is most worth trying with
#' models under about 4B parameters, or when plain extraction leaves fields wrong.
#'
#' @seealso \code{\link{edge_json_grammar}}, \code{\link{edge_grammar_completion}},
#'   \code{\link{edge_classify}}, \code{\link{edge_extract_batch}}
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#'
#' result <- edge_extract(ctx,
#'   "I absolutely love this product! Best purchase ever.",
#'   schema = list(
#'     sentiment = c("positive", "negative", "neutral"),
#'     confidence = "number"
#'   ))
#' # result$sentiment == "positive"
#' # result$confidence == 0.95
#'
#' edge_free_model(ctx)
#' }
#' @export
edge_extract <- function(ctx, text, schema, instruction = NULL, n_predict = 512L,
                          temperature = 0.2, draft = FALSE) {
  if (!is.character(text) || length(text) != 1L) {
    stop("text must be a single character string")
  }

  grammar <- edge_json_grammar(schema)
  instruction <- .extract_instruction(schema, instruction)

  notes <- ""
  if (isTRUE(draft)) {
    draft_prompt <- paste0(
      "### Instruction\n", instruction,
      " Think through the text and write each field on its own line as ",
      "\"field: value\".",
      "\n\n### Text\n", text,
      "\n\n### Notes\n"
    )
    draft_text <- edge_completion(ctx, draft_prompt,
                                  n_predict = n_predict,
                                  temperature = temperature)
    notes <- paste0("\n\n### Notes\n", trimws(draft_text))
  }

  prompt <- .extract_prompt(instruction, text, notes)
  raw_output <- edge_grammar_completion(ctx, prompt, grammar,
                                         n_predict = n_predict,
                                         temperature = temperature)
  .parse_extracted(raw_output)
}

.extract_instruction <- function(schema, instruction) {
  if (!is.null(instruction)) return(instruction)
  field_descriptions <- vapply(seq_along(schema), function(i) {
    nm <- names(schema)[i]
    spec <- schema[[i]]
    if (.is_enum(spec)) {
      paste0(nm, " (one of: ", paste(spec, collapse = ", "), ")")
    } else {
      paste0(nm, " (", spec, ")")
    }
  }, character(1))
  paste0(
    "Extract the following fields from the text: ",
    paste(field_descriptions, collapse = ", "), "."
  )
}

.extract_prompt <- function(instruction, text, notes = "") {
  paste0(
    "### Instruction\n", instruction,
    "\n\n### Text\n", text,
    notes,
    "\n\n### Response\n"
  )
}

# The grammar's JSON as an R list, or the raw text if it cannot be parsed.
.parse_extracted <- function(raw_output) {
  # Try to parse JSON
  tryCatch({
    # Find JSON object in output
    json_start <- regexpr("\\{", raw_output)
    if (json_start > 0) {
      json_str <- substring(raw_output, json_start)
      # Find matching closing brace
      depth <- 0
      for (i in seq_len(nchar(json_str))) {
        ch <- substr(json_str, i, i)
        if (ch == "{") depth <- depth + 1
        else if (ch == "}") {
          depth <- depth - 1
          if (depth == 0) {
            json_str <- substr(json_str, 1, i)
            break
          }
        }
      }

      if (requireNamespace("jsonlite", quietly = TRUE)) {
        return(jsonlite::fromJSON(json_str))
      } else {
        # Basic parsing fallback without jsonlite
        return(json_str)
      }
    }
    raw_output
  }, error = function(e) {
    warning("JSON parsing failed, returning raw output: ", e$message)
    raw_output
  })
}

#' Classify text into predefined categories
#'
#' Convenience function that uses grammar-constrained generation to classify
#' text into one of the provided categories.
#'
#' @param ctx Model context from edge_load_model()
#' @param text Text to classify (character string or vector for batch)
#' @param categories Character vector of allowed categories
#' @param instruction Optional classification instruction
#' @param temperature Sampling temperature (default: 0.1)
#' @return Character string (single text) or character vector (batch) with the predicted category
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#'
#' # Single classification
#' result <- edge_classify(ctx,
#'   "I love this product!",
#'   categories = c("positive", "negative", "neutral"))
#'
#' # Batch classification
#' texts <- c("Great product!", "Terrible experience", "It was okay")
#' results <- edge_classify(ctx, texts,
#'   categories = c("positive", "negative", "neutral"))
#'
#' edge_free_model(ctx)
#' }
#' @export
edge_classify <- function(ctx, text, categories, instruction = NULL,
                           temperature = 0.1) {
  if (!is.character(categories) || length(categories) < 2L) {
    stop("categories must be a character vector with at least 2 options")
  }

  # Build grammar that only allows one of the categories
  enum_options <- paste0('"', categories, '"', collapse = " | ")
  grammar <- paste0('root ::= ', enum_options, '\n')

  if (is.null(instruction)) {
    instruction <- paste0(
      "Classify the following text into exactly one of these categories: ",
      paste(categories, collapse = ", "), ".\n",
      "Respond with only the category name, nothing else."
    )
  }

  # Handle batch
  classify_one <- function(single_text) {
    prompt <- paste0(
      "### Instruction\n", instruction,
      "\n\n### Text\n", single_text,
      "\n\n### Category\n"
    )

    # Try grammar-constrained first, fall back to free generation + matching
    result <- tryCatch({
      out <- edge_grammar_completion(ctx, prompt, grammar,
                                      n_predict = 50L,
                                      temperature = temperature)
      trimws(out)
    }, error = function(e) {
      # Grammar sampler can fail on some models/tokenizers; fall back
      raw <- edge_completion_internal(ctx, prompt, 50L,
                                      .sampling_options(temperature, 0.95))
      if (startsWith(raw, prompt)) raw <- substring(raw, nchar(prompt) + 1L)
      raw <- trimws(raw)
      # Match to closest category
      raw_lower <- tolower(raw)
      for (cat in categories) {
        if (grepl(tolower(cat), raw_lower, fixed = TRUE)) return(cat)
      }
      # If no match, return first word and hope it matches
      first_word <- sub("\\s.*", "", raw)
      for (cat in categories) {
        if (startsWith(tolower(cat), tolower(first_word))) return(cat)
      }
      categories[1]  # last resort
    })
    result
  }

  if (length(text) == 1L) {
    classify_one(text)
  } else {
    vapply(text, classify_one, character(1), USE.NAMES = FALSE)
  }
}


# ============================================================================
# Embeddings API
# ============================================================================

#' Extract text embeddings from a model
#'
#' Computes dense vector embeddings for one or more text inputs using the
#' loaded model. These embeddings can be used for semantic search, clustering,
#' similarity comparison, and as input to downstream models.
#'
#' @param ctx Model context from edge_load_model()
#' @param texts Character vector of texts to embed
#' @param normalize Whether to L2-normalize the embeddings (default: TRUE).
#'   Normalized embeddings allow cosine similarity to be computed as a simple dot product.
#' @return A numeric matrix with dimensions (n_texts x n_embd), where each row
#'   is the embedding vector for the corresponding input text.
#'
#' @details
#' For best results, use a model designed for embeddings (e.g., nomic-embed-text).
#' However, generative models can also produce useful embeddings from their
#' hidden states.
#'
#' The embedding dimension depends on the model architecture (e.g., 768 for
#' small models, 4096+ for larger ones). Use \code{edge_model_n_embd()} to
#' query the dimension.
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("nomic-embed-text.gguf")
#'
#' # Single embedding
#' emb <- edge_embeddings(ctx, "The cat sat on the mat")
#' dim(emb)  # 1 x n_embd
#'
#' # Multiple embeddings
#' texts <- c("cats are great", "dogs are loyal", "the stock market crashed")
#' embs <- edge_embeddings(ctx, texts)
#' dim(embs)  # 3 x n_embd
#'
#' # Compute similarity
#' edge_similarity(embs[1,], embs[2,])  # high (both about pets)
#' edge_similarity(embs[1,], embs[3,])  # low (different topics)
#'
#' edge_free_model(ctx)
#' }
#' @export
edge_embeddings <- function(ctx, texts, normalize = TRUE) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(texts) || length(texts) == 0L) {
    stop("texts must be a non-empty character vector")
  }
  if (any(is.na(texts))) {
    stop("texts must not contain NA values")
  }

  result <- edge_embeddings_internal(ctx, texts, as.logical(normalize))

  # Check if we got all zeros (indicates model wasn't loaded with embeddings=TRUE)
  if (all(result == 0)) {
    warning("All embeddings are zero. The model must be loaded with embeddings=TRUE:\n",
            "  ctx <- edge_load_model('model.gguf', embeddings = TRUE)\n",
            "Returning zero matrix.")
  }

  result
}

#' Compute cosine similarity between two embedding vectors
#'
#' @param a Numeric vector (embedding)
#' @param b Numeric vector (embedding)
#' @return Cosine similarity score between -1 and 1
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#' embs <- edge_embeddings(ctx, c("happy cat", "joyful kitten", "stock market"))
#'
#' edge_similarity(embs[1,], embs[2,])  # ~0.9 (similar meaning)
#' edge_similarity(embs[1,], embs[3,])  # ~0.1 (different topics)
#' }
#' @export
edge_similarity <- function(a, b) {
  if (!is.numeric(a) || !is.numeric(b)) {
    stop("Both a and b must be numeric vectors")
  }
  if (length(a) != length(b)) {
    stop("Vectors must have the same length")
  }
  dot <- sum(a * b)
  norm_a <- sqrt(sum(a * a))
  norm_b <- sqrt(sum(b * b))
  if (norm_a == 0 || norm_b == 0) return(0)
  dot / (norm_a * norm_b)
}

#' Get the embedding dimension of a loaded model
#'
#' @param ctx Model context from edge_load_model()
#' @return Integer giving the embedding vector dimension
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#' edge_model_n_embd(ctx)  # e.g., 4096
#' }
#' @export
edge_model_n_embd <- function(ctx) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  edge_model_n_embd_internal(ctx)
}

#' Tokenize text with a model's tokenizer
#'
#' Splits text into the model's tokens, the same way the completion
#' functions read a prompt, so \code{length(edge_tokenize(ctx, prompt))} is
#' the number of tokens the prompt takes in the context.
#'
#' @param ctx Model context from \code{\link{edge_load_model}}
#' @param text A single character string
#' @param add_special Add the model's start-of-text token, as the completion
#'   functions do (default: TRUE)
#' @return An integer vector of token ids
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#' length(edge_tokenize(ctx, "The quick brown fox"))
#' edge_free_model(ctx)
#' }
#' @export
edge_tokenize <- function(ctx, text, add_special = TRUE) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(text) || length(text) != 1L || is.na(text)) {
    stop("text must be a single character string")
  }
  if (!is.logical(add_special) || length(add_special) != 1L || is.na(add_special)) {
    stop("add_special must be TRUE or FALSE")
  }
  edge_tokenize_internal(ctx, text, add_special)
}

#' Compute a similarity matrix for a set of embeddings
#'
#' @param embeddings A numeric matrix where each row is an embedding vector
#'   (as returned by \code{\link{edge_embeddings}})
#' @return A symmetric numeric matrix of pairwise cosine similarities
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#' embs <- edge_embeddings(ctx, c("cat", "kitten", "car", "automobile"))
#' sim_mat <- edge_similarity_matrix(embs)
#' # sim_mat[1,2] high (cat~kitten), sim_mat[1,3] low (cat~car)
#' }
#' @export
edge_similarity_matrix <- function(embeddings) {
  if (!is.matrix(embeddings) || !is.numeric(embeddings)) {
    stop("embeddings must be a numeric matrix")
  }
  n <- nrow(embeddings)
  # Normalize rows
  norms <- sqrt(rowSums(embeddings^2))
  norms[norms == 0] <- 1  # avoid division by zero
  normed <- embeddings / norms
  # Cosine similarity = dot product of normalized vectors
  tcrossprod(normed)
}


# ============================================================================
# Batch Processing / Vectorized API
# ============================================================================

#' Apply a prompt template to a vector of texts
#'
#' Maps a prompt template over a character vector, generating completions for
#' each element. This is the primary function for batch LLM operations on data
#' frames. A progress indicator is printed to the console.
#'
#' @param ctx Model context from edge_load_model()
#' @param texts Character vector of input texts
#' @param prompt_template A string with \code{\{text\}} as a placeholder for each input,
#'   or a function that takes a single text and returns a prompt string.
#' @param n_predict Maximum tokens to generate per text (default: 128)
#' @param temperature Sampling temperature (default: 0.7)
#' @param top_p Nucleus sampling threshold (default: 0.95)
#' @param grammar Optional GBNF grammar string to constrain output
#' @param progress Show progress messages (default: TRUE)
#' @inheritParams edge_completion
#' @return Character vector of completions, same length as \code{texts}
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#'
#' # Summarize a column
#' df <- data.frame(review = c("Great product!", "Terrible, broke on day 1"))
#' df$summary <- edge_map(ctx, df$review,
#'   "Summarize in 5 words: {text}")
#'
#' # With a custom prompt function
#' df$analysis <- edge_map(ctx, df$review, function(text) {
#'   paste0("Product review: ", text, "\nOne-word sentiment:")
#' })
#'
#' edge_free_model(ctx)
#' }
#' @seealso \code{\link{edge_classify}}, \code{\link{edge_extract_batch}}
#' @export
edge_map <- function(ctx, texts, prompt_template, n_predict = 128L,
                      temperature = 0.7, top_p = 0.95, grammar = NULL,
                      progress = TRUE, top_k = 0L, min_p = 0,
                      repeat_penalty = 1, seed = NULL, stop = NULL) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(texts) || length(texts) == 0L) {
    stop("texts must be a non-empty character vector")
  }

  n <- length(texts)
  results <- character(n)

  build_prompt <- if (is.function(prompt_template)) {
    prompt_template
  } else if (is.character(prompt_template) && length(prompt_template) == 1L) {
    function(text) gsub("{text}", text, prompt_template, fixed = TRUE)
  } else {
    stop("prompt_template must be a string with {text} placeholder or a function")
  }

  use_grammar <- !is.null(grammar) && nchar(grammar) > 0L
  # Checked once up front; with seed = NULL each text still gets its own seed.
  .sampling_options(temperature, top_p, top_k, min_p, repeat_penalty, seed, stop)

  if (n > 1L && edge_n_parallel_internal(ctx) > 1L) {
    if (progress) message(sprintf("Processing %d texts, up to %d at a time...",
                                  n, edge_n_parallel_internal(ctx)))
    prompts <- vapply(texts, build_prompt, character(1), USE.NAMES = FALSE)
    results <- .batch_completion(ctx, prompts, n_predict, temperature, top_p,
                                 top_k, min_p, repeat_penalty, seed, stop,
                                 grammar = if (use_grammar) grammar else "")
    if (progress) message(sprintf("Done. Processed %d texts.", n))
    return(results)
  }

  for (i in seq_len(n)) {
    if (progress && n > 1L) {
      message(sprintf("[%d/%d] Processing...", i, n))
      flush.console()
    }

    prompt <- build_prompt(texts[i])

    if (use_grammar) {
      results[i] <- edge_grammar_completion(ctx, prompt, grammar,
                                             n_predict = n_predict,
                                             temperature = temperature,
                                             top_p = top_p, top_k = top_k,
                                             min_p = min_p,
                                             repeat_penalty = repeat_penalty,
                                             seed = seed, stop = stop)
    } else {
      sampling <- .sampling_options(temperature, top_p, top_k, min_p,
                                    repeat_penalty, seed, stop)
      raw <- edge_completion_internal(ctx, prompt, as.integer(n_predict),
                                      sampling)
      # Strip the prompt prefix from the result
      if (startsWith(raw, prompt)) {
        results[i] <- substring(raw, nchar(prompt) + 1L)
      } else {
        results[i] <- raw
      }
    }
  }

  if (progress && n > 1L) message(sprintf("Done. Processed %d texts.", n))
  results
}

# Runs prompts through edge_completion_batch_internal(), n_parallel at a
# time. Each prompt gets its own seed, drawn as one call each would draw it,
# or the given seed for all of them.
.batch_completion <- function(ctx, prompts, n_predict, temperature, top_p,
                              top_k = 0L, min_p = 0, repeat_penalty = 1,
                              seed = NULL, stop = NULL, grammar = "") {
  n_predict <- max(1L, min(as.integer(n_predict), 4096L))
  temperature <- max(0.0, min(temperature, 2.0))
  top_p <- max(0.1, min(top_p, 1.0))
  sampling <- .sampling_options(temperature, top_p, top_k, min_p,
                                repeat_penalty, seed, stop)
  seeds <- if (is.null(seed)) {
    vapply(prompts, function(p) .sampling_options(temperature, top_p)$seed,
           numeric(1), USE.NAMES = FALSE)
  } else {
    rep(sampling$seed, length(prompts))
  }
  edge_completion_batch_internal(ctx, prompts, n_predict, sampling, seeds,
                                 grammar, "root")
}

#' Extract structured data from multiple texts
#'
#' Batch version of \code{\link{edge_extract}} that processes a vector of texts
#' and returns a data frame.
#'
#' @param ctx Model context from edge_load_model()
#' @param texts Character vector of texts to process
#' @param schema A named list defining the extraction schema (see \code{\link{edge_json_grammar}})
#' @param instruction Optional instruction to guide extraction
#' @param n_predict Maximum tokens to generate per text (default: 512)
#' @param temperature Sampling temperature (default: 0.2)
#' @param progress Show progress messages (default: TRUE)
#' @param draft Passed to \code{\link{edge_extract}}: draft in free text first,
#'   then apply the grammar (default: FALSE)
#' @return A data frame with one row per text and columns for each schema field
#'
#' @seealso \code{\link{edge_extract}}, \code{\link{edge_map}}
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#'
#' reviews <- c("Love it!", "Worst purchase ever", "It's okay I guess")
#' results <- edge_extract_batch(ctx, reviews,
#'   schema = list(
#'     sentiment = c("positive", "negative", "neutral"),
#'     confidence = "number"
#'   ))
#' # results is a data frame:
#' #   sentiment confidence
#' # 1  positive       0.95
#' # 2  negative       0.90
#' # 3   neutral       0.60
#'
#' edge_free_model(ctx)
#' }
#' @export
edge_extract_batch <- function(ctx, texts, schema, instruction = NULL,
                                n_predict = 512L, temperature = 0.2,
                                progress = TRUE, draft = FALSE) {
  if (!is.character(texts) || length(texts) == 0L) {
    stop("texts must be a non-empty character vector")
  }

  n <- length(texts)
  results_list <- vector("list", n)
  batched <- !isTRUE(draft) && n > 1L && edge_n_parallel_internal(ctx) > 1L
  if (batched) {
    if (progress) message(sprintf("Extracting from %d texts, up to %d at a time...",
                                  n, edge_n_parallel_internal(ctx)))
    instruction <- .extract_instruction(schema, instruction)
    prompts <- vapply(texts, function(t) .extract_prompt(instruction, t), character(1),
                      USE.NAMES = FALSE)
    raw <- .batch_completion(ctx, prompts, n_predict, temperature, 0.95,
                             grammar = edge_json_grammar(schema))
    results_list <- lapply(raw, function(r) suppressWarnings(.parse_extracted(r)))
  }

  for (i in seq_len(n)) {
    if (batched) {
      if (is.character(results_list[[i]])) {
        results_list[[i]] <- stats::setNames(as.list(rep(NA, length(schema))), names(schema))
      }
      next
    }
    if (progress && n > 1L) {
      message(sprintf("[%d/%d] Extracting...", i, n))
      flush.console()
    }

    result <- tryCatch(
      edge_extract(ctx, texts[i], schema,
                    instruction = instruction,
                    n_predict = n_predict,
                    temperature = temperature,
                    draft = draft),
      error = function(e) {
        warning("Extraction failed for text ", i, ": ", e$message)
        # Return NA-filled row
        stats::setNames(as.list(rep(NA, length(schema))), names(schema))
      }
    )

    if (is.character(result) && length(result) == 1L) {
      # JSON parsing failed, return NA row
      results_list[[i]] <- stats::setNames(as.list(rep(NA, length(schema))), names(schema))
    } else {
      results_list[[i]] <- result
    }
  }

  if (progress && n > 1L) message(sprintf("Done. Extracted from %d texts.", n))

  # Combine into data frame
  tryCatch(
    do.call(rbind.data.frame, c(results_list, stringsAsFactors = FALSE)),
    error = function(e) {
      # Fallback: build data frame manually
      df <- data.frame(matrix(NA, nrow = n, ncol = length(schema)))
      names(df) <- names(schema)
      for (i in seq_len(n)) {
        row <- results_list[[i]]
        if (is.list(row)) {
          for (nm in names(schema)) {
            if (nm %in% names(row)) df[i, nm] <- row[[nm]]
          }
        }
      }
      df
    }
  )
}


# ============================================================================
# Text-to-SQL, narration, and numeric back-verification
# ============================================================================

#' Generate a SQL query from a natural language question
#'
#' Convenience wrapper that constructs a text-to-SQL prompt with the supplied
#' schema, queries the model, and post-processes the output into a single
#' executable SQL statement. Optionally executes the query in a DBI connection
#' and returns both the SQL and its result.
#'
#' @param ctx Model context from edge_load_model()
#' @param question Natural language question (single character string)
#' @param schema One or more CREATE TABLE statements describing the relevant
#'   tables. Pass a single string or a character vector: vectors are
#'   concatenated with newlines. Pre-filter the schema to only the tables that
#'   are likely relevant to the question; large schemas degrade accuracy.
#' @param dialect SQL dialect label inserted into the prompt (default "sqlite").
#'   Examples: "sqlite", "postgresql", "snowflake".
#' @param examples Optional named list of question -> SQL pairs used as
#'   few-shot examples (e.g. \code{list("How many subjects?" = "SELECT COUNT(*) FROM ADSL;")}).
#' @param con Optional DBI connection. When non-NULL the generated SQL is
#'   executed in a read-only sense using \code{DBI::dbGetQuery}; failures are
#'   captured rather than re-thrown.
#' @param n_predict Maximum tokens to generate (default: 256)
#' @param temperature Sampling temperature (default: 0.1, low for SQL)
#' @return If \code{con} is NULL, a single character string containing the SQL.
#'   If \code{con} is supplied, a list with elements \code{sql}, \code{result},
#'   and \code{error} (NA on success).
#'
#' @details
#' This is a drafting helper, not a sandbox. Always review generated SQL before
#' executing it against any system of record. Pair this function with a
#' read-only connection and dedicated validation logic for production use.
#'
#' Output quality depends heavily on the model. Small general purpose models
#' often produce syntactically valid SQL that answers a different question than
#' the one asked. Use a model fine tuned for SQL, or a general model of at least
#' 7B parameters, and review every statement before running it.
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("sqlcoder-7b-2.Q4_K_M.gguf", n_ctx = 4096L)
#'
#' schema <- "
#' CREATE TABLE ADSL (USUBJID TEXT, AGE INTEGER, SEX TEXT, TRT01A TEXT, SAFFL TEXT);
#' CREATE TABLE ADAE (USUBJID TEXT, AETERM TEXT, AESER TEXT, AEREL TEXT);
#' "
#'
#' sql <- edge_text_to_sql(ctx,
#'   "How many subjects on active arm had at least one serious related AE?",
#'   schema = schema, dialect = "sqlite")
#' cat(sql)
#'
#' # With execution
#' con <- DBI::dbConnect(RSQLite::SQLite(), "study.db")
#' out <- edge_text_to_sql(ctx, "Count safety analysis subjects per arm",
#'                          schema = schema, con = con)
#' out$sql
#' out$result
#' DBI::dbDisconnect(con)
#' }
#' @export
edge_text_to_sql <- function(ctx, question, schema, dialect = "sqlite",
                              examples = NULL, con = NULL,
                              n_predict = 256L, temperature = 0.1) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(question) || length(question) != 1L || nchar(question) == 0L) {
    stop("question must be a non-empty single character string")
  }
  if (!is.character(schema) || length(schema) == 0L) {
    stop("schema must be a character vector of one or more CREATE TABLE statements")
  }
  if (!is.character(dialect) || length(dialect) != 1L) {
    stop("dialect must be a single character string")
  }

  schema_block <- paste(schema, collapse = "\n")

  example_block <- ""
  if (!is.null(examples)) {
    if (!is.list(examples) || is.null(names(examples))) {
      stop("examples must be a named list of question -> SQL pairs")
    }
    parts <- character()
    for (i in seq_along(examples)) {
      parts <- c(parts,
        paste0("### Question\n", names(examples)[i],
               "\n\n### SQL\n", examples[[i]]))
    }
    example_block <- paste0(paste(parts, collapse = "\n\n"), "\n\n")
  }

  prompt <- paste0(
    "### Task\n",
    "Generate a ", dialect, " SQL query that answers the question below. ",
    "Use only the tables and columns defined in the schema. ",
    "Return a single statement ending with a semicolon.\n\n",
    "### Database Schema\n", schema_block, "\n\n",
    example_block,
    "### Question\n", question, "\n\n",
    "### SQL\n"
  )

  raw <- edge_completion(ctx, prompt,
                          n_predict = n_predict,
                          temperature = temperature,
                          top_p = 0.95)
  sql <- .clean_sql_output(raw)

  if (is.null(con)) {
    return(sql)
  }

  if (!requireNamespace("DBI", quietly = TRUE)) {
    stop("Package 'DBI' is required when 'con' is supplied. Install with install.packages('DBI').")
  }

  result <- NULL
  err <- NA_character_
  tryCatch({
    result <- DBI::dbGetQuery(con, sql)
  }, error = function(e) {
    err <<- conditionMessage(e)
  })

  list(sql = sql, result = result, error = err)
}

#' Clean raw model output into an executable SQL statement
#' @keywords internal
.clean_sql_output <- function(raw) {
  if (!is.character(raw) || length(raw) != 1L) return(NA_character_)
  s <- raw
  # Strip common ```sql ... ``` fences (any case: models also write ```SQL)
  s <- sub("(?si)^.*?```(?:sql)?\\s*", "", s, perl = TRUE)
  s <- sub("(?s)```.*$", "", s, perl = TRUE)
  # Strip trailing model chatter after the first semicolon that ends the
  # statement, ignoring semicolons inside quoted literals
  semi <- .sql_terminator(s)
  if (semi > 0) {
    s <- substr(s, 1L, semi)
  } else {
    # No semicolon: cut at any "### " section header the model echoed
    s <- sub("\\s*###.*$", "", s)
    s <- sub("\\s*Explanation.*$", "", s, ignore.case = TRUE)
    s <- sub("\\s*Note:.*$", "", s, ignore.case = TRUE)
  }
  # Drop leading "SQL:" labels the model sometimes emits
  s <- sub("^\\s*(SQL|Query)\\s*:\\s*", "", s, ignore.case = TRUE)
  trimws(s)
}

# Position of the first semicolon outside quoted SQL literals, or 0.
.sql_terminator <- function(s) {
  chars <- strsplit(s, "", fixed = TRUE)[[1]]
  quote <- ""
  for (i in seq_along(chars)) {
    ch <- chars[[i]]
    if (nzchar(quote)) {
      # A doubled quote ('it''s') closes and reopens, which works out the same
      if (ch == quote) quote <- ""
    } else if (ch == "'" || ch == "\"") {
      quote <- ch
    } else if (ch == ";") {
      return(i)
    }
  }
  0L
}

#' Generate an English narrative from structured data
#'
#' Turns a named list, single-row data.frame, or multi-row data.frame into one
#' or more short English sentences describing the data. The function linearises
#' the input as labelled "field: value" lines and asks the model to narrate
#' them without inferring anything that is not in the input.
#'
#' @param ctx Model context from edge_load_model()
#' @param data A named list, a data.frame, or a character vector of pre-formatted
#'   "field: value" blocks. Multi-row data.frames are processed row-by-row and
#'   the function returns a character vector of the same length.
#' @param instruction Optional instruction overriding the default
#'   ("Write a single short paragraph describing this record. Use only the data
#'   provided. Do not infer causality or add information.")
#' @param max_words Soft limit communicated in the prompt (default: 60). Not
#'   enforced; the model is asked to stay under this.
#' @param n_predict Maximum tokens to generate per record (default: 200)
#' @param temperature Sampling temperature (default: 0.3)
#' @param progress Show progress messages for multi-row inputs (default: TRUE)
#' @return Character vector of narratives, one per input record.
#'
#' @details
#' This is a drafting helper. For regulated outputs (CSR, ICSR, PV) always run
#' \code{\link{edge_verify_narrative}} or an equivalent back-check before
#' surfacing the text to a human reviewer, and pass it the study's drug and
#' adverse-event terms. Unchecked output is often wrong: in a published study
#' of small models drafting case narratives from ICSRs, 60% of outputs had
#' wrong drug information and 71% wrong or missing adverse events (Hakim et
#' al., 2024, arXiv:2407.18322).
#'
#' Output quality depends heavily on the model. Small models tend to restate the
#' input fields rather than summarise them. A model of at least 7B parameters
#' gives noticeably better narratives.
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("Qwen3-1.7B-Q4_K_M.gguf", n_ctx = 2048L)
#'
#' lab <- list(
#'   USUBJID = "ABC-001-042", LBTESTCD = "ALT",
#'   LBORRES = 285, LBORNRHI = 55,
#'   LBDY = 28, VISIT = "Week 4", LBNRIND = "HIGH"
#' )
#' edge_narrate(ctx, lab)
#'
#' # Multi-row data.frame
#' df <- data.frame(
#'   subject = c("S001", "S002"),
#'   alt = c(285, 47),
#'   uln = c(55, 55),
#'   flag = c("HIGH", "NORMAL")
#' )
#' edge_narrate(ctx, df)
#' }
#' @export
edge_narrate <- function(ctx, data, instruction = NULL, max_words = 60L,
                          n_predict = 200L, temperature = 0.3,
                          progress = TRUE) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (is.null(instruction)) {
    instruction <- paste0(
      "Write a single short paragraph describing this record. ",
      "Use only the data provided. Do not infer causality or add information. ",
      "Stay under ", as.integer(max_words), " words."
    )
  }

  records <- .linearise_records(data)

  n <- length(records)
  out <- character(n)
  for (i in seq_len(n)) {
    if (progress && n > 1L) {
      message(sprintf("[%d/%d] Narrating...", i, n))
      flush.console()
    }
    prompt <- paste0(
      "### Instruction\n", instruction,
      "\n\n### Data\n", records[i],
      "\n\n### Narrative\n"
    )
    raw <- edge_completion(ctx, prompt,
                            n_predict = n_predict,
                            temperature = temperature,
                            top_p = 0.95)
    out[i] <- trimws(raw)
  }
  out
}

#' Linearise a record into a "field: value" block
#' @keywords internal
.linearise_records <- function(data) {
  if (is.character(data)) {
    return(data)
  }
  if (is.data.frame(data)) {
    out <- character(nrow(data))
    for (i in seq_len(nrow(data))) {
      row <- as.list(data[i, , drop = FALSE])
      out[i] <- .linearise_one(row)
    }
    return(out)
  }
  if (is.list(data) && !is.null(names(data))) {
    return(.linearise_one(data))
  }
  stop("data must be a named list, a data.frame, or a character vector")
}

.linearise_one <- function(rec) {
  nm <- names(rec)
  vals <- vapply(rec, function(v) {
    if (is.null(v) || length(v) == 0L) return("(missing)")
    if (is.na(v[1])) return("(missing)")
    as.character(v[1])
  }, character(1))
  paste(sprintf("- %s: %s", nm, vals), collapse = "\n")
}

#' Verify a generated narrative against source values
#'
#' Checks a generated narrative against the source record in two independent
#' ways. The \emph{literal} check needs no model: every numeric source value
#' must appear in the narrative, and, when \code{terms} is given, every term
#' from that dictionary (e.g. drug names, adverse-event terms) must occur in the
#' narrative if and only if it occurs in the source. The \emph{model} check
#' re-extracts the named fields from the narrative with \code{\link{edge_extract}}
#' and compares them to the source record.
#'
#' @param ctx Model context from edge_load_model(). May be NULL when
#'   \code{method = "literal"}.
#' @param narrative Character string produced by \code{\link{edge_narrate}} or
#'   any other generator.
#' @param expected Named list of source field values. The names define which
#'   fields to re-extract from the narrative. Values define the source-of-truth.
#' @param schema Optional named list overriding the schema passed to
#'   \code{edge_extract}. If NULL, types are inferred from \code{expected}
#'   (numeric -> "number", integer -> "integer", logical -> "boolean",
#'   character -> "string").
#' @param tolerance Numeric tolerance for number fields (default: 1e-6).
#'   Values within \code{tolerance} are considered equal.
#' @param temperature Sampling temperature for re-extraction (default: 0.1)
#' @param method Which checks to run: \code{"both"} (default), \code{"literal"}
#'   (deterministic only, no model call) or \code{"model"} (re-extraction only).
#' @param terms Optional character vector of controlled terms to guard, such as
#'   the drug names and MedDRA terms relevant to the study. Matching is
#'   case-insensitive on whole words. Ignored when \code{method = "model"}.
#' @return A list with:
#'   \itemize{
#'     \item \code{ok}: TRUE if every check that was run passed.
#'     \item \code{extracted}: the parsed extraction (as returned by
#'           \code{edge_extract}), or NULL if the model check was not run.
#'     \item \code{expected}: the supplied expected values.
#'     \item \code{mismatches}: data.frame with one row per field that failed
#'           the model check.
#'     \item \code{literal}: data.frame (\code{field}, \code{expected},
#'           \code{found}) for each numeric source value.
#'     \item \code{unsupported_terms}: terms that appear in the narrative but
#'           not in the source, i.e. likely hallucinations.
#'     \item \code{omitted_terms}: terms that appear in the source but not in
#'           the narrative.
#'   }
#'
#' @details
#' The model check re-extracts the fields with the model, so it inherits the
#' model's limitations. A small model can miss a field that is present and
#' report a mismatch that is not real, or confirm a value that is wrong. The
#' literal check has no such failure mode, but it is strict: it does not
#' recognise numbers written as words, and it reports a term as missing when
#' the narrative uses a synonym or a misspelling. Treat any failure as a reason
#' for a human to look rather than as proof of an error.
#'
#' Unchecked generated narratives are often wrong. In a published study of
#' small models drafting case narratives from ICSRs, 60% of outputs contained
#' wrong drug information and 71% wrong or missing adverse events. A simple
#' drug and adverse-event name comparison between source and output, which is
#' what \code{terms} provides, flagged every hallucinated drug name in the
#' sample that was checked (Hakim et al., 2024, arXiv:2407.18322).
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("Qwen3-1.7B-Q4_K_M.gguf", n_ctx = 2048L)
#'
#' rec <- list(subject = "S001", drug = "Drug X", event = "Hepatotoxicity",
#'             alt = 285, uln = 55, study_day = 28)
#' narrative <- edge_narrate(ctx, rec)
#' check <- edge_verify_narrative(ctx, narrative, expected = rec,
#'                                terms = c("Drug X", "Drug Y",
#'                                          "Hepatotoxicity", "Rash"))
#' if (!check$ok) {
#'   print(check$mismatches)
#'   print(check$literal[!check$literal$found, ])
#'   print(check$unsupported_terms)
#' }
#'
#' # Deterministic check only, no model needed
#' edge_verify_narrative(NULL, narrative, rec, method = "literal")
#' }
#' @export
edge_verify_narrative <- function(ctx, narrative, expected, schema = NULL,
                                   tolerance = 1e-6, temperature = 0.1,
                                   method = c("both", "literal", "model"),
                                   terms = NULL) {
  method <- match.arg(method)
  if (method != "literal" && !is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.character(narrative) || length(narrative) != 1L) {
    stop("narrative must be a single character string")
  }
  if (!is.list(expected) || is.null(names(expected)) || length(expected) == 0L) {
    stop("expected must be a non-empty named list")
  }
  if (!is.null(terms) && (!is.character(terms) || anyNA(terms))) {
    stop("terms must be a character vector without NA")
  }

  extracted <- NULL
  mismatch_df <- data.frame(field = character(), expected = character(),
                            got = character(), stringsAsFactors = FALSE)
  if (method != "literal") {
    if (is.null(schema)) {
      schema <- lapply(expected, function(v) {
        if (is.logical(v)) "boolean"
        else if (is.integer(v)) "integer"
        else if (is.numeric(v)) "number"
        else "string"
      })
      names(schema) <- names(expected)
    }

    extracted <- edge_extract(ctx, narrative, schema,
                               instruction = paste0(
                                 "Re-extract the following fields from the narrative. ",
                                 "Return only values that appear explicitly in the text. ",
                                 "Do not guess or infer. If a field is not stated in the text, ",
                                 "return an empty string for text, 0 for numbers and false ",
                                 "for true/false fields."
                               ),
                               temperature = temperature)

    mismatches <- list()
    for (nm in names(expected)) {
      got <- if (is.list(extracted) && nm %in% names(extracted)) extracted[[nm]] else NA
      exp <- expected[[nm]]
      eq <- .field_equal(got, exp, tolerance)
      if (!isTRUE(eq)) {
        mismatches[[length(mismatches) + 1L]] <- data.frame(
          field = nm,
          expected = .as_display(exp),
          got = .as_display(got),
          stringsAsFactors = FALSE
        )
      }
    }
    if (length(mismatches) > 0L) mismatch_df <- do.call(rbind, mismatches)
  }

  literal_df <- data.frame(field = character(), expected = character(),
                           found = logical(), stringsAsFactors = FALSE)
  unsupported <- character()
  omitted <- character()
  if (method != "model") {
    literal_df <- .literal_number_check(narrative, expected, tolerance)
    if (length(terms) > 0L) {
      source_text <- paste(vapply(expected, .as_display, character(1)),
                           collapse = "\n")
      in_narrative <- vapply(terms, .term_present, logical(1), text = narrative)
      in_source <- vapply(terms, .term_present, logical(1), text = source_text)
      unsupported <- unname(terms[in_narrative & !in_source])
      omitted <- unname(terms[in_source & !in_narrative])
    }
  }

  list(
    ok = nrow(mismatch_df) == 0L && all(literal_df$found) &&
      length(unsupported) == 0L && length(omitted) == 0L,
    extracted = extracted,
    expected = expected,
    mismatches = mismatch_df,
    literal = literal_df,
    unsupported_terms = unsupported,
    omitted_terms = omitted
  )
}

# Every numeric source value must appear as a number in the narrative.
# Thousands separators are dropped first; digits that are part of an
# identifier such as "ABC-001-042" are not treated as numbers. A unit may
# follow directly, as in "2.5mg" or "285U/L".
.literal_number_check <- function(narrative, expected, tolerance) {
  is_num <- vapply(expected, function(v) {
    is.numeric(v) && length(v) >= 1L && !is.na(v[1])
  }, logical(1))
  fields <- names(expected)[is_num]
  text <- gsub("(?<=\\d),(?=\\d{3}(?!\\d))", "", narrative, perl = TRUE)
  tokens <- regmatches(text, gregexpr("(?<![\\w.-])-?\\d+(?:\\.\\d+)?(?!\\.?\\d)",
                                      text, perl = TRUE))[[1]]
  found_numbers <- suppressWarnings(as.numeric(tokens))
  found_numbers <- found_numbers[!is.na(found_numbers)]
  found <- vapply(fields, function(nm) {
    any(abs(found_numbers - expected[[nm]][1]) <= tolerance)
  }, logical(1))
  data.frame(
    field = fields,
    expected = vapply(expected[fields], .as_display, character(1)),
    found = unname(found),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

# Case-insensitive whole-word match of a literal term.
.term_present <- function(term, text) {
  pattern <- paste0("(?<![[:alnum:]])\\Q", term, "\\E(?![[:alnum:]])")
  grepl(pattern, text, ignore.case = TRUE, perl = TRUE)
}

.field_equal <- function(got, exp, tolerance) {
  if (length(got) == 0L || (length(got) == 1L && is.na(got))) return(FALSE)
  if (is.logical(exp)) {
    return(isTRUE(as.logical(got) == exp))
  }
  if (is.numeric(exp)) {
    g <- suppressWarnings(as.numeric(got))
    if (is.na(g)) return(FALSE)
    return(abs(g - exp) <= tolerance)
  }
  identical(tolower(trimws(as.character(got))),
            tolower(trimws(as.character(exp))))
}

.as_display <- function(v) {
  if (length(v) == 0L) return("(empty)")
  if (is.na(v[1])) return("(NA)")
  as.character(v[1])
}


# ============================================================================
# RAG (Retrieval-Augmented Generation) Pipeline
# ============================================================================

#' Build an embedding index from text documents
#'
#' Reads text files from a directory (or accepts text directly), computes
#' embeddings for chunks, and returns an index object for retrieval.
#'
#' @param source Either a directory path containing text files, or a character
#'   vector of text chunks to index directly.
#' @param ctx Model context from edge_load_model() (used for embedding)
#' @param chunk_size Approximate number of characters per chunk (default: 500).
#'   Ignored when \code{source} is a character vector.
#' @param chunk_overlap Number of characters of overlap between chunks (default: 50).
#'   Ignored when \code{source} is a character vector.
#' @param file_pattern Glob pattern for files to read (default: "*.txt").
#'   Also supports "*.md", "*.csv", etc.
#' @param normalize Normalize embeddings (default: TRUE)
#' @param progress Show progress messages (default: TRUE)
#' @return An \code{edge_index} object (a list) containing:
#'   \itemize{
#'     \item \code{chunks}: character vector of text chunks
#'     \item \code{embeddings}: numeric matrix (n_chunks x n_embd)
#'     \item \code{sources}: character vector of source file paths (or NA for direct text)
#'     \item \code{n_chunks}: number of chunks
#'     \item \code{n_embd}: embedding dimension
#'   }
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#'
#' # Index a directory of text files
#' index <- edge_index_documents("./reports/", ctx)
#'
#' # Or index text directly
#' texts <- c("Revenue grew 15% in Q3", "New product launched in Asia")
#' index <- edge_index_documents(texts, ctx)
#'
#' # Search the index
#' results <- edge_search(index, ctx, "What happened with revenue?")
#' }
#' @export
edge_index_documents <- function(source, ctx, chunk_size = 500L,
                                  chunk_overlap = 50L,
                                  file_pattern = "*.txt",
                                  normalize = TRUE,
                                  progress = TRUE) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!is.numeric(chunk_size) || length(chunk_size) != 1L || is.na(chunk_size) || chunk_size < 1) {
    stop("chunk_size must be a positive number")
  }
  if (!is.numeric(chunk_overlap) || length(chunk_overlap) != 1L || is.na(chunk_overlap) ||
        chunk_overlap < 0 || chunk_overlap >= chunk_size) {
    stop("chunk_overlap must be at least 0 and smaller than chunk_size")
  }
  chunk_size <- as.integer(chunk_size)
  chunk_overlap <- as.integer(chunk_overlap)

  chunks <- character()
  sources <- character()

  if (length(source) == 1L && dir.exists(source)) {
    # Read files from directory
    files <- list.files(source, pattern = utils::glob2rx(file_pattern),
                        full.names = TRUE, recursive = TRUE)
    if (length(files) == 0L) {
      stop("No files matching '", file_pattern, "' found in: ", source)
    }

    if (progress) message(sprintf("Reading %d files...", length(files)))

    for (filepath in files) {
      text <- tryCatch(
        paste(readLines(filepath, warn = FALSE), collapse = "\n"),
        error = function(e) {
          warning("Could not read: ", filepath)
          NULL
        }
      )
      if (is.null(text) || nchar(text) == 0L) next

      file_chunks <- .chunk_text(text, chunk_size, chunk_overlap)
      chunks <- c(chunks, file_chunks)
      sources <- c(sources, rep(filepath, length(file_chunks)))
    }
  } else if (is.character(source) && length(source) > 0L) {
    # Direct text input - optionally chunk long texts
    for (i in seq_along(source)) {
      if (nchar(source[i]) > chunk_size * 2L) {
        text_chunks <- .chunk_text(source[i], chunk_size, chunk_overlap)
        chunks <- c(chunks, text_chunks)
        sources <- c(sources, rep(NA_character_, length(text_chunks)))
      } else {
        chunks <- c(chunks, source[i])
        sources <- c(sources, NA_character_)
      }
    }
  } else {
    stop("source must be a directory path or a character vector of texts")
  }

  if (length(chunks) == 0L) {
    stop("No text content found to index")
  }

  if (progress) message(sprintf("Computing embeddings for %d chunks...", length(chunks)))

  # Compute embeddings in batches to avoid memory issues
  batch_size <- 32L
  n_chunks <- length(chunks)
  emb_list <- list()

  for (start in seq(1L, n_chunks, by = batch_size)) {
    end <- min(start + batch_size - 1L, n_chunks)
    batch_chunks <- chunks[start:end]

    if (progress && n_chunks > batch_size) {
      message(sprintf("  Embedding batch %d-%d of %d...", start, end, n_chunks))
    }

    emb_list[[length(emb_list) + 1L]] <- edge_embeddings(ctx, batch_chunks,
                                                          normalize = normalize)
  }

  embeddings <- do.call(rbind, emb_list)

  if (progress) message(sprintf("Index built: %d chunks, %d-dim embeddings",
                                 n_chunks, ncol(embeddings)))

  structure(
    list(
      chunks = chunks,
      embeddings = embeddings,
      sources = sources,
      n_chunks = n_chunks,
      n_embd = ncol(embeddings)
    ),
    class = "edge_index"
  )
}

#' Search an embedding index for relevant chunks
#'
#' Finds the most similar text chunks to a query using cosine similarity.
#'
#' @param index An \code{edge_index} object from \code{\link{edge_index_documents}}
#' @param ctx Model context (same model used to build the index)
#' @param query Query text string
#' @param top_k Number of results to return (default: 5)
#' @return A data frame with columns:
#'   \itemize{
#'     \item \code{chunk}: the text content
#'     \item \code{score}: cosine similarity to the query
#'     \item \code{source}: source file path (or NA)
#'     \item \code{index}: position in the original index
#'   }
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#' index <- edge_index_documents("./docs/", ctx)
#'
#' results <- edge_search(index, ctx, "quarterly revenue growth")
#' print(results)  # top 5 most relevant chunks
#' }
#' @export
edge_search <- function(index, ctx, query, top_k = 5L) {
  if (!inherits(index, "edge_index")) {
    stop("index must be an edge_index object from edge_index_documents()")
  }
  if (!is_valid_model(ctx)) {
    stop("Invalid model context")
  }
  if (!is.character(query) || length(query) != 1L) {
    stop("query must be a single character string")
  }

  top_k <- min(as.integer(top_k), index$n_chunks)

  # Embed the query
  query_emb <- edge_embeddings(ctx, query, normalize = TRUE)

  # Compute similarities: query_emb (1 x d) %*% t(index_emb) (d x n)
  scores <- as.numeric(query_emb %*% t(index$embeddings))

  # Get top-k indices
  top_indices <- order(scores, decreasing = TRUE)[seq_len(top_k)]

  data.frame(
    chunk = index$chunks[top_indices],
    score = scores[top_indices],
    source = index$sources[top_indices],
    index = top_indices,
    stringsAsFactors = FALSE
  )
}

#' Ask a question using retrieval-augmented generation
#'
#' Combines semantic search with text generation: retrieves relevant context
#' from the index, then generates an answer grounded in that context.
#'
#' @param ctx Model context from edge_load_model()
#' @param question The question to answer
#' @param index An \code{edge_index} object from \code{\link{edge_index_documents}}
#' @param top_k Number of context chunks to retrieve (default: 3)
#' @param n_predict Maximum tokens for the answer (default: 256)
#' @param temperature Sampling temperature (default: 0.3)
#' @param system_prompt Optional system-level instruction for the model
#' @param return_context If TRUE, return both answer and retrieved context (default: FALSE)
#' @return If \code{return_context = FALSE}: a character string with the answer.
#'   If \code{return_context = TRUE}: a list with \code{answer}, \code{context} (data frame
#'   from \code{edge_search}), and \code{prompt} (the full constructed prompt).
#'
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("model.gguf")
#' index <- edge_index_documents("./reports/", ctx)
#'
#' # Simple Q&A
#' answer <- edge_ask(ctx, "What were Q3 revenues?", index)
#' cat(answer)
#'
#' # With context for debugging
#' result <- edge_ask(ctx, "What were Q3 revenues?", index,
#'                     return_context = TRUE)
#' cat(result$answer)
#' print(result$context)  # shows which chunks were used
#' }
#' @export
edge_ask <- function(ctx, question, index, top_k = 3L, n_predict = 256L,
                      temperature = 0.3, system_prompt = NULL,
                      return_context = FALSE) {
  if (!is_valid_model(ctx)) {
    stop("Invalid model context. Load a model first with edge_load_model()")
  }
  if (!inherits(index, "edge_index")) {
    stop("index must be an edge_index object from edge_index_documents()")
  }
  if (!is.character(question) || length(question) != 1L) {
    stop("question must be a single character string")
  }

  # Retrieve relevant context
  context_df <- edge_search(index, ctx, question, top_k = top_k)

  # Build context string
  context_texts <- context_df$chunk
  context_block <- paste(
    vapply(seq_along(context_texts), function(i) {
      paste0("[", i, "] ", context_texts[i])
    }, character(1)),
    collapse = "\n\n"
  )

  # Build prompt
  if (is.null(system_prompt)) {
    system_prompt <- paste0(
      "Answer the question based only on the provided context. ",
      "If the context does not contain enough information, say so. ",
      "Be concise and factual."
    )
  }

  prompt <- paste0(
    "### System\n", system_prompt,
    "\n\n### Context\n", context_block,
    "\n\n### Question\n", question,
    "\n\n### Answer\n"
  )

  # Generate answer
  raw <- edge_completion_internal(ctx, prompt, as.integer(n_predict),
                                  .sampling_options(temperature, 0.95))

  # Strip prompt
  answer <- if (startsWith(raw, prompt)) {
    substring(raw, nchar(prompt) + 1L)
  } else {
    raw
  }
  answer <- trimws(answer)

  if (return_context) {
    list(answer = answer, context = context_df, prompt = prompt)
  } else {
    answer
  }
}

#' Print method for edge_index objects
#'
#' @param x An edge_index object
#' @param ... Additional arguments (ignored)
#' @return The object (invisibly)
#' @export
print.edge_index <- function(x, ...) {
  cat(sprintf("edgemodelr RAG Index\n"))
  cat(sprintf("  Chunks: %d\n", x$n_chunks))
  cat(sprintf("  Embedding dim: %d\n", x$n_embd))

  unique_sources <- unique(x$sources[!is.na(x$sources)])
  if (length(unique_sources) > 0L) {
    cat(sprintf("  Source files: %d\n", length(unique_sources)))
    show_n <- min(5L, length(unique_sources))
    for (s in unique_sources[seq_len(show_n)]) {
      cat(sprintf("    - %s\n", basename(s)))
    }
    if (length(unique_sources) > show_n) {
      cat(sprintf("    ... and %d more\n", length(unique_sources) - show_n))
    }
  } else {
    cat("  Source: direct text input\n")
  }

  invisible(x)
}


# Internal helper: split text into overlapping chunks
.chunk_text <- function(text, chunk_size = 500L, overlap = 50L) {
  if (nchar(text) <= chunk_size) return(text)

  chunks <- character()
  pos <- 1L
  text_len <- nchar(text)

  while (pos <= text_len) {
    end <- min(pos + chunk_size - 1L, text_len)

    # Try to break at sentence boundary
    if (end < text_len) {
      chunk <- substr(text, pos, end)
      # Look for last sentence-ending punctuation
      breaks <- gregexpr("[.!?]\\s", chunk)[[1]]
      if (breaks[1] > 0 && max(breaks) > chunk_size * 0.5) {
        end <- pos + max(breaks)
      }
    }

    chunks <- c(chunks, trimws(substr(text, pos, end)))
    if (end >= text_len) break
    # Always move forward, even when the overlap is as long as the chunk.
    pos <- max(end + 1L - overlap, pos + 1L)
  }

  chunks[nchar(chunks) > 0L]
}


# ============================================================================
# Plumber API Wrapper (Model-as-a-Service)
# ============================================================================

# An error that edge_serve() answers with status 400 instead of 500.
.bad_request <- function(...) {
  stop(structure(class = c("edge_bad_request", "error", "condition"),
                 list(message = paste0(...), call = NULL)))
}

# The request body as plain lists. plumber's own parsing turns arrays of
# objects into data.frames, which loses nested fields such as tool_calls.
.oa_body <- function(req) {
  raw <- req$postBody
  if (is.null(raw) || !nzchar(raw)) return(list())
  body <- tryCatch(jsonlite::fromJSON(raw, simplifyVector = FALSE),
                   error = function(e) .bad_request("The request body is not valid JSON"))
  if (!is.list(body)) .bad_request("The request body must be a JSON object")
  body
}

# Message content: a string, null, or a list of parts of which the text
# parts are used.
.oa_text <- function(content) {
  if (is.null(content)) return("")
  if (is.character(content)) return(paste(content, collapse = ""))
  if (is.list(content)) {
    parts <- vapply(content, function(p) {
      if (is.character(p)) {
        paste(p, collapse = "")
      } else if (is.list(p) && identical(p$type, "text") && is.character(p$text)) {
        p$text
      } else {
        ""
      }
    }, character(1))
    return(paste(parts, collapse = ""))
  }
  as.character(content)
}

# A call as the model writes it in edge_chat_completion(), so a call sent
# back in the conversation looks the same as one the model made.
.oa_call_text <- function(name, args) {
  vals <- vapply(args, function(v) {
    as.character(jsonlite::toJSON(v, auto_unbox = TRUE))
  }, character(1))
  inner <- if (length(args)) paste0('"', names(args), '": ', vals, collapse = ", ") else ""
  paste0('{"tool": "', name, '", "arguments": {', inner, "}}")
}

# OpenAI chat messages to edge_chat_completion() messages. Assistant tool
# calls become the call text, and tool results get the tool's name from the
# call they answer.
.oa_messages <- function(messages) {
  if (!is.list(messages) || length(messages) == 0L) .bad_request("messages is required")
  call_names <- character()
  lapply(messages, function(m) {
    if (!is.list(m) || !is.character(m$role)) {
      .bad_request("each message must be an object with a role")
    }
    role <- if (identical(m$role, "developer")) "system" else m$role
    if (identical(role, "assistant") && length(m$tool_calls) > 0L) {
      call <- m$tool_calls[[1]]
      fn <- call[["function"]]
      args <- if (is.character(fn$arguments)) {
        tryCatch(jsonlite::fromJSON(fn$arguments, simplifyVector = FALSE),
                 error = function(e) .bad_request("tool call arguments are not valid JSON"))
      } else {
        fn$arguments
      }
      if (!is.null(call$id)) call_names[[call$id]] <<- fn$name
      return(list(role = "assistant", content = .oa_call_text(fn$name, args)))
    }
    if (identical(role, "tool")) {
      id <- m$tool_call_id
      name <- m$name %||% (if (!is.null(id) && id %in% names(call_names)) call_names[[id]])
      return(list(role = "tool", name = name, content = .oa_text(m$content)))
    }
    list(role = role, content = .oa_text(m$content))
  })
}

# The properties of a flat JSON schema object as edge_tool() and
# edge_json_grammar() fields: strings, numbers, integers, booleans or enums
# of strings. With descriptions, a field is list(type, description).
.oa_schema_fields <- function(schema, what, descriptions = TRUE) {
  props <- schema$properties %||% list()
  fields <- lapply(names(props), function(arg) {
    p <- props[[arg]]
    type <- p$type
    if (is.list(type) || length(type) > 1L) {
      type <- setdiff(unlist(type), "null")[1]
    }
    spec <- if (!is.null(p$enum)) {
      values <- as.character(unlist(p$enum))
      if (length(values) == 1L) I(values) else values
    } else if (is.character(type) && type %in% c("string", "number", "integer", "boolean")) {
      type
    } else {
      .bad_request("field '", arg, "' of ", what, " has an unsupported type; ",
                   "use string, number, integer, boolean or an enum")
    }
    if (!descriptions || is.null(p$description)) spec else list(type = spec, description = p$description)
  })
  names(fields) <- names(props)
  fields
}

# OpenAI function tools to edge_tool() objects. Every argument is required.
.oa_tools <- function(tools) {
  if (is.null(tools) || length(tools) == 0L) return(NULL)
  lapply(tools, function(t) {
    fn <- t[["function"]]
    if (!identical(t$type %||% "function", "function") || !is.list(fn) || is.null(fn$name)) {
      .bad_request("only tools of type \"function\" with a name are supported")
    }
    params <- .oa_schema_fields(fn$parameters, paste0("tool '", fn$name, "'"))
    tryCatch(edge_tool(fn$name, fn$description %||% "", params),
             error = function(e) .bad_request(conditionMessage(e)))
  })
}

# The grammar for response_format = json_schema, or NULL for plain text.
# The schema must be a flat object; json_object without a schema is not
# supported, since the reply could not be constrained.
.oa_response_grammar <- function(response_format) {
  type <- response_format$type %||% "text"
  if (identical(type, "text")) return(NULL)
  if (!identical(type, "json_schema")) {
    .bad_request("response_format must be \"text\" or \"json_schema\"")
  }
  schema <- response_format$json_schema$schema
  if (!is.list(schema) || !identical(schema$type, "object") || length(schema$properties) == 0L) {
    .bad_request("response_format json_schema must describe an object with properties")
  }
  edge_json_grammar(.oa_schema_fields(schema, "response_format", descriptions = FALSE))
}

.oa_tool_choice <- function(tool_choice) {
  if (is.null(tool_choice)) return("auto")
  if (is.character(tool_choice)) return(tool_choice)
  name <- tool_choice[["function"]]$name
  if (is.null(name)) .bad_request("tool_choice must be a string or name a function")
  name
}

.oa_id <- function(prefix) {
  paste0(prefix, paste(sample(c(letters, 0:9), 24L, replace = TRUE), collapse = ""))
}

# Serializes like the API's JSON serializer: no boxing, NULL and NA as null.
.oa_json <- function(x) {
  as.character(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "null",
                                digits = NA))
}

.oa_empty_object <- function() structure(list(), names = character())

# A whole response as server-sent events. The tokens are produced one by one
# but sent together when generation ends, since plumber cannot write a
# response in parts.
.oa_sse <- function(chunks) {
  paste0(paste0("data: ", vapply(chunks, .oa_json, character(1)), "\n\n", collapse = ""),
         "data: [DONE]\n\n")
}

.oa_sse_response <- function(res, chunks) {
  res$status <- 200L
  res$setHeader("Content-Type", "text/event-stream")
  res$setHeader("Cache-Control", "no-cache")
  res$body <- .oa_sse(chunks)
  res
}

.oa_sampling <- function(body) {
  stop <- unlist(body$stop)
  tryCatch(
    .sampling_options(as.numeric(body$temperature %||% 0.7),
                      as.numeric(body$top_p %||% 0.95),
                      as.integer(body$top_k %||% 0L),
                      as.numeric(body$min_p %||% 0),
                      as.numeric(body$repeat_penalty %||% 1),
                      body$seed, stop),
    error = function(e) .bad_request(conditionMessage(e)))
}

.oa_max_tokens <- function(body, default) {
  n <- body$max_completion_tokens %||% body$max_tokens %||% default
  n <- suppressWarnings(as.integer(n))
  if (length(n) != 1L || is.na(n) || n < 1L) .bad_request("max_tokens must be a positive integer")
  min(n, 4096L)
}

.oa_usage <- function(prompt_tokens, completion_tokens, cached_tokens = NULL) {
  usage <- list(prompt_tokens = prompt_tokens, completion_tokens = completion_tokens,
                total_tokens = prompt_tokens + completion_tokens)
  if (!is.null(cached_tokens)) {
    usage$prompt_tokens_details <- list(cached_tokens = as.integer(cached_tokens))
  }
  usage
}

# A finished text in word-sized pieces for streamed chunks. The engine
# hands over the text when it is done, so the pieces are cut afterwards.
.oa_pieces <- function(text) {
  if (is.null(text) || !nzchar(text)) return(character())
  pieces <- regmatches(text, gregexpr("\\s*\\S+|\\s+$", text, perl = TRUE))[[1]]
  if (!identical(paste(pieces, collapse = ""), text)) pieces <- text
  pieces
}

# How the endpoints reach a model: gen$prompt() formats chat messages and
# gen$generate() returns a promise of one generation (text, finish,
# prompt_tokens, completion_tokens, cached_tokens). .engine_gen() is the
# one edge_api() uses.
.engine_gen <- function(engine, on_submit = NULL) {
  list(
    prompt = function(messages) build_chat_prompt(messages, ctx = engine$ctx),
    generate = function(prompt, n_predict, sampling, grammar = NULL) {
      id <- .engine_submit(engine, prompt, n_predict, sampling, grammar)
      if (!is.null(on_submit)) on_submit(id)
      .engine_promise(engine, id)
    })
}

# POST /v1/chat/completions without the HTTP layer. Returns a promise of
# the response object and, for stream = true, the chunks to send instead.
.oa_chat <- function(gen, body, model_name) {
  messages <- .oa_messages(body$messages)
  tools <- .oa_tools(body$tools)
  tool_choice <- .oa_tool_choice(body$tool_choice)
  n_predict <- .oa_max_tokens(body, 256L)
  sampling <- .oa_sampling(body)
  id <- .oa_id("chatcmpl-")
  created <- as.integer(Sys.time())

  grammar <- .oa_response_grammar(body$response_format)
  use_tools <- !is.null(tools) && !identical(tool_choice, "none")
  if (!is.null(grammar) && use_tools) {
    .bad_request("response_format json_schema cannot be combined with tools")
  }
  if (use_tools) {
    checked <- tryCatch(.check_tools(tools), error = function(e) .bad_request(conditionMessage(e)))
    forced <- tryCatch(.check_tool_choice(tool_choice, checked),
                       error = function(e) .bad_request(conditionMessage(e)))
    if (identical(forced, character())) use_tools <- FALSE
  }

  answer <- function(out, tool_calls = list()) {
    content <- if (length(tool_calls)) NULL else out$text
    finish <- if (length(tool_calls)) "tool_calls" else out$finish
    usage <- .oa_usage(out$prompt_tokens, out$completion_tokens, out$cached_tokens)
    message <- list(role = "assistant", content = content)
    if (length(tool_calls)) message$tool_calls <- tool_calls
    response <- list(id = id, object = "chat.completion", created = created,
                     model = model_name,
                     choices = list(list(index = 0L, message = message,
                                         finish_reason = finish)),
                     usage = usage)
    if (!isTRUE(body$stream)) return(list(response = response))
    chunk <- function(delta, finish_reason = NA) {
      list(id = id, object = "chat.completion.chunk", created = created, model = model_name,
           choices = list(list(index = 0L, delta = delta, finish_reason = finish_reason)))
    }
    chunks <- list(chunk(list(role = "assistant", content = "")))
    chunks <- c(chunks, lapply(.oa_pieces(content), function(p) chunk(list(content = p))))
    if (length(tool_calls)) {
      chunks <- c(chunks, list(chunk(list(tool_calls = list(c(list(index = 0L), tool_calls[[1]]))))))
    }
    chunks <- c(chunks, list(chunk(.oa_empty_object(), finish)))
    if (isTRUE(body$stream_options$include_usage)) {
      chunks <- c(chunks, list(list(id = id, object = "chat.completion.chunk",
                                    created = created, model = model_name,
                                    choices = list(), usage = usage)))
    }
    list(response = response, chunks = chunks)
  }
  plain <- function() {
    prompt <- gen$prompt(.messages_with_tools(messages, NULL, NULL))
    promises::then(gen$generate(prompt, n_predict, sampling, grammar), answer)
  }
  if (!use_tools) return(plain())

  # The call itself is chosen greedily, as in edge_chat_completion().
  prompt <- gen$prompt(.messages_with_tools(messages, checked, forced))
  call_grammar <- .tool_grammar(checked, forced)
  promises::then(gen$generate(prompt, n_predict, .sampling_options(0, sampling$top_p), call_grammar),
                 function(out) {
    reply <- .parse_tool_reply(out$text, checked)
    if (isTRUE(reply$none) || (is.null(forced) && .repeats_last_call(reply, messages, checked))) {
      # No tool needed, or the model asked again for a result it has: answer
      # in text without the tool instructions.
      return(plain())
    }
    call <- reply$tool_calls[[1]]
    args <- if (length(call$arguments)) call$arguments else .oa_empty_object()
    answer(out, list(list(id = .oa_id("call_"), type = "function",
                          `function` = list(name = call$name, arguments = .oa_json(args)))))
  })
}

# POST /v1/completions without the HTTP layer; a promise like .oa_chat().
.oa_completion <- function(gen, body, model_name) {
  prompt <- .oa_text(body$prompt)
  n_predict <- .oa_max_tokens(body, 128L)
  sampling <- .oa_sampling(body)
  id <- .oa_id("cmpl-")
  created <- as.integer(Sys.time())
  promises::then(gen$generate(prompt, n_predict, sampling), function(out) {
    usage <- .oa_usage(out$prompt_tokens, out$completion_tokens, out$cached_tokens)
    response <- list(id = id, object = "text_completion", created = created, model = model_name,
                     choices = list(list(text = out$text, index = 0L,
                                         finish_reason = out$finish)),
                     usage = usage)
    if (!isTRUE(body$stream)) return(list(response = response))
    chunk <- function(text, finish_reason = NA) {
      list(id = id, object = "text_completion", created = created, model = model_name,
           choices = list(list(text = text, index = 0L, finish_reason = finish_reason)))
    }
    chunks <- c(lapply(.oa_pieces(out$text), chunk), list(chunk("", out$finish)))
    if (isTRUE(body$stream_options$include_usage)) {
      chunks <- c(chunks, list(list(id = id, object = "text_completion", created = created,
                                    model = model_name, choices = list(), usage = usage)))
    }
    list(response = response, chunks = chunks)
  })
}

# Sets the status for an error and returns its body: 400 for a bad
# request, 503 when the queue is full, 500 for anything else.
.oa_error <- function(res, e) {
  if (inherits(e, "edge_bad_request")) {
    res$status <- 400L
    type <- "invalid_request_error"
  } else if (inherits(e, "edge_busy")) {
    res$status <- 503L
    res$setHeader("Retry-After", "5")
    type <- "server_busy"
  } else {
    res$status <- 500L
    type <- "server_error"
  }
  list(error = list(message = conditionMessage(e), type = type))
}

# Runs one endpoint. fun returns the output of .oa_chat() or
# .oa_completion(), or a promise of it.
.oa_handle <- function(res, fun) {
  out <- tryCatch(fun(), error = function(e) e)
  if (inherits(out, "error")) return(.oa_error(res, out))
  if (!promises::is.promise(out)) out <- promises::promise_resolve(out)
  promises::catch(promises::then(out, function(out) {
    if (!is.null(out$chunks)) return(.oa_sse_response(res, out$chunks))
    out$response
  }), function(e) .oa_error(res, e))
}

#' Serve a model as a local OpenAI-compatible API
#'
#' Starts a local Plumber API server that exposes the loaded model through
#' endpoints compatible with the OpenAI API format. This allows other
#' applications (Python, JavaScript, curl) and OpenAI client libraries to
#' use the model over HTTP. \code{\link{chat_edge}} uses it to connect the
#' model to ellmer.
#'
#' @param model_path Path to a .gguf model file
#' @param host Host to bind to (default: "127.0.0.1" for local only)
#' @param port Port number (default: 8080)
#' @param n_ctx Context size (default: 2048)
#' @param n_gpu_layers GPU layers (default: 0, use -1 for full GPU offload)
#' @param embeddings Enable embeddings endpoint (default: FALSE)
#' @param api_key Optional API key for authentication. If set, requests must
#'   include an \code{Authorization: Bearer <key>} or \code{X-API-Key: <key>}
#'   header.
#' @param n_slots Requests generated at once (default: 1); see
#'   \code{\link{edge_api}}.
#' @param ... More arguments for \code{\link{edge_api}}, such as
#'   \code{n_threads}, \code{max_queue} or \code{cache_type}.
#'
#' @details
#' Endpoints served:
#' \itemize{
#'   \item \code{POST /v1/completions}: Text completion
#'   \item \code{POST /v1/chat/completions}: Chat completion (uses model's native template)
#'   \item \code{POST /v1/embeddings}: Text embeddings (if \code{embeddings = TRUE})
#'   \item \code{GET /v1/models}: List loaded model info
#'   \item \code{GET /health}: Health check
#' }
#'
#' Requests take \code{max_tokens} (or \code{max_completion_tokens}),
#' \code{temperature}, \code{top_p}, \code{seed} and \code{stop}, and also
#' \code{top_k}, \code{min_p} and \code{repeat_penalty}. \code{usage} counts
#' tokens with the model's tokenizer.
#'
#' Chat requests can pass \code{tools} and \code{tool_choice} in the OpenAI
#' format. Tool arguments may be strings, numbers, integers, booleans or
#' enums, and every argument is treated as required. A call comes back in
#' \code{message$tool_calls} with \code{finish_reason = "tool_calls"}; send
#' the result back as a message with \code{role = "tool"} and the call's
#' \code{tool_call_id}. The model makes at most one call per reply. See
#' \code{\link{edge_chat_completion}} for how calls are generated.
#'
#' \code{response_format} may be \code{{"type": "json_schema"}} with a
#' flat object schema of the same argument types; the reply is then
#' constrained to JSON of that shape by a grammar.
#'
#' With \code{stream = true} the response is sent as server-sent events in
#' the OpenAI chunk format, ending with \code{data: [DONE]}, and
#' \code{stream_options = {"include_usage": true}} adds a usage chunk. The
#' chunks are produced token by token but sent together when generation
#' ends, so clients get the format but not the text as it is written.
#'
#' The server is \code{\link{edge_api}} run with \code{plumber::pr_run()}:
#' requests wait in a queue and \code{n_slots} of them are generated at
#' once. It runs in the foreground; press Ctrl+C / Esc to stop. For many
#' users, or Posit Connect, use \code{\link{edge_api}} directly.
#'
#' @return Does not return while the server runs.
#' @examples
#' \dontrun{
#' # Serve a model locally
#' edge_serve("model.gguf", port = 8080)
#'
#' # Then from another terminal or Python:
#' # curl http://localhost:8080/v1/chat/completions \
#' #   -H "Content-Type: application/json" \
#' #   -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
#' }
#' @seealso \code{\link{edge_api}}, \code{\link{chat_edge}}
#' @export
edge_serve <- function(model_path, host = "127.0.0.1", port = 8080L,
                        n_ctx = 2048L, n_gpu_layers = 0L, embeddings = FALSE,
                        api_key = NULL, n_slots = 1L, ...) {
  pr <- edge_api(model_path, n_slots = n_slots, n_ctx = n_ctx, embeddings = embeddings,
                 api_key = api_key, n_gpu_layers = n_gpu_layers, ...)
  model_name <- tools::file_path_sans_ext(basename(model_path))
  message("
edgemodelr API server starting")
  message("  Model: ", model_name, " (", n_slots, if (n_slots == 1) " slot" else " slots", ")")
  message("  Endpoints:")
  message("    POST ", host, ":", port, "/v1/completions")
  message("    POST ", host, ":", port, "/v1/chat/completions")
  message("    POST ", host, ":", port, "/v1/jobs")
  if (!isFALSE(embeddings)) message("    POST ", host, ":", port, "/v1/embeddings")
  message("    GET  ", host, ":", port, "/v1/models")
  message("    GET  ", host, ":", port, "/health")
  if (!is.null(api_key)) message("  Auth: API key required")
  message("
Press Ctrl+C to stop.
")
  plumber::pr_run(pr, host = host, port = as.integer(port))
}

#' Run many requests at once on one model
#'
#' Starts a serving engine on a loaded model: a background thread that
#' generates for several requests at the same time with continuous batching,
#' the way llama.cpp's own server does. Each of \code{n_slots} slots holds
#' one request; every step decodes the next token of each running request
#' in one batch, so on CPU total throughput grows with the number of slots
#' while each request gets a little slower. Requests beyond the slots wait in
#' a queue, first come first served. A slot keeps its tokens after a request
#' ends, and a later request that starts the same way (the same system
#' prompt, or the next turn of a chat) reuses them instead of reading them
#' again.
#'
#' The engine runs beside R: \code{edge_engine_submit()} returns at once
#' with a request id, and R stays free while the request is generated.
#' \code{edge_engine_poll()} reports progress and the text so far. A
#' finished request is kept until \code{edge_engine_release()} frees it
#' (an unfinished one is cancelled first), so a client that polls should
#' release each request once it has read the result.
#' \code{edge_engine_result()} gives a promise of the finished request (for
#' plumber and Shiny) and \code{edge_engine_generate()} submits prompts and
#' waits for all of them; both free the requests themselves.
#'
#' While the engine runs, the model belongs to it: the direct completion
#' functions refuse to use \code{ctx} until \code{edge_engine_stop()}.
#' Tokenizing and chat templates still work. \code{\link{edge_api}} builds
#' an HTTP API on an engine. An internal llama.cpp error on the engine's
#' thread cannot become an R error and ends the R process, as it would end
#' llama.cpp's own server; on Posit Connect the process is then restarted.
#'
#' @param ctx A model loaded with \code{\link{edge_load_model}} with
#'   \code{n_parallel} of at least \code{n_slots} and
#'   \code{embeddings = FALSE}. The context is shared by the slots: each
#'   request needs room for its prompt plus \code{n_predict} tokens, and
#'   waits in the queue until there is room.
#' @param n_slots Requests generated at once (default: the model's
#'   \code{n_parallel}).
#' @param max_queue Requests that may wait (default: 256). A submit beyond
#'   it fails with an error of class \code{edge_busy}.
#' @param engine An engine from \code{edge_engine()}.
#' @param prompt,prompts Prompt text, already formatted (see
#'   \code{\link{build_chat_prompt}}).
#' @param n_predict Most tokens to generate (default: 256).
#' @param temperature,top_p,top_k,min_p,repeat_penalty,seed,stop Sampling,
#'   as in \code{\link{edge_completion}}. \code{temperature = 0} is greedy.
#' @param grammar Optional GBNF grammar with a \code{root} rule, such as
#'   one from \code{\link{edge_json_grammar}}.
#' @param ids,id Request ids from \code{edge_engine_submit()}.
#' @param timeout Seconds \code{edge_engine_generate()} waits before it
#'   cancels what is left (default: no limit).
#' @return \code{edge_engine()}: an \code{edge_engine} object.
#'   \code{edge_engine_submit()}: an integer request id.
#'   \code{edge_engine_poll()} and \code{edge_engine_generate()}: a data
#'   frame with one row per request: \code{id}, \code{status} ("queued",
#'   "running", "done" or "unknown"), \code{text} (so far, for an
#'   unfinished request), \code{finish} ("stop", "length", "cancelled" or
#'   "error"), \code{error}, \code{prompt_tokens},
#'   \code{completion_tokens}, \code{cached_tokens} (prompt tokens reused
#'   from a slot), \code{queue_position}, \code{queue_seconds},
#'   \code{first_token_seconds} and \code{total_seconds}.
#'   \code{edge_engine_result()}: a promise of a list with the same fields,
#'   rejected with an error of class \code{edge_cancelled} if the request is
#'   cancelled.
#'   \code{edge_engine_stats()}: a list of counters.
#' @examples
#' \dontrun{
#' ctx <- edge_load_model("gemma-3-1b-it-Q4_K_M.gguf", n_ctx = 8192, n_parallel = 4)
#' engine <- edge_engine(ctx)
#' prompts <- vapply(c("Rome", "Paris", "Oslo", "Lima"), function(city) {
#'   build_chat_prompt(list(list(role = "user",
#'     content = paste("One fact about", city))), ctx = ctx)
#' }, character(1))
#' out <- edge_engine_generate(engine, prompts, n_predict = 60, temperature = 0)
#' out$text
#' edge_engine_stats(engine)
#' edge_engine_stop(engine)
#' }
#' @seealso \code{\link{edge_api}}, \code{\link{edge_map}}
#' @export
edge_engine <- function(ctx, n_slots = NULL, max_queue = 256L) {
  if (!is_valid_model(ctx)) stop("Invalid model context. Load a model first with edge_load_model()")
  if (is.null(n_slots)) n_slots <- edge_n_parallel_internal(ctx)
  if (!is.numeric(n_slots) || length(n_slots) != 1L || is.na(n_slots) || n_slots < 1 ||
      n_slots != round(n_slots)) {
    stop("n_slots must be a positive whole number")
  }
  if (!is.numeric(max_queue) || length(max_queue) != 1L || is.na(max_queue) || max_queue < 1) {
    stop("max_queue must be a positive number")
  }
  ptr <- edge_engine_start_internal(ctx, as.integer(n_slots), as.integer(max_queue))
  state <- new.env(parent = emptyenv())
  state$waiters <- list()
  state$scheduled <- FALSE
  structure(list(ptr = ptr, ctx = ctx, state = state), class = "edge_engine")
}

#' @export
print.edge_engine <- function(x, ...) {
  s <- tryCatch(edge_engine_stats(x), error = function(e) NULL)
  if (is.null(s) || !isTRUE(s$running)) {
    cat("<edge_engine: stopped>\n")
  } else {
    cat(sprintf("<edge_engine: %d slots, %d running, %d queued, %d requests done>\n",
                s$slots, s$active, s$queued, as.integer(s$requests_done)))
  }
  invisible(x)
}

.check_engine <- function(engine) {
  if (!inherits(engine, "edge_engine")) stop("engine must come from edge_engine()")
}

.engine_submit <- function(engine, prompt, n_predict, sampling, grammar = NULL) {
  tryCatch(
    edge_engine_submit_internal(engine$ptr, enc2utf8(prompt), as.integer(n_predict), sampling,
                                if (is.null(grammar)) "" else grammar, "root"),
    error = function(e) {
      msg <- conditionMessage(e)
      if (grepl("queue_full", msg, fixed = TRUE)) {
        stop(structure(class = c("edge_busy", "error", "condition"),
                       list(message = "The server is busy: too many requests are waiting. Try again shortly.",
                            call = NULL)))
      }
      if (grepl("Prompt too long", msg, fixed = TRUE)) .bad_request(msg)
      stop(e)
    })
}

#' @rdname edge_engine
#' @export
edge_engine_submit <- function(engine, prompt, n_predict = 256L, temperature = 0.7,
                               top_p = 0.95, top_k = 0L, min_p = 0, repeat_penalty = 1,
                               seed = NULL, stop = NULL, grammar = NULL) {
  .check_engine(engine)
  if (!is.character(prompt) || length(prompt) != 1L || is.na(prompt)) {
    stop("prompt must be a single string")
  }
  sampling <- .sampling_options(temperature, top_p, top_k, min_p, repeat_penalty, seed, stop)
  .engine_submit(engine, prompt, n_predict, sampling, grammar)
}

#' @rdname edge_engine
#' @export
edge_engine_poll <- function(engine, ids) {
  .check_engine(engine)
  as.data.frame(edge_engine_poll_internal(engine$ptr, as.integer(ids)),
                stringsAsFactors = FALSE)
}

#' @rdname edge_engine
#' @export
edge_engine_release <- function(engine, ids) {
  .check_engine(engine)
  edge_engine_release_internal(engine$ptr, as.integer(ids))
  invisible(engine)
}

#' @rdname edge_engine
#' @export
edge_engine_cancel <- function(engine, ids) {
  .check_engine(engine)
  edge_engine_cancel_internal(engine$ptr, as.integer(ids))
  invisible(engine)
}

#' @rdname edge_engine
#' @export
edge_engine_result <- function(engine, id) {
  .check_engine(engine)
  if (!requireNamespace("promises", quietly = TRUE) || !requireNamespace("later", quietly = TRUE)) {
    stop("The 'promises' and 'later' packages are required for edge_engine_result()")
  }
  .engine_promise(engine, as.integer(id))
}

#' @rdname edge_engine
#' @export
edge_engine_generate <- function(engine, prompts, n_predict = 256L, temperature = 0.7,
                                 top_p = 0.95, top_k = 0L, min_p = 0, repeat_penalty = 1,
                                 seed = NULL, stop = NULL, grammar = NULL, timeout = Inf) {
  .check_engine(engine)
  if (!is.character(prompts) || anyNA(prompts)) stop("prompts must be a character vector")
  sampling <- .sampling_options(temperature, top_p, top_k, min_p, repeat_penalty, seed, stop)
  # Registered first, so a submit that fails part way (a full queue, a
  # prompt too long) cancels and releases the ones already submitted.
  ids <- integer()
  on.exit(if (length(ids)) edge_engine_release_internal(engine$ptr, ids), add = TRUE)
  for (p in prompts) ids <- c(ids, .engine_submit(engine, p, n_predict, sampling, grammar))
  deadline <- Sys.time() + timeout
  repeat {
    out <- edge_engine_poll(engine, ids)
    if (all(out$status %in% c("done", "unknown"))) break
    if (Sys.time() > deadline) {
      edge_engine_cancel_internal(engine$ptr, ids)
      deadline <- Inf
    }
    Sys.sleep(0.02)
  }
  out
}

#' @rdname edge_engine
#' @export
edge_engine_stats <- function(engine) {
  .check_engine(engine)
  edge_engine_stats_internal(engine$ptr)
}

#' @rdname edge_engine
#' @export
edge_engine_stop <- function(engine) {
  .check_engine(engine)
  edge_engine_stop_internal(engine$ptr)
  waiters <- engine$state$waiters
  engine$state$waiters <- list()
  for (ws in waiters) for (w in ws) w$reject(simpleError("The engine was stopped"))
  invisible(engine)
}

# A promise of one request. One poll loop per engine, run by later, checks
# every waiting request and settles the finished ones.
.engine_promise <- function(engine, id) {
  promises::promise(function(resolve, reject) {
    st <- engine$state
    key <- as.character(id)
    st$waiters[[key]] <- c(st$waiters[[key]], list(list(resolve = resolve, reject = reject)))
    .engine_schedule(engine)
  })
}

.engine_schedule <- function(engine) {
  st <- engine$state
  if (isTRUE(st$scheduled)) return(invisible())
  st$scheduled <- TRUE
  later::later(function() .engine_tick(engine), 0.01)
  invisible()
}

.engine_row <- function(p, i) lapply(p, function(col) col[[i]])

.engine_tick <- function(engine) {
  st <- engine$state
  st$scheduled <- FALSE
  keys <- names(st$waiters)
  if (length(keys) == 0L) return(invisible())
  p <- tryCatch(edge_engine_poll_internal(engine$ptr, as.integer(keys)), error = function(e) e)
  if (inherits(p, "error")) {
    waiters <- st$waiters
    st$waiters <- list()
    for (ws in waiters) for (w in ws) w$reject(p)
    return(invisible())
  }
  done <- which(p$status %in% c("done", "unknown"))
  for (i in done) {
    ws <- st$waiters[[keys[i]]]
    st$waiters[[keys[i]]] <- NULL
    row <- .engine_row(p, i)
    for (w in ws) {
      if (identical(row$status, "unknown")) {
        w$reject(simpleError("The request is no longer known to the engine"))
      } else if (identical(row$finish, "error")) {
        w$reject(simpleError(row$error))
      } else if (identical(row$finish, "cancelled")) {
        w$reject(structure(class = c("edge_cancelled", "error", "condition"),
                           list(message = "The request was cancelled", call = NULL)))
      } else {
        w$resolve(row)
      }
    }
  }
  if (length(done)) edge_engine_release_internal(engine$ptr, as.integer(keys[done]))
  if (length(st$waiters)) .engine_schedule(engine)
  invisible()
}

# CPUs this process may use when a container limits it (cgroup v2 or v1),
# or NA. parallel::detectCores() reports the host's cores, and llama.cpp
# slows down badly with more threads than the container has CPUs.
.cpu_quota <- function() {
  read1 <- function(f) tryCatch(readLines(f, n = 1L, warn = FALSE), error = function(e) "")
  v2 <- "/sys/fs/cgroup/cpu.max"
  if (file.exists(v2)) {
    parts <- strsplit(read1(v2), " ", fixed = TRUE)[[1]]
    if (length(parts) == 2L && parts[1] != "max") {
      q <- suppressWarnings(as.numeric(parts[1]) / as.numeric(parts[2]))
      if (is.finite(q) && q > 0) return(q)
    }
    return(NA_real_)
  }
  quota <- suppressWarnings(as.numeric(read1("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")))
  period <- suppressWarnings(as.numeric(read1("/sys/fs/cgroup/cpu/cpu.cfs_period_us")))
  if (length(quota) == 1L && length(period) == 1L && is.finite(quota) && quota > 0 &&
      is.finite(period) && period > 0) {
    return(quota / period)
  }
  NA_real_
}

#' Build an OpenAI-compatible API for many users
#'
#' Returns a plumber router that serves a local GGUF model to many users at
#' once, for Posit Connect or any server that runs plumber APIs. The model
#' is loaded once, and an \code{\link{edge_engine}} generates for up to
#' \code{n_slots} requests at the same time with continuous batching; more
#' requests wait in a queue, so a busy server answers later instead of
#' failing. Request handlers return promises, so the R process keeps
#' accepting requests while the engine works.
#'
#' Endpoints:
#' \itemize{
#'   \item \code{POST /v1/chat/completions} and \code{POST /v1/completions}:
#'     the same requests and replies as \code{\link{edge_serve}}, including
#'     tools, \code{response_format} and \code{stream}. \code{usage} adds
#'     \code{prompt_tokens_details$cached_tokens}.
#'   \item \code{POST /v1/jobs}: takes a chat completion request and returns
#'     at once (status 202) with a job id. \code{GET /v1/jobs/<id>} gives the
#'     status, the queue position and the text so far, then the whole chat
#'     completion as \code{response} once done. \code{DELETE /v1/jobs/<id>}
#'     cancels it. Jobs suit a Shiny app: it can show the answer as it is
#'     written, and no request has to stay open while the model works.
#'     Finished jobs are kept for 15 minutes.
#'   \item \code{POST /v1/embeddings} when \code{embeddings} is set.
#'   \item \code{GET /v1/models}, \code{GET /metrics} (engine counters) and
#'     \code{GET /health} (no key needed).
#' }
#'
#' A full queue answers 503 with \code{Retry-After}.
#'
#' Sizing: each slot gets \code{n_ctx} tokens of context, so the KV cache
#' holds \code{n_slots * n_ctx} tokens (at most 32768); \code{cache_type =
#' "q8_0"} halves its memory compared with "f16". On CPU, more slots give
#' more tokens per second in total until the cores are saturated, typically
#' at about 8 slots; each request then gets slower. Inside a container the
#' default thread count follows the container's CPU limit.
#'
#' @param model_path Path to a .gguf model file.
#' @param n_slots Requests generated at once (default: 8).
#' @param n_ctx Context per slot, in tokens (default: 2048): the longest
#'   prompt plus answer one request may use.
#' @param n_threads CPU threads (default: one per core, within the
#'   container's CPU limit). The environment variable
#'   \code{EDGEMODELR_THREADS} overrides the default.
#' @param max_queue Requests that may wait for a slot (default: 256).
#' @param cache_type KV cache type for keys and values (default: "q8_0");
#'   see \code{\link{edge_load_model}}.
#' @param embeddings FALSE (default), TRUE to also serve
#'   \code{/v1/embeddings} from a second copy of the model loaded for
#'   embeddings, or the path of a .gguf embedding model to serve instead.
#' @param api_key Optional key. When set, every endpoint but \code{/health}
#'   needs \code{Authorization: Bearer <key>} or \code{X-API-Key: <key>}.
#'   On Posit Connect, which uses \code{Authorization} for its own keys, use
#'   \code{X-API-Key}, or leave this unset and use Connect's access
#'   settings.
#' @param model_name Name reported as the model (default: the file name).
#' @param n_gpu_layers GPU layers (default: 0).
#' @return A plumber router. Run it with \code{plumber::pr_run()}, or return
#'   it from \code{entrypoint.R} on Posit Connect. The engine is in
#'   \code{attr(router, "edge_engine")}.
#' @examples
#' \dontrun{
#' api <- edge_api("gemma-3-1b-it-Q4_K_M.gguf", n_slots = 8)
#' plumber::pr_run(api, port = 8080)
#'
#' # entrypoint.R for Posit Connect
#' edgemodelr::edge_api(Sys.getenv("EDGEMODELR_MODEL"), n_slots = 8,
#'                      api_key = Sys.getenv("EDGEMODELR_API_KEY", unset = NA))
#' }
#' @seealso \code{\link{edge_serve}}, \code{\link{edge_engine}}
#' @export
edge_api <- function(model_path, n_slots = 8L, n_ctx = 2048L, n_threads = NULL,
                     max_queue = 256L, cache_type = "q8_0", embeddings = FALSE,
                     api_key = NULL, model_name = NULL, n_gpu_layers = 0L) {
  for (pkg in c("plumber", "jsonlite", "promises", "later")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      stop("The '", pkg, "' package is required for edge_api().\n",
           "Install it with: install.packages('", pkg, "')")
    }
  }
  if (!is.character(model_path) || length(model_path) != 1L || is.na(model_path) ||
      !nzchar(model_path)) {
    stop("model_path must be the path of a .gguf model")
  }
  if (!file.exists(model_path)) stop("Model file not found: ", model_path)
  if (!is.numeric(n_slots) || length(n_slots) != 1L || is.na(n_slots) || n_slots < 1 ||
      n_slots > 64 || n_slots != round(n_slots)) {
    stop("n_slots must be a whole number from 1 to 64")
  }
  if (!is.numeric(n_ctx) || length(n_ctx) != 1L || is.na(n_ctx) || n_ctx < 128) {
    stop("n_ctx must be at least 128")
  }
  total_ctx <- as.integer(n_slots * n_ctx)
  if (total_ctx > 32768L) {
    stop("n_slots * n_ctx is ", total_ctx, " tokens, more than the 32768 a context can hold; ",
         "lower n_slots or n_ctx")
  }
  if (length(api_key) != 1L || is.na(api_key) || !nzchar(api_key)) api_key <- NULL
  if (is.null(n_threads)) {
    env <- suppressWarnings(as.integer(Sys.getenv("EDGEMODELR_THREADS")))
    n_threads <- if (!is.na(env) && env > 0) env else .default_threads()
  }

  ctx <- edge_load_model(model_path, n_ctx = total_ctx, n_gpu_layers = n_gpu_layers,
                         n_threads = if (n_threads > 0) n_threads else NULL,
                         n_parallel = as.integer(n_slots),
                         cache_type_k = cache_type, cache_type_v = cache_type)
  engine <- edge_engine(ctx, n_slots = n_slots, max_queue = max_queue)

  embed_ctx <- NULL
  if (isTRUE(embeddings) || is.character(embeddings)) {
    embed_path <- if (is.character(embeddings)) embeddings else model_path
    embed_ctx <- edge_load_model(embed_path, n_ctx = n_ctx, n_gpu_layers = n_gpu_layers,
                                 n_threads = if (n_threads > 0) n_threads else NULL,
                                 embeddings = TRUE)
  }
  if (is.null(model_name)) model_name <- tools::file_path_sans_ext(basename(model_path))
  jobs <- new.env(parent = emptyenv())

  pr <- plumber::pr()
  pr <- plumber::pr_set_serializer(
    pr, plumber::serializer_unboxed_json(null = "null", na = "null", digits = NA))

  pr <- plumber::pr_filter(pr, "cors", function(req, res) {
    res$setHeader("Access-Control-Allow-Origin", "*")
    res$setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
    res$setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
    if (req$REQUEST_METHOD == "OPTIONS") {
      res$status <- 200L
      return(list())
    }
    plumber::forward()
  })
  if (!is.null(api_key)) {
    pr <- plumber::pr_filter(pr, "auth", function(req, res) {
      if (identical(req$PATH_INFO, "/health")) return(plumber::forward())
      bearer <- req$HTTP_AUTHORIZATION
      key <- req$HTTP_X_API_KEY
      if (!identical(key, api_key) && !identical(bearer, paste("Bearer", api_key))) {
        res$status <- 401L
        return(list(error = list(message = "Invalid API key", type = "authentication_error")))
      }
      plumber::forward()
    })
  }

  engine_stats <- function() {
    s <- edge_engine_stats(engine)
    s$tokens_per_second <- if (s$busy_seconds > 0) s$tokens_generated / s$busy_seconds else 0
    s
  }
  pr <- plumber::pr_get(pr, "/health", function(res) {
    s <- edge_engine_stats(engine)
    if (!isTRUE(s$running)) res$status <- 503L
    list(status = if (isTRUE(s$running)) "ok" else "stopped", model = model_name,
         slots = s$slots, active = s$active, queued = s$queued)
  })
  pr <- plumber::pr_get(pr, "/metrics", function() engine_stats())
  pr <- plumber::pr_get(pr, "/v1/models", function() {
    list(object = "list", data = list(
      list(id = model_name, object = "model", owned_by = "local")
    ))
  })
  pr <- plumber::pr_post(pr, "/v1/completions", function(req, res) {
    .oa_handle(res, function() .oa_completion(.engine_gen(engine), .oa_body(req), model_name))
  })
  pr <- plumber::pr_post(pr, "/v1/chat/completions", function(req, res) {
    .oa_handle(res, function() .oa_chat(.engine_gen(engine), .oa_body(req), model_name))
  })

  # Jobs: the chat completion runs in the background and is fetched later.
  forget_old_jobs <- function() {
    old <- Sys.time() - 15 * 60
    for (key in ls(jobs)) {
      job <- jobs[[key]]
      if (!is.null(job$ended) && job$ended < old) rm(list = key, envir = jobs)
    }
  }
  pr <- plumber::pr_post(pr, "/v1/jobs", function(req, res) {
    forget_old_jobs()
    job <- new.env(parent = emptyenv())
    job$id <- .oa_id("job-")
    job$status <- "queued"
    job$created <- as.integer(Sys.time())
    body <- tryCatch(.oa_body(req), error = function(e) e)
    if (inherits(body, "error")) return(.oa_error(res, body))
    body$stream <- FALSE
    gen <- .engine_gen(engine, on_submit = function(id) job$engine_id <- id)
    p <- tryCatch(.oa_chat(gen, body, model_name), error = function(e) e)
    if (inherits(p, "error")) return(.oa_error(res, p))
    promises::then(p, function(out) {
      if (isTRUE(job$cancelled)) return(invisible())
      job$status <- "done"
      job$response <- out$response
      job$ended <- Sys.time()
    }, onRejected = function(e) {
      job$status <- if (isTRUE(job$cancelled)) "cancelled" else "error"
      job$error <- conditionMessage(e)
      job$ended <- Sys.time()
    })
    jobs[[job$id]] <- job
    res$status <- 202L
    list(id = job$id, object = "chat.completion.job", status = "queued", created = job$created)
  })
  job_view <- function(job) {
    out <- list(id = job$id, object = "chat.completion.job", status = job$status,
                created = job$created)
    if (identical(job$status, "done")) {
      out$response <- job$response
    } else if (job$status %in% c("error", "cancelled")) {
      out$error <- list(message = job$error %||% job$status)
    } else if (!is.null(job$engine_id)) {
      p <- edge_engine_poll_internal(engine$ptr, job$engine_id)
      if (p$status %in% c("queued", "running")) {
        out$status <- p$status
        out$text <- p$text
        out$queue_position <- p$queue_position
        out$completion_tokens <- p$completion_tokens
      }
    }
    out
  }
  find_job <- function(id, res) {
    job <- if (exists(id, envir = jobs, inherits = FALSE)) jobs[[id]]
    if (is.null(job)) res$status <- 404L
    job
  }
  pr <- plumber::pr_get(pr, "/v1/jobs/<id>", function(id, res) {
    job <- find_job(id, res)
    if (is.null(job)) return(list(error = list(message = "No such job", type = "not_found")))
    job_view(job)
  })
  pr <- plumber::pr_delete(pr, "/v1/jobs/<id>", function(id, res) {
    job <- find_job(id, res)
    if (is.null(job)) return(list(error = list(message = "No such job", type = "not_found")))
    if (job$status %in% c("queued", "running")) {
      job$cancelled <- TRUE
      if (!is.null(job$engine_id)) edge_engine_cancel_internal(engine$ptr, job$engine_id)
      job$status <- "cancelled"
      job$ended <- Sys.time()
    }
    job_view(job)
  })

  if (!is.null(embed_ctx)) {
    pr <- plumber::pr_post(pr, "/v1/embeddings", function(req, res) {
      .oa_handle(res, function() {
        body <- .oa_body(req)
        if (is.null(body$input)) .bad_request("input is required")
        texts <- as.character(unlist(body$input))
        emb_matrix <- edge_embeddings(embed_ctx, texts)
        data_list <- lapply(seq_len(nrow(emb_matrix)), function(i) {
          list(object = "embedding", embedding = as.numeric(emb_matrix[i, ]),
               index = i - 1L)
        })
        n_tokens <- sum(vapply(texts, function(t) length(edge_tokenize_internal(embed_ctx, t, TRUE)),
                               integer(1)))
        list(response = list(object = "list", data = data_list, model = model_name,
                             usage = list(prompt_tokens = n_tokens, total_tokens = n_tokens)))
      })
    })
  }

  pr <- plumber::pr_hook(pr, "exit", function() {
    edge_engine_stop(engine)
    tryCatch(edge_free_model(ctx), error = function(e) NULL)
    if (!is.null(embed_ctx)) tryCatch(edge_free_model(embed_ctx), error = function(e) NULL)
  })
  attr(pr, "edge_engine") <- engine
  pr
}

# Servers started by chat_edge(), by model path, so a second chat on the
# same model reuses the running server.
.edge_servers <- new.env(parent = emptyenv())

#' Chat with a local model through ellmer
#'
#' Returns an ellmer chat object backed by a local GGUF model, so the
#' model works with ellmer's chat, streaming, tool calling and structured
#' data functions. With \code{model_path}, the model is served by
#' \code{\link{edge_serve}} in a background R process, which keeps running
#' until the R session ends and is reused by later calls with the same
#' model. With \code{base_url}, the chat connects to a server you started
#' yourself.
#'
#' @param model_path Path to a .gguf model file. Either this or
#'   \code{base_url} must be given.
#' @param system_prompt Optional system prompt for the chat.
#' @param base_url URL of a running \code{\link{edge_serve}} server, such as
#'   \code{"http://127.0.0.1:8080"}.
#' @param n_ctx Context size for a server started here (default: 4096).
#' @param n_gpu_layers GPU layers for a server started here (default: 0).
#' @param port Port for a server started here (default: a free one).
#' @param params Optional \code{ellmer::params()} with sampling settings such
#'   as \code{temperature}.
#' @param echo Passed to \code{ellmer::chat_openai_compatible()}.
#' @param embeddings Also serve \code{POST /v1/embeddings} from a server
#'   started here (default: FALSE), so the same server can embed texts, for
#'   example with \code{ragnar::embed_openai(base_url = )}. The model then
#'   keeps an output for every prompt token, which slows chat and takes more
#'   memory, and a chat model gives weaker embeddings than an embedding model.
#'   For embeddings inside R, \code{\link{embed_edge}} with an embedding
#'   model needs no server.
#' @return An ellmer \code{Chat} object.
#'
#' @details
#' Tool calling goes through \code{\link{edge_chat_completion}}: tool
#' arguments may be strings, numbers, integers, booleans or enums, and the
#' model makes one call per reply. Structured data works for flat objects
#' of those types. Streamed replies arrive in one piece when
#' generation ends (see \code{\link{edge_serve}}).
#'
#' @examples
#' \dontrun{
#' chat <- chat_edge("gemma-3-1b-it-Q4_K_M.gguf",
#'                   system_prompt = "Answer in one sentence.")
#' chat$chat("What is the capital of France?")
#'
#' # Tool calling with ellmer
#' get_weather <- ellmer::tool(
#'   function(city) "24 C and sunny",
#'   description = "Get the current weather for a city",
#'   arguments = list(city = ellmer::type_string("City name"))
#' )
#' chat$register_tool(get_weather)
#' chat$chat("What is the weather in Paris?")
#' }
#' @seealso \code{\link{edge_serve}}
#' @export
chat_edge <- function(model_path = NULL, system_prompt = NULL, base_url = NULL,
                      n_ctx = 4096L, n_gpu_layers = 0L, port = NULL,
                      params = NULL, echo = "none", embeddings = FALSE) {
  if (!requireNamespace("ellmer", quietly = TRUE)) {
    stop("The 'ellmer' package is required for chat_edge().\n",
         "Install it with: install.packages('ellmer')")
  }
  if (is.null(model_path) == is.null(base_url)) {
    stop("give either model_path or base_url")
  }
  if (is.null(base_url)) {
    base_url <- .edge_start_server(model_path, n_ctx = n_ctx,
                                   n_gpu_layers = n_gpu_layers, port = port,
                                   embeddings = isTRUE(embeddings))
  }
  base_url <- sub("/+$", "", base_url)
  base_url <- sub("/v1$", "", base_url)
  model_name <- if (is.null(model_path)) {
    "local"
  } else {
    tools::file_path_sans_ext(basename(model_path))
  }
  ellmer::chat_openai_compatible(
    base_url = paste0(base_url, "/v1"),
    name = "edgemodelr",
    system_prompt = system_prompt,
    credentials = function() "edgemodelr",
    model = model_name,
    params = params,
    echo = echo
  )
}

# Starts edge_serve() in a background R process, or reuses the one already
# serving this model, and waits until it answers /health.
.edge_start_server <- function(model_path, n_ctx, n_gpu_layers, port,
                               embeddings = FALSE) {
  if (!requireNamespace("callr", quietly = TRUE)) {
    stop("The 'callr' package is required to start a server for chat_edge().\n",
         "Install it with: install.packages('callr'), or pass base_url.")
  }
  if (!requireNamespace("plumber", quietly = TRUE)) {
    stop("The 'plumber' package is required to start a server for chat_edge().\n",
         "Install it with: install.packages('plumber'), or pass base_url.")
  }
  if (!file.exists(model_path)) stop("Model file not found: ", model_path)
  path <- normalizePath(model_path, winslash = "/")
  key <- paste0(path, if (embeddings) " +embeddings")
  running <- .edge_servers[[key]]
  if (!is.null(running) && running$process$is_alive()) return(running$server_url)

  if (is.null(port)) port <- httpuv::randomPort()
  server_url <- sprintf("http://127.0.0.1:%d", as.integer(port))
  process <- callr::r_bg(function(model_path, port, n_ctx, n_gpu_layers, embeddings) {
    edgemodelr::edge_serve(model_path, port = port, n_ctx = n_ctx,
                           n_gpu_layers = n_gpu_layers, embeddings = embeddings)
  }, args = list(model_path = path, port = as.integer(port), n_ctx = as.integer(n_ctx),
                 n_gpu_layers = as.integer(n_gpu_layers), embeddings = embeddings),
  supervise = TRUE)

  deadline <- Sys.time() + 300
  repeat {
    if (!process$is_alive()) {
      err <- tryCatch(process$read_all_error(), error = function(e) "")
      stop("The model server stopped while starting: ", err)
    }
    if (.server_answers(paste0(server_url, "/health"))) break
    if (Sys.time() > deadline) {
      process$kill()
      stop("The model server did not start within 5 minutes")
    }
    Sys.sleep(0.5)
  }
  .edge_servers[[key]] <- list(process = process, server_url = server_url)
  server_url
}

.server_answers <- function(health_url) {
  suppressWarnings(tryCatch({
    con <- url(health_url, open = "r")
    close(con)
    TRUE
  }, error = function(e) FALSE))
}

#' Embed texts with a local model, for ragnar and other packages
#'
#' Turns texts into embedding vectors with a local GGUF embedding model,
#' without a server. Called without \code{x}, it returns a function of the
#' texts, which is what \code{ragnar::ragnar_store_create(embed = )} takes.
#' That function refers to the model only by its path, so a ragnar store
#' saved with it embeds new texts and queries again in a later R session.
#'
#' The model is loaded the first time it is needed and kept for the rest of
#' the session, one per path.
#'
#' @param x A character vector of texts, a data frame with a \code{text}
#'   column, or missing.
#' @param model_path Path to a .gguf embedding model, such as
#'   Qwen3-Embedding-0.6B or nomic-embed-text.
#' @param n_ctx Context size for the model (default: 2048). Longer texts
#'   are cut to the batch size.
#' @param normalize Scale each vector to length 1 (default: TRUE), so the
#'   dot product of two vectors is their cosine similarity.
#' @return For a character vector, a numeric matrix with one row per text.
#'   For a data frame, the data frame with an \code{embedding} matrix column
#'   added. Without \code{x}, a function that takes \code{x}.
#'
#' @examples
#' \dontrun{
#' model <- "Qwen3-Embedding-0.6B-Q8_0.gguf"
#' m <- embed_edge(c("cats purr", "dogs bark"), model)
#' dim(m)
#'
#' # A ragnar store that embeds locally
#' store <- ragnar::ragnar_store_create(
#'   "docs.duckdb",
#'   embed = embed_edge(model_path = model)
#' )
#' }
#' @seealso \code{\link{edge_embeddings}}, \code{\link{chat_edge}}
#' @export
embed_edge <- function(x, model_path, n_ctx = 2048L, normalize = TRUE) {
  if (missing(model_path) || !is.character(model_path) || length(model_path) != 1L ||
      is.na(model_path)) {
    stop("model_path must be the path of a .gguf embedding model")
  }
  if (missing(x) || is.null(x)) {
    # Built from values only, in the base environment, so it still works
    # after being saved and loaded in a new session.
    path <- normalizePath(model_path, winslash = "/", mustWork = FALSE)
    fn <- as.function.default(c(alist(x = ), bquote(
      edgemodelr::embed_edge(x, model_path = .(path), n_ctx = .(as.integer(n_ctx)),
                             normalize = .(isTRUE(normalize))))),
      envir = baseenv())
    return(fn)
  }
  if (is.data.frame(x)) {
    if (!"text" %in% names(x)) stop("a data frame x needs a text column")
    x[["embedding"]] <- embed_edge(x[["text"]], model_path, n_ctx = n_ctx,
                                   normalize = normalize)
    return(x)
  }
  if (!is.character(x) || anyNA(x)) {
    stop("x must be a character vector without NA values, or a data frame with a text column")
  }
  ctx <- .edge_embed_model(model_path, n_ctx)
  if (length(x) == 0L) return(matrix(numeric(), 0L, edge_model_n_embd(ctx)))
  edge_embeddings(ctx, x, normalize = normalize)
}

# Embedding models loaded by embed_edge(), by path and context size.
.edge_embed_models <- new.env(parent = emptyenv())

.edge_embed_model <- function(model_path, n_ctx) {
  if (!file.exists(model_path)) stop("Model file not found: ", model_path)
  key <- paste0(normalizePath(model_path, winslash = "/"), "@", as.integer(n_ctx))
  ctx <- .edge_embed_models[[key]]
  if (is.null(ctx) || !is_valid_model(ctx)) {
    ctx <- edge_load_model(model_path, n_ctx = as.integer(n_ctx), embeddings = TRUE)
    .edge_embed_models[[key]] <- ctx
  }
  ctx
}

# Null coalescing operator for plumber body parsing
`%||%` <- function(x, y) if (is.null(x)) y else x

# Threads used when edge_load_model() gets n_threads = NULL: one per physical
# core, since token generation slows down when hyperthreads share a core.
# 0 lets the C++ side fall back to all hardware threads.
.default_threads <- function() {
  n <- tryCatch(parallel::detectCores(logical = FALSE), error = function(e) NA_integer_)
  if (length(n) != 1 || is.na(n) || n < 1) n <- NA_integer_
  # In a container, no more threads than the CPUs it may use.
  quota <- .cpu_quota()
  if (!is.na(quota)) n <- min(n, max(1L, floor(quota)), na.rm = TRUE)
  if (is.na(n)) return(0L)
  return(as.integer(n))
}
