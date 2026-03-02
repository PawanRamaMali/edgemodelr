#!/usr/bin/env Rscript
# =============================================================================
# Qwen3-14B Demo  --  edgemodelr
#
# Covers:
#   1. Download the model (Q4_K_M, ~9 GB)
#   2. CPU inference (streaming chat)
#   3. GPU inference (Windows / NVIDIA)
#   4. Thinking-mode control (/no_think)
#   5. Quick benchmark
#
# Run:
#   Rscript examples/qwen3_14b_demo.R [--gpu] [--skip-download]
#
# Flags:
#   --gpu            Use GPU (NVIDIA CUDA required; run setup once first)
#   --skip-download  Skip downloading the model if already present
# =============================================================================

library(edgemodelr)

# ── Parse simple CLI flags ────────────────────────────────────────────────────
args         <- commandArgs(trailingOnly = TRUE)
USE_GPU      <- "--gpu"           %in% args
SKIP_DL      <- "--skip-download" %in% args

MODEL_URL  <- paste0(
  "https://huggingface.co/bartowski/Qwen3-14B-GGUF/resolve/main/",
  "Qwen3-14B-Q4_K_M.gguf"
)
MODEL_FILE <- "Qwen3-14B-Q4_K_M.gguf"

# ── Helper: print a section header ───────────────────────────────────────────
section <- function(title) {
  cat("\n")
  cat(strrep("=", 60), "\n")
  cat(" ", title, "\n")
  cat(strrep("=", 60), "\n\n")
}

# =============================================================================
# STEP 1 — Download model
# =============================================================================
section("Step 1: Download Qwen3-14B-Q4_K_M (~9 GB)")

if (SKIP_DL) {
  # Resolve where edgemodelr would have cached it
  cache_dir  <- tools::R_user_dir("edgemodelr", which = "cache")
  model_path <- file.path(cache_dir, MODEL_FILE)
  if (!file.exists(model_path)) {
    stop(
      "--skip-download was set but model not found at: ", model_path,
      "\nRemove the flag to download it."
    )
  }
  cat("Using cached model:", model_path, "\n")
} else {
  cat("Downloading model (this may take a while on first run)...\n")
  model_path <- edge_download_url(url = MODEL_URL, filename = MODEL_FILE)
  cat("Saved to:", model_path, "\n")
}

# =============================================================================
# STEP 2 — GPU setup (optional, Windows / NVIDIA only)
# =============================================================================
if (USE_GPU) {
  section("Step 2: Activate CUDA GPU backend")

  info <- edge_cuda_info()
  if (!info$installed) {
    cat("CUDA backend not found. Running one-time setup...\n\n")

    cat("  [1/2] Installing ggml-cuda DLL and companion libraries...\n")
    edge_install_cuda()

    cat("  [2/2] Installing CUDA runtime (nvcudart + cublas)...\n")
    cat("        nvcudart_hybrid64.dll is copied from the Windows DriverStore\n")
    cat("        cublas64_13.dll is downloaded from NVIDIA redistrib (~400 MB)\n")
    edge_install_cuda_toolkit()

    cat("\nSetup complete. Activating CUDA backend...\n")
  } else {
    cat("CUDA backend already installed.\n")
  }

  edge_reload_cuda()

  info <- edge_cuda_info()
  cat(sprintf("CUDA installed: %s  |  active: %s  |  path: %s\n",
              info$installed, info$active,
              if (is.null(info$path)) "(none)" else info$path))

  if (!info$active) stop("Failed to activate CUDA backend.")
}

# =============================================================================
# STEP 3 — Load model
# =============================================================================
section(if (USE_GPU) "Step 3: Load model (GPU, full offload)" else
                     "Step 3: Load model (CPU)")

load_start <- proc.time()[["elapsed"]]

ctx <- edge_load_model(
  model_path,
  n_gpu_layers = if (USE_GPU) -1L else 0L,  # -1 = all layers to VRAM
  n_ctx        = 8192L,                      # context window
  n_threads    = NULL                        # auto-detect CPU cores
)

load_time <- round(proc.time()[["elapsed"]] - load_start, 2)
cat(sprintf("Model loaded in %.1f s\n", load_time))

if (!is_valid_model(ctx)) stop("Model failed to load.")

# =============================================================================
# STEP 4 — CPU / GPU streaming chat
# =============================================================================
section("Step 4: Streaming chat completion")

messages <- list(
  list(role = "system",
       content = paste(
         "You are a helpful R programming assistant.",
         "Be concise and give working code examples."
       )),
  list(role = "user",
       content = paste(
         "Write a short R function that downloads a CSV from a URL,",
         "reads it into a data frame, and returns a summary."
       ))
)

prompt <- build_chat_prompt(messages)

cat("--- Response ---\n")
invisible(edge_stream_completion(
  ctx,
  prompt      = prompt,
  n_predict   = 3000L,   # Qwen3 uses ~1000+ tokens for <think> before answering
  temperature = 0.6,
  callback    = function(data) {
    if (!data$is_final) {
      cat(data$token)
      flush.console()
    }
    TRUE
  }
))
cat("\n\n")

# =============================================================================
# STEP 5 — Thinking-mode control
# =============================================================================
section("Step 5: Suppress thinking mode (/no_think)")

cat("Asking a quick maths question without Qwen3's internal CoT...\n\n")

no_think_msgs <- list(
  list(role = "system",  content = "/no_think"),
  list(role = "user",    content = "What is 17 * 23? Answer with a single number.")
)

no_think_prompt <- build_chat_prompt(no_think_msgs)

invisible(edge_stream_completion(
  ctx,
  prompt      = no_think_prompt,
  n_predict   = 50L,
  temperature = 0.0,   # greedy for factual questions
  callback    = function(data) {
    if (!data$is_final) { cat(data$token); flush.console() }
    TRUE
  }
))
cat("\n\n")

# =============================================================================
# STEP 6 — Benchmark
# =============================================================================
section("Step 6: Benchmark (128 tokens)")

bm <- edge_benchmark(ctx, n_tokens = 128L)
cat(sprintf("Prompt eval : %.1f tok/s\n", bm$prompt_tps))
cat(sprintf("Generation  : %.1f tok/s\n", bm$gen_tps))

# =============================================================================
# Cleanup
# =============================================================================
section("Cleanup")
edge_free_model(ctx)
cat("Model context released.\n")
