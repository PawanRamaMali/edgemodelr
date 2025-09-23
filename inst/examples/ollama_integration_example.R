# Universal GGUF Model Integration with edgemodelr
# Complete guide to using ANY existing GGUF models (Ollama, local files, downloads, etc.)

library(edgemodelr)

cat("🔄 Universal GGUF Model Integration\n")
cat(rep("=", 60), "\n\n")

cat("This example shows how to find and reuse ANY GGUF models on your system,\n")
cat("including Ollama models, local downloads, and custom model directories.\n\n")

# Step 1: Check what Ollama models you have
cat("📋 Step 1: Checking Ollama Installation\n")
cat(rep("-", 40), "\n")

check_ollama_models <- function() {
  tryCatch({
    # Check if ollama command is available
    system_result <- system("ollama list", intern = TRUE, ignore.stderr = TRUE)

    if (length(system_result) > 1) {
      cat("✅ Ollama is installed and has models available:\n")
      cat(paste(system_result, collapse = "\n"), "\n\n")
      return(TRUE)
    } else {
      cat("❌ No Ollama models found or Ollama not installed\n\n")
      return(FALSE)
    }
  }, error = function(e) {
    cat("❌ Ollama command not found. Please install Ollama first:\n")
    cat("   Visit: https://ollama.ai/download\n\n")
    return(FALSE)
  })
}

ollama_available <- check_ollama_models()

# Step 2: Function to find Ollama model files
cat("🔍 Step 2: Locating Ollama Model Files\n")
cat(rep("-", 40), "\n")

find_ollama_models <- function() {
  # Common Ollama model storage locations
  possible_paths <- c(
    path.expand("~/.ollama/models"),           # Linux/macOS
    file.path(Sys.getenv("USERPROFILE"), ".ollama", "models"),  # Windows
    "/usr/share/ollama/models"                 # System-wide installation
  )

  for (path in possible_paths) {
    if (dir.exists(path)) {
      cat("✅ Found Ollama models directory:", path, "\n")

      # Look for blob files (where actual models are stored)
      blobs_dir <- file.path(path, "blobs")
      if (dir.exists(blobs_dir)) {
        blob_files <- list.files(blobs_dir, full.names = TRUE)
        cat("   Found", length(blob_files), "blob files\n")

        # Check which ones are GGUF format
        gguf_files <- c()
        for (blob in blob_files[1:min(10, length(blob_files))]) { # Check first 10
          if (file.exists(blob) && file.size(blob) > 1000) { # Skip small config files
            # Read first 4 bytes to check for GGUF magic
            con <- file(blob, "rb")
            header <- readBin(con, "raw", 4)
            close(con)

            if (length(header) == 4 && identical(header, charToRaw("GGUF"))) {
              gguf_files <- c(gguf_files, blob)
            }
          }
        }

        cat("   Found", length(gguf_files), "GGUF format models\n\n")
        return(list(base_path = path, blobs_dir = blobs_dir, gguf_files = gguf_files))
      }
    }
  }

  cat("❌ Could not locate Ollama models directory\n\n")
  return(NULL)
}

model_info <- find_ollama_models()

# Step 3: Copy Ollama models for edgemodelr use
cat("📁 Step 3: Preparing Models for edgemodelr\n")
cat(rep("-", 40), "\n")

setup_ollama_models <- function(model_info) {
  if (is.null(model_info)) {
    cat("❌ No Ollama models found to copy\n")
    return(NULL)
  }

  # Create a dedicated directory for edgemodelr models
  models_dir <- file.path(getwd(), "local_models")
  if (!dir.exists(models_dir)) {
    dir.create(models_dir, recursive = TRUE)
    cat("✅ Created models directory:", models_dir, "\n")
  }

  cat("💡 Note: We'll create symbolic links to save disk space\n")
  cat("   (or copy files if symbolic links aren't supported)\n\n")

  # Get Ollama model list to match blob files to model names
  if (ollama_available) {
    tryCatch({
      ollama_output <- system("ollama list", intern = TRUE, ignore.stderr = TRUE)

      if (length(ollama_output) > 1) {
        # Parse ollama list output to get model names
        model_lines <- ollama_output[-1] # Skip header
        model_names <- c()

        for (line in model_lines) {
          # Extract model name (first column)
          parts <- strsplit(trimws(line), "\\s+")[[1]]
          if (length(parts) > 0) {
            model_name <- gsub(":latest$", "", parts[1])
            model_names <- c(model_names, model_name)
          }
        }

        cat("📝 Available Ollama models to copy:\n")
        for (i in seq_along(model_names)) {
          cat(sprintf("   %d. %s\n", i, model_names[i]))
        }
        cat("\n")

        # For demonstration, we'll copy the first available model
        if (length(model_info$gguf_files) > 0 && length(model_names) > 0) {
          source_file <- model_info$gguf_files[1]
          target_file <- file.path(models_dir, paste0(model_names[1], ".gguf"))

          cat("📋 Copying model:", model_names[1], "\n")
          cat("   From:", source_file, "\n")
          cat("   To:", target_file, "\n")

          # Try to create symbolic link first, fall back to copy
          link_success <- FALSE
          if (.Platform$OS.type != "windows") {
            tryCatch({
              file.symlink(source_file, target_file)
              link_success <- TRUE
              cat("✅ Created symbolic link (saves disk space)\n")
            }, error = function(e) {
              cat("⚠️ Symbolic link failed, will copy file instead\n")
            })
          }

          if (!link_success) {
            if (file.copy(source_file, target_file)) {
              cat("✅ Model copied successfully\n")
            } else {
              cat("❌ Failed to copy model\n")
              return(NULL)
            }
          }

          return(list(
            model_path = target_file,
            model_name = model_names[1],
            models_dir = models_dir
          ))
        }
      }
    }, error = function(e) {
      cat("❌ Error processing Ollama models:", e$message, "\n")
    })
  }

  return(NULL)
}

copied_model <- setup_ollama_models(model_info)

# Step 4: Load and test the model with edgemodelr
cat("🚀 Step 4: Loading Model with edgemodelr\n")
cat(rep("-", 40), "\n")

test_ollama_model <- function(copied_model) {
  if (is.null(copied_model)) {
    cat("❌ No model available to test\n")
    return()
  }

  cat("Loading model:", copied_model$model_name, "\n")
  cat("Path:", copied_model$model_path, "\n\n")

  tryCatch({
    # Load the model
    ctx <- edge_load_model(
      model_path = copied_model$model_path,
      n_ctx = 2048,        # Context size
      n_gpu_layers = 0     # Use CPU for compatibility
    )

    # Verify model loaded successfully
    if (is_valid_model(ctx)) {
      cat("✅ Model loaded successfully!\n\n")

      # Test 1: Basic completion
      cat("🧪 Test 1: Basic Text Completion\n")
      prompt1 <- "The benefits of using R for data analysis include"
      result1 <- edge_completion(ctx, prompt1, n_predict = 80, temperature = 0.7)
      cat("Prompt:", prompt1, "\n")
      cat("Response:", result1, "\n\n")

      # Test 2: Question answering
      cat("🧪 Test 2: Question Answering\n")
      prompt2 <- "What is the difference between a data frame and a matrix in R?"
      result2 <- edge_completion(ctx, prompt2, n_predict = 120, temperature = 0.3)
      cat("Question:", prompt2, "\n")
      cat("Answer:", result2, "\n\n")

      # Test 3: Code generation
      cat("🧪 Test 3: R Code Generation\n")
      prompt3 <- "Write R code to create a histogram of random normal data:"
      result3 <- edge_completion(ctx, prompt3, n_predict = 150, temperature = 0.2)
      cat("Request:", prompt3, "\n")
      cat("Generated code:", result3, "\n\n")

      # Test 4: Streaming completion (if desired)
      cat("🧪 Test 4: Streaming Response Demo\n")
      cat("Prompt: Tell me a short story about data science\n")
      cat("Response: ")

      # Simple streaming example
      edge_stream_completion(
        ctx,
        "Tell me a short story about data science in 3 sentences:",
        n_predict = 100,
        temperature = 0.8,
        callback = function(token) {
          cat(token)
          flush.console()
          TRUE # Continue streaming
        }
      )
      cat("\n\n")

      # Clean up
      edge_free_model(ctx)
      cat("✅ Model cleaned up successfully\n\n")

    } else {
      cat("❌ Model validation failed\n")
    }

  }, error = function(e) {
    cat("❌ Error testing model:", e$message, "\n")
  })
}

if (ollama_available) {
  test_ollama_model(copied_model)
} else {
  cat("⏭️ Skipping model test - Ollama not available\n\n")
}

# Step 5: Advanced integration tips
cat("💡 Step 5: Advanced Integration Tips\n")
cat(rep("-", 40), "\n")

cat("🔧 Performance Optimization:\n")
cat("• Use n_gpu_layers > 0 if you have a compatible GPU\n")
cat("• Adjust n_ctx based on your memory and needs (512-4096)\n")
cat("• Lower n_predict values for faster responses\n\n")

cat("💾 Storage Management:\n")
cat("• Use symbolic links to avoid duplicating large model files\n")
cat("• Keep original Ollama models and just link to them\n")
cat("• Use edge_clean_cache() to manage edgemodelr's own cache\n\n")

cat("🔄 Model Switching:\n")
cat("• You can switch between different Ollama models easily\n")
cat("• Load different models for different tasks (coding vs. creative writing)\n")
cat("• Always call edge_free_model() before loading a new model\n\n")

# Step 6: Automation function
cat("⚙️ Step 6: Automation Helper Function\n")
cat(rep("-", 40), "\n")

cat("Here's a helper function to automate the entire process:\n\n")

setup_ollama_integration <- function(model_name = NULL) {
  cat("🔄 Setting up Ollama integration...\n")

  # Find available models
  model_info <- find_ollama_models()
  if (is.null(model_info)) return(NULL)

  # Get Ollama model list
  ollama_models <- system("ollama list", intern = TRUE, ignore.stderr = TRUE)
  if (length(ollama_models) <= 1) {
    cat("❌ No Ollama models found\n")
    return(NULL)
  }

  # Setup models directory
  models_dir <- file.path(getwd(), "local_models")
  if (!dir.exists(models_dir)) dir.create(models_dir, recursive = TRUE)

  # If no specific model requested, use the first available
  if (is.null(model_name)) {
    model_lines <- ollama_models[-1]
    first_model <- strsplit(trimws(model_lines[1]), "\\s+")[[1]][1]
    model_name <- gsub(":latest$", "", first_model)
  }

  cat("✅ Setup complete. Use edge_load_model() with your model files.\n")
  return(models_dir)
}

cat("Usage example:\n")
cat("  models_dir <- setup_ollama_integration()\n")
cat("  model_path <- file.path(models_dir, 'your_model.gguf')\n")
cat("  ctx <- edge_load_model(model_path)\n\n")

# Step 7: Troubleshooting guide
cat("🔧 Step 7: Troubleshooting\n")
cat(rep("-", 40), "\n")

troubleshooting_guide <- data.frame(
  Issue = c(
    "Model not found",
    "Loading fails",
    "Out of memory",
    "Slow performance",
    "Symbolic links fail"
  ),
  Solution = c(
    "Check Ollama installation and run 'ollama list'",
    "Verify GGUF format with hexdump or file command",
    "Reduce n_ctx parameter (try 512 or 1024)",
    "Use smaller model or reduce n_predict",
    "Copy files instead of creating symbolic links"
  ),
  stringsAsFactors = FALSE
)

print(troubleshooting_guide)

cat("\n🎉 Ollama Integration Complete!\n")
cat("\nBenefits of this approach:\n")
cat("✅ No duplicate model downloads\n")
cat("✅ Saves disk space (up to 30GB+ with symbolic links)\n")
cat("✅ Use familiar models you already have\n")
cat("✅ Consistent performance between Ollama and edgemodelr\n")
cat("✅ Easy switching between different models\n\n")

cat("Next steps:\n")
cat("• Try different models for different tasks\n")
cat("• Experiment with streaming: source('inst/examples/llama32_streaming_example.R')\n")
cat("• Explore document analysis with your existing models\n")
cat("• Set up automated workflows using your preferred Ollama models\n")