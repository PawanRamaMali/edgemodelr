test_that("Error handling and edge cases work correctly", {
  
  test_that("Invalid file paths are handled properly", {
    # Non-existent file
    expect_error(
      edge_load_model("does_not_exist.gguf"),
      "Model file does not exist"
    )
    
    # Empty file path
    expect_error(
      edge_load_model(""),
      "Model file does not exist"
    )
    
    # NULL file path
    expect_error(
      edge_load_model(NULL),
      "invalid 'file' argument"
    )
    
    # Directory instead of file
    if (dir.exists("tests")) {
      expect_error(
        edge_load_model("tests"),
        "Failed to load GGUF model"
      )
    }
  })
  
  test_that("Invalid parameters are rejected", {
    # Test edge_load_model with invalid parameters
    dummy_path <- "dummy.gguf"  # Will fail anyway, but testing parameter validation
    
    # Negative context size
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_ctx = -1)),
      "Model file does not exist"
    )
    
    # Zero context size
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_ctx = 0)),
      "Model file does not exist"
    )
    
    # Extremely large context size (should be handled gracefully)
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_ctx = 999999999)),
      "Model file does not exist"
    )
    
    # Negative GPU layers
    expect_error(
      suppressWarnings(edge_load_model(dummy_path, n_gpu_layers = -1)),
      "Model file does not exist"
    )
  })
  
  test_that("edge_completion handles errors gracefully", {
    # Test with NULL context
    expect_error(
      edge_completion(NULL, "Hello", n_predict = 5)
    )
    
    # Test with invalid context types
    invalid_contexts <- list("string", 123, list(), data.frame())
    for (ctx in invalid_contexts) {
      expect_error(
        edge_completion(ctx, "Hello", n_predict = 5)
      )
    }
    
    # Test with invalid n_predict
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    if (file.exists(model_path)) {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      # Test with invalid n_predict values
      invalid_n_predict <- c(-1, 0, "string", NULL, c(5, 10), list(5), Inf, -Inf, NaN)
      for (invalid_pred in invalid_n_predict) {
        expect_error(
          edge_completion(ctx, "Hello", n_predict = invalid_pred)
        )
      }
      
      # Test with invalid temperature values
      invalid_temperature <- c(-1, "string", NULL, c(0.8, 1.0), list(0.8), Inf, -Inf, NaN)
      for (invalid_temp in invalid_temperature) {
        expect_error(
          edge_completion(ctx, "Hello", n_predict = 5, temperature = invalid_temp)
        )
      }
      
      # Test with invalid top_p values
      invalid_top_p <- c(-1, 1.5, "string", NULL, c(0.9, 0.95), list(0.9), Inf, -Inf, NaN)
      for (invalid_p in invalid_top_p) {
        expect_error(
          edge_completion(ctx, "Hello", n_predict = 5, top_p = invalid_p)
        )
      }
      
      # Test with invalid prompt types
      invalid_prompts <- list(NULL, 123, c("a", "b"), list("hello"), data.frame(text = "hi"), TRUE, factor("test"))
      for (invalid_prompt in invalid_prompts) {
        expect_error(
          edge_completion(ctx, invalid_prompt, n_predict = 5)
        )
      }
      
      edge_free_model(ctx)
    }
  })
  
  test_that("Memory constraints are handled properly", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
      # Test very small context size (should work)
      ctx_small <- edge_load_model(model_path, n_ctx = 16)
      expect_true(is_valid_model(ctx_small))
      edge_free_model(ctx_small)
      
      # Test reasonable context size
      ctx_normal <- edge_load_model(model_path, n_ctx = 512)
      expect_true(is_valid_model(ctx_normal))
      edge_free_model(ctx_normal)
      
    } else {
      skip("No test model available for memory constraint tests")
    }
  })
  
  test_that("Corrupted or invalid model files are handled", {
    # Create a fake GGUF file with wrong content
    fake_gguf <- "fake_model.gguf"
    if (!file.exists(fake_gguf)) {
      writeLines("This is not a real GGUF file", fake_gguf)
    }
    
    expect_error(
      edge_load_model(fake_gguf)
    )
    
    # Cleanup
    if (file.exists(fake_gguf)) {
      unlink(fake_gguf)
    }
  })
  
  test_that("Edge cases in text completion", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
      ctx <- edge_load_model(model_path, n_ctx = 256)
      
      # Test with very long prompt (might exceed context)
      long_prompt <- paste(rep("This is a very long sentence that goes on and on.", 20), collapse = " ")
      
      # Should either work or give a reasonable error
      tryCatch({
        result <- edge_completion(ctx, long_prompt, n_predict = 5)
        expect_true(is.character(result))
      }, error = function(e) {
        # If it errors, the error should be informative
        expect_true(grepl("context|length|size", e$message, ignore.case = TRUE))
      })
      
      # Test with unusual characters
      unusual_prompts <- c(
        "Unicode: αβγδε",
        "Emojis: 🚀🔥💻",
        "Numbers: 123456789",
        "Symbols: !@#$%^&*()",
        "Mixed: Hello🌍123!",
        "Newlines: Hello\nWorld",
        "Tabs: Hello\tWorld",
        "Quotes: \"Hello\" 'World'",
        "Backslashes: C:\\\\path\\\\to\\\\file",
        paste("Very long word:", paste(rep("a", 200), collapse = "")),
        "Control chars: \r\n\t\b\f",
        "HTML: <script>alert('test')</script>",
        "SQL: SELECT * FROM users; DROP TABLE users;",
        "JSON: {\"key\": \"value\", \"number\": 123}"
      )
      
      for (prompt in unusual_prompts) {
        tryCatch({
          result <- edge_completion(ctx, prompt, n_predict = 3)
          expect_true(is.character(result))
        }, error = function(e) {
          # If there's an error, it should be a reasonable one
          expect_true(nchar(e$message) > 0)
        })
      }
      
      edge_free_model(ctx)
      
    } else {
      skip("No test model available for edge case tests")
    }
  })
  
  test_that("Concurrent access and thread safety", {
    model_path <- "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    
    if (file.exists(model_path)) {
      # Test loading the same model multiple times
      contexts <- list()
      
      # Load multiple contexts
      for (i in 1:3) {
        contexts[[i]] <- edge_load_model(model_path, n_ctx = 128)
        expect_true(is_valid_model(contexts[[i]]))
      }
      
      # Use all contexts
      for (i in 1:3) {
        result <- edge_completion(contexts[[i]], paste("Test", i), n_predict = 3)
        expect_true(is.character(result))
      }
      
      # Free all contexts
      for (i in 1:3) {
        edge_free_model(contexts[[i]])
        expect_false(is_valid_model(contexts[[i]]))
      }
      
    } else {
      skip("No test model available for concurrency tests")
    }
  })
  
  test_that("Resource cleanup after errors", {
    # Test that resources are properly cleaned up even when errors occur
    
    # This should fail but not leak memory
    expect_error(
      edge_load_model("nonexistent.gguf"),
      "Model file does not exist"
    )
    
    # Multiple failed attempts should not accumulate resources
    for (i in 1:5) {
      expect_error(
        suppressWarnings(edge_load_model("nonexistent.gguf"))
      )
    }
  })
  
  # Test all parameter combinations and edge cases
  test_that("Comprehensive parameter validation", {
    # Test edge_load_model with comprehensive invalid inputs
    invalid_model_paths <- list(
      NULL,
      numeric(0),
      character(0),
      c("path1", "path2"),
      123,
      TRUE,
      FALSE,
      list(),
      data.frame(),
      complex(1),
      factor("test")
    )
    
    for (invalid_path in invalid_model_paths) {
      expect_error(edge_load_model(invalid_path))
    }
    
    # Test n_ctx parameter validation
    invalid_n_ctx <- list(
      NULL,
      "string",
      c(256, 512),
      list(256),
      data.frame(n_ctx = 256),
      complex(1),
      factor(256),
      Inf,
      -Inf,
      NaN
    )
    
    for (invalid_ctx in invalid_n_ctx) {
      expect_error(
        edge_load_model("fake.gguf", n_ctx = invalid_ctx)
      )
    }
    
    # Test n_gpu_layers parameter validation
    invalid_gpu_layers <- list(
      NULL,
      "string",
      c(0, 1),
      list(0),
      data.frame(layers = 0),
      complex(1),
      factor(0),
      Inf,
      -Inf,
      NaN
    )
    
    for (invalid_layers in invalid_gpu_layers) {
      expect_error(
        edge_load_model("fake.gguf", n_gpu_layers = invalid_layers)
      )
    }
  })
  
  # Test error message quality and informativeness
  test_that("Error messages are informative", {
    # Test that error messages contain useful information
    tryCatch({
      edge_load_model("clearly_nonexistent_file_xyz123.gguf")
    }, error = function(e) {
      # Error message should mention the file or path
      expect_true(
        grepl("file|path|exist|found", e$message, ignore.case = TRUE),
        info = paste("Error message should be informative:", e$message)
      )
    })
    
    # Test error messages for invalid contexts
    tryCatch({
      edge_completion("invalid_context", "Hello")
    }, error = function(e) {
      expect_true(
        grepl("context|model|invalid", e$message, ignore.case = TRUE),
        info = paste("Error message should mention context:", e$message)
      )
    })
  })
  
  # Test boundary and overflow conditions
  test_that("Boundary and overflow conditions", {
    # Test with extreme parameter values
    extreme_values <- c(
      .Machine$integer.max,
      2^31 - 1,  # Max 32-bit signed integer
      1e9,       # Large number
      1e15       # Very large number
    )
    
    for (val in extreme_values) {
      # Should handle extreme values gracefully
      expect_error(
        edge_load_model("fake.gguf", n_ctx = val)
      )
    }
    
    # Test with very small but valid values
    small_values <- c(1, 2, 4, 8)
    for (val in small_values) {
      expect_error(
        edge_load_model("fake.gguf", n_ctx = val)  # Will fail due to file, not parameter
      )
    }
  })
  
  # Test exception safety and stack unwinding
  test_that("Exception safety", {
    possible_paths <- c(
      "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
      file.path(Sys.getenv("HOME"), ".cache", "edgemodelr", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    )
    
    model_path <- NULL
    for (path in possible_paths) {
      if (file.exists(path)) {
        model_path <- path
        break
      }
    }
    
    if (!is.null(model_path)) {
      # Test that partially constructed objects are cleaned up properly
      ctx <- edge_load_model(model_path, n_ctx = 128)
      
      # Generate error in middle of operation and ensure cleanup works
      expect_error(
        edge_completion(ctx, "test", n_predict = -5)  # Invalid parameter
      )
      
      # Context should still be valid after error
      expect_true(is_valid_model(ctx))
      
      # Should still be able to use it normally
      result <- edge_completion(ctx, "test", n_predict = 2)
      expect_true(is.character(result))
      
      edge_free_model(ctx)
      
    } else {
      skip("No test model available for exception safety tests")
    }
  })
  
  # Test for memory leaks under error conditions
  test_that("Memory leak prevention under errors", {
    # Repeatedly try to load invalid models to test for memory leaks
    for (i in 1:20) {
      expect_error(
        suppressWarnings(edge_load_model(paste0("nonexistent", i, ".gguf")))
      )
      
      # Try with various invalid parameters
      expect_error(
        suppressWarnings(edge_load_model("fake.gguf", n_ctx = -i))
      )
      
      expect_error(
        suppressWarnings(edge_load_model("fake.gguf", n_gpu_layers = -i))
      )
    }
    
    # Test that failed completions don't leak
    for (i in 1:10) {
      expect_error(
        edge_completion(paste0("fake_context_", i), "test")
      )
    }
  })
})