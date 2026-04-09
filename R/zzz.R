.onAttach <- function(libname, pkgname) {
  # Check SIMD optimization level and warn if running with generic (slow) code
  tryCatch({
    simd <- edge_simd_info()
    if (is.list(simd) && !is.null(simd$level)) {
      level <- toupper(simd$level)
      # Warn only if running generic (no SIMD) on x86_64
      if (level == "GENERIC" && .Platform$OS.type != "mac") {
        packageStartupMessage(
          "edgemodelr: Running without SIMD optimizations (generic mode). ",
          "Inference will be slower than optimal.\n",
          "For faster inference, reinstall from source with native optimizations:\n",
          "  Sys.setenv(EDGEMODELR_SIMD = 'NATIVE')\n",
          "  install.packages('edgemodelr', type = 'source')\n",
          "See edge_simd_info() for current SIMD status."
        )
      }
    }
  }, error = function(e) {
    # Silently ignore SIMD check errors
  })

  # Cache size check and optional auto-cleanup
  tryCatch({
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
    if (dir.exists(cache_dir)) {
      info <- edge_cache_info(cache_dir)
      max_size_mb <- getOption("edgemodelr.cache_max_size_mb", 5000)
      auto_clean <- isTRUE(getOption("edgemodelr.cache_auto_clean", FALSE))
      if (info$total_size_mb > max_size_mb) {
        if (auto_clean) {
          edge_clean_cache(cache_dir = cache_dir, ask = FALSE, verbose = FALSE)
        } else {
          packageStartupMessage(
            "edgemodelr cache size (", info$total_size_mb, " MB) exceeds limit (", max_size_mb,
            " MB). Use edge_clean_cache() or set options(edgemodelr.cache_auto_clean = TRUE)."
          )
        }
      }
    }
  }, error = function(e) {
    # Silently ignore cache errors on attach
  })
}

.onUnload <- function(libpath) {
  # Clean up any resources and suggest cache cleanup
  tryCatch({
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
    if (dir.exists(cache_dir)) {
      # Suggest cleanup but don't do it automatically
      packageStartupMessage("Note: Use edge_clean_cache() to manage cached model files")
    }
  }, error = function(e) {
    # Silently ignore cleanup errors during unload
  })
  
  library.dynam.unload("edgemodelr", libpath)
}
