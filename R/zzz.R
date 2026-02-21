.onAttach <- function(libname, pkgname) {
  # Cache size check and optional auto-cleanup
  tryCatch({
    cache_dir <- tools::R_user_dir("edgemodelr", "cache")
    if (dir.exists(cache_dir)) {
      info <- edge_cache_info(cache_dir)
      max_size_mb <- getOption("edgemodelr.cache_max_size_mb", 5000)
      auto_clean <- isTRUE(getOption("edgemodelr.cache_auto_clean", FALSE))
      if (info$total_size_mb > max_size_mb) {
        if (auto_clean) {
          edge_clean_cache(cache_dir = cache_dir, interactive = FALSE, verbose = FALSE)
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
