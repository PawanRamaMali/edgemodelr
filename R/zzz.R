.onAttach <- function(libname, pkgname) {
  # Removed startup messages to comply with CRAN policies
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