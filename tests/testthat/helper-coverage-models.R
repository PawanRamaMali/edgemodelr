# Find a GGUF already on disk for the model-backed tests. Nothing is
# downloaded: when the file is not found the caller skips.
coverage_model_path <- function(filename) {
  runner_tmp <- Sys.getenv("RUNNER_TEMP", "")
  local_app <- Sys.getenv("LOCALAPPDATA", "")
  dirs <- c(
    Sys.getenv("EDGEMODELR_TEST_MODEL_DIR", ""),
    file.path(tempdir(), "edgemodelr_integration_tests"),
    if (nzchar(runner_tmp)) file.path(runner_tmp, "edgemodelr_models"),
    if (nzchar(runner_tmp)) runner_tmp,
    tools::R_user_dir("edgemodelr", "cache"),
    if (nzchar(local_app)) file.path(local_app, "edgemodelr-cache")
  )
  for (d in dirs[nzchar(dirs)]) {
    p <- file.path(d, filename)
    if (file.exists(p)) return(p)
  }
  NA_character_
}

coverage_tinyllama <- function() {
  coverage_model_path("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
}

coverage_qwen3_embedding <- function() {
  coverage_model_path("Qwen3-Embedding-0.6B-Q8_0.gguf")
}

skip_without_model <- function(path, name) {
  skip_on_cran()
  skip_if(is.na(path), paste(name, "is not on disk"))
}

# Point tools::R_user_dir("edgemodelr", "cache") at a fresh directory under
# tempdir() for the rest of the calling test, so nothing outside tempdir()
# is read or written.
local_temp_cache <- function(env = parent.frame()) {
  root <- tempfile("edgemodelr-cache-")
  dir.create(root)
  old <- Sys.getenv("R_USER_CACHE_DIR", unset = NA)
  Sys.setenv(R_USER_CACHE_DIR = root)
  cleanup <- bquote({
    if (is.na(.(old))) Sys.unsetenv("R_USER_CACHE_DIR")
    else Sys.setenv(R_USER_CACHE_DIR = .(old))
    unlink(.(root), recursive = TRUE)
  })
  do.call(on.exit, list(cleanup, add = TRUE), envir = env)
  file.path(tools::R_user_dir("edgemodelr", "cache"), "cuda")
}

# Restore PATH when the calling test ends (edge_activate_cuda() prepends to it).
local_path_restore <- function(env = parent.frame()) {
  cleanup <- bquote(Sys.setenv(PATH = .(Sys.getenv("PATH"))))
  do.call(on.exit, list(cleanup, add = TRUE), envir = env)
  invisible()
}
