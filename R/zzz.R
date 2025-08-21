.onLoad <- function(libname, pkgname) {
  packageStartupMessage("edgemodelr: Local language model inference via llama.cpp")
  packageStartupMessage("Ensure you have GGUF model files for inference.")
}

.onUnload <- function(libpath) {
  library.dynam.unload("edgemodelr", libpath)
}