.onAttach <- function(libname, pkgname) {
  # Removed startup messages to comply with CRAN policies
}

.onUnload <- function(libpath) {
  library.dynam.unload("edgemodelr", libpath)
}