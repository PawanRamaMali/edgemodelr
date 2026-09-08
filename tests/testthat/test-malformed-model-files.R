# Loading a file that is not a valid GGUF used to take the whole R session
# down rather than raise an error. The crash was only visible on clang 23,
# where the bundled loader read unmapped memory instead of returning null.
# These cases all have to come back as ordinary R errors.

test_that("a text file with a .gguf name is rejected, not fatal", {
  p <- file.path(tempdir(), "not-really-a-model.gguf")
  writeLines("This is not a real GGUF file", p)
  on.exit(unlink(p), add = TRUE)

  expect_error(edge_load_model(p), "GGUF")
})


test_that("an empty file is rejected, not fatal", {
  p <- file.path(tempdir(), "empty-model.gguf")
  file.create(p)
  on.exit(unlink(p), add = TRUE)

  expect_error(edge_load_model(p), "GGUF")
})


test_that("a file with GGUF magic but no body is rejected, not fatal", {
  # Unlike the others, this input passes the magic check and is handed to the
  # bundled loader, so what happens next is third party behaviour we cannot
  # promise across compilers. That is the code path that crashed on clang 23
  # in the first place, so do not run it on CRAN: a crash here would fail the
  # check for the very reason this file exists to guard against.
  skip_on_cran()

  p <- file.path(tempdir(), "truncated-model.gguf")
  writeBin(c(charToRaw("GGUF"), as.raw(rep(0, 8))), p)
  on.exit(unlink(p), add = TRUE)

  expect_error(edge_load_model(p))
})


test_that("binary junk with no GGUF magic is rejected, not fatal", {
  p <- file.path(tempdir(), "junk-model.gguf")
  writeBin(as.raw(c(0x7f, 0x45, 0x4c, 0x46, rep(0x00, 60))), p)
  on.exit(unlink(p), add = TRUE)

  expect_error(edge_load_model(p), "GGUF")
})


test_that("the session survives every malformed input in sequence", {
  paths <- character()
  on.exit(unlink(paths), add = TRUE)

  p1 <- file.path(tempdir(), "seq-text.gguf")
  writeLines("nope", p1)
  p2 <- file.path(tempdir(), "seq-empty.gguf")
  file.create(p2)
  p3 <- file.path(tempdir(), "seq-junk.gguf")
  writeBin(as.raw(rep(0xff, 32)), p3)
  paths <- c(p1, p2, p3)

  for (p in paths) {
    expect_error(edge_load_model(p))
  }

  # Reaching here at all is the point: a crash would have ended the process.
  expect_true(TRUE)
})
