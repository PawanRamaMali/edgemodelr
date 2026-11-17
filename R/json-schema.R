# JSON Schema to GBNF, for edge_json_grammar(), edge_tool() and the
# response_format and tools of the OpenAI-compatible endpoints.
#
# The converter covers the subset of JSON Schema that structured output
# and tool definitions use: types (or a list of them), properties,
# required, additionalProperties, items, minItems/maxItems,
# minLength/maxLength, enum, const, anyOf/oneOf, a single allOf,
# nullable, local $ref to $defs or definitions, and the string formats
# date, time, date-time and uuid. Keywords it cannot enforce (pattern,
# minimum, ...) are an error rather than a grammar that ignores them.
#
# Rule names are built from the position of a value (field-2, item, alt-1,
# ref-Name), never from property names, so they only contain
# [a-zA-Z0-9-]. Properties are written in the order the schema lists them,
# and optional ones may be left out.

.json_types <- c("string", "number", "integer", "boolean", "null", "object", "array")

.schema_unsupported <- c(
  "pattern", "patternProperties", "minimum", "maximum", "exclusiveMinimum",
  "exclusiveMaximum", "multipleOf", "prefixItems", "uniqueItems", "contains",
  "minContains", "maxContains", "not", "if", "then", "else",
  "dependentRequired", "dependentSchemas", "dependencies", "propertyNames",
  "minProperties", "maxProperties", "unevaluatedProperties",
  "unevaluatedItems", "$dynamicRef", "$recursiveRef"
)

# Largest minItems/maxItems/minLength/maxLength: llama.cpp expands bounded
# repetitions into copies of the rule.
.schema_max_repeat <- 100L

.schema_error <- function(where, ...) {
  stop(where, ": ", ..., call. = FALSE)
}

# A string as JSON text, with the escapes JSON requires.
.json_quote <- function(s) {
  s <- enc2utf8(as.character(s))
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub("\"", "\\\"", s, fixed = TRUE)
  s <- gsub("\n", "\\n", s, fixed = TRUE)
  s <- gsub("\r", "\\r", s, fixed = TRUE)
  s <- gsub("\t", "\\t", s, fixed = TRUE)
  if (grepl("[[:cntrl:]]", s)) {
    chars <- strsplit(s, "")[[1]]
    ctl <- grepl("[[:cntrl:]]", chars)
    chars[ctl] <- sprintf("\\u%04x", vapply(chars[ctl], utf8ToInt, integer(1)))
    s <- paste(chars, collapse = "")
  }
  paste0("\"", s, "\"")
}

# A JSON scalar (as parsed from JSON or given in R) as JSON text.
.json_scalar <- function(v, where = "enum") {
  if (is.null(v)) return("null")
  if (is.list(v) || length(v) != 1L) {
    .schema_error(where, "enum and const values must be strings, numbers, booleans or null")
  }
  if (is.na(v)) return("null")
  if (is.logical(v)) return(if (v) "true" else "false")
  if (is.numeric(v)) return(as.character(v))
  .json_quote(v)
}

# A GBNF string literal that matches text exactly.
.gbnf_literal <- function(text) {
  text <- gsub("\\", "\\\\", text, fixed = TRUE)
  text <- gsub("\"", "\\\"", text, fixed = TRUE)
  text <- gsub("\n", "\\n", text, fixed = TRUE)
  text <- gsub("\r", "\\r", text, fixed = TRUE)
  text <- gsub("\t", "\\t", text, fixed = TRUE)
  paste0("\"", text, "\"")
}

# Collects the rules of one grammar. string_char is the character class of
# a string's contents; tool calls also leave out line breaks.
.gbnf_builder <- function(compact = FALSE, root = NULL, prefix = "",
                          string_char = "[^\"\\\\]") {
  b <- new.env(parent = emptyenv())
  b$compact <- isTRUE(compact)
  b$string_char <- string_char
  b$needed <- character()
  .gbnf_reset(b, root, prefix)
  b
}

# Starts the rules of a new root schema (one per tool) in the same grammar.
.gbnf_reset <- function(b, root, prefix = "") {
  b$root <- root
  b$prefix <- prefix
  b$rules <- character()
  b$refs <- character()
  b$resolving <- character()
  b$reserved <- character()
}

.gbnf_add <- function(b, name, body) {
  b$rules[name] <- body
  invisible(name)
}

.gbnf_need <- function(b, rules) {
  b$needed <- union(b$needed, rules)
}

.gbnf_comma <- function(b) if (b$compact) "\", \"" else "ws \",\" ws"
.gbnf_colon <- function(b) if (b$compact) "\": \"" else "ws \":\" ws"

# An expression as one element of a sequence.
.gbnf_group <- function(expr) {
  if (grepl(" | ", expr, fixed = TRUE)) paste0("( ", expr, " )") else expr
}

# An expression as a rule name, adding a rule when it is not one already.
.gbnf_named <- function(b, expr, name) {
  if (grepl("^[A-Za-z0-9-]+$", expr)) return(expr)
  .gbnf_add(b, name, expr)
}

# Repetition suffix for between lo and hi more copies (hi may be Inf).
.gbnf_repeat <- function(lo, hi) {
  if (lo == 0 && is.infinite(hi)) return("*")
  if (lo == 1 && is.infinite(hi)) return("+")
  if (lo == 0 && hi == 1) return("?")
  if (is.infinite(hi)) return(paste0("{", lo, ",}"))
  if (lo == hi) return(paste0("{", lo, "}"))
  paste0("{", lo, ",", hi, "}")
}

.schema_bounds <- function(s, lo_key, hi_key, where) {
  lo <- s[[lo_key]] %||% 0
  hi <- s[[hi_key]] %||% Inf
  ok <- function(x) is.numeric(x) && length(x) == 1L && !is.na(x) && x >= 0 && x == round(x)
  if (!ok(lo) || !ok(hi) || lo > hi) {
    .schema_error(where, lo_key, " and ", hi_key, " must be whole numbers with ",
                  lo_key, " <= ", hi_key)
  }
  if (lo > .schema_max_repeat || (is.finite(hi) && hi > .schema_max_repeat)) {
    .schema_error(where, lo_key, " and ", hi_key, " above ", .schema_max_repeat,
                  " are not supported")
  }
  c(lo, hi)
}

# The GBNF expression for a value matching schema s. Rules it needs are
# added to b with names starting with prefix.
.gbnf_value <- function(b, s, prefix, where) {
  if (isTRUE(s) || (is.list(s) && length(s) == 0L)) return(.gbnf_any(b))
  if (isFALSE(s)) .schema_error(where, "the schema false allows no value")
  if (!is.list(s) || is.null(names(s))) .schema_error(where, "a schema must be a JSON object")

  ref <- s[["$ref"]]
  if (!is.null(ref)) return(.gbnf_ref(b, ref, where))
  bad <- intersect(names(s), .schema_unsupported)
  if (length(bad)) {
    .schema_error(where, "the keyword '", bad[1], "' is not supported")
  }
  if (!is.null(s[["allOf"]])) {
    if (length(s[["allOf"]]) != 1L) {
      .schema_error(where, "allOf is supported only with a single schema")
    }
    s2 <- s[["allOf"]][[1]]
    if (isTRUE(s[["nullable"]]) && is.list(s2)) s2$nullable <- TRUE
    return(.gbnf_value(b, s2, prefix, where))
  }

  if ("const" %in% names(s)) {
    body <- .gbnf_literal(.json_scalar(s[["const"]], where))
  } else if ("enum" %in% names(s)) {
    values <- s[["enum"]]
    if (!is.list(values)) values <- as.list(values)
    if (length(values) == 0L) .schema_error(where, "enum must not be empty")
    body <- paste(vapply(values, function(v) .gbnf_literal(.json_scalar(v, where)),
                         character(1)), collapse = " | ")
  } else if (!is.null(s[["anyOf"]]) || !is.null(s[["oneOf"]])) {
    alts <- s[["anyOf"]] %||% s[["oneOf"]]
    if (!is.list(alts) || length(alts) == 0L || !is.null(names(alts))) {
      .schema_error(where, "anyOf and oneOf must be non-empty lists of schemas")
    }
    exprs <- vapply(seq_along(alts), function(k) {
      .gbnf_group(.gbnf_value(b, alts[[k]], paste0(prefix, "alt-", k, "-"),
                              paste0(where, " (alternative ", k, ")")))
    }, character(1))
    body <- paste(unique(exprs), collapse = " | ")
  } else {
    type <- s[["type"]]
    if (is.null(type)) {
      type <- if (!is.null(s[["properties"]]) || !is.null(s[["additionalProperties"]]) ||
                  !is.null(s[["required"]])) {
        "object"
      } else if (!is.null(s[["items"]])) {
        "array"
      }
    }
    if (is.null(type)) return(.gbnf_any(b))
    type <- unique(as.character(unlist(type)))
    unknown <- setdiff(type, .json_types)
    if (length(unknown) || length(type) == 0L) {
      .schema_error(where, "unsupported type '", unknown[1], "'; use ",
                    paste(.json_types, collapse = ", "))
    }
    exprs <- vapply(type, function(t) {
      sub_prefix <- if (length(type) > 1L) paste0(prefix, t, "-") else prefix
      .gbnf_typed(b, s, t, sub_prefix, where)
    }, character(1), USE.NAMES = FALSE)
    body <- if (length(exprs) > 1L) {
      paste(vapply(exprs, .gbnf_group, character(1)), collapse = " | ")
    } else {
      exprs
    }
  }
  if (isTRUE(s[["nullable"]]) && !identical(body, "null")) {
    .gbnf_need(b, "null")
    body <- paste0(.gbnf_group(body), " | null")
  }
  body
}

.gbnf_typed <- function(b, s, type, prefix, where) {
  switch(type,
         string = .gbnf_string(b, s, where),
         object = .gbnf_object(b, s, prefix, where),
         array = .gbnf_array(b, s, prefix, where),
         {
           .gbnf_need(b, type)
           type
         })
}

.gbnf_string <- function(b, s, where) {
  format <- s[["format"]]
  if (is.character(format) && length(format) == 1L) {
    quoted <- function(x) paste0("\"\\\"\" ", x, " \"\\\"\"")
    if (format == "date") {
      .gbnf_need(b, "date")
      return(quoted("date"))
    }
    if (format == "time") {
      .gbnf_need(b, "time")
      return(quoted("time"))
    }
    if (format == "date-time") {
      .gbnf_need(b, c("date", "time"))
      return(quoted("date \"T\" time"))
    }
    if (format == "uuid") {
      hex <- function(n) paste0("[0-9a-fA-F]{", n, "}")
      return(quoted(paste(hex(8), "\"-\"", hex(4), "\"-\"", hex(4), "\"-\"",
                          hex(4), "\"-\"", hex(12))))
    }
    # Other formats (email, uri, ...) are annotations: any string is allowed.
  }
  if (is.null(s[["minLength"]]) && is.null(s[["maxLength"]])) {
    .gbnf_need(b, "string")
    return("string")
  }
  n <- .schema_bounds(s, "minLength", "maxLength", where)
  paste0("\"\\\"\" ", b$string_char, .gbnf_repeat(n[1], n[2]), " \"\\\"\"")
}

.gbnf_object <- function(b, s, prefix, where) {
  props <- s[["properties"]]
  extra <- s[["additionalProperties"]]
  if (length(props) == 0L) {
    # Declared without properties: {}. Otherwise any keys, with values of
    # the additionalProperties schema.
    if (!is.null(props) || isFALSE(extra)) return("\"{}\"")
    value <- if (is.list(extra) && length(extra) > 0L) {
      .gbnf_named(b, .gbnf_value(b, extra, paste0(prefix, "value-"),
                                 paste0(where, " (additionalProperties)")),
                  paste0(prefix, "value"))
    } else {
      .gbnf_any(b)
    }
    .gbnf_need(b, "string")
    entry <- paste("string", .gbnf_colon(b), value)
    return(paste0("\"{\" ws ( ", entry, " ( ", .gbnf_comma(b), " ", entry,
                  " )* )? ws \"}\""))
  }
  keys <- names(props)
  if (!is.list(props) || is.null(keys) || any(!nzchar(keys)) || anyDuplicated(keys)) {
    .schema_error(where, "properties must be an object with unique names")
  }
  n <- length(props)
  req <- s[["required"]]
  # Without a required list every property is written: the output still
  # matches the schema, and small models fill fields more reliably.
  required <- if (is.null(req)) rep(TRUE, n) else keys %in% unlist(req)
  missing <- setdiff(as.character(unlist(req)), keys)
  if (length(missing)) {
    .schema_error(where, "required property '", missing[1], "' is not in properties")
  }

  kv <- character(n)
  for (i in seq_len(n)) {
    rule <- paste0(prefix, "field-", i)
    body <- .gbnf_value(b, props[[i]], paste0(rule, "-"), paste0(where, ".", keys[i]))
    .gbnf_add(b, rule, body)
    kv[i] <- if (b$compact) {
      paste0(.gbnf_literal(paste0(.json_quote(keys[i]), ": ")), " ", rule)
    } else {
      paste0(.gbnf_literal(.json_quote(keys[i])), " ws \":\" ws ", rule)
    }
  }
  comma <- .gbnf_comma(b)
  if (all(required)) {
    return(paste0("\"{\" ws ", paste(kv, collapse = paste0(" ", comma, " ")), " ws \"}\""))
  }

  # rest-k writes the fields from k on, after a field has been written.
  rest <- function(k) if (k > n) "" else paste0(" ", prefix, "rest-", k)
  for (k in rev(seq_len(n))[-n]) {
    item <- paste(comma, kv[k])
    if (!required[k]) item <- paste0("( ", item, " )?")
    .gbnf_add(b, paste0(prefix, "rest-", k), paste0(item, rest(k + 1L)))
  }
  # The first field written is an optional one or the first required one.
  first <- match(TRUE, required)
  last <- if (is.na(first)) n else first
  alts <- vapply(seq_len(last), function(j) paste0(kv[j], rest(j + 1L)), character(1))
  fields <- paste0("( ", paste(alts, collapse = " | "), " )", if (is.na(first)) "?" else "")
  paste0("\"{\" ws ", fields, " ws \"}\"")
}

.gbnf_array <- function(b, s, prefix, where) {
  items <- s[["items"]]
  if (is.list(items) && length(items) > 0L && is.null(names(items))) {
    .schema_error(where, "items must be a single schema; tuples are not supported")
  }
  n <- .schema_bounds(s, "minItems", "maxItems", where)
  if (n[2] == 0) return("\"[\" ws \"]\"")
  item <- .gbnf_value(b, items %||% TRUE, paste0(prefix, "item-"), paste0(where, "[]"))
  item <- .gbnf_named(b, item, paste0(prefix, "item"))
  inner <- item
  if (n[2] > 1) {
    inner <- paste0(item, " ( ", .gbnf_comma(b), " ", item, " )",
                    .gbnf_repeat(max(n[1] - 1, 0), n[2] - 1))
  }
  if (n[1] == 0) inner <- paste0("( ", inner, " )?")
  paste0("\"[\" ws ", inner, " ws \"]\"")
}

# Any JSON value.
.gbnf_any <- function(b) {
  .gbnf_need(b, c("json-value", "string", "number", "boolean", "null"))
  "json-value"
}

# The definition a local $ref points to, or NULL.
.schema_lookup <- function(root, ref) {
  m <- regmatches(ref, regexec("^#/(\\$defs|definitions)/(.+)$", ref))[[1]]
  if (length(m) == 0L) return(NULL)
  name <- gsub("~0", "~", gsub("~1", "/", m[3], fixed = TRUE), fixed = TRUE)
  defs <- root[[m[2]]]
  if (!is.list(defs)) return(NULL)
  defs[[name]]
}

.gbnf_ref <- function(b, ref, where) {
  if (!is.character(ref) || length(ref) != 1L ||
      !grepl("^#/(\\$defs|definitions)/.+$", ref)) {
    .schema_error(where, "only local $ref to #/$defs/... or #/definitions/... is supported")
  }
  if (ref %in% names(b$refs)) return(b$refs[[ref]])
  if (ref %in% b$resolving) {
    .schema_error(where, "the $ref '", ref, "' refers to itself; recursive schemas are not supported")
  }
  def <- .schema_lookup(b$root, ref)
  if (is.null(def)) .schema_error(where, "the $ref '", ref, "' is not defined")
  base <- gsub("[^A-Za-z0-9]", "", sub(".*/", "", ref))
  if (!nzchar(base)) base <- "def"
  rule <- paste0(b$prefix, "ref-", base)
  k <- 1L
  while (rule %in% c(b$refs, names(b$rules), b$reserved)) {
    k <- k + 1L
    rule <- paste0(b$prefix, "ref-", base, k)
  }
  b$reserved <- c(b$reserved, rule)
  b$resolving <- c(b$resolving, ref)
  body <- .gbnf_value(b, def, paste0(rule, "-"), ref)
  b$resolving <- setdiff(b$resolving, ref)
  .gbnf_add(b, rule, body)
  b$refs[ref] <- rule
  rule
}

# The shared rules the grammar's rules use, with ws first.
.gbnf_shared <- function(b) {
  ws <- if (b$compact) "ws ::= \"\"" else "ws ::= \"\" | \" \" | \"\\n\" [ \\t]{0,20}"
  need <- b$needed
  colon <- .gbnf_colon(b)
  comma <- .gbnf_comma(b)
  c(ws,
    if ("string" %in% need) paste0("string ::= \"\\\"\" ", b$string_char, "* \"\\\"\""),
    if ("number" %in% need) "number ::= \"-\"? [0-9]+ (\".\" [0-9]+)?",
    if ("integer" %in% need) "integer ::= \"-\"? [0-9]+",
    if ("boolean" %in% need) "boolean ::= \"true\" | \"false\"",
    if ("null" %in% need) "null ::= \"null\"",
    if ("date" %in% need) paste0(
      "date ::= [0-9]{4} \"-\" ( \"0\" [1-9] | \"1\" [0-2] ) \"-\" ",
      "( \"0\" [1-9] | [12] [0-9] | \"3\" [01] )"),
    if ("time" %in% need) paste0(
      "time ::= ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] \":\" [0-5] [0-9] ",
      "( \".\" [0-9]{1,6} )? ( \"Z\" | [+-] ( [01] [0-9] | \"2\" [0-3] ) \":\" [0-5] [0-9] )"),
    if ("json-value" %in% need) c(
      "json-value ::= json-object | json-array | string | number | boolean | null",
      paste0("json-object ::= \"{\" ws ( string ", colon, " json-value ( ", comma,
             " string ", colon, " json-value )* )? ws \"}\""),
      paste0("json-array ::= \"[\" ws ( json-value ( ", comma,
             " json-value )* )? ws \"]\""))
  )
}

# TRUE for a list that is a JSON Schema rather than edge_json_grammar()'s
# named list of fields: its type is object, array or null, or a list of
# types, or it has $ref, $defs, anyOf, oneOf or allOf.
.is_json_schema <- function(x) {
  if (!is.list(x) || is.null(names(x))) return(FALSE)
  type <- x[["type"]]
  if (is.character(type) && length(type) == 1L && !inherits(type, "AsIs") &&
      type %in% c("object", "array", "null")) {
    return(TRUE)
  }
  if (is.list(type)) return(TRUE)
  if (!is.null(x[["$ref"]]) || !is.null(x[["$defs"]]) || !is.null(x[["definitions"]])) {
    return(TRUE)
  }
  for (k in c("anyOf", "oneOf", "allOf")) {
    v <- x[[k]]
    if (is.list(v) && length(v) > 0L && is.null(names(v))) return(TRUE)
  }
  FALSE
}

.legacy_types <- c("string", "number", "integer", "boolean")

# One field of a named list of fields (edge_json_grammar(), edge_tool()):
# a type name, a character vector of allowed values, list(type =,
# description =) with one of those, or a JSON Schema. Returns the field's
# JSON Schema, the short form edge_tool() keeps in $parameters (the type
# name or allowed values when there is one) and its description.
.field_schema <- function(spec, name) {
  bad_type <- function() {
    stop("Unknown type for field '", name, "'. Use 'string', 'number', 'integer', ",
         "'boolean', a character vector of allowed values, or a JSON Schema list.",
         call. = FALSE)
  }
  if (is.character(spec)) {
    if (length(spec) == 0L || anyNA(spec)) bad_type()
    if (.is_enum(spec)) {
      return(list(schema = list(enum = as.list(unname(as.character(spec)))),
                  short = spec, note = ""))
    }
    type <- tolower(spec)
    if (!type %in% .legacy_types) bad_type()
    return(list(schema = list(type = type), short = type, note = ""))
  }
  if (!is.list(spec) || (length(spec) > 0L && is.null(names(spec)))) bad_type()
  note <- spec[["description"]]
  note <- if (is.character(note) && length(note) == 1L && !is.na(note)) note else ""
  type <- spec[["type"]]
  plain <- all(names(spec) %in% c("type", "description"))
  if (plain && is.character(type) && length(type) > 0L &&
      (inherits(type, "AsIs") || !all(type %in% .json_types) ||
       (length(type) == 1L && tolower(type) %in% .legacy_types))) {
    out <- .field_schema(type, name)
    out$note <- note
    if (nzchar(note)) out$schema$description <- note
    return(out)
  }
  # A string enum from a JSON Schema keeps the short form of allowed values.
  values <- spec[["enum"]]
  if (all(names(spec) %in% c("type", "description", "enum")) &&
      (is.null(type) || identical(type, "string")) && length(values) > 0L &&
      all(vapply(values, function(v) is.character(v) && length(v) == 1L && !is.na(v),
                 logical(1)))) {
    values <- as.character(unlist(values))
    short <- if (length(values) == 1L) I(values) else values
    return(list(schema = spec, short = short, note = note))
  }
  list(schema = spec, short = spec, note = note)
}

# A named list of fields as an object schema with every field required.
.fields_schema <- function(fields) {
  parts <- lapply(seq_along(fields), function(i) .field_schema(fields[[i]], names(fields)[i]))
  props <- lapply(parts, `[[`, "schema")
  names(props) <- names(fields)
  list(type = "object", properties = props, additionalProperties = FALSE)
}

# JSON text as a schema list.
.parse_schema_text <- function(x) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("The 'jsonlite' package is needed to read a schema given as JSON text.")
  }
  jsonlite::fromJSON(x, simplifyVector = FALSE)
}

# edge_json_grammar()'s schema argument as a JSON Schema.
.as_schema <- function(schema) {
  if (is.character(schema) && length(schema) == 1L && !.is_enum(schema) &&
      grepl("^\\s*[{]", schema)) {
    schema <- .parse_schema_text(schema)
    if (!is.list(schema)) stop("schema must be a JSON object")
    return(schema)
  }
  if (.is_json_schema(schema)) return(schema)
  if (!is.list(schema) || is.null(names(schema))) {
    stop("schema must be a named list of fields or a JSON Schema")
  }
  .fields_schema(schema)
}

# The GBNF grammar for values matching a JSON Schema, with root rule
# "root".
.json_schema_grammar <- function(schema, compact = FALSE, where = "schema") {
  b <- .gbnf_builder(compact, root = schema)
  root <- .gbnf_value(b, schema, "", where)
  lines <- c(paste0("root ::= ", root),
             if (length(b$rules)) paste0(names(b$rules), " ::= ", b$rules),
             .gbnf_shared(b))
  paste0(paste(lines, collapse = "\n"), "\n")
}

# A short description of a schema for prompts: "string", "array of
# object {name: string, qty?: integer}", 'one of "a", "b"'.
.describe_schema <- function(s, root = NULL, depth = 0L) {
  if (depth > 6L) return("...")
  if (!is.list(s) || length(s) == 0L) return("any")
  ref <- s[["$ref"]]
  if (is.character(ref) && length(ref) == 1L) {
    def <- .schema_lookup(root, ref)
    if (is.null(def)) return(sub(".*/", "", ref))
    return(.describe_schema(def, root, depth + 1L))
  }
  if (length(s[["allOf"]]) == 1L) return(.describe_schema(s[["allOf"]][[1]], root, depth + 1L))
  out <- if ("const" %in% names(s)) {
    tryCatch(.json_scalar(s[["const"]]), error = function(e) "a constant")
  } else if (!is.null(s[["enum"]])) {
    values <- tryCatch(vapply(as.list(s[["enum"]]), .json_scalar, character(1)),
                       error = function(e) "...")
    paste0("one of ", paste(values, collapse = ", "))
  } else if (!is.null(s[["anyOf"]]) || !is.null(s[["oneOf"]])) {
    alts <- s[["anyOf"]] %||% s[["oneOf"]]
    paste(vapply(alts, .describe_schema, character(1), root = root, depth = depth + 1L),
          collapse = " or ")
  } else {
    type <- as.character(unlist(s[["type"]]))
    if (length(type) == 0L) {
      type <- if (!is.null(s[["properties"]])) "object" else if (!is.null(s[["items"]])) "array" else "any"
    }
    parts <- vapply(type, function(t) {
      if (t == "array") {
        paste0("array of ", .describe_schema(s[["items"]], root, depth + 1L))
      } else if (t == "object" && length(s[["properties"]]) > 0L) {
        props <- s[["properties"]]
        req <- s[["required"]]
        req <- if (is.null(req)) names(props) else as.character(unlist(req))
        fields <- vapply(names(props), function(k) {
          paste0(k, if (!k %in% req) "?", ": ",
                 .describe_schema(props[[k]], root, depth + 1L))
        }, character(1))
        paste0("object {", paste(fields, collapse = ", "), "}")
      } else if (t == "string" && is.character(s[["format"]])) {
        paste0("string (", s[["format"]], ")")
      } else {
        t
      }
    }, character(1))
    paste(parts, collapse = " or ")
  }
  if (isTRUE(s[["nullable"]])) out <- paste(out, "or null")
  out
}

# A JSON parser for model output and tool calls, so edge_chat_completion()
# needs no extra package. Objects become named lists, arrays unnamed lists
# and null NULL, as jsonlite::fromJSON(simplifyVector = FALSE) gives them;
# whole numbers in integer range become integers.
.json_parse <- function(text) {
  p <- new.env(parent = emptyenv())
  p$c <- strsplit(enc2utf8(text), "")[[1]]
  p$i <- 1L
  value <- .json_parse_value(p)
  .json_parse_ws(p)
  if (p$i <= length(p$c)) stop("unexpected text after the JSON value")
  value
}

.json_parse_ws <- function(p) {
  while (p$i <= length(p$c) && p$c[p$i] %in% c(" ", "\t", "\n", "\r")) p$i <- p$i + 1L
}

.json_parse_value <- function(p) {
  .json_parse_ws(p)
  if (p$i > length(p$c)) stop("unexpected end of JSON")
  ch <- p$c[p$i]
  if (ch == "{") {
    p$i <- p$i + 1L
    out <- list()
    keys <- character()
    .json_parse_ws(p)
    if (identical(p$c[p$i], "}")) {
      p$i <- p$i + 1L
      return(structure(list(), names = character()))
    }
    repeat {
      .json_parse_ws(p)
      if (!identical(p$c[p$i], "\"")) stop("expected a key")
      keys <- c(keys, .json_parse_string(p))
      .json_parse_ws(p)
      if (!identical(p$c[p$i], ":")) stop("expected ':'")
      p$i <- p$i + 1L
      out[length(out) + 1L] <- list(.json_parse_value(p))
      .json_parse_ws(p)
      ch <- p$c[p$i]
      p$i <- p$i + 1L
      if (identical(ch, "}")) break
      if (!identical(ch, ",")) stop("expected ',' or '}'")
    }
    names(out) <- keys
    return(out)
  }
  if (ch == "[") {
    p$i <- p$i + 1L
    out <- list()
    .json_parse_ws(p)
    if (identical(p$c[p$i], "]")) {
      p$i <- p$i + 1L
      return(out)
    }
    repeat {
      out[length(out) + 1L] <- list(.json_parse_value(p))
      .json_parse_ws(p)
      ch <- p$c[p$i]
      p$i <- p$i + 1L
      if (identical(ch, "]")) break
      if (!identical(ch, ",")) stop("expected ',' or ']'")
    }
    return(out)
  }
  if (ch == "\"") return(.json_parse_string(p))
  for (word in c("true", "false", "null")) {
    end <- p$i + nchar(word) - 1L
    if (end <= length(p$c) && paste(p$c[p$i:end], collapse = "") == word) {
      p$i <- end + 1L
      return(switch(word, true = TRUE, false = FALSE, null = NULL))
    }
  }
  start <- p$i
  while (p$i <= length(p$c) && p$c[p$i] %in% c("-", "+", ".", "e", "E", 0:9)) p$i <- p$i + 1L
  num <- paste(p$c[start:(p$i - 1L)], collapse = "")
  if (p$i == start || !grepl("^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$", num)) {
    stop("invalid JSON value")
  }
  value <- as.numeric(num)
  if (grepl("^-?[0-9]+$", num) && abs(value) <= .Machine$integer.max) value <- as.integer(value)
  value
}

.json_parse_string <- function(p) {
  p$i <- p$i + 1L
  chars <- p$c
  n <- length(chars)
  parts <- character()
  start <- p$i
  i <- p$i
  repeat {
    if (i > n) stop("unterminated string")
    ch <- chars[i]
    if (ch == "\"") break
    if (ch == "\\") {
      if (i > start) parts <- c(parts, chars[start:(i - 1L)])
      esc <- if (i + 1L <= n) chars[i + 1L] else stop("unterminated string")
      if (esc == "u") {
        if (i + 5L > n) stop("bad \\u escape")
        code <- strtoi(paste(chars[(i + 2L):(i + 5L)], collapse = ""), 16L)
        i <- i + 6L
        if (!is.na(code) && code >= 0xD800 && code <= 0xDBFF && i + 5L <= n &&
            chars[i] == "\\" && chars[i + 1L] == "u") {
          low <- strtoi(paste(chars[(i + 2L):(i + 5L)], collapse = ""), 16L)
          if (!is.na(low) && low >= 0xDC00 && low <= 0xDFFF) {
            code <- 0x10000 + (code - 0xD800) * 1024 + (low - 0xDC00)
            i <- i + 6L
          }
        }
        if (is.na(code)) stop("bad \\u escape")
        parts <- c(parts, intToUtf8(code))
      } else {
        parts <- c(parts, switch(esc, n = "\n", t = "\t", r = "\r", b = "\b",
                                 f = "\f", "/" = "/", "\\" = "\\", "\"" = "\"",
                                 stop("bad escape")))
        i <- i + 2L
      }
      start <- i
      next
    }
    i <- i + 1L
  }
  if (i > start) parts <- c(parts, chars[start:(i - 1L)])
  p$i <- i + 1L
  out <- paste(parts, collapse = "")
  Encoding(out) <- "UTF-8"
  out
}

# Gives parsed JSON the R types its schema declares: integer fields become
# integers and number fields doubles, at any depth.
.json_coerce <- function(v, s, root, depth = 0L) {
  if (is.null(v) || !is.list(s) || depth > 50L) return(v)
  ref <- s[["$ref"]]
  if (is.character(ref) && length(ref) == 1L) {
    return(.json_coerce(v, .schema_lookup(root, ref), root, depth + 1L))
  }
  if (length(s[["allOf"]]) == 1L) return(.json_coerce(v, s[["allOf"]][[1]], root, depth + 1L))
  type <- as.character(unlist(s[["type"]]))
  if (is.numeric(v) && length(v) == 1L) {
    if ("integer" %in% type && !"number" %in% type) {
      if (v == round(v) && abs(v) <= .Machine$integer.max) v <- as.integer(v)
    } else if ("number" %in% type) {
      v <- as.numeric(v)
    }
    return(v)
  }
  if (is.list(v)) {
    if (!is.null(names(v))) {
      props <- s[["properties"]]
      for (k in names(v)) {
        sub <- if (is.list(props)) props[[k]] else NULL
        if (is.null(sub) && is.list(s[["additionalProperties"]])) sub <- s[["additionalProperties"]]
        if (!is.null(sub)) v[k] <- list(.json_coerce(v[[k]], sub, root, depth + 1L))
      }
    } else if (is.list(s[["items"]])) {
      v <- lapply(v, .json_coerce, s[["items"]], root, depth + 1L)
    }
  }
  v
}
