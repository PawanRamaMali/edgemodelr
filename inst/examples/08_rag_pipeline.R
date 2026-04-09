#!/usr/bin/env Rscript
# =============================================================================
# edgemodelr RAG Pipeline — End-to-End Test
# =============================================================================
#
# This script tests the complete RAG (Retrieval-Augmented Generation) pipeline:
#   1. Download / locate a model
#   2. Create sample documents
#   3. Build embedding index
#   4. Semantic search
#   5. Question answering with retrieved context
#   6. Batch classification and extraction
#
# Usage:
#   Rscript inst/examples/08_rag_pipeline.R
#
# Models tested:
#   - TinyLlama-1.1B (default, auto-downloaded ~637MB)
#   - Ollama qwen2.5:7b (if available, much better quality)
# =============================================================================

library(edgemodelr)

cat("
============================================================
   edgemodelr RAG Pipeline Test
============================================================
")

# --- Helper ---
divider <- function(title) {
  cat(sprintf("\n--- %s %s\n", title, strrep("-", max(1, 56 - nchar(title)))))
}

# =============================================================================
# Step 0: Locate or download a model
# =============================================================================
divider("Step 0: Model Setup")

model_path <- NULL
ctx <- NULL

# Check for a user-specified model via environment variable
env_model <- Sys.getenv("EDGEMODELR_TEST_MODEL", "")
if (nchar(env_model) > 0 && file.exists(env_model)) {
  model_path <- env_model
  cat(sprintf("Using user-specified model: %s\n", basename(model_path)))
}

# Default: TinyLlama-1.1B (fast enough for pipeline testing, ~637MB)
if (is.null(model_path)) {
  cache_dir <- tools::R_user_dir("edgemodelr", "cache")
  tiny_path <- file.path(cache_dir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

  if (file.exists(tiny_path)) {
    cat("Using cached TinyLlama-1.1B\n")
    model_path <- tiny_path
  } else {
    cat("Downloading TinyLlama-1.1B (637MB)...\n")
    setup <- edge_quick_setup("TinyLlama-1.1B")
    model_path <- setup$model_path
  }
}

cat(sprintf("Loading model: %s\n", basename(model_path)))
ctx <- tryCatch(
  edge_load_model(model_path, n_ctx = 2048L, n_gpu_layers = -1L, embeddings = TRUE),
  error = function(e) {
    cat("GPU offload failed, using CPU...\n")
    edge_load_model(model_path, n_ctx = 2048L, n_gpu_layers = 0L, embeddings = TRUE)
  }
)

cat(sprintf("Embedding dimension: %d\n", edge_model_n_embd(ctx)))

# =============================================================================
# Step 1: Create sample documents (simulated company reports)
# =============================================================================
divider("Step 1: Create Sample Documents")

docs_dir <- file.path(tempdir(), "edgemodelr_rag_test")
dir.create(docs_dir, showWarnings = FALSE, recursive = TRUE)

# Q3 Financial Report
writeLines(c(
  "Quarterly Financial Report - Q3 2025",
  "",
  "Revenue for Q3 2025 reached $42.8 million, representing a 15% increase",
  "over Q3 2024. The growth was primarily driven by the enterprise segment,",
  "which grew 28% year-over-year to $18.3 million. Consumer revenue remained",
  "flat at $24.5 million due to increased competition in the mobile market.",
  "",
  "Operating expenses totaled $35.2 million, up 8% from the prior year.",
  "The largest increase was in R&D spending, which rose to $14.1 million",
  "as the company invested heavily in its next-generation AI platform.",
  "",
  "Net income was $5.1 million, or $0.34 per diluted share, compared to",
  "$3.2 million ($0.22 per share) in Q3 2024. Cash and equivalents stood",
  "at $127.4 million at quarter end.",
  "",
  "The company raised its full-year revenue guidance to $165-170 million,",
  "up from the prior range of $158-163 million."
), file.path(docs_dir, "q3_financial_report.txt"))

# Product Launch
writeLines(c(
  "Product Update - October 2025",
  "",
  "The company launched EdgeAI Pro, its flagship enterprise product, on",
  "October 1st, 2025. EdgeAI Pro enables organizations to run large language",
  "models entirely on-premises with zero data leaving the network.",
  "",
  "Key features include:",
  "- Support for models up to 70B parameters on consumer GPUs",
  "- Automatic model quantization and optimization",
  "- REST API compatible with OpenAI format",
  "- Built-in RAG pipeline with document ingestion",
  "- Role-based access control and audit logging",
  "",
  "Early customer feedback has been overwhelmingly positive. The company",
  "signed 14 enterprise contracts in the first two weeks, including three",
  "Fortune 500 companies. Average contract value was $285,000 annually.",
  "",
  "The product is available starting at $5,000/month for teams under 50 users."
), file.path(docs_dir, "product_update.txt"))

# Team Update
writeLines(c(
  "Team and Operations Update - Q3 2025",
  "",
  "The engineering team grew to 128 members, up from 95 at the start of",
  "the year. Key hires included Dr. Sarah Chen as VP of AI Research",
  "(previously at DeepMind) and Marcus Rivera as Head of Infrastructure.",
  "",
  "The company opened a new office in Austin, Texas to complement the",
  "San Francisco headquarters. The Austin office will focus on the",
  "inference optimization team and customer success.",
  "",
  "Employee satisfaction scores reached 4.2 out of 5, the highest in",
  "company history. Voluntary turnover dropped to 8% annualized,",
  "well below the industry average of 15%.",
  "",
  "The board approved a new equity refresh program providing additional",
  "RSUs to all employees who have been with the company for 2+ years."
), file.path(docs_dir, "team_update.txt"))

cat(sprintf("Created %d sample documents in %s\n",
            length(list.files(docs_dir, pattern = "*.txt")), docs_dir))

# =============================================================================
# Step 2: Build embedding index
# =============================================================================
divider("Step 2: Build Embedding Index")

t0 <- Sys.time()
index <- edge_index_documents(docs_dir, ctx,
                               chunk_size = 400L,
                               chunk_overlap = 50L,
                               file_pattern = "*.txt")
t1 <- Sys.time()

print(index)
cat(sprintf("Index built in %.1f seconds\n", as.numeric(t1 - t0, units = "secs")))

# Show what chunks look like
cat("\nFirst 3 chunks (truncated):\n")
for (i in seq_len(min(3, index$n_chunks))) {
  snippet <- substr(index$chunks[i], 1, 80)
  cat(sprintf("  [%d] %s...\n", i, snippet))
}

# =============================================================================
# Step 3: Semantic Search
# =============================================================================
divider("Step 3: Semantic Search")

queries <- c(
  "What was the revenue in Q3?",
  "Who joined the leadership team?",
  "Tell me about the new product",
  "How many employees are there?"
)

for (q in queries) {
  cat(sprintf("\nQuery: \"%s\"\n", q))
  results <- edge_search(index, ctx, q, top_k = 3L)
  for (j in seq_len(nrow(results))) {
    snippet <- substr(results$chunk[j], 1, 70)
    cat(sprintf("  [%.3f] %s...\n", results$score[j], snippet))
  }
}

# =============================================================================
# Step 4: Similarity Matrix
# =============================================================================
divider("Step 4: Document Similarity")

# Embed the document titles/summaries for a similarity view
doc_summaries <- c(
  "quarterly financial revenue earnings",
  "product launch enterprise AI",
  "team hiring employees office"
)
doc_embs <- edge_embeddings(ctx, doc_summaries)
sim_mat <- edge_similarity_matrix(doc_embs)
rownames(sim_mat) <- c("Finance", "Product", "Team")
colnames(sim_mat) <- c("Finance", "Product", "Team")
cat("Document similarity matrix:\n")
print(round(sim_mat, 3))

# =============================================================================
# Step 5: RAG Question Answering
# =============================================================================
divider("Step 5: RAG Question Answering")

questions <- c(
  "What was the Q3 2025 revenue and how did it compare to last year?",
  "What is EdgeAI Pro and when was it launched?",
  "Who is the new VP of AI Research?"
)

for (q in questions) {
  cat(sprintf("\nQ: %s\n", q))
  t0 <- Sys.time()
  result <- edge_ask(ctx, q, index,
                      top_k = 3L,
                      n_predict = 150L,
                      temperature = 0.3,
                      return_context = TRUE)
  elapsed <- as.numeric(Sys.time() - t0, units = "secs")

  cat(sprintf("A: %s\n", result$answer))
  cat(sprintf("   (%.1fs, used %d context chunks from: %s)\n",
              elapsed,
              nrow(result$context),
              paste(basename(na.omit(unique(result$context$source))), collapse = ", ")))
}

# =============================================================================
# Step 6: Text Classification (grammar-constrained)
# =============================================================================
divider("Step 6: Text Classification")

review_texts <- c(
  "This product is absolutely amazing, best purchase I've ever made!",
  "Terrible quality, broke after two days. Complete waste of money.",
  "It works as expected, nothing special but gets the job done.",
  "Exceeded all my expectations, would recommend to everyone!"
)

cat("Classifying reviews into positive/negative/neutral...\n")
t0 <- Sys.time()
labels <- edge_classify(ctx, review_texts,
                         categories = c("positive", "negative", "neutral"),
                         temperature = 0.1)
elapsed <- as.numeric(Sys.time() - t0, units = "secs")

for (i in seq_along(review_texts)) {
  cat(sprintf("  [%s] %s\n", labels[i], substr(review_texts[i], 1, 60)))
}
cat(sprintf("Classified %d texts in %.1f seconds\n", length(review_texts), elapsed))

# =============================================================================
# Step 7: Structured Extraction
# =============================================================================
divider("Step 7: Structured Extraction")

cat("Extracting structured data from financial text...\n")
t0 <- Sys.time()
result <- edge_extract(ctx,
  "Revenue for Q3 2025 reached $42.8 million, a 15% increase year-over-year.",
  schema = list(
    revenue = "string",
    growth = "string",
    period = "string"
  ),
  temperature = 0.2)
elapsed <- as.numeric(Sys.time() - t0, units = "secs")

cat(sprintf("Extracted in %.1f seconds:\n", elapsed))
if (is.list(result)) {
  str(result)
} else {
  cat("  Raw output:", result, "\n")
}

# =============================================================================
# Step 8: Batch Map
# =============================================================================
divider("Step 8: Batch Map")

texts_to_summarize <- c(
  "The company reported strong Q3 earnings with revenue up 15%.",
  "A new product called EdgeAI Pro was launched for enterprise customers.",
  "The engineering team expanded to 128 members with key leadership hires."
)

cat("Generating one-line summaries...\n")
t0 <- Sys.time()
summaries <- edge_map(ctx, texts_to_summarize,
  "Rewrite this in exactly 5 words: {text}\nFive-word summary:",
  n_predict = 30L,
  temperature = 0.3)
elapsed <- as.numeric(Sys.time() - t0, units = "secs")

for (i in seq_along(summaries)) {
  cat(sprintf("  Input:   %s\n", substr(texts_to_summarize[i], 1, 65)))
  cat(sprintf("  Summary: %s\n\n", trimws(summaries[i])))
}
cat(sprintf("Mapped %d texts in %.1f seconds\n", length(texts_to_summarize), elapsed))

# =============================================================================
# Cleanup
# =============================================================================
divider("Cleanup")

edge_free_model(ctx)
unlink(docs_dir, recursive = TRUE)
cat("Model freed, temp files cleaned up.\n")

cat("
============================================================
   RAG Pipeline Test Complete
============================================================
")
