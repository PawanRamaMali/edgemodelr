# =============================================================================
# edgemodelr — base image
#
# Installs R + edgemodelr from source. No model is pre-downloaded.
# Mount a local model cache or use the companion Dockerfile.tinyllama for
# a self-contained image with TinyLlama-1.1B baked in.
#
# Build:
#   docker build -f docker/Dockerfile -t edgemodelr .
#
# Run interactive R:
#   docker run --rm -it edgemodelr
#
# Run a script:
#   docker run --rm -v $(pwd)/examples:/scripts edgemodelr Rscript /scripts/qwen3_14b_demo.R
#
# Mount a local model cache (avoids re-downloading):
#   docker run --rm -it \
#     -v ~/.cache/R/edgemodelr:/root/.cache/R/edgemodelr \
#     edgemodelr
# =============================================================================

FROM rocker/r-ver:4.5.2

LABEL org.opencontainers.image.title="edgemodelr"
LABEL org.opencontainers.image.description="Local LLM inference in R — base image (no model)"
LABEL org.opencontainers.image.source="https://github.com/PawanRamaMali/edgemodelr"
LABEL org.opencontainers.image.licenses="MIT"

# ── System build dependencies ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        libcurl4-openssl-dev \
        libssl-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── R package dependencies ────────────────────────────────────────────────────
RUN install2.r --error --skipinstalled \
        Rcpp \
        curl \
    && rm -rf /tmp/downloaded_packages

# ── Install edgemodelr from source ───────────────────────────────────────────
WORKDIR /build
COPY . /build

RUN R CMD INSTALL --no-test-load . \
    && rm -rf /build

# ── Model cache directory (mount a volume here to reuse downloaded models) ───
RUN mkdir -p /root/.cache/R/edgemodelr
VOLUME ["/root/.cache/R/edgemodelr"]

WORKDIR /workspace

CMD ["R"]
