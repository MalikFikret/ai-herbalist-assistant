# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Production Dockerfile for the AI Herbalist Assistant.
#
# This image is intended for PRODUCTION deployments only. Local development
# is expected to continue using the DevContainer provided by the course /
# instructor -- this file does not reference `.devcontainer/` and should not
# be used as one.
#
# Build:  docker build -t herbalist-assistant:latest .
# Run:    docker run --rm -p 8501:8501 --env-file .env herbalist-assistant:latest
# -----------------------------------------------------------------------------

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HA_HOME=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# System packages required by chromadb / sentence-transformers at runtime.
# build-essential only lives in the builder layer; the final image does not
# ship a C toolchain.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${HA_HOME}

# Install Python deps first for better layer caching. We install the package
# itself in editable-equivalent mode after copying the source tree below.
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project sources.
COPY pyproject.toml README.md ./
COPY src ./src
COPY .streamlit ./.streamlit
COPY langgraph.json ./

# Install the package so `import herbalist_assistant` works without PYTHONPATH.
RUN pip install --no-deps .

# Create a non-root user and make the runtime dirs writable so we can persist
# the Chroma DB and chat history at container runtime (mount volumes if you
# want them to survive container restarts).
RUN useradd --create-home --shell /usr/sbin/nologin herbalist \
    && mkdir -p ${HA_HOME}/data ${HA_HOME}/.chroma_db \
    && chown -R herbalist:herbalist ${HA_HOME}

USER herbalist

EXPOSE 8501

# Streamlit health probe. `/_stcore/health` returns 200 when the UI is ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${STREAMLIT_SERVER_PORT}/_stcore/health" || exit 1

ENTRYPOINT ["streamlit", "run", "src/herbalist_assistant/ui/streamlit_app.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false"]
