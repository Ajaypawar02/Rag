# syntax=docker/dockerfile:1
# Two images from one file (docker-compose builds both):
#   docker build --target api -t conversational-rag-api .
#   docker build --target ui  -t conversational-rag-ui  .

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --create-home --uid 1000 app
WORKDIR /app


# ---------------------------------------------------------------------------- #
# API: FastAPI + LangGraph + retrieval (also runs the one-shot seed job)
# ---------------------------------------------------------------------------- #
FROM base AS api

# CPU-only PyTorch first: the default wheel bundles ~2 GB of CUDA libraries this image never uses.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-api.txt

# Bake the embedding model into the image so containers start without downloading it.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY scripts ./scripts
COPY data/sample_docs ./data/sample_docs
COPY data/eval ./data/eval
RUN mkdir -p data/uploads && chown -R app:app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://localhost:8080/health', timeout=4).status != 200)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]


# ---------------------------------------------------------------------------- #
# UI: Streamlit (talks to the API over HTTP only)
# ---------------------------------------------------------------------------- #
FROM base AS ui

COPY ui/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY ui ./ui
USER app

EXPOSE 8501
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
