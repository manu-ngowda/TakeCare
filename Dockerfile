# Dockerfile
# ──────────
# Builds a lean Python container for Cloud Run.
# Cloud Run expects the app to listen on PORT (default 8080).

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY main.py rag_pipeline.py rag_config.py rag_sanitize.py rag_rerank.py knowledge_base.json ./

# Cloud Run injects PORT env var — uvicorn must bind to it
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
