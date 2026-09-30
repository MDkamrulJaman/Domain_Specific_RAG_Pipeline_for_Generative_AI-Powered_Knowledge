# syntax=docker/dockerfile:1
# Production image build and run:
#   docker build -t rag-knowledge-assistant .
#   docker run --rm -p 8000:8000 rag-knowledge-assistant
# Verify readiness at http://localhost:8000/health.
# Dependencies are installed in an isolated virtual environment in the builder stage; only that
# environment and the application source are copied into the slim runtime image. Uvicorn runs as
# the non-root appuser.

# =========================
# STAGE 1: Builder (dependency installation)
# =========================
FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/pip check

# =========================
# STAGE 2: Runtime (production image)
# =========================
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    GRADIO_ANALYTICS_ENABLED=False \
    PATH="/opt/venv/bin:$PATH"

RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid appuser --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
# Include only runtime source and UI assets, not local deployment metadata.
COPY --chown=appuser:appuser app/ /app/app/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import json, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); assert json.load(r)['status'] == 'healthy'" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--log-level", "info"]
