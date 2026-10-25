# syntax=docker/dockerfile:1

# ───────────── Étape 1 : construction de l'interface React ─────────────
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ───────────── Étape 2 : API FastAPI + interface construite ─────────────
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data "$HF_HOME" \
    && chown -R app:app /app /home/app
WORKDIR /app

# PyTorch en version CPU (≈ 200 Mo au lieu de plusieurs Go avec CUDA), puis le reste.
COPY requirements.docker.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.docker.txt

COPY --chown=app:app src/ ./src/
COPY --chown=app:app scripts/ ./scripts/
# FastAPI sert l'interface depuis frontend/dist (cf. src/api/app.py)
COPY --from=frontend --chown=app:app /build/dist ./frontend/dist

USER app
EXPOSE 8000

# Premier démarrage : les modèles Hugging Face se téléchargent, d'où le long start-period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

CMD ["python", "scripts/run_api.py", "--host", "0.0.0.0", "--port", "8000"]
