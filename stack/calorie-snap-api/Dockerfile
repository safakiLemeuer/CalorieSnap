# CalorieSnap API — Orchestrator
# BHTLabs · Lightweight — no GPU needed
# Routes /snap to FastVLM, /snap/base to Ollama moondream

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    httpx \
    Pillow \
    python-multipart

COPY main.py .

EXPOSE 8020

CMD ["python", "-m", "uvicorn", "main:app", \
     "--host", "0.0.0.0", "--port", "8020", "--reload"]
