# Build on the existing API image so pip never runs (SSL inspection).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
WORKDIR /srv/whatsapp
RUN python3 -c "import fastapi, uvicorn" 2>/dev/null \
    || pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29"
COPY *.py ./
ENV PYTHONUNBUFFERED=1
EXPOSE 8022
ENTRYPOINT []
CMD ["python3", "-m", "uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8022"]
