# FastVLM Inference Service
# BHTLabs · CUDA 12.1 · RTX 3080
#
# Runs FastVLM 0.5B via LLaVA inference pipeline
# Model weights mounted from Windows host at /model
# FastVLM repo mounted at /fastvlm

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# LLaVA dependencies (FastVLM is built on LLaVA)
RUN pip install --no-cache-dir \
    transformers==4.37.2 \
    tokenizers==0.15.2 \
    accelerate \
    safetensors \
    Pillow \
    einops \
    timm \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    httpx \
    sentencepiece \
    shortuuid \
    ninja

COPY fastvlm_server.py .

EXPOSE 8031

CMD ["python3", "-m", "uvicorn", "fastvlm_server:app", \
     "--host", "0.0.0.0", "--port", "8031", "--workers", "1"]
