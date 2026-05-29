"""
FastVLM Inference Server
BHTLabs · Google Python Style Guide

Serves FastVLM 0.5B for Indian food identification and calorie estimation.
Uses LLaVA inference pipeline — the same codebase FastVLM is built on.
Model weights mounted at /model, FastVLM repo at /fastvlm.

Endpoints:
  POST /query   { image_base64: str } → { dish_name, total_calories, ... }
  GET  /health
"""

import base64
import io
import logging
import os
import re
import sys
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

MODEL_PATH   = Path(os.environ.get("MODEL_PATH",   "/model"))
FASTVLM_REPO = Path(os.environ.get("FASTVLM_REPO", "/fastvlm"))

PROMPT = (
    "You are CalorieSnap, an expert in Indian cuisine nutrition. "
    "Look at this food image carefully.\n"
    "Identify the Indian dish and estimate total calories including ghee and cooking oil.\n"
    "Reply in exactly this format:\n"
    "DISH: poha\n"
    "CALORIES: 220\n"
    "CONFIDENCE: HIGH"
)

_INDIAN_DISHES = [
    "poha", "upma", "idli", "dosa", "vada", "sambar", "rasam",
    "dal makhani", "dal tadka", "dal fry", "chana masala", "chole",
    "chole bhature", "aloo paratha", "paratha", "roti", "naan",
    "biryani", "pulao", "khichdi", "butter chicken", "chicken tikka",
    "chicken curry", "paneer butter masala", "palak paneer", "paneer",
    "rajma", "pav bhaji", "samosa", "pakora", "gulab jamun",
    "halwa", "kheer", "thali", "rice", "chapati",
]

app = FastAPI(title="FastVLM Inference", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_tokenizer = None
_image_processor = None
_device = None


def _extract_dish(raw: str) -> str:
    m = re.search(r"^DISH\s*[:\-=]\s*(.+)", raw, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    raw_lower = raw.lower()
    for dish in _INDIAN_DISHES:
        if dish in raw_lower:
            return dish.title()
    return "Unknown dish"


def _extract_calories(raw: str) -> int:
    for pattern in [
        r"^CALORIES\s*[:\-=]\s*(\d+)",
        r"(\d{2,4})\s*(?:cal|kcal|calories?)",
    ]:
        m = re.search(pattern, raw, re.MULTILINE | re.IGNORECASE)
        if m:
            return int(m.group(1))
    nums = re.findall(r"\b(\d{3,4})\b", raw)
    valid = [int(n) for n in nums if 50 <= int(n) <= 5000]
    return valid[0] if valid else 0


@app.on_event("startup")
async def startup():
    global _model, _tokenizer, _image_processor, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", _device)
    if _device == "cuda":
        log.info("GPU: %s — %.1fGB VRAM",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Add FastVLM repo and LLaVA subdir to path
    sys.path.insert(0, str(FASTVLM_REPO))
    sys.path.insert(0, str(FASTVLM_REPO / "llava"))

    log.info("Loading FastVLM from %s", MODEL_PATH)

    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_name = get_model_name_from_path(str(MODEL_PATH))
        log.info("Model name: %s", model_name)

        _tokenizer, _model, _image_processor, _ = load_pretrained_model(
            model_path=str(MODEL_PATH),
            model_base=None,
            model_name=model_name,
            device_map=_device,
        )
        _model.eval()
        log.info("FastVLM loaded — ready on :8031")

    except Exception as e:
        log.exception("Failed to load model: %s", e)
        raise


@app.post("/query")
async def query(req: "QueryRequest"):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        img.thumbnail((512, 512), Image.LANCZOS)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    try:
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates

        t0 = time.monotonic()

        # Process image
        image_tensor = process_images(
            [img], _image_processor, _model.config
        ).to(_device, dtype=torch.float16)

        # Build prompt
        if getattr(_model.config, "mm_use_im_start_end", False):
            prompt = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
                      + DEFAULT_IM_END_TOKEN + "\n" + PROMPT)
        else:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT

        conv = conv_templates["qwen_1_5"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            full_prompt, _tokenizer,
            IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(_device)

        with torch.inference_mode():
            output_ids = _model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[img.size],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=64,
                use_cache=True,
            )

        raw = _tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0].strip()

        elapsed = time.monotonic() - t0
        log.info("FastVLM %.1fs | %s", elapsed, raw[:150])

        dish      = _extract_dish(raw)
        calories  = _extract_calories(raw)
        conf_m    = re.search(
            r"CONFIDENCE\s*[:\-=]\s*(LOW|MEDIUM|HIGH)", raw, re.IGNORECASE
        )
        confidence = conf_m.group(1).upper() if conf_m else (
            "MEDIUM" if calories > 0 else "LOW"
        )

        return {
            "dish_name":      dish,
            "total_calories": calories,
            "confidence":     confidence,
            "elapsed_ms":     int(elapsed * 1000),
            "raw":            raw,
            "model":          "fastvlm-0.5b-caloriesnap",
        }

    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(500, f"Inference failed: {e}")


@app.get("/health")
async def health():
    return {
        "status":       "ok" if _model is not None else "loading",
        "model_loaded": _model is not None,
        "model_path":   str(MODEL_PATH),
        "device":       _device,
    }


class QueryRequest(BaseModel):
    image_base64: str
