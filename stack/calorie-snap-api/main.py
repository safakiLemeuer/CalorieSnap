"""
CalorieSnap API — Orchestrator
BHTLabs · Google Python Style Guide

Routes food photo requests across all three demo stages:
  POST /snap        → FastVLM fine-tuned  (Stage 3 — correct answer)
  POST /snap/base   → Ollama moondream    (Stage 1 — no Indian knowledge)
  POST /snap/all    → All three stages in one call (for demo comparison)
  GET  /health
"""

import base64
import logging
import os
import time
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

FASTVLM_URL  = os.environ.get("FASTVLM_URL",  "http://fastvlm-inference:8031")
OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://ollama:11434")
BASE_MODEL   = os.environ.get("BASE_MODEL",   "moondream:latest")
TIMEOUT      = 120.0

INDIAN_PROMPT = (
    "You are a nutrition expert. Look at this Indian food image. "
    "Identify the dish and estimate total calories including cooking oil and ghee. "
    "Reply in this format:\nDISH: dal makhani\nCALORIES: 340\nCONFIDENCE: MEDIUM"
)

app = FastAPI(title="CalorieSnap API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SnapRequest(BaseModel):
    image_base64: str
    hint: str = "indian"


class StageResult(BaseModel):
    stage: int
    model: str
    dish_name: str
    total_calories: int
    confidence: str
    elapsed_ms: int
    raw: str


class SnapResponse(BaseModel):
    dish_name: str
    total_calories: int
    confidence: str
    elapsed_ms: int
    model: str
    raw: str
    note: str = "Air-gapped inference. Zero data sent to cloud. BHTLabs CalorieSnap."


class AllStagesResponse(BaseModel):
    stage1_base: StageResult
    stage2_prompt_tuned: StageResult
    stage3_fine_tuned: StageResult
    note: str = "Three stages of the same problem. Same photo. Same model family. Different training."


async def _call_fastvlm(b64: str) -> dict:
    """Call FastVLM fine-tuned inference service."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(
            f"{FASTVLM_URL}/query",
            json={"image_base64": b64},
        )
        r.raise_for_status()
        return r.json()


async def _call_ollama(b64: str, model: str, prompt: str) -> tuple[str, int]:
    """Call Ollama with an image."""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model":  model,
                "stream": False,
                "messages": [{
                    "role":    "user",
                    "content": prompt,
                    "images":  [b64],
                }],
            },
        )
        r.raise_for_status()
    elapsed = int((time.monotonic() - t0) * 1000)
    content = r.json()["message"]["content"]
    return content, elapsed


def _parse_ollama(raw: str, model: str, elapsed: int) -> dict:
    """Parse Ollama response into structured result."""
    import re

    dish_m = re.search(r"^DISH\s*[:\-=]\s*(.+)", raw, re.MULTILINE | re.IGNORECASE)
    dish = dish_m.group(1).strip() if dish_m else _extract_dish_nl(raw)

    cal_m = re.search(r"(?:^CALORIES|approximately|about)\s*[:\-=]?\s*(\d{2,4})",
                      raw, re.MULTILINE | re.IGNORECASE)
    if not cal_m:
        cal_m = re.search(r"(\d{2,4})\s*(?:cal|kcal|calories?)", raw, re.IGNORECASE)
    calories = int(cal_m.group(1)) if cal_m else 0

    conf_m = re.search(r"CONFIDENCE\s*[:\-=]\s*(LOW|MEDIUM|HIGH)", raw, re.IGNORECASE)
    confidence = conf_m.group(1).upper() if conf_m else ("MEDIUM" if calories > 0 else "LOW")

    return {
        "dish_name":      dish,
        "total_calories": calories,
        "confidence":     confidence,
        "elapsed_ms":     elapsed,
        "model":          model,
        "raw":            raw,
    }


_DISHES = [
    "poha", "upma", "idli", "dosa", "dal makhani", "dal tadka",
    "chole", "chole bhature", "aloo paratha", "paratha",
    "biryani", "pulao", "butter chicken", "paneer", "rajma",
    "pav bhaji", "samosa", "thali", "rice", "roti",
]


def _extract_dish_nl(raw: str) -> str:
    for dish in _DISHES:
        if dish in raw.lower():
            return dish.title()
    return "Unknown dish"


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    services = {}

    # Check FastVLM
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{FASTVLM_URL}/health")
            services["fastvlm"] = r.json()
    except Exception as e:
        services["fastvlm"] = {"status": "unreachable", "error": str(e)}

    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            services["ollama"] = {"status": "ok", "models": models}
    except Exception as e:
        services["ollama"] = {"status": "unreachable", "error": str(e)}

    all_ok = all(s.get("status") == "ok" for s in services.values())
    return {
        "status":   "ok" if all_ok else "degraded",
        "services": services,
        "ts":       datetime.utcnow().isoformat() + "Z",
    }


@app.post("/snap", response_model=SnapResponse)
async def snap(req: SnapRequest) -> SnapResponse:
    """Stage 3 — FastVLM fine-tuned. Best answer."""
    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64")

    try:
        result = await _call_fastvlm(b64)
        return SnapResponse(**result)
    except httpx.ConnectError:
        raise HTTPException(503, "FastVLM service unreachable")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/snap/base", response_model=SnapResponse)
async def snap_base(req: SnapRequest) -> SnapResponse:
    """Stage 1 — Base moondream. No Indian food knowledge."""
    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    try:
        raw, elapsed = await _call_ollama(
            b64, BASE_MODEL,
            "What food is in this image? How many calories does it have?"
        )
        parsed = _parse_ollama(raw, BASE_MODEL, elapsed)
        return SnapResponse(**parsed)
    except httpx.ConnectError:
        raise HTTPException(503, "Ollama unreachable")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/snap/all", response_model=AllStagesResponse)
async def snap_all(req: SnapRequest) -> AllStagesResponse:
    """
    All three stages in one call — for the demo comparison screen.
    Stage 1: base moondream — no Indian food knowledge
    Stage 2: moondream-indian prompt-tuned — knows Indian food, may be wrong dish
    Stage 3: FastVLM fine-tuned — correct answer
    """
    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    # Stage 1 — base moondream
    try:
        raw1, el1 = await _call_ollama(
            b64, BASE_MODEL,
            "What food is in this image? How many calories?"
        )
        s1 = _parse_ollama(raw1, BASE_MODEL, el1)
    except Exception as e:
        s1 = {"dish_name": "error", "total_calories": 0,
              "confidence": "LOW", "elapsed_ms": 0,
              "model": BASE_MODEL, "raw": str(e)}

    # Stage 2 — moondream-indian prompt-tuned
    try:
        raw2, el2 = await _call_ollama(b64, "moondream-indian:latest", INDIAN_PROMPT)
        s2 = _parse_ollama(raw2, "moondream-indian:latest", el2)
    except Exception as e:
        s2 = {"dish_name": "error", "total_calories": 0,
              "confidence": "LOW", "elapsed_ms": 0,
              "model": "moondream-indian", "raw": str(e)}

    # Stage 3 — FastVLM fine-tuned
    try:
        s3 = await _call_fastvlm(b64)
    except Exception as e:
        s3 = {"dish_name": "error", "total_calories": 0,
              "confidence": "LOW", "elapsed_ms": 0,
              "model": "fastvlm-0.5b", "raw": str(e)}

    return AllStagesResponse(
        stage1_base=StageResult(stage=1, **s1),
        stage2_prompt_tuned=StageResult(stage=2, **s2),
        stage3_fine_tuned=StageResult(stage=3, **s3),
    )


@app.get("/ping")
async def ping():
    return {"status": "ok", "ts": datetime.utcnow().isoformat() + "Z"}
