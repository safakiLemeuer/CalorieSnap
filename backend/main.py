"""
CalorieSnap Backend — Demo Ready
FastAPI + Moondream via Ollama
BHTLabs · June 5 2026 Demo
Google Python Style Guide compliant.
"""

import base64
import logging
import os
import re
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

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
MODEL      = os.environ.get("VISION_MODEL", "moondream-indian")
TIMEOUT    = 120.0

log.info(f"Ollama: {OLLAMA_URL} | Model: {MODEL}")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
INDIAN_PROMPT = """You are a nutrition expert. Look at this Indian food image.

Identify the dish and estimate total calories including cooking oil and ghee.

Reply in exactly this format:
DISH: dal makhani
CALORIES: 340
CONFIDENCE: MEDIUM"""

CALORIE_PROMPT = """Look at this food image carefully.

What food do you see? How many calories does it contain?

Answer in exactly this format:
DISH: poha
CALORIES: 250
CONFIDENCE: MEDIUM"""

app = FastAPI(title="CalorieSnap API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SnapRequest(BaseModel):
    image_base64: str
    hint: str = "indian"


class FoodItem(BaseModel):
    name: str
    portion_g: str
    calories: str


class SnapResponse(BaseModel):
    total_calories: int
    dish_name: str
    items: list[FoodItem]
    confidence: str
    note: str
    elapsed_ms: int
    model: str
    raw: str


# ---------------------------------------------------------------------------
# Dish extraction from natural language
# ---------------------------------------------------------------------------

_INDIAN_DISHES = [
    "poha", "upma", "idli", "dosa", "vada", "sambar", "rasam",
    "dal makhani", "dal tadka", "dal fry", "chana masala", "chole",
    "chole bhature", "bhature", "aloo paratha", "paratha", "roti", "naan",
    "biryani", "pulao", "khichdi", "pongal",
    "butter chicken", "chicken tikka", "chicken curry", "mutton curry",
    "paneer butter masala", "palak paneer", "paneer tikka", "paneer",
    "rajma", "kadhi", "baingan bharta", "aloo gobi", "matar paneer",
    "pav bhaji", "vada pav", "samosa", "pakora", "bhajiya",
    "halwa", "kheer", "gulab jamun", "jalebi", "ladoo",
    "thali", "rice", "chapati",
]


def _extract_dish_from_text(raw: str) -> str:
    """Extract dish name from natural language model output."""
    if not raw:
        return "Unknown dish"
    raw_lower = raw.lower()
    for dish in _INDIAN_DISHES:
        if dish in raw_lower:
            return dish.title()
    m = re.search(
        r"(?:plate|bowl|dish|serving)\s+of\s+([\w\s]{3,30}?)(?:\.|,|\s+which|\s+that|\s+with)",
        raw_lower,
    )
    if m:
        return m.group(1).strip().title()
    words = raw.strip().split()
    return " ".join(words[:4]).strip(".,") if words else "Unknown dish"


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> dict:
    """Parse Moondream output — structured format then natural language fallback."""
    log.info(f"Parsing raw: {raw[:300]}")

    m = re.search(r"^CALORIES\s*[:\-=]\s*(\d+)", raw, re.MULTILINE | re.IGNORECASE)
    if m:
        total = int(m.group(1))
        log.info(f"Strategy 1: {total}")
    else:
        m = re.search(r"TOTAL[_\s]CALORIES?\s*[:\-=]\s*(\d+)", raw, re.IGNORECASE)
        if m:
            total = int(m.group(1))
            log.info(f"Strategy 2: {total}")
        else:
            m = re.search(
                r"(?:approximately|about|around|total|contains?|has)\s+(\d{2,4})\s*(?:cal|kcal|calories?)",
                raw, re.IGNORECASE,
            )
            if m:
                total = int(m.group(1))
                log.info(f"Strategy 3: {total}")
            else:
                m = re.search(r"(\d{2,4})\s*(?:cal|kcal|calories?)", raw, re.IGNORECASE)
                if m:
                    total = int(m.group(1))
                    log.info(f"Strategy 4: {total}")
                else:
                    nums = re.findall(r"\b(\d{3,4})\b", raw)
                    valid = [int(n) for n in nums if 50 <= int(n) <= 5000]
                    total = valid[0] if valid else 0
                    log.info(f"Strategy 5: {total} from {nums}")

    dish_match = re.search(r"^DISH\s*[:\-=]\s*(.+)", raw, re.MULTILINE | re.IGNORECASE)
    dish_name = dish_match.group(1).strip() if dish_match else _extract_dish_from_text(raw)
    log.info(f"Dish: {dish_name} | Calories: {total}")

    conf_match = re.search(r"CONFIDENCE\s*[:\-=]\s*(LOW|MEDIUM|HIGH)", raw, re.IGNORECASE)
    confidence = conf_match.group(1).upper() if conf_match else ("MEDIUM" if total > 0 else "LOW")

    items = [FoodItem(name=dish_name, portion_g="estimated portion", calories=str(total))]

    return {
        "total_calories": total,
        "dish_name":      dish_name,
        "items":          items,
        "confidence":     confidence,
        "note":           "Air-gapped inference. Zero data sent to cloud. BHTLabs CalorieSnap.",
    }


# ---------------------------------------------------------------------------
# Ollama query
# ---------------------------------------------------------------------------

async def _query_ollama(image_b64: str, model: str, prompt: str) -> tuple[str, int]:
    """Send image to a model via Ollama."""
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
                    "images":  [image_b64],
                }],
            },
        )
        r.raise_for_status()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    content = r.json()["message"]["content"]
    log.info(f"[{model}] raw ({elapsed_ms}ms): {content[:500]}")
    return content, elapsed_ms


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        moondream_ready = any("moondream" in m.lower() for m in models)
        return {
            "status":           "ok",
            "ollama_reachable":  True,
            "moondream_ready":   moondream_ready,
            "available_models":  models,
            "model_in_use":     MODEL,
        }
    except Exception as e:
        return {"status": "error", "ollama_reachable": False, "error": str(e)}


@app.post("/snap", response_model=SnapResponse)
async def snap(req: SnapRequest) -> SnapResponse:
    """Fine-tuned moondream-indian — Indian cuisine aware."""
    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")
    try:
        raw, elapsed_ms = await _query_ollama(b64, MODEL, INDIAN_PROMPT)
    except httpx.ConnectError:
        raise HTTPException(503, "Cannot reach Ollama.")
    except httpx.TimeoutException:
        raise HTTPException(504, "Moondream timed out.")
    except Exception as e:
        raise HTTPException(500, f"Vision model error: {e}")

    parsed = _parse_response(raw)
    return SnapResponse(**parsed, elapsed_ms=elapsed_ms, model=MODEL, raw=raw)


@app.post("/snap/base", response_model=SnapResponse)
async def snap_base(req: SnapRequest) -> SnapResponse:
    """Base moondream — no fine-tuning. For before/after demo comparison."""
    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data.")
    try:
        raw, elapsed_ms = await _query_ollama(b64, "moondream:latest", CALORIE_PROMPT)
    except httpx.ConnectError:
        raise HTTPException(503, "Cannot reach Ollama.")
    except httpx.TimeoutException:
        raise HTTPException(504, "Moondream timed out.")
    except Exception as e:
        raise HTTPException(500, f"Vision model error: {e}")

    parsed = _parse_response(raw)
    return SnapResponse(**parsed, elapsed_ms=elapsed_ms, model="moondream:latest", raw=raw)


@app.get("/ping")
async def ping() -> dict:
    return {"status": "ok", "ts": datetime.utcnow().isoformat() + "Z"}
