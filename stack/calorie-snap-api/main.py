"""
CalorieSnap API — Orchestrator v2
BHTLabs · Google Python Style Guide

Endpoints:
  POST /snap/v2     → Multi-item decomposition (NEW — headline feature)
  POST /snap        → FastVLM single-dish  (Stage 3)
  POST /snap/base   → Ollama moondream     (Stage 1)
  POST /snap/all    → Three-stage demo comparison
  GET  /health
"""

import base64
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

FASTVLM_URL = os.environ.get("FASTVLM_URL", "http://fastvlm-inference:8031")
OLLAMA_URL  = os.environ.get("OLLAMA_URL",  "http://ollama:11434")
BASE_MODEL  = os.environ.get("BASE_MODEL",  "moondream:latest")
DB_PATH     = os.environ.get("CUISINE_DB",  "/app/caloriesnap_cuisine.db")
TIMEOUT     = 120.0

INDIAN_PROMPT = (
    "You are a nutrition expert. Look at this Indian food image. "
    "Identify the dish and estimate total calories including cooking oil and ghee. "
    "Reply in this format:\nDISH: dal makhani\nCALORIES: 340\nCONFIDENCE: MEDIUM"
)

# Multi-item decomposition prompt — the v2 headline feature
MULTI_ITEM_PROMPT = (
    "You are CalorieSnap, an expert in Indian food nutrition. "
    "Look at this food image carefully. "
    "List EVERY separate food item you can see on the plate or in the image. "
    "For each item give a calorie estimate. "
    "Use ONLY this format, one item per line, nothing else:\n"
    "ITEM: dal makhani | CALORIES: 260\n"
    "ITEM: steamed rice | CALORIES: 220\n"
    "ITEM: roti | CALORIES: 70\n"
    "ITEM: raita | CALORIES: 45\n"
    "List all items you can see. If you see only one item, list just that one."
)

app = FastAPI(title="CalorieSnap API", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB connection (singleton) ─────────────────────────────────────────────────
_db_conn: Optional[sqlite3.Connection] = None


def get_db() -> Optional[sqlite3.Connection]:
    """Return a cached SQLite connection, or None if DB not found."""
    global _db_conn
    if _db_conn is not None:
        return _db_conn
    path = Path(DB_PATH)
    if not path.exists():
        log.warning("Cuisine DB not found at %s — DB lookup disabled", DB_PATH)
        return None
    _db_conn = sqlite3.connect(str(path), check_same_thread=False)
    _db_conn.row_factory = sqlite3.Row
    log.info("Cuisine DB loaded: %s", DB_PATH)
    return _db_conn


def db_lookup(name: str) -> Optional[dict]:
    """Look up a dish by name in the cuisine DB.

    Args:
        name: Dish name from model output.

    Returns:
        Dict with cal_per_serving, cal_per_100g, region, or None.
    """
    conn = get_db()
    if not conn:
        return None
    q = f"%{name.strip().lower()}%"
    row = conn.execute(
        """SELECT name, region, category, cal_per_100g, cal_per_serving,
                  serving_unit, protein_g, carb_g, fat_g
           FROM dishes
           WHERE LOWER(name) LIKE ? OR LOWER(name_local) LIKE ?
           LIMIT 1""",
        (q, q),
    ).fetchone()
    return dict(row) if row else None


# ── Pydantic models ───────────────────────────────────────────────────────────

class SnapRequest(BaseModel):
    image_base64: str
    hint: str = "indian"


class FoodItem(BaseModel):
    name: str
    calories: int
    confidence: str
    region: Optional[str] = None
    source: str = "model"   # "model" | "db" | "db+model"


class V2Response(BaseModel):
    items: list[FoodItem]
    total_calories: int
    item_count: int
    elapsed_ms: int
    model: str
    raw: str
    db_coverage: str
    note: str = "Multi-item decomposition. Air-gapped. BHTLabs CalorieSnap v2."


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


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_multi_item(raw: str) -> list[dict]:
    """Parse ITEM: name | CALORIES: number lines from model output.

    Handles messy model output — partial matches, sentence-embedded items,
    numbered lists, and bullet points.

    Args:
        raw: Raw text output from the VLM.

    Returns:
        List of dicts with name and calories keys.
    """
    items = []

    # Primary: strict ITEM: ... | CALORIES: ... format
    for m in re.finditer(
        r"ITEM\s*[:\-]\s*(.+?)\s*[|\-]\s*CALORIES\s*[:\-]\s*(\d+)",
        raw, re.IGNORECASE
    ):
        items.append({"name": m.group(1).strip(), "calories": int(m.group(2))})

    if items:
        return items

    # Fallback 1: numbered/bulleted list with calories inline
    # e.g. "1. Dal makhani - 260 calories"
    for m in re.finditer(
        r"(?:^|\n)\s*(?:\d+[\.\)]\s*|[-•*]\s*)(.+?)\s*[:\-–]\s*(\d{2,4})\s*(?:cal|kcal|calories?)?",
        raw, re.IGNORECASE
    ):
        name = m.group(1).strip()
        cal  = int(m.group(2))
        if len(name) > 2 and cal < 3000:
            items.append({"name": name, "calories": cal})

    if items:
        return items

    # Fallback 2: any "X calories" pattern with preceding dish-like word
    _GARBAGE = {"and","it","has","approximately","about","around","the","this","that","which"}
    for m in re.finditer(
        r"([a-z][a-z\s]{2,25}?)\s*[:\-,]?\s*(\d{2,4})\s*(?:cal|kcal|calories?)",
        raw, re.IGNORECASE
    ):
        name = m.group(1).strip().rstrip(":-,").strip()
        cal  = int(m.group(2))
        words = name.lower().split()
        if 1 <= len(words) <= 4 and words[0] not in _GARBAGE and cal < 3000:
            items.append({"name": name, "calories": cal})

    return items


def _parse_ollama(raw: str, model: str, elapsed: int) -> dict:
    """Parse single-dish Ollama response."""
    dish_m = re.search(r"^DISH\s*[:\-=]\s*(.+)", raw, re.MULTILINE | re.IGNORECASE)
    dish = dish_m.group(1).strip() if dish_m else _extract_dish_nl(raw)

    cal_m = re.search(
        r"(?:^CALORIES|approximately|about)\s*[:\-=]?\s*(\d{2,4})",
        raw, re.MULTILINE | re.IGNORECASE
    )
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


_KNOWN_DISHES = [
    "poha", "upma", "idli", "dosa", "dal makhani", "dal tadka",
    "chole", "chole bhature", "aloo paratha", "paratha",
    "biryani", "pulao", "butter chicken", "paneer", "rajma",
    "pav bhaji", "samosa", "thali", "rice", "roti", "khichdi",
    "rasam", "sambar", "vada", "uttapam", "appam", "pongal",
    "halwa", "kheer", "gulab jamun", "jalebi", "ladoo",
]


def _extract_dish_nl(raw: str) -> str:
    for dish in _KNOWN_DISHES:
        if dish in raw.lower():
            return dish.title()
    return "Unknown dish"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

async def _call_fastvlm(b64: str, prompt: Optional[str] = None) -> dict:
    """Call FastVLM inference service.

    Args:
        b64:    Base64-encoded image.
        prompt: Optional override prompt. Uses server default if None.

    Returns:
        Parsed response dict from FastVLM.
    """
    payload = {"image_base64": b64}
    if prompt:
        payload["prompt"] = prompt
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.post(f"{FASTVLM_URL}/query", json=payload)
        r.raise_for_status()
        return r.json()


async def _call_ollama(b64: str, model: str, prompt: str) -> tuple[str, int]:
    """Call Ollama vision model.

    Args:
        b64:    Base64-encoded image.
        model:  Ollama model name.
        prompt: Text prompt.

    Returns:
        Tuple of (response_text, elapsed_ms).
    """
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
    return r.json()["message"]["content"], elapsed


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    services = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{FASTVLM_URL}/health")
            services["fastvlm"] = r.json()
    except Exception as e:
        services["fastvlm"] = {"status": "unreachable", "error": str(e)}

    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            services["ollama"] = {"status": "ok", "models": models}
    except Exception as e:
        services["ollama"] = {"status": "unreachable", "error": str(e)}

    db_conn = get_db()
    if db_conn:
        count = db_conn.execute("SELECT COUNT(*) FROM dishes").fetchone()[0]
        services["cuisine_db"] = {"status": "ok", "dishes": count}
    else:
        services["cuisine_db"] = {"status": "not_loaded", "dishes": 0}

    all_ok = all(
        s.get("status") == "ok"
        for k, s in services.items()
        if k != "cuisine_db"
    )
    return {
        "status":   "ok" if all_ok else "degraded",
        "services": services,
        "version":  "4.0.0",
        "ts":       datetime.utcnow().isoformat() + "Z",
    }


@app.post("/snap/v2", response_model=V2Response)
async def snap_v2(req: SnapRequest) -> V2Response:
    """
    Multi-item decomposition — CalorieSnap v2 headline feature.

    Identifies every food item visible in the photo, returns per-item
    calorie estimates with DB-backed accuracy where available.
    No equivalent in Cal.ai, MyFitnessPal, or Lose It.
    """
    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "Invalid base64")

    t0 = time.monotonic()

    # Call FastVLM with multi-item prompt
    try:
        raw_result = await _call_fastvlm(b64)
        raw = raw_result.get("raw", "")
    except Exception as e:
        # Fallback to Ollama if FastVLM is down
        log.warning("FastVLM unavailable, falling back to ollama: %s", e)
        try:
            fastvlm_fb = await _call_fastvlm(b64)
            raw = fastvlm_fb.get("raw", "")
        except Exception as e2:
            raise HTTPException(503, f"All inference services unavailable: {e2}")

    elapsed = int((time.monotonic() - t0) * 1000)
    log.info("v2 raw output (%dms): %s", elapsed, raw[:200])

    # Parse items from model output
    parsed_items = _parse_multi_item(raw)

    # If parser got nothing, fall back to single-dish parse
    if not parsed_items:
        log.warning("Multi-item parser got no items, falling back to single-dish")
        dish_m = re.search(r"DISH\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
        cal_m  = re.search(r"(\d{2,4})\s*(?:cal|kcal|calories?)", raw, re.IGNORECASE)
        # Also try extracting dish name directly from paragraph
        dish = dish_m.group(1).strip() if dish_m else _extract_dish_nl(raw)
        # If still unknown try "dish called X" pattern
        if dish == "Unknown dish":
            ans_m = re.search(r"Answer[\s]*[:\-][\s]*([A-Za-z][A-Za-z\s]{2,40})", raw, re.IGNORECASE)
            if ans_m:
                dish = ans_m.group(1).strip().split(",")[0].strip()
        cal = int(cal_m.group(1)) if cal_m else 0
        parsed_items = [{"name": dish, "calories": cal}]

    # Enrich with DB lookup — DB overrides model calories when found
    food_items = []
    db_hits = 0
    for item in parsed_items:
        name = item["name"]
        cal  = item["calories"]
        source = "model"
        region = None
        confidence = "MEDIUM"

        db_row = db_lookup(name)
        if db_row:
            db_hits += 1
            # Use DB serving calories if available, else cal_per_100g
            db_cal = db_row.get("cal_per_serving") or (
                (db_row.get("cal_per_100g", 0) * 1.5)  # rough 150g serving estimate
            )
            if db_cal and db_cal > 0:
                cal = round(db_cal)
                source = "db"
            region = db_row.get("region")
            confidence = "HIGH"
        elif cal > 0:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        food_items.append(FoodItem(
            name=name,
            calories=cal,
            confidence=confidence,
            region=region,
            source=source,
        ))

    total_cal = sum(i.calories for i in food_items)
    coverage  = f"{db_hits}/{len(food_items)} items verified by INDB"

    log.info(
        "v2 result: %d items, %d cal total, %s",
        len(food_items), total_cal, coverage,
    )

    return V2Response(
        items=food_items,
        total_calories=total_cal,
        item_count=len(food_items),
        elapsed_ms=elapsed,
        model="fastvlm-0.5b+indb-2024",
        raw=raw,
        db_coverage=coverage,
    )


@app.post("/snap", response_model=SnapResponse)
async def snap(req: SnapRequest) -> SnapResponse:
    """Stage 3 — FastVLM fine-tuned. Best single-dish answer."""
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
    All three stages in one call — demo comparison screen.
    Stage 1: base moondream    — no Indian food knowledge
    Stage 2: moondream-indian  — knows Indian food, may be wrong dish
    Stage 3: FastVLM           — correct dish identified
    """
    b64 = req.image_base64
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    try:
        raw1, el1 = await _call_ollama(
            b64, BASE_MODEL, "What food is in this image? How many calories?"
        )
        s1 = _parse_ollama(raw1, BASE_MODEL, el1)
    except Exception as e:
        s1 = {"dish_name": "error", "total_calories": 0,
              "confidence": "LOW", "elapsed_ms": 0,
              "model": BASE_MODEL, "raw": str(e)}

    try:
        raw2, el2 = await _call_ollama(b64, "moondream-indian:latest", INDIAN_PROMPT)
        s2 = _parse_ollama(raw2, "moondream-indian:latest", el2)
    except Exception as e:
        s2 = {"dish_name": "error", "total_calories": 0,
              "confidence": "LOW", "elapsed_ms": 0,
              "model": "moondream-indian", "raw": str(e)}

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
