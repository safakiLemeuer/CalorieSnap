"""CalorieSnap API v3.

Pipeline:
    photo -> vision model (Foundry cloud or air-gapped FastVLM)
          -> items[] -> INDB matching (alias, exact, model-picked candidate)
          -> calories from INDB only, with a data plausibility guard
          -> telemetry row -> response

Endpoints:
    GET  /                       single-page UI (Snap, Log, Admin)
    GET  /api/health             configuration state
    POST /api/snap               identify and verify one photo
    POST /api/feedback           user verdict on an item (the flywheel)
    GET  /api/dishes?q=          INDB autocomplete
    GET  /api/history            recent meals
    GET  /api/admin/summary      KPIs and insights
    GET  /api/admin/match?dish=  explain how a name is matched to INDB
    POST /api/admin/alias        pin or clear a name-to-INDB match
    GET  /api/admin/events       drill-down and review queue
    GET  /api/admin/export.csv   labeled dataset
"""

import base64
import csv
import hashlib
import io
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

import config
import indb
import matching
import telemetry
import vision
from errors import PipelineError

_HTML_PATH = os.path.join(config.BASE_DIR, "static", "ui.html")
_MAX_IMAGE_B64 = 4_000_000  # ~3 MB decoded. The UI resizes to 768 px.
_MAX_THUMB_B64 = 40_000

app = FastAPI(title="CalorieSnap API", version="3.6.0")


class SnapRequest(BaseModel):
    """Body of POST /api/snap."""

    image_b64: str
    media_type: str = "image/jpeg"
    thumb_b64: Optional[str] = None
    model: Optional[str] = None
    source: str = "web"  # Channel: web, whatsapp, ios.


class AliasRequest(BaseModel):
    """Body of POST /api/admin/alias. indb_name=None clears the alias."""

    dish: str
    indb_name: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Body of POST /api/feedback."""

    event_id: int
    verdict: str
    item_id: Optional[int] = None
    corrected_dish: Optional[str] = None
    corrected_portion_g: Optional[float] = None


def _cost(model: str, tok_in: int, tok_out: int, picker: str,
          pick_in: int, pick_out: int) -> Optional[float]:
    """USD for one snap. None when the vision model has no price set."""
    prices = config.prices()
    if model not in prices:
        return None
    pick_price = prices.get(picker, [0.0, 0.0])
    return (tok_in * prices[model][0] + tok_out * prices[model][1]
            + pick_in * pick_price[0] + pick_out * pick_price[1]) / 1e6


@app.get("/", response_class=HTMLResponse)
def ui_page() -> HTMLResponse:
    """Serves the single-page UI. Never cached."""
    try:
        with open(_HTML_PATH, "r", encoding="utf-8") as handle:
            return HTMLResponse(handle.read(),
                                headers={"cache-control": "no-store"})
    except OSError:
        raise HTTPException(status_code=500,
                            detail=f"UI file missing: {_HTML_PATH}")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """Reports configuration state for the UI and for monitoring."""
    return {
        "ok": True,
        "version": app.version,
        "foundry_configured": config.cloud_configured(),
        "local_configured": bool(config.cfg("FASTVLM_URL")),
        "models": config.models(),
        "indb_dishes": indb.dish_count(),
    }


@app.post("/api/snap")
def snap(req: SnapRequest) -> JSONResponse:
    """Runs the pipeline on one photo and logs the outcome either way."""
    started = time.time()
    model = req.model or config.models()[0]
    if model not in config.models():
        raise HTTPException(status_code=400, detail="Unknown model")
    if len(req.image_b64) > _MAX_IMAGE_B64:
        raise HTTPException(status_code=413, detail="Image too large")
    try:
        image = base64.b64decode(req.image_b64, validate=True)
    except ValueError:
        raise HTTPException(status_code=400, detail="Bad base64 image")

    record: Dict[str, Any] = {
        "ts": started, "model": model, "version": app.version,
        "source": req.source[:20],
        "image_sha1": hashlib.sha1(image).hexdigest(),
        "thumb_b64": (req.thumb_b64 or "")[:_MAX_THUMB_B64] or None,
        "image_bytes": len(image),
    }
    items: List[Dict[str, Any]] = []
    error: Optional[PipelineError] = None
    try:
        t0 = time.time()
        items, raw, tok_in, tok_out = vision.identify(
            model, req.image_b64, req.media_type)
        record.update(llm_ms=int((time.time() - t0) * 1000),
                      raw_output=raw[:4000], input_tokens=tok_in,
                      output_tokens=tok_out)
        t1 = time.time()
        items, pick_in, pick_out = matching.verify(items)
        record["db_ms"] = int((time.time() - t1) * 1000)
        record.update(input_tokens=tok_in + pick_in,
                      output_tokens=tok_out + pick_out)
        record["cost_usd"] = _cost(model, tok_in, tok_out, vision.
                                   picker_model(), pick_in, pick_out)
    except PipelineError as err:
        error, items = err, []
        record.update(error_stage=err.stage, error_msg=err.message[:500])

    record.update(
        total_ms=int((time.time() - started) * 1000),
        n_items=len(items),
        n_matched=sum(1 for i in items if i.get("matched_id")),
        # Implausible INDB values are shown on the item but never enter the
        # meal total or the daily log.
        total_cal=round(sum(i.get("cal") or 0 for i in items
                            if not i.get("suspect")), 1))
    event_id = telemetry.store_event(record, items)

    if error:
        return JSONResponse(status_code=error.status, content={
            "event_id": event_id,
            "error": {"stage": error.stage, "message": error.message}})
    return JSONResponse(content={
        "event_id": event_id, "model": model, "items": items,
        "total_calories": record["total_cal"],
        "total_ms": record["total_ms"]})


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    """Records a user verdict. This is the correction flywheel."""
    try:
        return telemetry.apply_feedback(
            req.event_id, req.verdict, req.item_id, req.corrected_dish,
            req.corrected_portion_g)
    except telemetry.FeedbackError as err:
        raise HTTPException(status_code=err.status, detail=err.message)


@app.get("/api/dishes")
def dishes(q: str = "") -> List[str]:
    """INDB name autocomplete for the correction editor."""
    return indb.search_names(q)


@app.get("/api/history")
def history(limit: int = 30) -> List[Dict[str, Any]]:
    """Recent successful snaps for the food log."""
    rows = telemetry.events("ok", min(limit, 100)) or []
    for row in rows:
        row.pop("raw_output", None)
    return rows


@app.get("/api/admin/summary")
def admin_summary(days: int = 7) -> Dict[str, Any]:
    """KPIs and insights for the admin dashboard."""
    return telemetry.summary(days)


@app.get("/api/admin/match")
def admin_match(dish: str) -> Dict[str, Any]:
    """Explains how a dish name gets matched: candidates and picker choice."""
    return matching.explain(dish)


@app.post("/api/admin/alias")
def admin_alias(req: AliasRequest) -> Dict[str, Any]:
    """Pins a predicted name to an INDB row, or clears the saved match."""
    if not req.indb_name:
        telemetry.alias_delete(req.dish)
        return {"dish": req.dish, "alias": None}
    row = indb.get_by_name(req.indb_name)
    if not row:
        raise HTTPException(status_code=400,
                            detail="No INDB dish with that exact name")
    telemetry.alias_put(req.dish, row["id"], row["name"], "manual")
    return {"dish": req.dish, "alias": row["name"]}


@app.get("/api/admin/events")
def admin_events(kind: str = "all",
                 limit: int = 40) -> List[Dict[str, Any]]:
    """Event drill-down. kind: all, errors, unreviewed, low_conf."""
    rows = telemetry.events(kind, limit)
    if rows is None or kind == "ok":
        raise HTTPException(status_code=400, detail="Unknown kind")
    return rows


@app.get("/api/admin/export.csv")
def admin_export() -> Response:
    """Exports every rated item as CSV."""
    rows = telemetry.labeled_rows()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if rows:
        writer.writerow(rows[0].keys())
    writer.writerows([tuple(r) for r in rows])
    return Response(buffer.getvalue(), media_type="text/csv", headers={
        "content-disposition":
            "attachment; filename=caloriesnap_labels.csv"})
