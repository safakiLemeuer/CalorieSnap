"""Matches identified dishes to INDB rows.

Order per item:
    1. alias     a cached earlier decision for this exact predicted name
    2. exact     the predicted name equals an INDB name
    3. picker    token retrieval shortlists candidates, then a text-only
                 model call picks the same dish ("picker"), the nearest
                 nutritional equivalent ("picker_close"), or none
    4. fallback  substring rules, used only when the picker is unavailable
                 (air-gapped mode) or its call fails

Picker decisions are cached as aliases, so each distinct name costs one
model call ever, and repeat snaps match deterministically.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import indb
import telemetry
import vision
from errors import PipelineError


def verify(items: List[Dict[str, Any]]
           ) -> Tuple[List[Dict[str, Any]], int, int]:
    """Attaches INDB match fields and calories to each item in place.

    Returns:
        (items, picker_input_tokens, picker_output_tokens)
    """
    tok_in = tok_out = 0
    try:
        conn = indb.connect()
        try:
            pending: List[Dict[str, Any]] = []
            shortlist: Dict[int, List[sqlite3.Row]] = {}
            for idx, item in enumerate(items):
                alias = telemetry.alias_get(item["dish"])
                row = indb.get(conn, alias[0]) if alias else None
                method: Optional[str] = None
                if row:
                    method = ("alias_close" if alias[1] == "picker_close"
                              else "alias")
                if not row:
                    row, method = indb.lookup(conn, item["dish"])
                    if method != "exact":
                        fallback = (row, method)
                        row, method = None, None
                        rows = indb.candidates(conn, item["dish"])
                        if fallback[0] is not None and all(
                                r["id"] != fallback[0]["id"] for r in rows):
                            rows.append(fallback[0])
                        item["_fallback"] = fallback
                        if rows:
                            shortlist[idx] = rows
                            pending.append({
                                "item": idx, "dish": item["dish"],
                                "candidates": [r["name"] for r in rows]})
                _attach(item, row, method)

            picked: Optional[Dict[int, Tuple[Optional[int], str]]] = None
            if pending and vision.picker_available():
                try:
                    picked, tok_in, tok_out = vision.pick_matches(pending)
                except PipelineError:
                    picked = None  # Fall back below. The snap still works.
            for idx, rows in shortlist.items():
                item = items[idx]
                if picked is None:
                    _attach(item, *item["_fallback"])
                    continue
                choice, quality = picked.get(idx, (None, "same"))
                if choice is not None and 0 <= choice < len(rows):
                    method = "picker" if quality == "same" else "picker_close"
                    _attach(item, rows[choice], method)
                    telemetry.alias_put(item["dish"], rows[choice]["id"],
                                        rows[choice]["name"], method)
            for item in items:
                item.pop("_fallback", None)
        finally:
            conn.close()
    except sqlite3.Error as err:
        raise PipelineError("db", f"SQLite: {err}", 500)
    return items, tok_in, tok_out


def _attach(item: Dict[str, Any], row: Optional[sqlite3.Row],
            method: Optional[str]) -> None:
    """Writes match fields onto an item."""
    item["matched_id"] = row["id"] if row else None
    item["matched_name"] = row["name"] if row else None
    item["match_method"] = method if row else None
    item["cal"] = indb.calories(row, item["portion_g"]) if row else None
    item["suspect"] = bool(row) and indb.is_suspect(row, item["portion_g"])
    item["approx"] = bool(row) and (method or "").endswith("_close")


def explain(dish: str) -> Dict[str, Any]:
    """Shows how one dish name would be matched. For the admin debugger.

    Does not write aliases or telemetry.
    """
    conn = indb.connect()
    try:
        alias = telemetry.alias_get(dish)
        row, method = indb.lookup(conn, dish)
        rows = indb.candidates(conn, dish)
        names = [r["name"] for r in rows]
    finally:
        conn.close()
    out: Dict[str, Any] = {
        "dish": dish, "alias": alias,
        "string_match": {"name": row["name"], "method": method}
                        if row else None,
        "candidates": names, "picker": None}
    if names and vision.picker_available():
        try:
            picked, _, _ = vision.pick_matches(
                [{"item": 0, "dish": dish, "candidates": names}])
            choice, quality = picked.get(0, (None, "same"))
            out["picker"] = {
                "choice": names[choice] if choice is not None
                          and 0 <= choice < len(names) else None,
                "quality": quality}
        except PipelineError as err:
            out["picker"] = {"error": err.message}
    return out
