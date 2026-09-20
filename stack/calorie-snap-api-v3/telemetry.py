"""Telemetry storage, the correction flywheel, and admin analytics.

Telemetry lives in its own SQLite file (TELEMETRY_DB, default
/data/caloriesnap_telemetry.db) so the cuisine DB stays read-only and the
data survives container recreation when /data is a volume.
"""

import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import config
import indb

VERDICTS = ("correct", "wrong_dish", "wrong_portion", "wrong_match",
            "extra", "missing")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model TEXT,
    image_sha1 TEXT,
    thumb_b64 TEXT,
    image_bytes INTEGER,
    llm_ms INTEGER,
    db_ms INTEGER,
    total_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    raw_output TEXT,
    n_items INTEGER DEFAULT 0,
    n_matched INTEGER DEFAULT 0,
    total_cal REAL DEFAULT 0,
    error_stage TEXT,
    error_msg TEXT
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    idx INTEGER NOT NULL,
    dish_pred TEXT,
    name_local TEXT,
    portion_g REAL,
    confidence REAL,
    matched_id INTEGER,
    matched_name TEXT,
    match_method TEXT,
    cal REAL,
    verdict TEXT,
    corrected_dish TEXT,
    corrected_portion_g REAL,
    feedback_ts REAL
);
CREATE TABLE IF NOT EXISTS aliases (
    name_lower TEXT PRIMARY KEY,
    dish_id INTEGER NOT NULL,
    dish_name TEXT,
    source TEXT,
    ts REAL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_items_event ON items(event_id);
"""


def connect() -> sqlite3.Connection:
    """Opens the telemetry DB, creating the schema on first use."""
    path = config.cfg("TELEMETRY_DB", "/data/caloriesnap_telemetry.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(items)")}
    if "suspect" not in columns:  # Migration for pre-3.2 databases.
        conn.execute("ALTER TABLE items ADD COLUMN suspect INTEGER DEFAULT 0")
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(events)")}
    if "version" not in columns:  # Rows from before 3.3 stay NULL.
        conn.execute("ALTER TABLE events ADD COLUMN version TEXT")
    if "source" not in columns:  # Rows from before 3.6 stay NULL (web).
        conn.execute("ALTER TABLE events ADD COLUMN source TEXT")
    return conn


def alias_get(name: str) -> Optional[Tuple[int, str]]:
    """Cached (INDB dish id, source) for a predicted name, if any."""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT dish_id, source FROM aliases WHERE name_lower = ?",
            (name.strip().lower(),)).fetchone()
        return (row["dish_id"], row["source"] or "") if row else None
    finally:
        conn.close()


def confusion_hints(limit: int = 8) -> List[Tuple[str, str, int]]:
    """Most frequent user-corrected misidentifications, recent first.

    Returns [(predicted, actual, count)] from the last 90 days with at
    least CONFUSION_MIN_COUNT corrections. These feed the vision prompt,
    so every correction improves the next photo without retraining.
    Raise CONFUSION_MIN_COUNT before opening the app to the public, so one
    careless or hostile correction cannot steer the prompt.
    """
    try:
        floor = max(1, int(config.cfg("CONFUSION_MIN_COUNT", "1")))
    except ValueError:
        floor = 1
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT LOWER(i.dish_pred) AS pred, "
            "LOWER(i.corrected_dish) AS actual, COUNT(*) AS n "
            "FROM items i JOIN events e ON e.id = i.event_id "
            "WHERE i.verdict = 'wrong_dish' AND i.corrected_dish IS NOT NULL"
            " AND i.dish_pred IS NOT NULL AND e.ts >= ? "
            "GROUP BY pred, actual HAVING n >= ? "
            "ORDER BY n DESC, MAX(e.ts) DESC LIMIT ?",
            (time.time() - 90 * 86400, floor, limit)).fetchall()
        return [(r["pred"], r["actual"], r["n"]) for r in rows]
    finally:
        conn.close()


def missed_hints(limit: int = 6) -> List[Tuple[str, int]]:
    """Dishes users most often had to add by hand, last 90 days."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT LOWER(i.corrected_dish) AS dish, COUNT(*) AS n "
            "FROM items i JOIN events e ON e.id = i.event_id "
            "WHERE i.verdict = 'missing' AND i.corrected_dish IS NOT NULL "
            "AND e.ts >= ? GROUP BY dish ORDER BY n DESC LIMIT ?",
            (time.time() - 90 * 86400, limit)).fetchall()
        return [(r["dish"], r["n"]) for r in rows]
    finally:
        conn.close()


def alias_delete(name: str) -> None:
    """Forgets the saved match for a predicted name."""
    conn = connect()
    try:
        conn.execute("DELETE FROM aliases WHERE name_lower = ?",
                     (name.strip().lower(),))
        conn.commit()
    finally:
        conn.close()


def alias_names() -> set:
    """Lowercased predicted names that currently have a saved match."""
    conn = connect()
    try:
        return {r["name_lower"] for r in conn.execute(
            "SELECT name_lower FROM aliases")}
    finally:
        conn.close()


def alias_put(name: str, dish_id: int, dish_name: str,
              source: str) -> None:
    """Caches a name-to-dish decision."""
    conn = connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO aliases VALUES (?,?,?,?,?)",
            (name.strip().lower(), dish_id, dish_name, source, time.time()))
        conn.commit()
    finally:
        conn.close()


def store_event(record: Dict[str, Any],
                 items: List[Dict[str, Any]]) -> int:
    """Writes one event and its items. Sets item["id"] on each item."""
    conn = connect()
    try:
        cols = list(record.keys())
        cur = conn.execute(
            f"INSERT INTO events ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            [record[c] for c in cols])
        event_id = cur.lastrowid
        for idx, item in enumerate(items):
            cur = conn.execute(
                "INSERT INTO items (event_id, idx, dish_pred, name_local,"
                " portion_g, confidence, matched_id, matched_name,"
                " match_method, cal, suspect) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, idx, item["dish"], item["name_local"],
                 item["portion_g"], item["confidence"],
                 item.get("matched_id"), item.get("matched_name"),
                 item.get("match_method"), item.get("cal"),
                 int(bool(item.get("suspect")))))
            item["id"] = cur.lastrowid
        conn.commit()
        return event_id
    finally:
        conn.close()


class FeedbackError(Exception):
    """Invalid feedback. status is the HTTP code to return."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def apply_feedback(event_id: int, verdict: str, item_id: Optional[int],
                   corrected_dish: Optional[str],
                   corrected_portion_g: Optional[float]) -> Dict[str, Any]:
    """Records a user verdict and recomputes the meal total.

    A corrected dish or portion is re-verified against INDB so the log
    reflects what was eaten. verdict="missing" adds a new item.
    verdict="wrong_match" means the model named the dish correctly but it
    was tied to the wrong INDB row: corrected_dish is the right INDB name,
    which is pinned as a manual alias so the mistake never repeats.
    """
    if verdict not in VERDICTS:
        raise FeedbackError(400, "Unknown verdict")
    now = time.time()
    conn = connect()
    try:
        if verdict == "missing":
            if not corrected_dish:
                raise FeedbackError(400, "corrected_dish required")
            next_idx = conn.execute(
                "SELECT COALESCE(MAX(idx), -1) + 1 FROM items "
                "WHERE event_id = ?", (event_id,)).fetchone()[0]
            item_id = conn.execute(
                "INSERT INTO items (event_id, idx, verdict, feedback_ts) "
                "VALUES (?,?,?,?)",
                (event_id, next_idx, "missing", now)).lastrowid
        else:
            found = conn.execute(
                "SELECT id FROM items WHERE id = ? AND event_id = ?",
                (item_id, event_id)).fetchone()
            if not found:
                raise FeedbackError(404, "Item not found")
            conn.execute(
                "UPDATE items SET verdict = ?, feedback_ts = ? "
                "WHERE id = ?", (verdict, now, item_id))

        current = conn.execute("SELECT * FROM items WHERE id = ?",
                               (item_id,)).fetchone()
        if (verdict == "wrong_dish" and current["dish_pred"]
                and (current["match_method"] or "").startswith(
                    ("picker", "alias"))):
            conn.execute("DELETE FROM aliases WHERE name_lower = ?",
                         (current["dish_pred"].strip().lower(),))
        cal: Optional[float] = current["cal"]
        if verdict == "wrong_match":
            row = indb.get_by_name(corrected_dish or "")
            if not row or not current["dish_pred"]:
                raise FeedbackError(400, "Pick an existing INDB dish name")
            conn.execute(
                "UPDATE items SET matched_id = ?, matched_name = ?, "
                "suspect = ? WHERE id = ?",
                (row["id"], row["name"],
                 int(indb.is_suspect(row, current["portion_g"])), item_id))
            conn.commit()
            alias_put(current["dish_pred"], row["id"], row["name"], "manual")
        if verdict == "extra":
            cal = 0.0
        elif corrected_dish or corrected_portion_g:
            cal = indb.calories_for(
                corrected_dish or current["dish_pred"],
                corrected_portion_g or current["portion_g"])
        conn.execute(
            "UPDATE items SET corrected_dish = ?, corrected_portion_g = ?,"
            " cal = ? WHERE id = ?",
            (corrected_dish, corrected_portion_g, cal, item_id))
        total = round(conn.execute(
            "SELECT COALESCE(SUM(cal), 0) FROM items WHERE event_id = ? "
            "AND COALESCE(suspect, 0) = 0",
            (event_id,)).fetchone()[0], 1)
        conn.execute("UPDATE events SET total_cal = ? WHERE id = ?",
                     (total, event_id))
        conn.commit()
        return {"item_id": item_id, "cal": cal, "total_calories": total}
    finally:
        conn.close()


_EVENT_FILTERS = {
    "all": "1=1",
    "ok": "error_stage IS NULL",
    "errors": "error_stage IS NOT NULL",
    "unreviewed": ("error_stage IS NULL AND id IN (SELECT event_id "
                   "FROM items WHERE verdict IS NULL)"),
    "low_conf": ("id IN (SELECT event_id FROM items "
                 "WHERE confidence < 0.7 OR matched_id IS NULL)"),
}


def events(kind: str, limit: int) -> Optional[List[Dict[str, Any]]]:
    """Events with their items, newest first. None for an unknown kind."""
    where = _EVENT_FILTERS.get(kind)
    if where is None:
        return None
    conn = connect()
    try:
        rows = [dict(r) for r in conn.execute(
            f"SELECT * FROM events WHERE {where} ORDER BY ts DESC LIMIT ?",
            (min(limit, 200),)).fetchall()]
        for row in rows:
            row["items"] = [dict(r) for r in conn.execute(
                "SELECT * FROM items WHERE event_id = ? ORDER BY idx",
                (row["id"],)).fetchall()]
        return rows
    finally:
        conn.close()


def labeled_rows() -> List[sqlite3.Row]:
    """Every rated item. This is the labeled dataset."""
    conn = connect()
    try:
        return conn.execute(
            "SELECT e.id AS event_id, e.ts, e.model, e.image_sha1, "
            "i.dish_pred, i.confidence, i.portion_g, i.matched_name, "
            "i.match_method, i.verdict, i.corrected_dish, "
            "i.corrected_portion_g FROM items i JOIN events e "
            "ON e.id = i.event_id WHERE i.verdict IS NOT NULL "
            "ORDER BY e.ts").fetchall()
    finally:
        conn.close()


def summary(days: int = 7) -> Dict[str, Any]:
    """Everything the admin dashboard needs in one response."""
    since = time.time() - max(1, min(days, 365)) * 86400
    conn = connect()
    try:
        events = conn.execute(
            "SELECT * FROM events WHERE ts >= ?", (since,)).fetchall()
        items = conn.execute(
            "SELECT i.*, e.model AS model, e.image_sha1 AS image_sha1, "
            "e.version AS version "
            "FROM items i JOIN events e ON e.id = i.event_id "
            "WHERE e.ts >= ?", (since,)).fetchall()
    finally:
        conn.close()

    solved = alias_names()
    ok = [e for e in events if not e["error_stage"]]
    failed = [e for e in events if e["error_stage"]]
    predicted = [i for i in items if i["dish_pred"]]
    costs = [e["cost_usd"] for e in ok if e["cost_usd"] is not None]
    acc = _accuracy(items)

    kpis = {
        "snaps": len(events),
        "errors": len(failed),
        "error_rate": round(len(failed) / len(events), 3) if events
                      else None,
        "p50_ms": _percentile([e["total_ms"] for e in ok], 50),
        "p95_ms": _percentile([e["total_ms"] for e in ok], 95),
        "avg_llm_ms": _mean([e["llm_ms"] for e in ok]),
        "avg_db_ms": _mean([e["db_ms"] for e in ok]),
        "avg_cost_usd": _mean(costs, 6),
        "total_cost_usd": round(sum(costs), 4),
        "avg_input_tokens": _mean([e["input_tokens"] for e in ok]),
        "avg_output_tokens": _mean([e["output_tokens"] for e in ok]),
        "items": len(predicted),
        "match_rate": (round(sum(1 for i in predicted if i["matched_id"])
                             / len(predicted), 3) if predicted else None),
        "rated_items": acc["rated"],
        "accuracy": acc["accuracy"],
        "review_coverage": (round(acc["rated"] / len(predicted), 3)
                            if predicted else None),
        "missing_items": sum(1 for i in items
                             if i["verdict"] == "missing"),
        "portion_fixes": sum(1 for i in items
                             if i["verdict"] == "wrong_portion"),
        "suspect_hits": sum(1 for i in predicted if i["suspect"]),
        "wrong_matches": sum(1 for i in items
                             if i["verdict"] == "wrong_match"),
    }

    return {
        "days": days,
        "kpis": kpis,
        "daily": _daily(events),
        "errors_by_stage": _count_by(failed, "error_stage"),
        "recent_errors": [
            {k: e[k] for k in ("id", "ts", "model", "error_stage",
                               "error_msg", "raw_output")}
            for e in sorted(failed, key=lambda e: -e["ts"])[:10]],
        # Names that have since gained a saved match are no longer misses.
        "db_misses": _top(
            [i["dish_pred"].strip().lower() for i in predicted
             if not i["matched_id"]
             and i["dish_pred"].strip().lower() not in solved], 20),
        "bad_matches": _top(
            [f'{i["dish_pred"]} -> {i["corrected_dish"]}' for i in items
             if i["verdict"] == "wrong_match"], 20),
        "confusions": _top(
            [f'{i["dish_pred"]} -> {i["corrected_dish"]}' for i in items
             if i["verdict"] == "wrong_dish" and i["corrected_dish"]],
            20),
        "missed_dishes": _top(
            [i["corrected_dish"].strip().lower() for i in items
             if i["verdict"] == "missing" and i["corrected_dish"]], 20),
        "suspect_rows": _top(
            [i["matched_name"] for i in predicted if i["suspect"]], 20),
        "calibration": _calibration(predicted),
        "by_match_method": _grouped_accuracy(
            predicted, lambda i: i["match_method"] or "no match"),
        "by_model": _by_model(events, items),
        "by_version": _by_version(events, predicted),
        "by_source": _top([e["source"] or "web" for e in events], 10),
        "consistency": _consistency(predicted),
    }


def _percentile(values: List[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile. Returns None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1,
                      int(round(pct / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[rank]


def _accuracy(rows: List[sqlite3.Row]) -> Dict[str, Any]:
    """Identification accuracy over rated, model-predicted items.

    wrong_portion counts as a correct identification. missing items are
    excluded here and reported separately as recall failures. wrong_match
    is a lookup failure, not a model failure, so it counts as correct.
    """
    rated = [r for r in rows if r["verdict"] and r["verdict"] != "missing"]
    good = sum(1 for r in rated if r["verdict"] in (
        "correct", "wrong_portion", "wrong_match"))
    return {"rated": len(rated),
            "accuracy": round(good / len(rated), 3) if rated else None}


def _mean(values: List[Optional[float]],
          digits: int = 1) -> Optional[float]:
    """Mean of the non-null values, or None."""
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), digits) if present else None


def _count_by(rows: List[sqlite3.Row], key: str) -> List[Dict[str, Any]]:
    """Counts rows per value of key, most common first."""
    return _top([r[key] for r in rows], 20)


def _top(values: List[str], limit: int) -> List[Dict[str, Any]]:
    """Returns [{name, n}] for the most common values."""
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"name": k, "n": v} for k, v in ranked[:limit]]


def _daily(events: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    """Per-day volume, errors, and median latency."""
    days: Dict[str, Dict[str, Any]] = {}
    for event in events:
        day = time.strftime("%Y-%m-%d", time.localtime(event["ts"]))
        slot = days.setdefault(day, {"day": day, "snaps": 0,
                                     "errors": 0, "_ms": []})
        slot["snaps"] += 1
        if event["error_stage"]:
            slot["errors"] += 1
        elif event["total_ms"] is not None:
            slot["_ms"].append(event["total_ms"])
    out = []
    for day in sorted(days):
        slot = days[day]
        slot["p50_ms"] = _percentile(slot.pop("_ms"), 50)
        out.append(slot)
    return out


def _calibration(items: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    """Accuracy per confidence bucket. Shows if confidence is usable."""
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
    out = []
    for low, high in buckets:
        rows = [i for i in items if i["confidence"] is not None
                and low <= i["confidence"] < high]
        out.append({"bucket": f"{low:.2f}-{min(high, 1.0):.2f}",
                    "items": len(rows), **_accuracy(rows)})
    return out


def _grouped_accuracy(items: List[sqlite3.Row],
                      key_fn) -> List[Dict[str, Any]]:
    """Accuracy per group."""
    groups: Dict[str, List[sqlite3.Row]] = {}
    for item in items:
        groups.setdefault(key_fn(item), []).append(item)
    return [{"name": name, "items": len(rows), **_accuracy(rows)}
            for name, rows in sorted(groups.items())]


def _by_model(events: List[sqlite3.Row],
              items: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    """Side-by-side model comparison for benchmarking."""
    out = []
    for model in sorted({e["model"] or "unknown" for e in events}):
        mine = [e for e in events if (e["model"] or "unknown") == model]
        ok = [e for e in mine if not e["error_stage"]]
        rows = [i for i in items if (i["model"] or "unknown") == model]
        out.append({
            "model": model, "snaps": len(mine),
            "error_rate": round(1 - len(ok) / len(mine), 3),
            "p50_ms": _percentile([e["total_ms"] for e in ok], 50),
            "avg_cost_usd": _mean([e["cost_usd"] for e in ok], 6),
            **_accuracy(rows)})
    return out


def _by_version(events: List[sqlite3.Row],
                predicted: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    """Match rate and accuracy per release, so each change is measurable."""
    out = []
    for version in sorted({e["version"] or "pre-3.3" for e in events}):
        rows = [i for i in predicted
                if (i["version"] or "pre-3.3") == version]
        matched = sum(1 for i in rows if i["matched_id"])
        out.append({
            "version": version,
            "snaps": sum(1 for e in events
                         if (e["version"] or "pre-3.3") == version),
            "items": len(rows),
            "match_rate": round(matched / len(rows), 3) if rows else None,
            **_accuracy(rows)})
    return out


def _consistency(items: List[sqlite3.Row]) -> Dict[str, Any]:
    """Same photo, model, and release: did the dish set change?

    Runs from different releases are not compared, because a prompt change
    is supposed to change the answer.
    """
    runs: Dict[Tuple[str, str, str], Dict[int, List[str]]] = {}
    for item in items:
        key = (item["image_sha1"], item["model"] or "unknown",
               item["version"] or "pre-3.3")
        runs.setdefault(key, {}).setdefault(
            item["event_id"], []).append(item["dish_pred"].lower())
    repeated = {k: v for k, v in runs.items() if len(v) >= 2}
    unstable = sum(
        1 for v in repeated.values()
        if len({"|".join(sorted(d)) for d in v.values()}) > 1)
    return {"repeated_photos": len(repeated), "unstable": unstable}
