"""Bot state in SQLite: message dedup, per-user sessions, daily totals.

Phone numbers are never stored. Users are keyed by a salted SHA-256 of
their WhatsApp id.
"""

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (msg_id TEXT PRIMARY KEY, ts REAL);
CREATE TABLE IF NOT EXISTS sessions (
    user_key TEXT PRIMARY KEY, event_id INTEGER, items TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS meals (
    event_id INTEGER PRIMARY KEY, user_key TEXT, ts REAL, total REAL,
    partial INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_meals_user ON meals(user_key, ts);
"""


def _connect() -> sqlite3.Connection:
    path = settings.cfg("WA_DB", "/data/whatsapp_bot.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def user_key(wa_id: str) -> str:
    """Stable pseudonymous key for a WhatsApp id."""
    salt = settings.cfg("WA_HASH_SALT", "caloriesnap")
    return hashlib.sha256(f"{salt}:{wa_id}".encode("utf-8")).hexdigest()[:32]


def first_time(msg_id: str) -> bool:
    """True once per message id. Meta retries deliveries."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM seen WHERE ts < ?",
                     (time.time() - 3 * 86400,))
        try:
            conn.execute("INSERT INTO seen VALUES (?, ?)",
                         (msg_id, time.time()))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    finally:
        conn.close()


def save_session(key: str, event_id: int,
                 items: List[Dict[str, Any]]) -> None:
    """Remembers the user's latest meal so text replies can correct it."""
    conn = _connect()
    try:
        conn.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?)",
                     (key, event_id, json.dumps(items), time.time()))
        conn.commit()
    finally:
        conn.close()


def load_session(key: str) -> Optional[Dict[str, Any]]:
    """The latest meal, if it is under 12 hours old."""
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE user_key = ?",
                           (key,)).fetchone()
    finally:
        conn.close()
    if not row or row["ts"] < time.time() - 12 * 3600:
        return None
    return {"event_id": row["event_id"], "items": json.loads(row["items"])}


def save_meal(key: str, event_id: int, total: float,
              partial: bool) -> None:
    """Upserts a meal total for the daily summary."""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO meals VALUES (?,?,?,?,?) ON CONFLICT(event_id) "
            "DO UPDATE SET total = excluded.total, "
            "partial = excluded.partial",
            (event_id, key, time.time(), total, int(partial)))
        conn.commit()
    finally:
        conn.close()


def today(key: str) -> Dict[str, Any]:
    """Meals and calories since local midnight, plus photos used today."""
    midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT total, partial FROM meals WHERE user_key = ? "
            "AND ts >= ?", (key, midnight)).fetchall()
    finally:
        conn.close()
    return {"meals": len(rows),
            "total": round(sum(r["total"] or 0 for r in rows)),
            "partial": any(r["partial"] for r in rows)}
