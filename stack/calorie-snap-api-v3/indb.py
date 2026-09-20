"""INDB (ICMR-NIN) lookups. This module is the only calorie source."""

import math
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import config
from errors import PipelineError

_DISH_COLS = "id, name, name_local, cal_per_100g, cal_per_serving"

# Words that carry no dish identity.
_STOPWORDS = {"with", "and", "the", "of", "in", "ka", "ki", "ke", "style"}
# Retrieval-only synonyms. They widen the candidate set. The picker model
# makes the final call, so a loose synonym cannot cause a wrong match.
_SYNONYMS = [
    {"roti", "chapati", "phulka"}, {"dal", "dhal", "daal", "lentil"},
    {"chana", "chole", "chickpea", "chane"}, {"curd", "dahi", "yogurt"},
    {"aloo", "potato"}, {"palak", "spinach"}, {"matar", "pea", "mutter"},
    {"gobhi", "gobi", "cauliflower"}, {"bhindi", "okra"},
    {"baingan", "brinjal", "eggplant"}, {"chawal", "rice"},
    {"paratha", "parantha"}, {"rajma", "kidney"}, {"murgh", "chicken"},
]
_SYN = {word: group for group in _SYNONYMS for word in group}
# Plausibility bounds for the data audit guard.
_MAX_KCAL_PER_100G = 500.0
_SERVING_GRAMS_RANGE = (10.0, 700.0)
_MAX_FULL = 80
_index_cache: Dict[str, Any] = {}


def connect() -> sqlite3.Connection:
    """Opens the cuisine DB read-only."""
    path = config.cfg(
        "CUISINE_DB", os.path.join(config.BASE_DIR,
                                   "caloriesnap_cuisine.db"))
    if not os.path.exists(path):
        raise PipelineError("db", f"Cuisine DB not found at {path}", 500)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def lookup(conn: sqlite3.Connection,
           name: str) -> Tuple[Optional[sqlite3.Row], Optional[str]]:
    """Finds the best INDB row for a predicted dish name.

    Order: exact name, substring (shortest name wins, which prefers
    "Poha" over "Poha cutlet"), then reverse containment (the longest
    INDB name found inside the prediction, so "Kanda Poha" -> "Poha").

    Returns:
        (row, method). method is logged so accuracy can be measured per
        method. Both are None when nothing matches.
    """
    needle = re.sub(r"\s+", " ", name.strip().lower())
    if not needle:
        return None, None
    row = conn.execute(
        f"SELECT {_DISH_COLS} FROM dishes WHERE LOWER(name) = ? "
        "OR LOWER(name_local) = ? LIMIT 1", (needle, needle)).fetchone()
    if row:
        return row, "exact"
    row = conn.execute(
        f"SELECT {_DISH_COLS} FROM dishes WHERE LOWER(name) LIKE ? "
        "ORDER BY LENGTH(name) ASC LIMIT 1", (f"%{needle}%",)).fetchone()
    if row:
        return row, "like"
    row = conn.execute(
        f"SELECT {_DISH_COLS} FROM dishes "
        "WHERE LENGTH(name) >= 4 AND ? LIKE '%' || LOWER(name) || '%' "
        "ORDER BY LENGTH(name) DESC LIMIT 1", (needle,)).fetchone()
    if row:
        return row, "contained"
    return None, None


def _tokens(text: str) -> set:
    """Lowercased identity tokens with plural stripping and synonyms."""
    out = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        if word in _STOPWORDS or len(word) < 3:
            continue
        out |= _SYN.get(word, {word})
    return out


def _index(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Token index over every dish name. Built once per DB file version."""
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    stamp = (path, os.path.getmtime(path))
    if _index_cache.get("stamp") != stamp:
        rows = conn.execute(f"SELECT {_DISH_COLS} FROM dishes").fetchall()
        docs = [(row, _tokens(f"{row['name']} {row['name_local'] or ''}"))
                for row in rows]
        freq: Dict[str, int] = {}
        for _, toks in docs:
            for tok in toks:
                freq[tok] = freq.get(tok, 0) + 1
        idf = {tok: math.log(1 + len(docs) / n) for tok, n in freq.items()}
        _index_cache.update(stamp=stamp, docs=docs, idf=idf)
    return _index_cache


def candidates(conn: sqlite3.Connection, name: str,
               limit: int = 30) -> List[sqlite3.Row]:
    """Top INDB rows by IDF-weighted token overlap with a dish name.

    INDB names are descriptive ("Boiled rice (Uble chawal)", "Paneer
    shaslik/tikka"), so string equality misses most real matches. This
    returns a shortlist for the picker model to judge.

    The limit is generous on purpose. A generic head noun ("rice") ties
    dozens of rows on score, and the right one ("Boiled rice (Uble
    chawal)") ranks below shorter names. Thirty short names cost about
    $0.0003 to judge, which is cheaper than a missed match.
    """
    index = _index(conn)
    idf = index["idf"]
    # Words INDB never uses ("basmati", "masala") cannot match any row, so
    # they must not dilute the score of the words that can.
    needle = {tok for tok in _tokens(name) if tok in idf}
    total = sum(idf[tok] for tok in needle)
    if not total:
        return []
    scored = []
    for row, toks in index["docs"]:
        shared = needle & toks
        if not shared:
            continue
        recall = sum(idf[tok] for tok in shared) / total
        precision = (sum(idf[tok] for tok in shared)
                     / sum(idf[tok] for tok in toks))
        if recall >= 0.3:
            scored.append((recall + 0.5 * precision, -len(row["name"]),
                           row["id"], row, recall >= 0.999))
    scored.sort(reverse=True)
    # Rows that cover every known word of the name are all kept (up to
    # _MAX_FULL): for "Basmati Rice" that is every rice row, and the right
    # one can rank anywhere among them. Partial matches fill up to limit.
    full = [e[3] for e in scored if e[4]][:_MAX_FULL]
    partial = [e[3] for e in scored if not e[4]]
    return full + partial[:max(0, limit - len(full))]


def get(conn: sqlite3.Connection, dish_id: int) -> Optional[sqlite3.Row]:
    """One dish row by id."""
    return conn.execute(f"SELECT {_DISH_COLS} FROM dishes WHERE id = ?",
                        (dish_id,)).fetchone()


def is_suspect(row: sqlite3.Row, portion_g: Optional[float]) -> bool:
    """True when the energy value used for this item is implausible.

    With a portion, calories come from cal_per_100g, so only that value is
    judged. Without one, cal_per_serving is used, so the implied serving
    weight is judged instead. A whole tandoori chicken is a legitimate
    large serving and must not flag a correct per-100 g value.
    """
    per_100g, serving = row["cal_per_100g"], row["cal_per_serving"]
    if per_100g and portion_g:
        return per_100g > _MAX_KCAL_PER_100G
    if per_100g and serving:
        grams = serving / per_100g * 100.0
        low, high = _SERVING_GRAMS_RANGE
        return per_100g > _MAX_KCAL_PER_100G or not low <= grams <= high
    return False


def calories(row: sqlite3.Row, portion_g: Optional[float]) -> float:
    """Calories for a portion, from INDB values only."""
    per_100g = row["cal_per_100g"]
    if per_100g and portion_g:
        return round(per_100g * portion_g / 100.0, 1)
    return round(row["cal_per_serving"] or 0.0, 1)


def calories_for(dish: str, portion_g: Optional[float]) -> Optional[float]:
    """Calories for a user-corrected dish, or None when not in INDB."""
    conn = connect()
    try:
        row, _ = lookup(conn, dish)
        return calories(row, portion_g) if row else None
    finally:
        conn.close()


def search_names(query: str, limit: int = 12) -> List[str]:
    """Autocomplete for the correction editor."""
    needle = query.strip().lower()
    if len(needle) < 2:
        return []
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT name FROM dishes WHERE LOWER(name) LIKE ? "
            "ORDER BY LENGTH(name) ASC LIMIT ?",
            (f"%{needle}%", limit)).fetchall()
        return [r["name"] for r in rows]
    finally:
        conn.close()


def all_names() -> List[str]:
    """Every canonical dish name, for prompt grounding."""
    conn = connect()
    try:
        return [r["name"] for r in conn.execute(
            "SELECT name FROM dishes ORDER BY name").fetchall()]
    finally:
        conn.close()


def get_by_name(name: str) -> Optional[sqlite3.Row]:
    """One dish row by its exact INDB name, case-insensitive."""
    conn = connect()
    try:
        return conn.execute(
            f"SELECT {_DISH_COLS} FROM dishes WHERE LOWER(name) = ? LIMIT 1",
            (name.strip().lower(),)).fetchone()
    finally:
        conn.close()


def dish_count() -> Optional[int]:
    """Number of dishes, or None when the DB is unavailable."""
    try:
        conn = connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM dishes").fetchone()[0]
        finally:
            conn.close()
    except (PipelineError, sqlite3.Error):
        return None
