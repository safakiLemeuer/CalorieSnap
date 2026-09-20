"""Audits INDB energy values for implausible rows.

Usage (PowerShell):
    Get-Content scripts\audit_indb.py -Raw | docker exec -i calorie-snap-api-v3 python3 > indb_audit.csv

Flags rows over 500 kcal per 100 g, and rows whose implied serving weight
(cal_per_serving / cal_per_100g * 100) is under 10 g or over 700 g. Output
is CSV, worst first. Fix flagged rows at the source, then rebuild the DB.
"""
import csv
import os
import sqlite3
import sys

path = os.environ.get("CUISINE_DB", "/indb/caloriesnap_cuisine.db")
conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
rows = conn.execute(
    "SELECT id, indb_code, name, cal_per_100g, cal_per_serving, "
    "serving_unit FROM dishes").fetchall()
flagged = []
for dish_id, code, name, per_100g, serving, unit in rows:
    grams = serving / per_100g * 100 if per_100g and serving else None
    reasons = []
    if per_100g and per_100g > 500:
        reasons.append("over 500 kcal/100g")
    if grams is not None and not 10 <= grams <= 700:
        reasons.append(f"implied serving {grams:.0f} g")
    if not per_100g and not serving:
        reasons.append("no energy data")
    if reasons:
        flagged.append((per_100g or 0, dish_id, code, name, per_100g,
                        serving, unit, round(grams) if grams else "",
                        "; ".join(reasons)))
writer = csv.writer(sys.stdout)
writer.writerow(["id", "indb_code", "name", "cal_per_100g",
                 "cal_per_serving", "serving_unit", "implied_serving_g",
                 "reason"])
for row in sorted(flagged, reverse=True):
    writer.writerow(row[1:])
print(f"# {len(flagged)} of {len(rows)} rows flagged", file=sys.stderr)
