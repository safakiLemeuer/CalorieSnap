"""
CalorieSnap v2 — INDB Import Pipeline
BHTLabs · Google Python Style Guide

Builds caloriesnap_cuisine.db from INDB open-access data.
Run once on your Windows machine where all XLSX files are present.

Usage:
    py -3.12 indb_import.py --indb INDB.xlsx --names recipes_names.xlsx --out caloriesnap_cuisine.db
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd


CATEGORY_MAP = [
    (r'tea|coffee|lassi|milkshake|juice|sharbat|cooler|lemonade|panna|cocoa|drink|squash', 'Beverage'),
    (r'sandwich|toast', 'Snack'),
    (r'soup|consomme|rasam|shorba', 'Soup'),
    (r'chapati|roti|paratha|parantha|poori|puri|naan|kulcha|bhatura|thepla|makki|bajra roti|phulka', 'Bread'),
    (r'pulao|biryani|rice|khichdi|khichri|pongal|chitranna|sadam|chawal', 'Rice'),
    (r'idli|dosa|dosai|uttapam|appam|pesarattu|cheela|chilla|upma|poha|vada|medu|puttu|porridge|daliya|oatmeal|cornflake|rava idli', 'Breakfast'),
    (r'sambar|dal |dhal |lentil|moong|masoor|arhar|urad dal|channa dal|rajmah|lobia|horsegram|kulthi|panchmel|makhani dal', 'Dal'),
    (r'curry|sabzi|subji|bhaji|kofta|matar|paneer |palak|gobi|bhindi|baingan|karela|jackfruit|yam|mushroom|korma|kadhai|shahi|butter chicken|chicken curry|fish curry|prawn|machli|keema|stew|aloo ki|jhol|kosha', 'Curry'),
    (r'kheer|halwa|ladoo|laddoo|barfi|burfi|jalebi|gulab jamun|rasgulla|sandesh|modak|payasam|pudding|sweet|meetha|mithai|khoa|peda|mysore pak|mysore pak|sheera', 'Dessert'),
    (r'samosa|pakora|pakoda|bhajiya|cutlet|tikka|kebab|kabab|bonda|vada pav|kachori|roll|chaat|bhel|puri chaat|sev|fritter|murukku', 'Snack'),
    (r'raita|salad|chutney|pickle|achaar|papad|chokha|salsa', 'Accompaniment'),
    (r'sauce|stock|white sauce|brown sauce|bechamel', 'Condiment'),
    (r'jam|squash|preserve|murabba', 'Preserve'),
]

REGION_MAP = [
    (r'dosa|idli|sambar|rasam|upma|appam|pongal|pesarattu|uttapam|avial|thoran|olan|puttu|stew|chettinad|vangi bhat|chitranna|puliyodharai|sadam|pulihora|bisi bele|medu vada|kozhukattai|murukku|payasam|rasavangi|mor kuzhambu|jowar dosa|rice puttu', 'South India'),
    (r'paratha|parantha|dal makhani|rajmah|chole|bhature|aloo gobi|palak paneer|shahi paneer|kadhai paneer|matar paneer|butter chicken|nihari|haleem|seekh|biryani lucknow|awadhi|punjabi|sarson|makki|keema|nargisi|moti mahal|litti|sattu|besan gatte|laal maas|dal baati|ker sangri', 'North India'),
    (r'pav bhaji|vada pav|dhokla|thepla|undhiyu|modak|sol kadhi|kombdi|kolhapuri|maharashtrian|gujarati|rajasthani|dabeli|misal|sabudana|puran poli|aamras|shrikhand', 'West India'),
    (r'macher|machli bengali|kosha|posto|doi|rasgulla|sandesh|mishti|hilsa|chingri|kolkata|bengali|odia|dalma|chhena|chak-hao|black rice|payesh|bhapa|patishapta|litti|sattu|jharkhand', 'East India'),
    (r'nagaland|manipur|assam|northeast|bamboo|smoked pork|black sesame', 'Northeast India'),
]


def infer_category(name: str) -> str:
    name_lower = name.lower()
    for pattern, cat in CATEGORY_MAP:
        if re.search(pattern, name_lower):
            return cat
    return 'Main'


def infer_region(name: str) -> str:
    name_lower = name.lower()
    for pattern, region in REGION_MAP:
        if re.search(pattern, name_lower):
            return region
    return 'Pan-India'


SCHEMA = """
CREATE TABLE IF NOT EXISTS dishes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    indb_code       TEXT UNIQUE,
    name            TEXT NOT NULL,
    name_local      TEXT,
    region          TEXT NOT NULL,
    category        TEXT NOT NULL,
    cal_per_100g    REAL NOT NULL,
    cal_per_serving REAL,
    serving_unit    TEXT,
    protein_g       REAL,
    carb_g          REAL,
    fat_g           REAL,
    fiber_g         REAL,
    data_source     TEXT DEFAULT 'INDB-2024',
    confidence      TEXT DEFAULT 'HIGH'
);
CREATE INDEX IF NOT EXISTS idx_name     ON dishes(name);
CREATE INDEX IF NOT EXISTS idx_region   ON dishes(region);
CREATE INDEX IF NOT EXISTS idx_category ON dishes(category);
"""


def build(indb_path: Path, names_path: Path, out_path: Path) -> None:
    """Build SQLite database from INDB xlsx files.

    Args:
        indb_path:  Path to INDB.xlsx (nutrient values per recipe).
        names_path: Path to recipes_names.xlsx (recipe names).
        out_path:   Output SQLite database path.
    """
    print(f"Loading {indb_path.name}...")
    df_indb  = pd.read_excel(str(indb_path))
    df_names = pd.read_excel(str(names_path))

    df = df_indb.merge(
        df_names[['recipe_code', 'recipe_name_org', 'recipe_name']],
        left_on='food_code', right_on='recipe_code', how='left',
    )

    if out_path.exists():
        out_path.unlink()
    conn = sqlite3.connect(str(out_path))
    conn.executescript(SCHEMA)

    inserted = skipped = 0
    for _, row in df.iterrows():
        kcal = row.get('energy_kcal')
        if not kcal or pd.isna(kcal) or float(kcal) <= 0:
            skipped += 1
            continue

        name = str(row.get('recipe_name') or row.get('food_name', '')).strip()
        if not name:
            skipped += 1
            continue

        local_m = re.search(r'\(([^)]+)\)', name)
        name_local = local_m.group(1) if local_m else None

        serving_kcal = row.get('unit_serving_energy_kcal')
        serving_unit = row.get('servings_unit')

        def safe_float(val):
            try:
                return round(float(val), 2) if val and not pd.isna(val) else None
            except (TypeError, ValueError):
                return None

        try:
            conn.execute(
                """INSERT OR IGNORE INTO dishes
                   (indb_code, name, name_local, region, category,
                    cal_per_100g, cal_per_serving, serving_unit,
                    protein_g, carb_g, fat_g, fiber_g)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    str(row.get('food_code', '')),
                    name,
                    name_local,
                    infer_region(name),
                    infer_category(name),
                    round(float(kcal), 1),
                    safe_float(serving_kcal),
                    str(serving_unit) if serving_unit and not pd.isna(serving_unit) else None,
                    safe_float(row.get('protein_g')),
                    safe_float(row.get('carb_g')),
                    safe_float(row.get('fat_g')),
                    safe_float(row.get('fibre_g')),
                ),
            )
            inserted += 1
        except Exception as e:
            print(f"  Skip {name[:40]}: {e}")
            skipped += 1

    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM dishes").fetchone()[0]
    print(f"\nImported {inserted} dishes ({skipped} skipped). Total in DB: {total}")

    print("\nBy category:")
    for r in conn.execute("SELECT category, COUNT(*) n FROM dishes GROUP BY category ORDER BY n DESC"):
        print(f"  {r[0]:<20} {r[1]}")

    print("\nBy region:")
    for r in conn.execute("SELECT region, COUNT(*) n FROM dishes GROUP BY region ORDER BY n DESC"):
        print(f"  {r[0]:<25} {r[1]}")

    conn.close()
    print(f"\nSaved to {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CalorieSnap cuisine DB from INDB")
    parser.add_argument("--indb",  default="INDB.xlsx",           help="Path to INDB.xlsx")
    parser.add_argument("--names", default="recipes_names.xlsx",  help="Path to recipes_names.xlsx")
    parser.add_argument("--out",   default="caloriesnap_cuisine.db", help="Output DB path")
    args = parser.parse_args()
    build(Path(args.indb), Path(args.names), Path(args.out))
