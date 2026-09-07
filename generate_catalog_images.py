"""
generate_catalog_images.py — one-time script that generates a product image for
every REAL row in the `catalog` table, using ONLY that row's own fields
(name, category, color_finish, style_tags). Never invents SKUs — it only ever
loops over what already exists in the database.

Safe to re-run: skips any item that already has an image_url set.

Usage:
    python generate_catalog_images.py
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from tools import _get_connection
from room_image import _call_flux


def ensure_image_column(conn):
    cols = [row[1] for row in conn.execute("PRAGMA table_info(catalog)").fetchall()]
    if "image_url" not in cols:
        print("Adding image_url column to catalog table...")
        conn.execute("ALTER TABLE catalog ADD COLUMN image_url TEXT")
        conn.commit()


def build_sku_prompt(row: dict) -> str:
    """Template-based prompt using ONLY this row's real fields — no LLM needed,
    since these fields are already clean and factual."""
    name = row.get("name", "")
    category = row.get("category", "")
    finish = row.get("color_finish", "")
    style = row.get("style_tags", "")
    parts = [f"A professional product photo of a {name}"]
    if category:
        parts.append(f", a {category.lower()}")
    if finish:
        parts.append(f", {finish.lower()} finish")
    if style:
        parts.append(f", {style.lower()} style")
    parts.append(", studio lighting, plain white background, catalog photography, no text, no watermark")
    return "".join(parts)


def main():
    conn = _get_connection()
    ensure_image_column(conn)

    rows = conn.execute(
        "SELECT * FROM catalog WHERE image_url IS NULL OR image_url = ''"
    ).fetchall()
    total = len(rows)
    print(f"Found {total} catalog items without an image.")

    succeeded, failed = 0, 0
    for i, row in enumerate(rows, 1):
        row = dict(row)
        item_id = row["item_id"]
        prompt = build_sku_prompt(row)
        print(f"[{i}/{total}] {item_id} — {row['name']!r} ... ", end="", flush=True)
        try:
            data_uri = _call_flux(prompt, width=768, height=768, steps=25)
            conn.execute(
                "UPDATE catalog SET image_url = ? WHERE item_id = ?", (data_uri, item_id)
            )
            conn.commit()
            succeeded += 1
            print("OK")
        except Exception as e:
            failed += 1
            print(f"FAILED: {e}")
        time.sleep(1)  # be polite to the free endpoint

    conn.close()
    print(f"\nDone. {succeeded} succeeded, {failed} failed out of {total}.")


if __name__ == "__main__":
    main()
