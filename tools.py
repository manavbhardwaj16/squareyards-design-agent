"""
tools.py — The 3 graded tools for the AI Interior Design Agent.

Tool 1: catalog_search  — parameterized SQL against the catalog
Tool 2: budget_calculator — running budget check with NULL-price handling
Tool 3: fit_check — footprint heuristic against room floor area (35% threshold)
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interior_company_catalog.db")


def _get_connection():
    """Returns a new SQLite connection with row_factory for dict results."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Tool 1 — catalog_search
# ---------------------------------------------------------------------------
def catalog_search(
    category: str = None,
    style: str = None,
    max_price: float = None,
    room_type: str = "Living Room",
    in_stock_only: bool = True,
) -> list[dict]:
    """
    Search the catalog with optional filters. All filters are combinable.

    Returns full rows INCLUDING NULL-price items — never filtered out silently.
    Returns [] when nothing matches (never raises).
    """
    conn = _get_connection()
    try:
        clauses = []
        params = []

        if category:
            clauses.append("category = ?")
            params.append(category)

        if style:
            # style_tags is comma-separated; use LIKE for substring match
            clauses.append("style_tags LIKE ?")
            params.append(f"%{style}%")

        if max_price is not None:
            # Include NULL-price items — they pass the filter (we don't know their price)
            clauses.append("(price_inr <= ? OR price_inr IS NULL)")
            params.append(max_price)

        if room_type:
            clauses.append("room_types LIKE ?")
            params.append(f"%{room_type}%")

        if in_stock_only:
            clauses.append("in_stock = 1")

        where = " AND ".join(clauses) if clauses else "1=1"
        query = f"SELECT * FROM catalog WHERE {where} ORDER BY category, price_inr"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool 2 — budget_calculator
# ---------------------------------------------------------------------------
def budget_calculator(selected_items: list[dict], budget_inr: float) -> dict:
    """
    Calculate budget usage for a list of selected items.

    - NULL-price items are excluded from total_spent, listed separately in items_missing_price.
    - over_budget = True the instant total_spent > budget_inr.
    """
    total_spent = 0
    items_missing_price = []

    for item in selected_items:
        price = item.get("price_inr")
        if price is None:
            items_missing_price.append(item.get("item_id", "UNKNOWN"))
        else:
            total_spent += price

    remaining = budget_inr - total_spent
    over_budget = total_spent > budget_inr

    return {
        "total_spent": total_spent,
        "budget": budget_inr,
        "remaining": remaining,
        "over_budget": over_budget,
        "items_missing_price": items_missing_price,
    }


# ---------------------------------------------------------------------------
# Tool 3 — fit_check
# ---------------------------------------------------------------------------
FOOTPRINT_THRESHOLD_PCT = 35.0  # Author's judgment call: 35% of floor area for circulation


def fit_check(
    selected_items: list[dict],
    room_length_cm: float,
    room_width_cm: float,
) -> dict:
    """
    Check whether selected items fit the room.

    Footprint heuristic:
      - Sum (width_cm × depth_cm) per item vs. room floor area.
      - fits = False if total footprint exceeds 35% of floor area.
      - Flags oversized_items — any item whose single dimension exceeds
        the room's corresponding dimension.

    Returns {"fits": bool, "footprint_used_pct": float, "oversized_items": []}.
    """
    room_area = room_length_cm * room_width_cm
    total_footprint = 0
    oversized_items = []

    for item in selected_items:
        item_w = item.get("width_cm", 0) or 0
        item_d = item.get("depth_cm", 0) or 0
        item_footprint = item_w * item_d
        total_footprint += item_footprint

        # Check if any single dimension exceeds room dimensions
        # Compare the item's larger dimension against the room's larger dimension,
        # and the item's smaller dimension against the room's smaller dimension.
        item_max = max(item_w, item_d)
        item_min = min(item_w, item_d)
        room_max = max(room_length_cm, room_width_cm)
        room_min = min(room_length_cm, room_width_cm)

        if item_max > room_max or item_min > room_min:
            oversized_items.append({
                "item_id": item.get("item_id", "UNKNOWN"),
                "name": item.get("name", "UNKNOWN"),
                "item_dims": f"{item_w}x{item_d} cm",
                "room_dims": f"{room_length_cm}x{room_width_cm} cm",
            })

    footprint_pct = (total_footprint / room_area * 100) if room_area > 0 else 100.0
    fits = footprint_pct <= FOOTPRINT_THRESHOLD_PCT and len(oversized_items) == 0

    return {
        "fits": fits,
        "footprint_used_pct": round(footprint_pct, 2),
        "oversized_items": oversized_items,
    }


# ---------------------------------------------------------------------------
# Helper: get brief by ID
# ---------------------------------------------------------------------------
def get_brief(brief_id: str) -> dict | None:
    """Fetch a room brief by ID. Returns None if not found."""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT * FROM room_briefs WHERE brief_id = ?", (brief_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_living_room_briefs() -> list[dict]:
    """Fetch all Living Room briefs. Kept for backward compatibility."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM room_briefs WHERE room_type LIKE '%Living%' ORDER BY brief_id"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_all_briefs() -> list[dict]:
    """Fetch ALL briefs across every room_type (Living Room, Bedroom, Dining, Study, Kids, etc.)."""
    conn = _get_connection()
    try:
        cursor = conn.execute("SELECT * FROM room_briefs ORDER BY brief_id")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tool metadata for LLM function-calling declarations
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS = [
    {
        "name": "catalog_search",
        "description": (
            "Search the interior design catalog for items. All filters are optional and combinable. "
            "Returns full item rows including NULL-price items. Returns empty list if nothing matches. "
            "Use in_stock_only=false to check if an item exists anywhere (even out of stock) for honest messaging."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Filter by exact category name (e.g. 'Sofa', 'Coffee Table', 'TV Unit', 'Rug', 'Floor Lamp', 'Armchair', 'Bookshelf', 'Wall Art', 'Cushions', 'Ottoman', 'Side Table', 'Curtains', 'Planter', 'Mirror', 'Pendant Light', 'Table Lamp', 'Console', 'Bean Bag').",
                },
                "style": {
                    "type": "string",
                    "description": "Filter by style tag substring (e.g. 'Scandinavian', 'Mid-Century', 'Bohemian', 'Industrial', 'Contemporary', 'Traditional', 'Minimalist', 'Coastal').",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price in INR. NULL-price items are still included.",
                },
                "room_type": {
                    "type": "string",
                    "description": "Filter by room type substring. Defaults to 'Living Room'.",
                },
                "in_stock_only": {
                    "type": "boolean",
                    "description": "If true (default), only return in-stock items. Set to false to check existence regardless of stock.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "budget_calculator",
        "description": (
            "Calculate budget usage for selected items. NULL-price items are excluded from total "
            "and listed separately. Returns over_budget=true the instant total exceeds budget."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "selected_items": {
                    "type": "array",
                    "description": "List of selected item dicts with at least 'item_id' and 'price_inr' fields.",
                    "items": {"type": "object"},
                },
                "budget_inr": {
                    "type": "number",
                    "description": "The customer's total budget in INR.",
                },
            },
            "required": ["selected_items", "budget_inr"],
        },
    },
    {
        "name": "fit_check",
        "description": (
            "Check whether selected items fit a room. Uses footprint heuristic: sum of (width×depth) "
            "per item vs room floor area. fits=false if footprint exceeds 35% of floor area. "
            "Also flags oversized items whose dimensions exceed the room."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "selected_items": {
                    "type": "array",
                    "description": "List of selected item dicts with 'width_cm' and 'depth_cm' fields.",
                    "items": {"type": "object"},
                },
                "room_length_cm": {
                    "type": "number",
                    "description": "Room length in cm.",
                },
                "room_width_cm": {
                    "type": "number",
                    "description": "Room width in cm.",
                },
            },
            "required": ["selected_items", "room_length_cm", "room_width_cm"],
        },
    },
]
