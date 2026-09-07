"""
api.py — FastAPI wrapper exposing the design agent to the Lovable frontend.

Endpoints:
  GET  /briefs      -> list of available Living Room briefs (for the intake screen)
  POST /run-agent    -> runs the agent on a brief_id, returns the AgentResult JSON

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from tools import get_all_briefs, get_brief, _get_connection
from agent import run_agent

app = FastAPI(title="Interior Design Agent API")

# Allow the local Vite dev server (and any other origins listed via env) to call this API.
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
_extra_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
allow_origins = _default_origins + [o.strip() for o in _extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunAgentRequest(BaseModel):
    brief_id: str


class CustomBriefRequest(BaseModel):
    room_type: str
    length_cm: float
    width_cm: float
    ceiling_cm: float | None = None
    budget_inr: float
    style_preference: str
    must_haves: str
    constraints: str
    customer_notes: str = Field(default="", alias="customer_notes")

    class Config:
        populate_by_name = True


@app.post("/custom-brief")
def create_custom_brief(payload: CustomBriefRequest):
    """Insert a customer-submitted custom brief into room_briefs and return its new brief_id."""
    brief_id = f"CUSTOM-{uuid.uuid4().hex[:8].upper()}"
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO room_briefs
                (brief_id, room_type, length_cm, width_cm, ceiling_cm, budget_inr,
                 style_preference, must_haves, constraints, customer_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                brief_id,
                payload.room_type,
                payload.length_cm,
                payload.width_cm,
                payload.ceiling_cm,
                payload.budget_inr,
                payload.style_preference,
                payload.must_haves,
                payload.constraints,
                payload.customer_notes,
            ),
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save custom brief: {e}")
    finally:
        conn.close()

    return {"brief_id": brief_id}


@app.get("/room-types")
def list_room_types():
    """Distinct room types available, derived live from room_briefs (never hardcoded)."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "SELECT DISTINCT room_type FROM room_briefs WHERE room_type IS NOT NULL ORDER BY room_type"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


@app.get("/catalog")
def list_catalog(room_type: str | None = None, category: str | None = None):
    """Catalog items, optionally filtered by room_type and/or category."""
    from tools import catalog_search
    return catalog_search(room_type=room_type, category=category, in_stock_only=False)


class RoomImageRequest(BaseModel):
    brief_id: str
    selected_items: list[dict]


@app.post("/room-image")
def create_room_image(payload: RoomImageRequest):
    """
    Generate a full-room render image, grounded only in the brief's own fields
    and the SKUs actually selected by the agent for this run.
    """
    from room_image import generate_room_image

    brief = get_brief(payload.brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{payload.brief_id}' not found.")

    try:
        image_url = generate_room_image(brief, payload.selected_items)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Room image generation failed: {e}")

    return {"image_url": image_url}


@app.get("/sku-image/{item_id}")
def get_sku_image(item_id: str):
    """
    Return the cached image_url for a real catalog SKU. Never generates on the
    fly — only returns what generate_catalog_images.py already produced and
    stored against this item_id.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT image_url FROM catalog WHERE item_id = ?", (item_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail=f"SKU '{item_id}' not found in catalog.")

    return {"item_id": item_id, "image_url": row[0]}


class GenerateSkuImageRequest(BaseModel):
    item_id: str
    room_type: str | None = "Living Room"
    style_preference: str | None = ""
    customer_notes: str | None = ""


@app.post("/generate-sku-image")
def generate_sku_image_alias(payload: GenerateSkuImageRequest):
    """
    Generate or return image for catalog SKU using fallback pipeline.
    """
    from room_image import _call_flux
    from tools import get_catalog_item
    item = get_catalog_item(payload.item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"SKU '{payload.item_id}' not found in catalog.")

    # Check if DB has cached image_url
    conn = _get_connection()
    try:
        row = conn.execute("SELECT image_url FROM catalog WHERE item_id = ?", (payload.item_id,)).fetchone()
        if row and row[0]:
            return {"item_id": payload.item_id, "name": item.get("name"), "category": item.get("category"), "image_url": row[0]}
    finally:
        conn.close()

    # Generate image using _call_flux with 3-tier fallback
    prompt = f"Professional studio photo of {item.get('name')}, {item.get('category')} in {payload.style_preference or item.get('style_tags', '')} style."
    data_url = _call_flux(prompt)
    return {"item_id": payload.item_id, "name": item.get("name"), "category": item.get("category"), "image_url": data_url}


class GenerateRoomRenderRequest(BaseModel):
    brief_id: str | None = None
    room_type: str | None = "Living Room"
    style_preference: str | None = ""
    item_ids: list[str] | None = None
    customer_notes: str | None = ""


@app.post("/generate-room-render")
def generate_room_render_alias(payload: GenerateRoomRenderRequest):
    """
    Generate room render using fallback pipeline.
    """
    from room_image import generate_room_image
    brief = get_brief(payload.brief_id) if payload.brief_id else None
    if not brief:
        brief = {
            "room_type": payload.room_type or "Living Room",
            "style_preference": payload.style_preference or "Modern",
            "customer_note": payload.customer_notes or "",
        }

    selected_items = []
    if payload.item_ids:
        from tools import get_catalog_item
        for iid in payload.item_ids:
            it = get_catalog_item(iid)
            if it: selected_items.append(it)

    image_url = generate_room_image(brief, selected_items)
    return {"room_type": payload.room_type, "style_preference": payload.style_preference, "items_included": [it.get("name") for it in selected_items], "image_url": image_url}



@app.get("/briefs")
def list_briefs():
    """Return all briefs (every room_type) for the intake screen's brief picker."""
    try:
        return get_all_briefs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load briefs: {e}")


@app.post("/run-agent")
def run_agent_endpoint(payload: RunAgentRequest):
    """Run the design agent on a given brief_id and return the structured result."""
    brief = get_brief(payload.brief_id)
    if brief is None:
        raise HTTPException(status_code=404, detail=f"Brief '{payload.brief_id}' not found.")

    try:
        result = run_agent(brief_id=payload.brief_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent run failed: {e}")

    return result


@app.get("/health")
def health():
    return {"status": "ok"}
