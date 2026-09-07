"""
room_image.py — builds a grounded text-to-image prompt for the full room render,
using ONLY the customer's brief fields and the agent's actual selected_items —
never inventing furniture, colors, or layout details that aren't in that data.

Image generation uses NVIDIA Build's hosted FLUX.1-dev endpoint.
"""

import os
import base64
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

FLUX_INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"


def _get_client():
    return OpenAI(
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=os.getenv("NVIDIA_API_KEY", ""),
    )


def _call_flux(prompt: str, width: int = 1024, height: int = 1024, steps: int = 30) -> str:
    """
    Calls NVIDIA Build's hosted FLUX.1-dev endpoint and returns a data URI.
    Supports automatic fallback to NVIDIA_FALLBACK_API_KEY if primary API key fails or hits limits.
    """
    keys_to_try = [
        os.getenv("NVIDIA_API_KEY", ""),
        os.getenv("NVIDIA_FALLBACK_API_KEY", "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"),
        os.getenv("NVIDIA_FALLBACK_API_KEY_2", "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"),
    ]
    keys_to_try = [k for k in keys_to_try if k]

    if not keys_to_try:
        raise RuntimeError("No valid NVIDIA API key is configured.")

    last_err = None
    for api_key in keys_to_try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "mode": "base",
            "cfg_scale": 3.5,
            "width": width,
            "height": height,
            "seed": 0,
            "steps": steps,
        }

        try:
            resp = requests.post(FLUX_INVOKE_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code != 200:
                last_err = RuntimeError(f"NVIDIA Image Generation API failed ({resp.status_code}): {resp.text[:200]}")
                continue
            body = resp.json()

            b64_image = None
            if isinstance(body, dict):
                if "artifacts" in body and body["artifacts"]:
                    b64_image = body["artifacts"][0].get("base64")
                elif "image" in body:
                    b64_image = body["image"]
                elif "b64_json" in body:
                    b64_image = body["b64_json"]

            if b64_image:
                return f"data:image/png;base64,{b64_image}"
            else:
                last_err = RuntimeError(f"Unexpected FLUX response shape: {list(body.keys()) if isinstance(body, dict) else type(body)}")
        except Exception as e:
            last_err = e

    raise last_err if last_err else RuntimeError("All NVIDIA API keys failed.")


def build_room_image_prompt(brief: dict, selected_items: list[dict]) -> str:
    """
    Step 1 of the pipeline: ask an LLM to compose a single, well-formed
    text-to-image prompt, but constrain its inputs to only real fields —
    the brief's own tags/notes and the SKUs actually selected by the agent.
    """
    room_type = brief.get("room_type", "room")
    style = brief.get("style_preference", "")
    notes = brief.get("customer_note") or brief.get("customer_notes") or ""
    must_haves = brief.get("must_haves", "")
    length_cm = brief.get("length_cm")
    width_cm = brief.get("width_cm")

    # Only real, selected item names/categories/colors go into the prompt —
    # nothing the agent didn't actually pick.
    item_lines = []
    for item in selected_items:
        name = item.get("name", "")
        category = item.get("category", "")
        finish = item.get("color_finish", "")
        if name:
            line = f"- {name} ({category})" + (f", {finish}" if finish else "")
            item_lines.append(line)
    items_block = "\n".join(item_lines) if item_lines else "no items selected yet"

    composer_prompt = f"""You are writing a single text-to-image generation prompt for an interior
design render. Use ONLY the facts given below — do not invent furniture, colors, or
layout details that aren't listed. Do not add items that aren't in the selected items list.

Room type: {room_type}
Room size: {length_cm}cm x {width_cm}cm (if known)
Style preference: {style}
Customer notes/tags: {notes}
Must-haves mentioned: {must_haves}

Selected items actually in this design:
{items_block}

Write ONE paragraph (60-90 words) describing this exact room as a photorealistic
interior design render: the room type, the style, and the specific selected items
with their real colors/finishes, arranged naturally. Do not mention brand names,
prices, or anything not listed above. Output ONLY the prompt text, nothing else."""

    client = _get_client()
    model_name = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-nano-30b-a3b")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "detailed thinking off"},
            {"role": "user", "content": composer_prompt},
        ],
        temperature=0.4,
        max_tokens=200,
    )
    image_prompt = (resp.choices[0].message.content or "").strip()
    return image_prompt


def generate_room_image(brief: dict, selected_items: list[dict]) -> str:
    """
    Step 2: send the composed prompt to FLUX.1-dev on NVIDIA Build.
    Returns a data URI the frontend can drop straight into an <img src="...">.
    """
    image_prompt = build_room_image_prompt(brief, selected_items)
    return _call_flux(image_prompt, width=1024, height=1024, steps=30)
