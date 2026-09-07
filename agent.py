"""
agent.py — Core AI Interior Design Agent.

Implements the 8-step agent loop using OpenAI / NVIDIA NIM API for orchestration.
Uses exactly 3 tools: catalog_search, budget_calculator, fit_check.

Steps:
  0. Normalize input (brief_id lookup or free-text extraction)
  2. Parse must-haves into inclusions/exclusions
  3. Screen guardrails (hardcoded + LLM)
  4. Search, select, check incrementally (per item)
  5. Re-plan on failure (5-iteration hard cap per category)
  6. Feasibility stop condition
  7. Disclose out-of-stock items
  8. Structured output + designer-voice rationale
"""

import json
import os
import copy
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from tools import (
    catalog_search,
    budget_calculator,
    fit_check,
    get_brief,
    get_all_living_room_briefs,
    _get_connection,
    TOOL_DECLARATIONS,
)
from guardrails import run_all_guardrails


# ---------------------------------------------------------------------------
# OpenAI / NVIDIA NIM client setup
# ---------------------------------------------------------------------------
def _get_client():
    """Returns initialized OpenAI client and model name."""
    base_url = os.getenv("NVIDIA_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    api_key = os.getenv("NVIDIA_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    model_name = os.getenv("NVIDIA_MODEL", os.getenv("OPENAI_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"))

    if not api_key:
        raise ValueError("No API key found. Set NVIDIA_API_KEY or OPENAI_API_KEY in .env.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name


# ---------------------------------------------------------------------------
# System prompt — encodes all agent discipline
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert interior designer AI agent for a home furnishing company.

## YOUR ROLE
You receive a customer brief for ANY room type (Living Room, Bedroom, Dining, Study, Kids, etc.) and produce a ready-to-buy Bill of Quantities (BOQ) using ONLY items from our catalog database. You have 3 tools: catalog_search, budget_calculator, fit_check.

The brief tells you the room_type — always pass that exact room_type value to catalog_search so results are filtered to items valid for that room. Category expectations differ by room_type (e.g., a Kids room may need a study table + bed + storage; a Dining room needs a dining table + chairs + sideboard; a Study needs a desk + chair + shelving) — infer sensible categories from the room_type and the brief's must-haves, never from a fixed template.

## CRITICAL RULES — NEVER VIOLATE
1. **NEVER FABRICATE**: Only recommend items returned by catalog_search. Never invent item IDs, names, or prices.
2. **NEVER HIDE TRADE-OFFS**: If budget is exceeded, room doesn't fit, or style doesn't match — say so explicitly.
3. **INCREMENTAL CHECKING**: After selecting EACH item, call BOTH budget_calculator AND fit_check immediately with all items selected so far. Do not batch-check only at the end.
4. **MUST-HAVES FIRST**: Always fund and place must-have items before nice-to-haves/accessories.
5. **EXCLUSIONS**: If the brief says "no X" (e.g., "no TV"), suppress that category entirely.
6. **RUNNING TOTAL**: Track running_total after each item addition in the BOQ.

## PROCESS (follow strictly)

### Pre-Planning (internal reasoning)
Estimate a rough budget split and footprint split across planned categories. Must-haves get priority allocation.

### Item Selection Loop
For each planned category (must-haves first):
1. Call `catalog_search` with appropriate filters (category, style, max_price, room_type="Living Room", in_stock_only=True)
2. Pick the best matching item from the search results.
3. Call `budget_calculator` with ALL currently selected items + customer's budget.
4. Call `fit_check` with ALL currently selected items + room length & width.
5. If over_budget or fits=false:
   - Try a cheaper/smaller alternative in the same category (up to 5 attempts per category).
   - If no alternative works and it is a non-must-have, drop it.
   - If it is a must-have and cannot fit/afford, mark feasible=false and explain why.

### Hard Cap & Infeasibility
- Maximum 5 re-plan iterations per category.
- If must-haves cannot be satisfied within budget/space or do not exist in catalog, set `"feasible": false` and present the closest realistic partial plan.

### Out-of-Stock Handling
If an out-of-stock item is the only viable option, call `catalog_search(..., in_stock_only=False)`, select it, and explicitly disclose its lead_time_days and why it was selected.

## FINAL RESPONSE FORMAT
When you are done with all tool calls and have your final selection, output ONLY a valid JSON object matching this exact schema:

```json
{
  "brief_id": "string or null",
  "selected_items": [
    {
      "item_id": "SOF-001",
      "category": "Sofa",
      "name": "Nordby 3-Seater Fabric Sofa",
      "price_inr": 58000,
      "in_stock": 1,
      "lead_time_days": 21,
      "running_total": 58000,
      "justification": "must_have"
    }
  ],
  "budget_summary": {
    "total_spent": 58000,
    "budget": 250000,
    "remaining": 192000,
    "over_budget": false,
    "items_missing_price": [],
    "sub_budget_allocation": {"Sofa": 58000}
  },
  "fit_summary": {
    "fits": true,
    "footprint_used_pct": 10.9,
    "oversized_items": [],
    "sub_footprint_allocation": {"Sofa": 10.9}
  },
  "rationale": "Designer explanation explaining why each piece was chosen, why this budget split, why this footprint split.",
  "trade_offs": "Explanation of trade-offs made or none.",
  "preference_deviation_flagged": false,
  "guardrails_triggered": [],
  "assumptions_made": [],
  "feasible": true,
  "iterations_used": 1
}
```
"""


# ---------------------------------------------------------------------------
# Tool execution dispatcher
# ---------------------------------------------------------------------------
def _execute_tool(tool_name: str, args: dict, default_room_type: str = None) -> dict | list:
    """Execute a tool by name and return the result."""
    if tool_name == "catalog_search":
        category = args.get("category")
        style = args.get("style")
        max_price = args.get("max_price")
        room_type = args.get("room_type") or default_room_type or "Living Room"
        in_stock_only = args.get("in_stock_only", True)
        if isinstance(in_stock_only, str):
            in_stock_only = in_stock_only.lower() in ("true", "1", "yes")
        return catalog_search(
            category=category,
            style=style,
            max_price=max_price,
            room_type=room_type,
            in_stock_only=in_stock_only,
        )
    elif tool_name == "budget_calculator":
        items = args.get("selected_items", [])
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:
                items = []
        budget = float(args.get("budget_inr", 0))
        return budget_calculator(items, budget)
    elif tool_name == "fit_check":
        items = args.get("selected_items", [])
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:
                items = []
        length = float(args.get("room_length_cm", 0))
        width = float(args.get("room_width_cm", 0))
        return fit_check(items, length, width)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _get_openai_tools():
    """Convert TOOL_DECLARATIONS to OpenAI tools schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": td["name"],
                "description": td["description"],
                "parameters": td["parameters"],
            },
        }
        for td in TOOL_DECLARATIONS
    ]


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------
def run_agent(brief_id: str = None, free_text: str = None) -> dict:
    """
    Run the design agent on a brief.

    Args:
        brief_id: A brief ID (e.g., "BR-01") to look up from the database.
        free_text: Free-text brief description (used if brief_id is None).

    Returns:
        The structured output dict (Section 4 schema) with tool_call_log appended.
    """
    client, model_name = _get_client()
    tool_call_log = []

    # -----------------------------------------------------------------------
    # Step 0 — Normalize input
    # -----------------------------------------------------------------------
    brief = None
    assumptions_made = []

    if brief_id:
        brief = get_brief(brief_id)
        if not brief:
            return {
                "brief_id": brief_id,
                "error": f"Brief '{brief_id}' not found in database.",
                "feasible": False,
                "tool_call_log": [],
            }
        # Room type is no longer restricted to Living Room — the agent now
        # handles any room_type present in room_briefs (Living Room, Bedroom,
        # Dining, Study, Kids, etc.), driven entirely by the brief itself.
    elif free_text:
        brief = _extract_brief_from_text(free_text, client, model_name)
        assumptions_made.append("Brief fields extracted from free text via LLM — verify accuracy.")
    else:
        return {
            "error": "No brief_id or free_text provided.",
            "feasible": False,
            "tool_call_log": [],
        }

    # -----------------------------------------------------------------------
    # Step 3 — Run hardcoded guardrails BEFORE planning
    # -----------------------------------------------------------------------
    guardrail_results = run_all_guardrails(brief)
    guardrails_triggered = list(guardrail_results["guardrails_triggered"])
    guardrail_messages = guardrail_results["messages"]

    # -----------------------------------------------------------------------
    # Build initial message
    # -----------------------------------------------------------------------
    brief_summary = (
        f"Brief ID: {brief.get('brief_id', 'N/A')}\n"
        f"Room Type: {brief.get('room_type', 'Living Room')}\n"
        f"Room Dimensions: {brief.get('length_cm', 0)}cm (L) × {brief.get('width_cm', 0)}cm (W) × {brief.get('ceiling_cm', 0)}cm (ceiling)\n"
        f"Budget: ₹{brief.get('budget_inr', 0):,}\n"
        f"Style Preference: {brief.get('style_preference', 'Not specified')}\n"
        f"Must-Haves: {brief.get('must_haves', 'None specified')}\n"
        f"Constraints: {brief.get('constraints', 'None')}\n"
        f"Customer Note: {brief.get('customer_note', 'None')}\n"
    )

    guardrail_context = ""
    if guardrails_triggered:
        guardrail_context = (
            "\n\n## PRE-CHECKED GUARDRAILS TRIGGERED:\n"
            + "\n".join(f"- [{g}]: {m}" for g, m in zip(guardrails_triggered, guardrail_messages))
            + "\n\n(Make sure to include these in guardrails_triggered in your output JSON and address them in rationale/trade_offs.)"
        )

    room_type_for_prompt = brief.get("room_type", "room")
    user_prompt = (
        f"## Customer Brief\n{brief_summary}{guardrail_context}\n\n"
        f"Please build the complete {room_type_for_prompt} design. Follow the agent loop: search catalog "
        f"filtering by room_type=\"{room_type_for_prompt}\", add items one by one, verify budget and fit "
        "after each item. Return the final JSON output."
    )

    messages = [
        {"role": "system", "content": "detailed thinking off"},
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    tools = _get_openai_tools()

    # -----------------------------------------------------------------------
    # Steps 4-8 — Agent loop with tool calling
    # -----------------------------------------------------------------------
    max_turns = 35
    turn = 0
    final_text = ""

    while turn < max_turns:
        turn += 1
        import sys as _sys
        print(f"[turn {turn}/{max_turns}] calling model...", file=_sys.stderr, flush=True)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as e:
            final_text = f"Error during model completion: {e}"
            break

        choice = response.choices[0]
        message = choice.message

        # Check if the model requested tool calls
        if message.tool_calls:
            # Append cleanly formatted assistant message
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except Exception:
                    fn_args = {}

                # Execute tool
                tool_result = _execute_tool(fn_name, fn_args, default_room_type=brief.get("room_type"))
                summary = _summarize_output(fn_name, tool_result)

                # Live progress to stderr
                import sys
                print(f"  ↳ [{turn}] {fn_name}: {summary}", file=sys.stderr, flush=True)

                # Log tool call
                tool_call_log.append({
                    "tool": fn_name,
                    "input": fn_args,
                    "output_summary": summary,
                    "output": tool_result,
                })

                # Append tool response message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": json.dumps(tool_result, default=str),
                })
        else:
            # Final text response reached
            final_text = message.content or ""
            break

    # If loop ended without final text (e.g. hit max turns), request final JSON explicitly
    if not final_text.strip() or "selected_items" not in final_text:
        messages.append({
            "role": "user",
            "content": "Please generate your final response JSON now matching the required schema based on the catalog searches and checks performed above. Output only the JSON object.",
        })
        try:
            wrap_resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            final_text = wrap_resp.choices[0].message.content or ""
        except Exception:
            # Some models/endpoints reject response_format — retry without it.
            try:
                wrap_resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2048,
                )
                final_text = wrap_resp.choices[0].message.content or ""
            except Exception as e:
                final_text = f"Error during final response generation: {e}"

    # Parse and format output schema
    result = _parse_agent_output(
        final_text, brief, guardrails_triggered, assumptions_made, tool_call_log
    )
    return result


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _extract_brief_from_text(free_text: str, client: OpenAI, model_name: str) -> dict:
    """Extract brief fields from unstructured free text."""
    prompt = f"""Extract the following fields from this customer request for a room design.
Return ONLY a valid JSON object with these exact keys:
- room_type: string (infer from the text — e.g. "Living Room", "Bedroom", "Dining", "Study", "Kids". Default to "Living Room" only if genuinely unclear.)
- length_cm: number (room length in cm; 1 foot = 30.48 cm)
- width_cm: number (room width in cm)
- ceiling_cm: number (default 300)
- budget_inr: number (budget in INR)
- style_preference: string (style name)
- must_haves: string (comma-separated list of must-have items)
- constraints: string
- customer_note: string

Customer request:
{free_text}
"""
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=512,
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)
    except Exception:
        return {
            "room_type": "Living Room",
            "length_cm": 450,
            "width_cm": 350,
            "ceiling_cm": 300,
            "budget_inr": 200000,
            "style_preference": "Contemporary",
            "must_haves": "sofa, coffee table, tv unit",
            "constraints": "",
            "customer_note": free_text,
        }


def _summarize_output(tool_name: str, result) -> str:
    """Create a short summary of tool output for the log."""
    if tool_name == "catalog_search":
        if isinstance(result, list):
            if len(result) == 0:
                return "No items found (0 matches)"
            ids = [r.get("item_id", "?") for r in result[:4]]
            return f"Found {len(result)} items ({', '.join(ids)}{'...' if len(result) > 4 else ''})"
        return str(result)[:80]
    elif tool_name == "budget_calculator":
        return f"Spent ₹{result.get('total_spent',0):,} / ₹{result.get('budget',0):,} | Over: {result.get('over_budget', False)}"
    elif tool_name == "fit_check":
        return f"Footprint: {result.get('footprint_used_pct',0)}% | Fits: {result.get('fits', False)} | Oversized: {len(result.get('oversized_items', []))}"
    return str(result)[:80]


def _clean_json_string(text: str) -> str:
    """Extract JSON from model response text, removing thinking tokens and markdown."""
    if not text:
        return "{}"

    # Remove <think>...</think> blocks if present
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Look for ```json ... ``` blocks
    if "```json" in cleaned:
        match = re.search(r"```json\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()
        parts = cleaned.split("```json")
        if len(parts) > 1:
            sub = parts[-1].split("```")[0].strip()
            if "{" in sub and "}" in sub:
                return sub

    # Look for generic ``` ... ``` blocks
    if "```" in cleaned:
        matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if matches:
            return matches[-1].strip()

    # Look for the JSON object that actually contains "selected_items" by
    # scanning brace depth from each '{' — this avoids grabbing prose that
    # merely happens to contain stray braces before/after the real JSON.
    candidates = []
    for idx, ch in enumerate(cleaned):
        if ch == "{":
            depth = 0
            for j in range(idx, len(cleaned)):
                if cleaned[j] == "{":
                    depth += 1
                elif cleaned[j] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(cleaned[idx : j + 1])
                        break
    for c in candidates:
        if '"selected_items"' in c:
            return c

    # Fall back to largest JSON-looking block enclosed in { ... }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]

    return cleaned


def _parse_agent_output(
    text: str,
    brief: dict,
    guardrails_triggered: list,
    assumptions_made: list,
    tool_call_log: list,
) -> dict:
    """Parse the LLM's final text output into the structured schema."""
    json_str = _clean_json_string(text)
    result = None

    try:
        result = json.loads(json_str)
    except Exception:
        # Try finding json with regex if malformed
        import re
        m = re.search(r"(\{.*\})", json_str, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(1))
            except Exception:
                result = None

    if result is None or not isinstance(result, dict):
        # Log the raw text for debugging, but NEVER surface the model's raw
        # planning/reasoning narrative to the customer-facing rationale field.
        import sys as _sys
        print(f"[WARN] Could not parse agent JSON output. Raw text (truncated): {text[:500]}", file=_sys.stderr)
        result = {
            "brief_id": brief.get("brief_id"),
            "selected_items": [],
            "budget_summary": {},
            "fit_summary": {},
            "rationale": "We couldn't finalize a design plan for this brief. Please try again.",
            "trade_offs": "",
            "preference_deviation_flagged": False,
            "feasible": False,
            "iterations_used": 1,
        }

    # If selected_items is empty, recover from the latest valid fit_check/budget_calculator tool call
    if not result.get("selected_items"):
        for tc in reversed(tool_call_log):
            if tc.get("tool") in ("budget_calculator", "fit_check"):
                inp = tc.get("input", {})
                items_in_call = inp.get("selected_items", [])
                if isinstance(items_in_call, str):
                    try:
                        items_in_call = json.loads(items_in_call)
                    except Exception:
                        items_in_call = []
                if items_in_call and isinstance(items_in_call, list):
                    result["selected_items"] = items_in_call
                    break

    # Enrich selected items from catalog if fields are missing
    conn = _get_connection()
    try:
        cur = conn.cursor()
        for it in result.get("selected_items", []):
            iid = it.get("item_id")
            if iid:
                cur.execute("SELECT * FROM catalog WHERE item_id = ?", (iid,))
                row = cur.fetchone()
                if row:
                    row_dict = dict(row)
                    for k, v in row_dict.items():
                        if it.get(k) is None:
                            it[k] = v
            it.setdefault("justification", "must_have")
    finally:
        conn.close()

    # Reconcile running totals, budget, and fit
    items = result.get("selected_items", [])
    running = 0
    for it in items:
        p = it.get("price_inr")
        if p is not None:
            running += p
        it["running_total"] = running

    b_calc = budget_calculator(items, brief.get("budget_inr", 0))
    f_calc = fit_check(items, brief.get("length_cm", 0), brief.get("width_cm", 0))

    sub_budget = {}
    for it in items:
        cat = it.get("category", "Other")
        sub_budget[cat] = sub_budget.get(cat, 0) + (it.get("price_inr") or 0)

    room_area = (brief.get("length_cm", 0) * brief.get("width_cm", 0)) or 1
    sub_footprint = {}
    for it in items:
        cat = it.get("category", "Other")
        it_area = (it.get("width_cm", 0) or 0) * (it.get("depth_cm", 0) or 0)
        pct = round((it_area / room_area) * 100, 2)
        sub_footprint[cat] = round(sub_footprint.get(cat, 0.0) + pct, 2)

    result["budget_summary"] = {
        "total_spent": b_calc["total_spent"],
        "budget": b_calc["budget"],
        "remaining": b_calc["remaining"],
        "over_budget": b_calc["over_budget"],
        "items_missing_price": b_calc["items_missing_price"],
        "sub_budget_allocation": sub_budget,
    }
    result["fit_summary"] = {
        "fits": f_calc["fits"],
        "footprint_used_pct": f_calc["footprint_used_pct"],
        "oversized_items": f_calc["oversized_items"],
        "sub_footprint_allocation": sub_footprint,
    }

    # Check feasibility: if over budget or does not fit, set feasible = false
    if b_calc["over_budget"] or not f_calc["fits"]:
        result["feasible"] = False

    # Ensure all required top-level schema keys
    result.setdefault("brief_id", brief.get("brief_id"))
    result.setdefault("rationale", "Design curated to fit room footprint and budget constraints.")
    result.setdefault("trade_offs", "")
    result.setdefault("preference_deviation_flagged", False)
    result.setdefault("feasible", True)
    result.setdefault("iterations_used", 1)

    # Clean <think> tags from rationale and trade_offs
    import re
    if result.get("rationale"):
        result["rationale"] = re.sub(r"<think>.*?</think>", "", str(result["rationale"]), flags=re.DOTALL).strip()
    if result.get("trade_offs"):
        result["trade_offs"] = re.sub(r"<think>.*?</think>", "", str(result["trade_offs"]), flags=re.DOTALL).strip()

    # Merge guardrails
    existing_guardrails = result.get("guardrails_triggered", [])
    if isinstance(existing_guardrails, list):
        merged_guardrails = list(set(guardrails_triggered + existing_guardrails))
    else:
        merged_guardrails = guardrails_triggered
    result["guardrails_triggered"] = merged_guardrails

    # Merge assumptions
    existing_assumptions = result.get("assumptions_made", [])
    if isinstance(existing_assumptions, list):
        merged_assumptions = list(set(assumptions_made + existing_assumptions))
    else:
        merged_assumptions = assumptions_made
    result["assumptions_made"] = merged_assumptions

    # Append clean tool call log
    clean_log = []
    for entry in tool_call_log:
        clean_log.append({
            "tool": entry["tool"],
            "input": entry["input"],
            "output_summary": entry["output_summary"],
        })
    result["tool_call_log"] = clean_log

    return result
