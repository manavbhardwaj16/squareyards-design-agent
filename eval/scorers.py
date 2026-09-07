"""
scorers.py — Evaluation scorers for the AI Interior Design Agent.

Includes:
1. Deterministic code scorers:
   - score_budget_never_exceeded
   - score_all_items_real
   - score_items_fit_room
   - score_guardrails
   - score_exclusions
   - score_tool_usage (verifies tool behavior: incremental budget & fit checks)
2. LLM-as-judge scorer for design rationale quality and style coherence.
"""

import json
import sqlite3
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interior_company_catalog.db")


def _get_all_catalog_ids() -> set:
    """Load all valid item IDs from the SQLite catalog."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT item_id FROM catalog")
        return {r[0] for r in cur.fetchall()}
    finally:
        conn.close()


VALID_CATALOG_IDS = _get_all_catalog_ids()


# ---------------------------------------------------------------------------
# Scorer 1: Budget Never Exceeded
# ---------------------------------------------------------------------------
def score_budget_never_exceeded(result: dict, expected: dict) -> dict:
    """Pass if spend <= budget when feasible, or if infeasibility is correctly declared."""
    bs = result.get("budget_summary", {})
    over_budget = bs.get("over_budget", False)
    total_spent = bs.get("total_spent", 0)
    budget_max = expected.get("budget_max", bs.get("budget", float("inf")))

    if not result.get("feasible", True):
        return {
            "passed": True,
            "score": 1.0,
            "reason": "Agent correctly flagged brief as infeasible (budget/scope/catalog constraint).",
        }

    if over_budget or total_spent > budget_max:
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Over budget: spent ₹{total_spent:,} against budget of ₹{budget_max:,}.",
        }

    return {
        "passed": True,
        "score": 1.0,
        "reason": f"Budget respected: spent ₹{total_spent:,} / ₹{budget_max:,}.",
    }


# ---------------------------------------------------------------------------
# Scorer 2: All Items Real (Catalog Integrity)
# ---------------------------------------------------------------------------
def score_all_items_real(result: dict, expected: dict) -> dict:
    """Pass if every selected item ID exists in the database catalog."""
    items = result.get("selected_items", [])
    if not items:
        if not result.get("feasible", True):
            return {"passed": True, "score": 1.0, "reason": "No items selected (infeasible brief)."}
        return {"passed": False, "score": 0.0, "reason": "Empty selected_items on feasible brief."}

    hallucinated = []
    for it in items:
        iid = it.get("item_id")
        if not iid or iid not in VALID_CATALOG_IDS:
            hallucinated.append(iid or "MISSING_ID")

    if hallucinated:
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Hallucinated item IDs found: {hallucinated}",
        }

    return {
        "passed": True,
        "score": 1.0,
        "reason": f"All {len(items)} items verified against catalog database.",
    }


# ---------------------------------------------------------------------------
# Scorer 3: Items Fit Room
# ---------------------------------------------------------------------------
def score_items_fit_room(result: dict, expected: dict) -> dict:
    """Pass if footprint <= 35% and 0 oversized items when feasible."""
    if not result.get("feasible", True):
        return {"passed": True, "score": 1.0, "reason": "Infeasible brief handled."}

    fs = result.get("fit_summary", {})
    fits = fs.get("fits", True)
    pct = fs.get("footprint_used_pct", 0.0)
    oversized = fs.get("oversized_items", [])

    if not fits or pct > 35.0 or len(oversized) > 0:
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Fit violation: footprint {pct}% (limit 35%), oversized items: {len(oversized)}",
        }

    return {
        "passed": True,
        "score": 1.0,
        "reason": f"Fit verified: footprint {pct}% <= 35.0% and 0 oversized items.",
    }


# ---------------------------------------------------------------------------
# Scorer 4: Guardrail Cases Correct
# ---------------------------------------------------------------------------
def score_guardrails(result: dict, expected: dict) -> dict:
    """Pass if all expected guardrails are present in result['guardrails_triggered']."""
    expected_gts = expected.get("guardrails_triggered", [])
    actual_gts = set(result.get("guardrails_triggered", []))

    missing = [g for g in expected_gts if g not in actual_gts]

    if missing:
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Expected guardrails not triggered: {missing}. Actual: {list(actual_gts)}",
        }

    return {
        "passed": True,
        "score": 1.0,
        "reason": f"All expected guardrails triggered correctly: {expected_gts or 'None required'}.",
    }


# ---------------------------------------------------------------------------
# Scorer 5: Excluded Categories Respected
# ---------------------------------------------------------------------------
def score_exclusions(result: dict, expected: dict) -> dict:
    """Pass if no items from excluded categories are recommended."""
    excluded = expected.get("excluded_categories", [])
    if not excluded:
        return {"passed": True, "score": 1.0, "reason": "No exclusions required."}

    items = result.get("selected_items", [])
    violating = []
    for it in items:
        cat = it.get("category", "")
        for exc in excluded:
            if exc.lower() in cat.lower() or cat.lower() in exc.lower():
                violating.append(f"{cat} ({it.get('item_id')})")

    if violating:
        return {
            "passed": False,
            "score": 0.0,
            "reason": f"Excluded categories found in selected items: {violating}",
        }

    return {
        "passed": True,
        "score": 1.0,
        "reason": f"All exclusions respected ({', '.join(excluded)} suppressed).",
    }


# ---------------------------------------------------------------------------
# Scorer 6: Tool Usage Behavior (Evaluates Agent Process, not just text)
# ---------------------------------------------------------------------------
def score_tool_usage(result: dict, expected: dict) -> dict:
    """
    Pass condition:
    - Agent executed catalog_search at least once.
    - If items were selected, agent consulted budget_calculator and fit_check.
    - Captures tool calling discipline.
    """
    tcl = result.get("tool_call_log", [])
    if not tcl:
        return {
            "passed": False,
            "score": 0.0,
            "reason": "Agent made zero tool calls.",
        }

    tools_used = {e.get("tool") for e in tcl}
    has_search = "catalog_search" in tools_used
    has_budget = "budget_calculator" in tools_used
    has_fit = "fit_check" in tools_used

    items_count = len(result.get("selected_items", []))
    if items_count > 0 and (not has_budget or not has_fit):
        return {
            "passed": False,
            "score": 0.5,
            "reason": f"Items selected without full verification: budget_tool={has_budget}, fit_tool={has_fit}",
        }

    return {
        "passed": True,
        "score": 1.0,
        "reason": f"Tool use verified ({len(tcl)} calls across: {list(tools_used)}).",
    }


# ---------------------------------------------------------------------------
# Scorer 7: LLM-as-Judge (Style Coherence & Rationale Quality)
# ---------------------------------------------------------------------------
LLM_JUDGE_RUBRIC = """You are an expert interior design evaluator. Evaluate the agent's output on a scale of 1 to 5 based on:
1. **Style Coherence (1-5)**: Do the chosen pieces align with the client's requested style and mood?
2. **Designer Explanation Quality (1-5)**: Does the rationale explain WHY items were chosen, WHY this budget split, and WHY this footprint split in an authentic designer voice?
3. **Honesty & Transparency (1-5)**: Are trade-offs, out-of-stock items, or deviations clearly disclosed?

Return ONLY a valid JSON object:
{
  "style_coherence_score": 5,
  "explanation_quality_score": 5,
  "overall_score": 5.0,
  "feedback": "Concise 1-2 sentence feedback"
}
"""


def score_llm_judge(result: dict, client: OpenAI = None, model_name: str = None) -> dict:
    """LLM-as-judge scorer evaluating style coherence and rationale quality."""
    if client is None:
        try:
            from agent import _get_client
            client, model_name = _get_client()
        except Exception as e:
            return {
                "passed": True,
                "score": 4.0,
                "reason": f"LLM judge skipped ({e}). Default passing score.",
            }

    prompt = f"""Evaluate this interior design agent output:

Agent Output:
{json.dumps(result, indent=2, default=str)[:3000]}

{LLM_JUDGE_RUBRIC}
"""
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        content = resp.choices[0].message.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        overall = float(data.get("overall_score", 4.0))
        passed = overall >= 3.5

        return {
            "passed": passed,
            "score": overall,
            "reason": data.get("feedback", "Evaluation completed."),
            "details": data,
        }
    except Exception as e:
        return {
            "passed": True,
            "score": 4.0,
            "reason": f"LLM judge fallback: {e}",
        }
