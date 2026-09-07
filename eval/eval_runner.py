"""
eval_runner.py — Automated evaluation harness for AI Interior Design Agent.

Runs all test cases from golden_cases.json, executes deterministic code scorers
+ LLM-as-judge scoring, checks the ship gate, and produces an evidence report.

Ship gate:
- Deterministic code scorers pass rate >= 95.0%
- LLM-as-judge average score >= 3.5 / 5.0
"""

import json
import os
import sys
import time

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent import run_agent, _get_client
from eval.scorers import (
    score_budget_never_exceeded,
    score_all_items_real,
    score_items_fit_room,
    score_guardrails,
    score_exclusions,
    score_tool_usage,
    score_llm_judge,
)

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_CASES_PATH = os.path.join(EVAL_DIR, "golden_cases.json")


def run_eval(cases_to_run: list = None, run_llm_judge: bool = True) -> dict:
    """Run evaluation on golden cases and return comprehensive results."""
    with open(GOLDEN_CASES_PATH, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    if cases_to_run:
        cases = [c for c in all_cases if c["id"] in cases_to_run]
    else:
        cases = all_cases

    client, model_name = None, None
    if run_llm_judge:
        try:
            client, model_name = _get_client()
        except Exception as e:
            print(f"Warning: could not initialize LLM judge client ({e}).")

    print(f"\n{'='*75}")
    print(f"  AI INTERIOR DESIGN AGENT — EVALUATION HARNESS")
    print(f"  Total Test Cases: {len(cases)} golden cases")
    print(f"{'='*75}\n")

    results_table = []
    code_scorer_passes = 0
    total_code_tests = 0
    llm_scores = []

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        cname = case["name"]
        print(f"[{i}/{len(cases)}] Running {cid}: {cname}...", end="", flush=True)

        start_time = time.time()
        try:
            if "brief_id" in case:
                agent_output = run_agent(brief_id=case["brief_id"])
            else:
                agent_output = run_agent(free_text=case["free_text"])
        except Exception as e:
            agent_output = {"error": str(e), "feasible": False, "tool_call_log": []}

        elapsed = round(time.time() - start_time, 2)
        expected = case.get("expected", {})

        # Run 6 deterministic code scorers
        s_budget = score_budget_never_exceeded(agent_output, expected)
        s_catalog = score_all_items_real(agent_output, expected)
        s_fit = score_items_fit_room(agent_output, expected)
        s_guardrails = score_guardrails(agent_output, expected)
        s_exclusions = score_exclusions(agent_output, expected)
        s_tools = score_tool_usage(agent_output, expected)

        case_code_passes = sum([
            s_budget["passed"],
            s_catalog["passed"],
            s_fit["passed"],
            s_guardrails["passed"],
            s_exclusions["passed"],
            s_tools["passed"],
        ])
        code_scorer_passes += case_code_passes
        total_code_tests += 6

        # Run LLM judge
        s_llm = {"passed": True, "score": 4.0, "reason": "Skipped"}
        if run_llm_judge:
            s_llm = score_llm_judge(agent_output, client, model_name)
            llm_scores.append(s_llm["score"])

        all_passed = case_code_passes == 6 and s_llm["passed"]
        status_sym = "✅ PASS" if all_passed else "❌ FAIL"
        print(f" {status_sym} ({elapsed}s, {len(agent_output.get('tool_call_log', []))} tool calls)")

        results_table.append({
            "id": cid,
            "name": cname,
            "feasible": agent_output.get("feasible", False),
            "tool_calls": len(agent_output.get("tool_call_log", [])),
            "elapsed_s": elapsed,
            "budget_pass": s_budget["passed"],
            "catalog_pass": s_catalog["passed"],
            "fit_pass": s_fit["passed"],
            "guardrail_pass": s_guardrails["passed"],
            "exclusion_pass": s_exclusions["passed"],
            "tool_use_pass": s_tools["passed"],
            "llm_score": s_llm["score"],
            "llm_feedback": s_llm["reason"][:80],
            "agent_output": agent_output,
        })

    # Summary calculations
    code_pass_rate = round((code_scorer_passes / total_code_tests) * 100, 1) if total_code_tests > 0 else 0.0
    avg_llm_score = round(sum(llm_scores) / len(llm_scores), 2) if llm_scores else 0.0
    ship_gate_met = code_pass_rate >= 95.0 and (avg_llm_score >= 3.5 if run_llm_judge else True)

    print(f"\n{'='*75}")
    print(f"  EVALUATION SUMMARY & SHIP GATE REPORT")
    print(f"{'='*75}")
    print(f"  Total Code Tests:        {total_code_tests} tests across {len(cases)} cases")
    print(f"  Code Scorers Pass Rate:  {code_pass_rate}% ({code_scorer_passes}/{total_code_tests} passed)")
    print(f"  Target Code Pass Rate:   >= 95.0%")
    if run_llm_judge:
        print(f"  Average LLM-Judge Score: {avg_llm_score} / 5.0 (Target: >= 3.5)")
    print(f"  Ship Gate Decision:      {'🚀 READY TO SHIP' if ship_gate_met else '🛑 SHIP GATE BLOCKED'}")
    print(f"{'='*75}\n")

    return {
        "code_pass_rate": code_pass_rate,
        "avg_llm_score": avg_llm_score,
        "ship_gate_met": ship_gate_met,
        "results": results_table,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--cases", nargs="+", help="Specific case IDs to run (e.g. BR-01 ADV-01)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring")
    args = parser.parse_args()

    eval_result = run_eval(cases_to_run=args.cases, run_llm_judge=not args.no_judge)
    sys.exit(0 if eval_result["ship_gate_met"] else 1)
