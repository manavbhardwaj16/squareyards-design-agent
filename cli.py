"""
cli.py — CLI runner for the AI Interior Design Agent.

Usage:
  python cli.py --brief BR-01              # Single brief by ID
  python cli.py --all                       # All Living Room briefs
  python cli.py --text "I need a ..."       # Free-text brief
"""

import argparse
import json
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agent import run_agent
from tools import get_all_briefs


def _print_human_summary(result: dict, file=sys.stderr):
    """Print a human-readable summary to stderr."""
    bid = result.get("brief_id", "N/A")
    feasible = result.get("feasible", False)

    print(f"\n{'='*60}", file=file)
    print(f"  DESIGN RESULT — Brief: {bid}", file=file)
    print(f"{'='*60}", file=file)

    if result.get("error"):
        print(f"  ❌ ERROR: {result['error']}", file=file)
        return

    status = "✅ FEASIBLE" if feasible else "⚠️  INFEASIBLE"
    print(f"  Status: {status}", file=file)

    # Budget
    bs = result.get("budget_summary", {})
    budget = bs.get("budget", 0)
    spent = bs.get("total_spent", 0)
    remaining = bs.get("remaining", 0)
    over = bs.get("over_budget", False)
    print(f"\n  💰 Budget: ₹{budget:,.0f}", file=file)
    print(f"     Spent:  ₹{spent:,.0f}", file=file)
    print(f"     Left:   ₹{remaining:,.0f} {'🔴 OVER BUDGET!' if over else '🟢'}", file=file)
    if bs.get("items_missing_price"):
        print(f"     ⚠️  Missing prices: {', '.join(bs['items_missing_price'])}", file=file)

    # Fit
    fs = result.get("fit_summary", {})
    fits = fs.get("fits", True)
    pct = fs.get("footprint_used_pct", 0)
    print(f"\n  📐 Footprint: {pct}% of floor area {'🟢' if fits else '🔴 DOES NOT FIT!'}", file=file)
    if fs.get("oversized_items"):
        for oi in fs["oversized_items"]:
            print(f"     ⚠️  Oversized: {oi.get('name','?')} ({oi.get('item_dims','?')})", file=file)

    # Selected items
    items = result.get("selected_items", [])
    if items:
        print(f"\n  📋 BOQ ({len(items)} items):", file=file)
        print(f"  {'─'*56}", file=file)
        for it in items:
            price = it.get("price_inr")
            price_str = f"₹{price:,.0f}" if price is not None else "₹ TBD"
            stock = "✅" if it.get("in_stock", 1) else f"⏳ {it.get('lead_time_days', '?')}d"
            rt = it.get("running_total")
            rt_str = f"  (running: ₹{rt:,.0f})" if rt is not None else ""
            just = it.get("justification", "")
            print(f"    {it.get('item_id','?'):8s} {it.get('name','?'):35s} {price_str:>10s} {stock} [{just}]{rt_str}", file=file)

    # Guardrails
    gts = result.get("guardrails_triggered", [])
    if gts:
        print(f"\n  🛡️  Guardrails triggered:", file=file)
        for g in gts:
            print(f"    • {g}", file=file)

    # Trade-offs
    tradeoffs = result.get("trade_offs", "")
    if tradeoffs:
        print(f"\n  ⚖️  Trade-offs: {tradeoffs[:200]}{'...' if len(tradeoffs) > 200 else ''}", file=file)

    # Tool calls
    tcl = result.get("tool_call_log", [])
    print(f"\n  🔧 Tool calls: {len(tcl)}", file=file)
    for tc in tcl:
        print(f"    {tc['tool']}({json.dumps(tc['input'], default=str)[:60]}) → {tc.get('output_summary','')[:60]}", file=file)

    print(f"\n{'='*60}\n", file=file)


def main():
    parser = argparse.ArgumentParser(description="AI Interior Design Agent — Living Room MVP")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--brief", type=str, help="Run a single brief by ID (e.g., BR-01)")
    group.add_argument("--all", action="store_true", help="Run all Living Room briefs")
    group.add_argument("--text", type=str, help="Run with a free-text brief description")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")

    args = parser.parse_args()

    if args.brief:
        result = run_agent(brief_id=args.brief)
        _print_human_summary(result)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, default=str, ensure_ascii=False))

    elif args.all:
        briefs = get_all_briefs()
        results = []
        for b in briefs:
            bid = b["brief_id"]
            print(f"\n>>> Processing {bid}...", file=sys.stderr)
            result = run_agent(brief_id=bid)
            _print_human_summary(result)
            results.append(result)
        indent = 2 if args.pretty else None
        print(json.dumps(results, indent=indent, default=str, ensure_ascii=False))

    elif args.text:
        result = run_agent(free_text=args.text)
        _print_human_summary(result)
        indent = 2 if args.pretty else None
        print(json.dumps(result, indent=indent, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
