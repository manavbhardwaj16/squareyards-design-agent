"""
app.py — Interactive Streamlit Dashboard for AI Interior Design Agent (Living Room MVP).

Features:
- Room Brief selection (all seeded living room briefs + free-text mode)
- Real-time agent execution with tool call log streaming
- Visual BOQ (Bill of Quantities) with running totals & justification badges
- Budget meter & 35% footprint utilization gauge
- Guardrail alerts & designer voice rationale cards
- Raw JSON export & evaluation view
"""

import json
import os
import sqlite3
import streamlit as st

from agent import run_agent
from tools import get_all_briefs, DB_PATH

# ---------------------------------------------------------------------------
# Streamlit Page Config & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Interior Design Agent — Living Room MVP",
    page_icon="🛋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .rationale-box {
        background: #0f172a;
        border-left: 4px solid #6366f1;
        padding: 1.2rem;
        border-radius: 0 12px 12px 0;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .badge-must-have {
        background-color: #ef4444;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-pref {
        background-color: #3b82f6;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-assumption {
        background-color: #eab308;
        color: black;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🛋️ AI Interior Design Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Living Room MVP — Zero-Hallucination, Guardrail-Enforced Bill of Quantities</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Brief Input & Catalog Browser
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Select or Input Brief")

briefs = get_all_briefs()
brief_options = {
    f"{b['brief_id']} — {b['style_preference']} (₹{b['budget_inr']:,})": b['brief_id']
    for b in briefs
}
brief_options["Custom Free-Text Request"] = "custom"

selected_label = st.sidebar.selectbox("Choose Room Brief", list(brief_options.keys()))
selected_mode = brief_options[selected_label]

custom_text = ""
selected_brief_id = None

if selected_mode == "custom":
    custom_text = st.sidebar.text_area(
        "Enter Free-Text Room Description",
        placeholder="e.g. Living room 450x350cm, budget 200000. Mid-Century style with sofa and coffee table.",
        height=150,
    )
else:
    selected_brief_id = selected_mode
    # Show brief details in sidebar
    brief_data = next((b for b in briefs if b["brief_id"] == selected_brief_id), None)
    if brief_data:
        st.sidebar.markdown(f"**Room Size:** {brief_data['length_cm']}×{brief_data['width_cm']} cm (Ceiling: {brief_data['ceiling_cm']} cm)")
        st.sidebar.markdown(f"**Budget:** ₹{brief_data['budget_inr']:,}")
        st.sidebar.markdown(f"**Style:** {brief_data['style_preference']}")
        st.sidebar.markdown(f"**Must-Haves:** {brief_data['must_haves']}")
        if brief_data.get("constraints"):
            st.sidebar.markdown(f"**Constraints:** {brief_data['constraints']}")
        if brief_data.get("customer_note"):
            st.sidebar.caption(f"Note: \"{brief_data['customer_note']}\"")

run_btn = st.sidebar.button("🚀 Generate Living Room Design", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main Execution Area
# ---------------------------------------------------------------------------
if run_btn:
    with st.spinner("🤖 Agent analyzing brief, querying catalog, and validating constraints..."):
        if selected_mode == "custom":
            if not custom_text.strip():
                st.error("Please provide a room description.")
                st.stop()
            result = run_agent(free_text=custom_text)
        else:
            result = run_agent(brief_id=selected_brief_id)

    st.session_state["last_result"] = result

if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    # Top Status Bar
    feasible = res.get("feasible", False)
    bs = res.get("budget_summary", {})
    fs = res.get("fit_summary", {})
    items = res.get("selected_items", [])
    gts = res.get("guardrails_triggered", [])

    # Feasibility Alert
    if feasible:
        st.success("✅ **Design Feasible** — All selected items fit within room constraints and budget.")
    else:
        st.error("⚠️ **Design Infeasible** — Constraints could not be satisfied. Showing closest partial plan.")

    # Top Metric Columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label="Total Budget",
            value=f"₹{bs.get('budget', 0):,}",
            delta=f"Spent: ₹{bs.get('total_spent', 0):,}",
            delta_color="off",
        )
    with c2:
        rem = bs.get("remaining", 0)
        over = bs.get("over_budget", False)
        st.metric(
            label="Remaining Budget",
            value=f"₹{rem:,}",
            delta="Over Budget!" if over else "Within Budget",
            delta_color="inverse" if over else "normal",
        )
    with c3:
        pct = fs.get("footprint_used_pct", 0.0)
        st.metric(
            label="Floor Footprint Used",
            value=f"{pct}%",
            delta=f"Threshold: 35.0%",
            delta_color="inverse" if pct > 35 else "normal",
        )
    with c4:
        st.metric(
            label="Items in BOQ",
            value=len(items),
            delta=f"Tool calls: {len(res.get('tool_call_log', []))}",
            delta_color="off",
        )

    # Guardrails Banner if triggered
    if gts:
        st.warning(f"🛡️ **Guardrails Triggered ({len(gts)}):** " + ", ".join(f"`{g}`" for g in gts))

    st.divider()

    # Two columns: BOQ & Visualizer on Left, Rationale & Details on Right
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📋 Bill of Quantities (BOQ)")
        if items:
            table_data = []
            for it in items:
                p = it.get("price_inr")
                p_str = f"₹{p:,}" if p is not None else "₹ TBD (Quote)"
                rt = it.get("running_total")
                rt_str = f"₹{rt:,}" if rt is not None else "—"
                stock = "✅ In Stock" if it.get("in_stock", 1) else f"⏳ {it.get('lead_time_days', '?')}d lead"
                table_data.append({
                    "Item ID": it.get("item_id", ""),
                    "Category": it.get("category", ""),
                    "Name": it.get("name", ""),
                    "Price (INR)": p_str,
                    "Running Total": rt_str,
                    "Stock Status": stock,
                    "Justification": it.get("justification", ""),
                })
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info("No items in final selection.")

        # Breakdown charts / allocations
        sub_b = bs.get("sub_budget_allocation", {})
        sub_f = fs.get("sub_footprint_allocation", {})

        if sub_b:
            st.markdown("#### 📊 Category Spend Breakdown")
            st.bar_chart(sub_b)

    with col_right:
        st.subheader("✨ Designer Voice & Rationale")
        rationale = res.get("rationale", "")
        st.markdown(f'<div class="rationale-box">{rationale}</div>', unsafe_allow_html=True)

        if res.get("trade_offs"):
            st.markdown("#### ⚖️ Trade-Offs & Compromises")
            st.info(res["trade_offs"])

        if res.get("assumptions_made"):
            st.markdown("#### 📝 Assumptions Made")
            for asm in res["assumptions_made"]:
                st.caption(f"• {asm}")

    st.divider()

    # Tool Call Log Inspector
    with st.expander("🔧 Inspect Tool Call Execution Log", expanded=False):
        tcl = res.get("tool_call_log", [])
        if tcl:
            for idx, tc in enumerate(tcl, 1):
                st.markdown(f"**Step {idx}: `{tc['tool']}`**")
                st.json(tc["input"])
                st.caption(f"Output: {tc.get('output_summary', '')}")
                st.markdown("---")
        else:
            st.write("No tool calls recorded.")

    # Raw Output JSON
    with st.expander("📄 View Full Output JSON Schema", expanded=False):
        st.json(res)
