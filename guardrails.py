"""
guardrails.py — Hardcoded first-pass guardrail filters.

These are CODE checks, not LLM-judgment-only. The spec requires guardrail #3
(scope) to be hardcoded as a keyword/intent filter in addition to model judgment.

Guardrails implemented:
  G1  — Catalog integrity: designer piece detection + "-style" alternative lookup
  G3  — Scope: civil/structural/electrical/plumbing keyword scan
  G10 — Cultural practice: never apply without explicit customer mention
  G11 — Commercial promise: guaranteed date/locked price detection
"""

import re
from tools import catalog_search

# ---------------------------------------------------------------------------
# G3 — Scope guardrail: civil / structural / electrical / plumbing
# ---------------------------------------------------------------------------

_SCOPE_KEYWORDS = [
    # Structural
    r"\bknock\s*down\b", r"\bload[- ]?bearing\b", r"\bwall\s*removal\b",
    r"\bremove\s*(the\s*)?wall\b", r"\bdemolish\b", r"\bdemolition\b",
    r"\bstructural\b", r"\btear\s*down\b", r"\bknock\s*out\b",
    # Electrical
    r"\brewiring\b", r"\brewire\b", r"\belectrical\s*(work|panel|wiring)\b",
    r"\bcircuit\s*breaker\b", r"\bfuse\s*box\b",
    # Plumbing
    r"\bplumbing\b", r"\breplumb\b", r"\bpipe\s*(work|fitting)\b",
    r"\bdrain(age)?\b", r"\bsewer\b",
    # Civil / construction
    r"\bcivil\s*(work|engineer)\b", r"\bfoundation\b", r"\bconstruction\b",
    r"\bRCC\b", r"\bconcrete\s*pour\b",
]

_SCOPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SCOPE_KEYWORDS]


def check_scope_guardrail(brief: dict) -> dict:
    """
    Scan must_haves, constraints, and customer_note for out-of-scope
    civil/structural/electrical/plumbing content.

    Returns:
        {
            "triggered": bool,
            "flagged_terms": [str],
            "message": str  — redirect message for the customer
        }
    """
    text_fields = [
        brief.get("must_haves", ""),
        brief.get("constraints", ""),
        brief.get("customer_note", ""),
    ]
    combined = " ".join(text_fields)
    flagged = []

    for pattern in _SCOPE_PATTERNS:
        matches = pattern.findall(combined)
        if matches:
            flagged.extend(matches)

    if flagged:
        return {
            "triggered": True,
            "flagged_terms": list(set(flagged)),
            "message": (
                "Your brief includes requests that fall outside our furnishing scope "
                f"(detected: {', '.join(set(flagged))}). These involve civil, structural, "
                "electrical, or plumbing work and should be assessed by a qualified "
                "professional (structural engineer, licensed electrician, or plumber). "
                "We'll proceed with the furnishing-only portion of your request."
            ),
        }

    return {"triggered": False, "flagged_terms": [], "message": ""}


# ---------------------------------------------------------------------------
# G1 — Designer piece detection
# ---------------------------------------------------------------------------

_DESIGNER_BRANDS = [
    "Togo", "Noguchi", "Eames", "Herman Miller", "Kartell", "Cassina",
    "Knoll", "B&B Italia", "Vitra", "Fritz Hansen", "Artek",
    "Le Corbusier", "Barcelona Chair", "Wassily Chair", "Tulip Table",
    "Egg Chair", "Swan Chair", "Platner", "Saarinen",
    "Ligne Roset", "Roche Bobois",
]

_DESIGNER_PATTERNS = [
    re.compile(rf"\b{re.escape(brand)}\b", re.IGNORECASE) for brand in _DESIGNER_BRANDS
]


def check_designer_piece_guardrail(brief: dict) -> dict:
    """
    Check for named branded/designer piece requests.
    For each detected brand, search catalog (including out of stock) for "-style" alternatives.

    Returns:
        {
            "triggered": bool,
            "detected_brands": [str],
            "alternatives": {brand: [catalog_items]},
            "message": str
        }
    """
    text_fields = [
        brief.get("must_haves", ""),
        brief.get("constraints", ""),
        brief.get("customer_note", ""),
    ]
    combined = " ".join(text_fields)
    detected = []

    for i, pattern in enumerate(_DESIGNER_PATTERNS):
        if pattern.search(combined):
            detected.append(_DESIGNER_BRANDS[i])

    if not detected:
        return {"triggered": False, "detected_brands": [], "alternatives": {}, "message": ""}

    # Search for -style alternatives in the catalog
    alternatives = {}
    for brand in detected:
        # Search with in_stock_only=False to find any homage pieces
        results = catalog_search(
            style=None, category=None, room_type=brief.get("room_type"), in_stock_only=False
        )
        # Filter for items whose name contains the brand name + "-style" or "style"
        brand_lower = brand.lower()
        alt_items = [
            item for item in results
            if brand_lower in item.get("name", "").lower()
            or f"{brand_lower}-style" in item.get("name", "").lower()
            or f"{brand_lower} style" in item.get("name", "").lower()
        ]
        alternatives[brand] = alt_items

    alt_messages = []
    for brand, items in alternatives.items():
        if items:
            names = ", ".join(f"'{it['name']}' ({it['item_id']})" for it in items)
            alt_messages.append(
                f"  • {brand}: Not available as a genuine piece. "
                f"We have inspired-by alternatives: {names}"
            )
        else:
            alt_messages.append(
                f"  • {brand}: Not available as a genuine piece, and no -style "
                f"alternative exists in our catalog."
            )

    message = (
        "Your brief requests specific designer/branded pieces. Our catalog does not "
        "carry genuine branded furniture, but we may have inspired-by homage pieces:\n"
        + "\n".join(alt_messages)
    )

    return {
        "triggered": True,
        "detected_brands": detected,
        "alternatives": alternatives,
        "message": message,
    }


# ---------------------------------------------------------------------------
# G11 — Commercial promise guardrail
# ---------------------------------------------------------------------------

_PROMISE_KEYWORDS = [
    r"\bguarantee[ds]?\b", r"\blocked[- ]?price\b", r"\bfinal\s*price\b",
    r"\bcommitted\s*delivery\b", r"\bpromise[ds]?\b", r"\bfixed\s*price\b",
    r"\bprice\s*lock\b", r"\bdelivery\s*guarantee\b", r"\bexact\s*date\b",
    r"\bconfirm(ed)?\s*delivery\b",
]

_PROMISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _PROMISE_KEYWORDS]


def check_commercial_promise_guardrail(brief: dict) -> dict:
    """
    Detect demands for guaranteed delivery dates or locked/final prices.

    Returns:
        {
            "triggered": bool,
            "flagged_terms": [str],
            "message": str
        }
    """
    text_fields = [
        brief.get("must_haves", ""),
        brief.get("constraints", ""),
        brief.get("customer_note", ""),
    ]
    combined = " ".join(text_fields)
    flagged = []

    for pattern in _PROMISE_PATTERNS:
        matches = pattern.findall(combined)
        if matches:
            flagged.extend(matches)

    if flagged:
        return {
            "triggered": True,
            "flagged_terms": list(set(flagged)),
            "message": (
                "We understand you'd like firm commitments on pricing and/or delivery dates. "
                "The prices and lead times shown are current catalog figures and are not "
                "final negotiated prices or guaranteed delivery dates. Final pricing and "
                "delivery commitments are confirmed at the order stage by our sales team."
            ),
        }

    return {"triggered": False, "flagged_terms": [], "message": ""}


# ---------------------------------------------------------------------------
# G10 — Cultural practice guardrail
# ---------------------------------------------------------------------------

_CULTURAL_KEYWORDS = [
    r"\bvastu\b", r"\bfeng\s*shui\b", r"\bvaastu\b", r"\bvastu\s*shastra\b",
]

_CULTURAL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _CULTURAL_KEYWORDS]


def check_cultural_practice_guardrail(brief: dict) -> dict:
    """
    Only apply cultural/religious placement practices if EXPLICITLY mentioned
    by the customer. Geography/demographics alone is NEVER sufficient.

    Returns:
        {
            "triggered": bool,
            "detected_practices": [str],
            "message": str
        }
    """
    text_fields = [
        brief.get("must_haves", ""),
        brief.get("constraints", ""),
        brief.get("customer_note", ""),
    ]
    combined = " ".join(text_fields)
    detected = []

    for pattern in _CULTURAL_PATTERNS:
        if pattern.search(combined):
            detected.append(pattern.pattern.strip(r"\b").replace(r"\s*", " "))

    if detected:
        return {
            "triggered": True,
            "detected_practices": detected,
            "message": (
                f"We noted your mention of {', '.join(detected)}. We'll incorporate "
                "these placement guidelines where applicable and will clearly disclose "
                "where they influenced our recommendations."
            ),
        }

    return {"triggered": False, "detected_practices": [], "message": ""}


# ---------------------------------------------------------------------------
# Master guardrail runner
# ---------------------------------------------------------------------------

def run_all_guardrails(brief: dict) -> dict:
    """
    Run all guardrail checks on a brief. Returns a consolidated result.
    """
    scope = check_scope_guardrail(brief)
    designer = check_designer_piece_guardrail(brief)
    commercial = check_commercial_promise_guardrail(brief)
    cultural = check_cultural_practice_guardrail(brief)

    triggered = []
    messages = []

    if scope["triggered"]:
        triggered.append("out_of_scope_civil_structural_electrical_plumbing")
        messages.append(scope["message"])

    if designer["triggered"]:
        triggered.append("designer_piece_substituted")
        messages.append(designer["message"])

    if commercial["triggered"]:
        triggered.append("promise_declined_date_or_price")
        messages.append(commercial["message"])

    if cultural["triggered"]:
        triggered.append("cultural_practice_signal_applied")
        messages.append(cultural["message"])

    return {
        "guardrails_triggered": triggered,
        "messages": messages,
        "details": {
            "scope": scope,
            "designer": designer,
            "commercial": commercial,
            "cultural": cultural,
        },
    }
