"""
Cognivis OS — AI Insights Engine (Production)
=============================================
Generates real, data-driven business intelligence from invoice data.
Zero randomness — every output is derived from the actual numbers.

Public API:
  revenue_insights(df)              → revenue breakdown dict
  risk_insights(df)                 → compliance risk analysis dict
  behavioral_insights(df, memory)   → pattern analysis from past actions
  generate_actions(insights)        → actionable recommendations list
  generate_all_recommendations(df)  → unified list for the AI Brain UI
"""

import pandas as pd
import numpy as np
from data.schemas import DecisionRecommendation


# ══════════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCALE  (based on data volume — more data = higher confidence)
# ══════════════════════════════════════════════════════════════════════════════════

def _confidence(n: int) -> float:
    if n < 5:  return 0.40
    if n < 15: return 0.60
    if n < 30: return 0.75
    if n < 50: return 0.85
    return 0.94


# ══════════════════════════════════════════════════════════════════════════════════
# PART 1A — REVENUE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════════

def revenue_insights(df: pd.DataFrame) -> dict:
    """
    Analyses revenue from invoice data.

    Returns:
    {
        total_revenue     : float
        top_category      : str
        bottom_category   : str
        avg_order_value   : float
        category_breakdown: { category: { revenue, count, avg } }
        invoice_count     : int
        confidence        : float
    }
    """
    if df is None or len(df) == 0:
        return {"error": "No data available"}

    positive = df[df["amount_sar"] > 0].copy()
    if len(positive) == 0:
        return {"error": "No positive-value invoices"}

    total   = float(positive["amount_sar"].sum())
    aov     = float(positive["amount_sar"].mean())
    count   = len(positive)

    # Category breakdown
    breakdown = {}
    if "category" in positive.columns:
        grp = positive.groupby("category")["amount_sar"].agg(["sum", "count", "mean"])
        for cat, row in grp.iterrows():
            breakdown[cat] = {
                "revenue": round(float(row["sum"]), 2),
                "count":   int(row["count"]),
                "avg":     round(float(row["mean"]), 2),
            }
        top_cat    = grp["sum"].idxmax() if len(grp) > 0 else "N/A"
        bottom_cat = grp["sum"].idxmin() if len(grp) > 0 else "N/A"
    else:
        top_cat = bottom_cat = "N/A"

    return {
        "total_revenue":      round(total, 2),
        "top_category":       top_cat,
        "bottom_category":    bottom_cat,
        "avg_order_value":    round(aov, 2),
        "category_breakdown": breakdown,
        "invoice_count":      count,
        "confidence":         _confidence(count),
    }


# ══════════════════════════════════════════════════════════════════════════════════
# PART 1B — RISK INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════════

def risk_insights(df: pd.DataFrame) -> dict:
    """
    Analyses compliance risk patterns in the invoice data.

    Violation condition:
      amount >= 1000 AND customer_vat_id is empty

    Returns:
    {
        total_invoices        : int
        violation_count       : int
        violation_rate_pct    : float
        capital_at_risk       : float  (SAR 5,000 per violation)
        high_risk_categories  : list of categories with most violations
        high_risk_amounts     : list of amounts that triggered violations
        patterns              : list of plain-English pattern descriptions
        confidence            : float
    }
    """
    if df is None or len(df) == 0:
        return {"error": "No data available"}

    total = len(df)

    # Identify violations directly from data (not relying on pre-scored column)
    violations = df[
        (df["amount_sar"] >= 1000) &
        (df["customer_vat_id"].fillna("").astype(str).str.strip() == "")
    ].copy()

    v_count = len(violations)
    v_rate  = round((v_count / total) * 100, 1) if total > 0 else 0.0
    capital = v_count * 5000

    # High-risk categories
    high_risk_cats = []
    if "category" in violations.columns and v_count > 0:
        cat_counts     = violations["category"].value_counts()
        high_risk_cats = cat_counts.index.tolist()[:3]

    # High-risk amounts
    high_risk_amounts = violations["amount_sar"].sort_values(ascending=False).head(5).tolist()

    # Pattern detection
    patterns = []
    if v_count > 0:
        patterns.append(f"{v_count} invoices over SAR 1,000 are missing a customer VAT ID.")
    if v_rate > 20:
        patterns.append(f"Violation rate of {v_rate}% is above the 20% danger threshold.")
    if high_risk_cats:
        patterns.append(f"Most violations come from: {', '.join(high_risk_cats)}.")
    if v_count == 0:
        patterns.append("No violations detected. Your business is ZATCA-clean.")

    return {
        "total_invoices":       total,
        "violation_count":      v_count,
        "violation_rate_pct":   v_rate,
        "capital_at_risk":      capital,
        "high_risk_categories": high_risk_cats,
        "high_risk_amounts":    high_risk_amounts,
        "patterns":             patterns,
        "confidence":           _confidence(total),
    }


# ══════════════════════════════════════════════════════════════════════════════════
# PART 1C — BEHAVIORAL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════════

def behavioral_insights(df: pd.DataFrame, memory: dict) -> dict:
    """
    Analyses past user actions to detect dominant resolution patterns.

    memory dict expected keys:
      b2b_conversions : int  (how many times VAT was added to fix a violation)

    Returns:
    {
        dominant_pattern   : str
        b2b_conversions    : int
        auto_suggest_b2b   : bool   (True if pattern is strong enough)
        pattern_confidence : float
        summary            : str
    }
    """
    if memory is None:
        memory = {}

    conversions = int(memory.get("b2b_conversions", 0))

    # Detect dominant pattern from data
    resolved = df[df.get("status", pd.Series(dtype=str)) == "Resolved"] \
               if "status" in df.columns else pd.DataFrame()

    pattern_conf = min(0.95, 0.40 + (conversions * 0.08))
    auto_suggest = conversions >= 3

    if conversions == 0:
        dominant = "No resolutions recorded yet"
        summary  = "Start resolving violations to build behavioral memory."
    elif conversions < 3:
        dominant = "B2B VAT capture (early pattern)"
        summary  = f"You have resolved {conversions} violations by adding a VAT ID. Keep going to unlock auto-suggestions."
    else:
        dominant = "B2B VAT capture (strong pattern)"
        summary  = (
            f"You have resolved {conversions} violations by converting to B2B. "
            f"Cognivis will now pre-suggest this on new violations automatically."
        )

    return {
        "dominant_pattern":   dominant,
        "b2b_conversions":    conversions,
        "auto_suggest_b2b":   auto_suggest,
        "pattern_confidence": round(pattern_conf, 2),
        "summary":            summary,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# PART 1D — GENERATE ACTIONS  (insights → actionable recommendations)
# ══════════════════════════════════════════════════════════════════════════════════

def generate_actions(insights: dict) -> list[dict]:
    """
    Converts the structured insights dict into a list of actionable recommendations.

    Each action:
    {
        title          : str
        description    : str
        system_action  : str  (what Cognivis can do automatically)
        priority       : "HIGH" | "MEDIUM" | "LOW"
        category       : "COMPLIANCE" | "REVENUE" | "OPERATIONS"
    }
    """
    actions = []

    rev  = insights.get("revenue", {})
    risk = insights.get("risk", {})
    beh  = insights.get("behavior", {})

    # ── Compliance actions (always highest priority)
    v_count = risk.get("violation_count", 0)
    capital = risk.get("capital_at_risk", 0)
    if v_count > 0:
        actions.append({
            "title":         f"Resolve {v_count} ZATCA Violations",
            "description":   (
                f"You have {v_count} invoices with missing VAT IDs on orders over SAR 1,000. "
                f"Total fine exposure: SAR {capital:,}. Each takes under 30 seconds to fix."
            ),
            "system_action": "Open ZATCA Shield → Resolution Queue to fix all violations now.",
            "priority":      "HIGH",
            "category":      "COMPLIANCE",
        })

    for pattern in risk.get("patterns", []):
        if "danger threshold" in pattern:
            actions.append({
                "title":         "Enable Proactive POS Interception",
                "description":   (
                    f"Your violation rate ({risk.get('violation_rate_pct', 0)}%) is high. "
                    "Activate the Cashier Terminal to block violations at the point of sale."
                ),
                "system_action": "Go to Cashier Terminal → Enable Shield interception on all terminals.",
                "priority":      "HIGH",
                "category":      "COMPLIANCE",
            })
            break

    # ── Revenue actions
    top_cat    = rev.get("top_category", "")
    bottom_cat = rev.get("bottom_category", "")
    aov        = rev.get("avg_order_value", 0)
    if top_cat and bottom_cat and top_cat != bottom_cat:
        breakdown   = rev.get("category_breakdown", {})
        top_rev     = breakdown.get(top_cat, {}).get("revenue", 0)
        bottom_rev  = breakdown.get(bottom_cat, {}).get("revenue", 0)
        actions.append({
            "title":         f"Grow Your Top Category: {top_cat}",
            "description":   (
                f"{top_cat} generates SAR {top_rev:,.0f} vs {bottom_cat}'s SAR {bottom_rev:,.0f}. "
                "Shift floor space, staffing, or marketing toward your top performer."
            ),
            "system_action": "Review category performance in AI Brain → Revenue by Category chart.",
            "priority":      "MEDIUM",
            "category":      "REVENUE",
        })

    if 0 < aov < 800:
        actions.append({
            "title":         "Increase Average Order Value",
            "description":   (
                f"Your average invoice is SAR {aov:,.0f}. "
                "Bundling items or offering add-ons could push this above SAR 1,000, "
                "which also enables stronger B2B relationships."
            ),
            "system_action": "Use What-If Engine → Price Change to model the impact of bundling.",
            "priority":      "MEDIUM",
            "category":      "REVENUE",
        })

    # ── Behavioral actions
    if beh.get("auto_suggest_b2b"):
        actions.append({
            "title":         "Auto-Suggest B2B Conversion Enabled",
            "description":   beh.get("summary", ""),
            "system_action": "Cognivis will now pre-fill the B2B recommendation on new violations.",
            "priority":      "LOW",
            "category":      "OPERATIONS",
        })

    return actions


# ══════════════════════════════════════════════════════════════════════════════════
# UNIFIED MASTER FUNCTION  (called by the AI Brain UI)
# ══════════════════════════════════════════════════════════════════════════════════

def generate_all_insights(df: pd.DataFrame, memory: dict = None) -> dict:
    """
    Runs all insight engines and returns the full structured output.

    Returns:
    {
        revenue  : { ... }
        risk     : { ... }
        behavior : { ... }
        actions  : [ ... ]
    }
    """
    if memory is None:
        memory = {}

    rev  = revenue_insights(df)
    risk = risk_insights(df)
    beh  = behavioral_insights(df, memory)
    acts = generate_actions({"revenue": rev, "risk": risk, "behavior": beh})

    return {
        "revenue":  rev,
        "risk":     risk,
        "behavior": beh,
        "actions":  acts,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# UI ADAPTER  (converts actions → DecisionRecommendation for the Brain UI cards)
# ══════════════════════════════════════════════════════════════════════════════════

def generate_all_recommendations(df: pd.DataFrame, memory: dict = None) -> list[DecisionRecommendation]:
    """
    Wrapper that returns a list of DecisionRecommendation objects
    for direct use in the AI Brain module UI.
    """
    if memory is None:
        try:
            import streamlit as st
            memory = st.session_state.get("behavior_memory", {})
        except Exception:
            memory = {}

    insights = generate_all_insights(df, memory)
    actions  = insights.get("actions", [])

    recs = []
    priority_map = {"HIGH": 0.95, "MEDIUM": 0.78, "LOW": 0.60}

    for act in actions:
        recs.append(DecisionRecommendation(
            title        = act["title"],
            body         = f"{act['description']}<br><br><i style='color:#64748b;font-size:12px;'>💡 {act['system_action']}</i>",
            confidence   = priority_map.get(act["priority"], 0.70),
            category     = act["category"],
            action_label = act["system_action"][:50],
            action_prompt= act["title"],
        ))

    return recs
