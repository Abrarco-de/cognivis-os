"""
Cognivis OS — Simulation Engine (What-If)
Runs scenario projections on invoice data without modifying real data.
"""
import pandas as pd
import numpy as np


def simulate_price_change(df: pd.DataFrame, pct_change: float) -> dict:
    """
    Simulates applying a percentage price change across all transactions.
    Returns revenue impact AND compliance risk delta.

    pct_change: e.g. 15 for +15%, -10 for -10%
    """
    sim = df[df['amount_sar'] > 0].copy()
    multiplier = 1 + (pct_change / 100)

    base_revenue = sim['amount_sar'].sum()
    sim['sim_amount'] = sim['amount_sar'] * multiplier
    sim_revenue = sim['sim_amount'].sum()
    revenue_delta = sim_revenue - base_revenue

    # Transactions newly crossing the SAR 1,000 compliance threshold
    new_risks = sim[
        (sim['sim_amount'] >= 1000) &
        (sim['amount_sar'] < 1000) &
        (sim['customer_vat_id'].astype(str).str.strip() == "")
    ]

    # Transactions that drop below threshold (compliance risk removed)
    risks_removed = sim[
        (sim['sim_amount'] < 1000) &
        (sim['amount_sar'] >= 1000) &
        (sim['customer_vat_id'].astype(str).str.strip() == "")
    ]

    return {
        "base_revenue": base_revenue,
        "sim_revenue": sim_revenue,
        "revenue_delta": revenue_delta,
        "pct_change": pct_change,
        "new_risk_count": len(new_risks),
        "risks_removed": len(risks_removed),
        "net_risk_delta": len(new_risks) - len(risks_removed),
        "fine_exposure_delta": len(new_risks) * 5000,
    }


def simulate_vat_enforcement(df: pd.DataFrame, capture_rate: float) -> dict:
    """
    Simulates what happens if the business captures VAT IDs at a given rate
    on all transactions over SAR 1,000.

    capture_rate: 0.0 – 1.0 (e.g. 0.80 = 80% of eligible transactions get VAT)
    """
    high_value = df[(df['amount_sar'] >= 1000) & (df['customer_vat_id'].astype(str).str.strip() == "")]
    total_at_risk = len(high_value)
    resolved = int(total_at_risk * capture_rate)
    remaining_violations = total_at_risk - resolved

    fine_saved = resolved * 5000
    fine_remaining = remaining_violations * 5000

    return {
        "total_at_risk": total_at_risk,
        "resolved_by_capture": resolved,
        "remaining_violations": remaining_violations,
        "fine_saved": fine_saved,
        "fine_remaining": fine_remaining,
        "capture_rate_pct": round(capture_rate * 100, 0)
    }
