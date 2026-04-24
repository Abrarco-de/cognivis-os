"""
Cognivis OS — Trust Score Engine
Dynamically calculates a business's compliance health score.
This score is designed to feed into future lending/fintech integrations.
"""
import pandas as pd
import streamlit as st
from data.schemas import TrustScore


def calculate_trust_score(df: pd.DataFrame) -> TrustScore:
    """
    Derives the Cognivis Trust Score from real invoice data.

    Components:
    - Violation Rate (0–40 pts): lower = better
    - Resolution Activity (0–30 pts): based on resolved vs pending
    - Consistency (0–30 pts): based on clean invoice ratio over time
    """
    if df is None or len(df) == 0:
        return TrustScore(score=50, label="INSUFFICIENT DATA",
                          violation_rate=0, resolution_speed_avg=0, consistency_score=0)

    total = len(df)
    high_risk = len(df[df['ai_risk_score'] >= 80])
    resolved = len(df[df.get('status', pd.Series()).eq("Resolved")]) if 'status' in df.columns else 0
    clean = len(df[df['ai_risk_score'] == 0])

    violation_rate = high_risk / total if total > 0 else 0
    consistency = clean / total if total > 0 else 0
    resolution_ratio = resolved / max(high_risk, 1)

    # Score calculation
    violation_score = max(0, 40 - int(violation_rate * 100))
    resolution_score = int(resolution_ratio * 30)
    consistency_score_pts = int(consistency * 30)

    # Behavioral memory bonus (from session state)
    memory_bonus = min(10, st.session_state.get('behavior_memory', {}).get('b2b_conversions', 0) * 2)

    raw_score = violation_score + resolution_score + consistency_score_pts + memory_bonus
    score = min(100, max(0, raw_score))

    if score >= 85:
        label = "PRIME"
    elif score >= 70:
        label = "GOOD"
    elif score >= 50:
        label = "FAIR"
    else:
        label = "AT RISK"

    return TrustScore(
        score=score,
        label=label,
        violation_rate=round(violation_rate * 100, 1),
        resolution_speed_avg=resolution_ratio,
        consistency_score=round(consistency * 100, 1)
    )
