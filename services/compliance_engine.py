"""
Cognivis OS — Compliance Engine
ZATCA Phase 2 rule enforcement. Extensible rule objects.
Each rule is a dataclass with a condition callable.
"""
import pandas as pd
from dataclasses import dataclass
from typing import Callable
from data.schemas import ComplianceRule


# ── Rule Definitions ────────────────────────────────────────────────────────────

def _rule_ksa14(row) -> bool:
    """B2C invoice over SAR 1,000 must be converted to B2B with VAT."""
    return (
        float(row.get('amount_sar', 0)) >= 1000
        and str(row.get('customer_vat_id', "")).strip() == ""
    )

def _rule_ksa26(row) -> bool:
    """Missing invoice date field."""
    return str(row.get('date', "")).strip() == ""

def _rule_ksa09(row) -> bool:
    """Invoice amount is zero or negative."""
    return float(row.get('amount_sar', 0)) <= 0

def _rule_ksa31(row) -> bool:
    """VAT ID present but wrong format (not 15 digits starting with 3)."""
    vat = str(row.get('customer_vat_id', "")).strip()
    if not vat:
        return False  # No VAT = separate rule (BR-KSA-14)
    return not (len(vat) == 15 and vat.isdigit() and vat.startswith("3"))


RULES: list[ComplianceRule] = [
    ComplianceRule(
        rule_id="BR-KSA-14",
        description="B2C invoice over SAR 1,000 requires customer VAT ID",
        severity="HIGH",
        action="BLOCK",
        condition=_rule_ksa14
    ),
    ComplianceRule(
        rule_id="BR-KSA-09",
        description="Invoice amount must be greater than zero",
        severity="HIGH",
        action="BLOCK",
        condition=_rule_ksa09
    ),
    ComplianceRule(
        rule_id="BR-KSA-31",
        description="Customer VAT ID format is invalid",
        severity="HIGH",
        action="BLOCK",
        condition=_rule_ksa31
    ),
    ComplianceRule(
        rule_id="BR-KSA-26",
        description="Invoice is missing a date field",
        severity="MEDIUM",
        action="WARN",
        condition=_rule_ksa26
    ),
]


# ── Risk Scorer ─────────────────────────────────────────────────────────────────

def score_invoice(row: dict) -> int:
    """
    Returns an AI risk score 0–100 for a single invoice.
    Based on which rules are triggered and their severity weights.
    """
    weights = {"HIGH": 40, "MEDIUM": 20, "LOW": 10}
    score = 0
    for rule in RULES:
        if rule.condition(row):
            score += weights.get(rule.severity, 10)
    return min(100, score)


def get_triggered_rules(row: dict) -> list[ComplianceRule]:
    """Returns all rules that fire for a given invoice."""
    return [r for r in RULES if r.condition(row)]


# ── Batch Auditor ───────────────────────────────────────────────────────────────

def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all compliance rules across a full invoice DataFrame.
    Adds 'ai_risk_score' and 'violations' columns.
    """
    df = df.copy()
    df['ai_risk_score'] = df.apply(lambda row: score_invoice(row.to_dict()), axis=1)
    df['violations'] = df.apply(
        lambda row: ", ".join([r.rule_id for r in get_triggered_rules(row.to_dict())]) or "None",
        axis=1
    )
    return df


# ── Real-Time Single Invoice Check ──────────────────────────────────────────────

def check_transaction(amount: float, vat_id: str) -> dict:
    """
    Validates a single transaction BEFORE invoice creation.
    Returns: { allowed: bool, blocking_rule: str|None, warnings: list }
    """
    fake_row = {"amount_sar": amount, "customer_vat_id": vat_id, "date": "2025-01-01"}
    triggered = get_triggered_rules(fake_row)

    blocking = [r for r in triggered if r.action == "BLOCK"]
    warnings = [r for r in triggered if r.action == "WARN"]

    return {
        "allowed": len(blocking) == 0,
        "blocking_rule": blocking[0] if blocking else None,
        "warnings": warnings,
        "risk_score": score_invoice(fake_row)
    }


# ── Summary Stats ───────────────────────────────────────────────────────────────

def get_compliance_summary(df: pd.DataFrame) -> dict:
    """Returns key compliance KPIs for the dashboard."""
    total = len(df)
    if total == 0:
        return {"total": 0, "violations": 0, "violation_rate": 0.0, "high_risk": 0, "capital_at_risk": 0}

    violations = df[df['ai_risk_score'] >= 80]
    high_risk = len(violations)
    violation_rate = round((high_risk / total) * 100, 1)
    capital_at_risk = high_risk * 5000  # SAR 5,000 minimum fine per ZATCA

    return {
        "total": total,
        "violations": high_risk,
        "violation_rate": violation_rate,
        "high_risk": high_risk,
        "capital_at_risk": capital_at_risk
    }
