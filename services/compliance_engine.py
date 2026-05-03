"""
Cognivis OS — ZATCA Rule Engine (Production)
============================================
Modular, extensible compliance enforcement for ZATCA Phase 2.

Rules implemented:
  BR-KSA-14            High-value invoice missing VAT ID
  VAT_FORMAT           VAT ID present but wrong format
  MISSING_FIELDS       Invoice ID or amount is missing/zero
  INVOICE_TYPE_CHECK   B2B doc type declared but no VAT ID

Public API:
  validate_transaction(row)   → real-time single invoice check
  check_transaction(amt, vat) → simple POS terminal wrapper
  audit_dataframe(df)         → batch scan of all invoices
  get_compliance_summary(df)  → KPI dict for dashboard
"""

import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════════

def _str(val) -> str:
    """Safely convert any value to a clean string. Removes .0 float artifacts."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _amt(val) -> float:
    """Safely convert any value to a float."""
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _valid_vat(vat: str) -> bool:
    """Saudi VAT ID: exactly 15 digits, starts with 3."""
    v = _str(vat)
    return len(v) == 15 and v.isdigit() and v.startswith("3")


# ══════════════════════════════════════════════════════════════════════════════════
# RULE CONDITIONS  (each takes a row dict → returns bool)
# ══════════════════════════════════════════════════════════════════════════════════

def _cond_missing_fields(row: dict) -> bool:
    """Invoice ID is blank OR amount is zero/negative."""
    return _str(row.get("invoice_id", "")) == "" or _amt(row.get("amount_sar", 0)) <= 0


def _cond_ksa14(row: dict) -> bool:
    """
    BR-KSA-14 — Core ZATCA hook.
    Invoice >= SAR 1,000 with no VAT ID captured.
    """
    return _amt(row.get("amount_sar", 0)) >= 1000 and _str(row.get("customer_vat_id", "")) == ""


def _cond_invoice_type(row: dict) -> bool:
    """B2B Tax Invoice declared but VAT ID missing — document type mismatch."""
    doc  = _str(row.get("doc_type", "")).lower()
    vat  = _str(row.get("customer_vat_id", ""))
    is_b2b = "388" in doc and "simplified" not in doc and "b2c" not in doc
    return is_b2b and vat == ""


def _cond_vat_format(row: dict) -> bool:
    """VAT ID is present but invalid format — only fires when something was entered."""
    vat = _str(row.get("customer_vat_id", ""))
    if vat == "":
        return False   # Missing VAT is BR-KSA-14's job
    return not _valid_vat(vat)


# ══════════════════════════════════════════════════════════════════════════════════
# MASTER RULE LIST  (order = priority)
# ══════════════════════════════════════════════════════════════════════════════════

RULES: list[dict] = [
    {
        "rule_id":      "MISSING_FIELDS",
        "description":  "Invoice ID or amount is missing or zero",
        "severity":     "HIGH",
        "action":       "BLOCK",
        "condition":    _cond_missing_fields,
        "user_message": "This invoice is incomplete — it is missing an ID or a valid amount.",
    },
    {
        "rule_id":      "BR-KSA-14",
        "description":  "Invoice >= SAR 1,000 requires customer VAT ID",
        "severity":     "HIGH",
        "action":       "BLOCK",
        "condition":    _cond_ksa14,
        "user_message": (
            "This invoice is over SAR 1,000. Saudi tax law (ZATCA BR-KSA-14) requires "
            "the customer's VAT registration number. Ask the customer for their "
            "15-digit VAT number to proceed."
        ),
    },
    {
        "rule_id":      "INVOICE_TYPE_CHECK",
        "description":  "B2B Tax Invoice declared but no customer VAT ID provided",
        "severity":     "HIGH",
        "action":       "BLOCK",
        "condition":    _cond_invoice_type,
        "user_message": (
            "This is marked as a B2B Tax Invoice but has no customer VAT ID. "
            "Add the VAT ID or switch to a Simplified Invoice."
        ),
    },
    {
        "rule_id":      "VAT_FORMAT",
        "description":  "VAT ID format invalid (must be 15 digits starting with 3)",
        "severity":     "HIGH",
        "action":       "BLOCK",
        "condition":    _cond_vat_format,
        "user_message": (
            "The VAT ID entered is not valid. Saudi VAT numbers are exactly 15 digits "
            "and start with '3'. Please double-check with the customer."
        ),
    },
]

_WEIGHT = {"HIGH": 45, "MEDIUM": 20, "LOW": 10}


# ══════════════════════════════════════════════════════════════════════════════════
# CORE — validate_transaction
# ══════════════════════════════════════════════════════════════════════════════════

def validate_transaction(row: dict) -> dict:
    """
    Validates one invoice row against all ZATCA rules.

    Returns:
    {
        status         : "APPROVED" | "BLOCKED"
        violations     : [rule_ids that fired]
        blocking_rules : [rule_ids that are BLOCK type]
        warning_rules  : [rule_ids that are WARN type]
        messages       : [human-readable message per rule]
        first_message  : str  (most important message)
        first_rule_id  : str  (most important rule ID)
        risk_score     : int 0-100
    }
    """
    blocking, warnings, messages = [], [], []
    score = 0

    for rule in RULES:
        try:
            fired = rule["condition"](row)
        except Exception:
            fired = False   # Never crash on a bad row

        if fired:
            score += _WEIGHT.get(rule["severity"], 10)
            messages.append(rule["user_message"])
            if rule["action"] == "BLOCK":
                blocking.append(rule["rule_id"])
            else:
                warnings.append(rule["rule_id"])

    return {
        "status":         "BLOCKED" if blocking else "APPROVED",
        "violations":     blocking + warnings,
        "blocking_rules": blocking,
        "warning_rules":  warnings,
        "messages":       messages,
        "first_message":  messages[0] if messages else "",
        "first_rule_id":  blocking[0] if blocking else "",
        "risk_score":     min(100, score),
    }


# ══════════════════════════════════════════════════════════════════════════════════
# POS TERMINAL WRAPPER
# ══════════════════════════════════════════════════════════════════════════════════

def check_transaction(amount: float, vat_id: str, doc_type: str = "") -> dict:
    """
    Quick check for the live Cashier Terminal.
    Doc type is auto-detected: B2B (Tax Invoice) if VAT provided, else Simplified.
    Returns UI-friendly dict with allowed bool and plain-English message.
    """
    vat_clean = str(vat_id).strip()
    # Auto-detect doc type from VAT presence — mirrors real cashier behaviour
    if not doc_type:
        doc_type = "Tax Invoice (388)" if vat_clean else "Simplified Invoice (388-B2C)"

    row = {
        "invoice_id":      f"POS-{int(amount)}",
        "amount_sar":      amount,
        "customer_vat_id": vat_clean,
        "doc_type":        doc_type,
    }
    r = validate_transaction(row)
    return {
        "allowed":       r["status"] == "APPROVED",
        "status":        r["status"],
        "violations":    r["violations"],
        "messages":      r["messages"],
        "first_message": r["first_message"],
        "first_rule_id": r["first_rule_id"],
        "risk_score":    r["risk_score"],
        # Legacy key some UI parts still reference
        "blocking_rule": type("R", (), {
            "rule_id":     r["first_rule_id"],
            "description": r["first_message"],
        })() if r["first_rule_id"] else None,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# BATCH AUDITOR
# ══════════════════════════════════════════════════════════════════════════════════

def audit_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the rule engine on every row of a DataFrame.
    Adds: ai_risk_score, violations, status columns.
    Safe against duplicate columns.
    """
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]   # deduplicate columns

    results = df.apply(lambda row: validate_transaction(row.to_dict()), axis=1)

    df["ai_risk_score"] = results.apply(lambda r: r["risk_score"])
    df["violations"]    = results.apply(
        lambda r: ", ".join(r["violations"]) if r["violations"] else "None"
    )

    # Only set status on rows not already manually Resolved
    if "status" not in df.columns:
        df["status"] = "Pending"

    mask = df["status"] != "Resolved"
    df.loc[mask, "status"] = results[mask].apply(
        lambda r: "Violation" if r["status"] == "BLOCKED" else "Clean"
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════════
# COMPLIANCE SUMMARY  (KPI dict for dashboard)
# ══════════════════════════════════════════════════════════════════════════════════

def get_compliance_summary(df: pd.DataFrame) -> dict:
    """Returns dashboard KPIs. Threshold: risk_score >= 45 = one HIGH rule fired."""
    if df is None or len(df) == 0:
        return {"total": 0, "violations": 0, "violation_rate": 0.0,
                "capital_at_risk": 0, "resolved": 0}

    total      = len(df)
    violations = df[df["ai_risk_score"] >= 45]
    high_risk  = len(violations)
    resolved   = int((df.get("status", pd.Series(dtype=str)) == "Resolved").sum())
    rate       = round((high_risk / total) * 100, 1) if total > 0 else 0.0

    return {
        "total":           total,
        "violations":      high_risk,
        "violation_rate":  rate,
        "capital_at_risk": high_risk * 5000,
        "resolved":        resolved,
    }
