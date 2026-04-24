"""
Cognivis OS — Validators & Logger
Input validation and structured audit logging.
"""
import re
import pandas as pd
from datetime import datetime
import streamlit as st


# ── VAT Validation ─────────────────────────────────────────────────────────────

def validate_vat_id(vat_id: str) -> tuple[bool, str]:
    """
    Validates a Saudi ZATCA VAT ID.
    Rules: exactly 15 digits, starts with 3.
    Returns (is_valid, error_message).
    """
    vat_id = str(vat_id).strip()
    if not vat_id:
        return False, "VAT ID is required for B2B invoices over SAR 1,000."
    if not vat_id.isdigit():
        return False, "VAT ID must contain digits only."
    if len(vat_id) != 15:
        return False, f"VAT ID must be exactly 15 digits (got {len(vat_id)})."
    if not vat_id.startswith("3"):
        return False, "Saudi VAT IDs must start with '3'."
    return True, ""


def validate_amount(amount) -> tuple[bool, str]:
    """Validates a transaction amount."""
    try:
        val = float(str(amount).replace(",", ""))
        if val <= 0:
            return False, "Amount must be greater than 0."
        return True, ""
    except (ValueError, TypeError):
        return False, "Invalid amount format."


# ── Column Normalizer ───────────────────────────────────────────────────────────

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps any POS CSV/API column names to the Cognivis standard schema.
    Works with Foodics, Salla, and generic uploads.
    """
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    col_map = {}
    for col in df.columns:
        if re.search(r'invoice|inv|bill|receipt|order_id', col):
            col_map[col] = 'invoice_id'
        elif re.search(r'amount|total|price|sar|order_total', col):
            col_map[col] = 'amount_sar'
        elif re.search(r'vat|tax|tin|buyer_vat|customer_vat', col):
            col_map[col] = 'customer_vat_id'
        elif re.search(r'category|group|type|product_category|item_group', col):
            col_map[col] = 'category'
        elif re.search(r'date|time|created', col):
            col_map[col] = 'date'

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    if 'invoice_id' not in df.columns:
        df['invoice_id'] = [f"SYS-{i:04d}" for i in range(len(df))]
    if 'amount_sar' in df.columns:
        df['amount_sar'] = pd.to_numeric(
            df['amount_sar'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        ).fillna(0)
    if 'customer_vat_id' not in df.columns:
        df['customer_vat_id'] = ""
    else:
        df['customer_vat_id'] = (
            df['customer_vat_id'].fillna("").astype(str).str.strip()
            .str.replace(r'\.0$', '', regex=True)
        )
    if 'category' not in df.columns:
        df['category'] = "General"
    if 'doc_type' not in df.columns:
        df['doc_type'] = df.apply(
            lambda r: "Tax Invoice (388)" if r.get('customer_vat_id', "") != "" else "Simplified Invoice (388-B2C)",
            axis=1
        )
    if 'date' not in df.columns:
        df['date'] = datetime.now().strftime("%Y-%m-%d")
    if 'status' not in df.columns:
        df['status'] = "Pending"

    return df


# ── Audit Logger ────────────────────────────────────────────────────────────────

def log_audit(action: str, invoice_id: str, status: str, user: str = "System Engine"):
    """Appends an immutable entry to the session audit ledger."""
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Invoice ID": invoice_id,
        "Action": action,
        "Status": status,
        "Authorized By": user
    }
    if 'audit_ledger' not in st.session_state:
        st.session_state.audit_ledger = []
    st.session_state.audit_ledger.append(entry)
