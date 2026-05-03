"""
Cognivis OS — Persistence Layer
=================================
Stores audit logs and invoice data per business.

Two modes (auto-detected):
  SUPABASE mode  — when SUPABASE_URL + SUPABASE_KEY are in Streamlit secrets
  FILE mode      — falls back to local JSON files (good for pilots and local dev)

This means the app works offline and upgrades to real DB with zero code change.

Supabase setup (free tier):
  1. Create project at supabase.com
  2. Create table: cognivis_audit
     columns: id (uuid), business_name (text), timestamp (text),
              invoice_id (text), action (text), status (text), authorized_by (text)
  3. Add to Streamlit secrets:
     SUPABASE_URL = "https://xxx.supabase.co"
     SUPABASE_KEY = "your-anon-key"
"""

import json
import os
import streamlit as st
from datetime import datetime
from pathlib import Path


# ── Mode detection ────────────────────────────────────────────────────────────

def _supabase_configured() -> bool:
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        return bool(url and key)
    except Exception:
        return False


def _get_supabase_client():
    """Returns a Supabase client. Only called when configured."""
    try:
        from supabase import create_client
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except ImportError:
        return None
    except Exception:
        return None


# ── File-based fallback ───────────────────────────────────────────────────────

def _get_audit_file(business_name: str) -> Path:
    """Returns path to local audit JSON file for a business."""
    safe_name = "".join(c if c.isalnum() else "_" for c in business_name)
    data_dir  = Path("data/audit_logs")
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{safe_name}_audit.json"


def _load_file_audit(business_name: str) -> list:
    f = _get_audit_file(business_name)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return []
    return []


def _save_file_audit(business_name: str, entries: list):
    f = _get_audit_file(business_name)
    try:
        f.write_text(json.dumps(entries, indent=2))
    except Exception:
        pass  # Never crash on persistence failure


# ── Public API ────────────────────────────────────────────────────────────────

def save_audit_entry(
    business_name: str,
    invoice_id: str,
    action: str,
    status: str,
    authorized_by: str = "System"
):
    """
    Persists one audit entry for a business.
    Tries Supabase first, falls back to file silently.
    Never crashes the app.
    """
    entry = {
        "business_name": business_name,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "invoice_id":    invoice_id,
        "action":        action,
        "status":        status,
        "authorized_by": authorized_by,
    }

    # Try Supabase
    if _supabase_configured():
        try:
            client = _get_supabase_client()
            if client:
                client.table("cognivis_audit").insert(entry).execute()
                return
        except Exception:
            pass  # Fall through to file

    # File fallback
    try:
        existing = _load_file_audit(business_name)
        existing.append(entry)
        _save_file_audit(business_name, existing)
    except Exception:
        pass  # Never crash


def load_audit_log(business_name: str) -> list:
    """
    Loads the full audit log for a business.
    Returns list of entry dicts.
    """
    # Try Supabase
    if _supabase_configured():
        try:
            client = _get_supabase_client()
            if client:
                resp = (
                    client.table("cognivis_audit")
                    .select("*")
                    .eq("business_name", business_name)
                    .order("timestamp", desc=True)
                    .limit(500)
                    .execute()
                )
                return resp.data or []
        except Exception:
            pass

    # File fallback
    return _load_file_audit(business_name)


def get_persistence_mode() -> str:
    """Returns 'supabase' or 'local' — shown in the UI for transparency."""
    return "supabase" if _supabase_configured() else "local"
