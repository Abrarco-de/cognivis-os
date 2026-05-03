"""
Cognivis OS — Failure Handler
================================
Wraps all external calls with retry logic, timeouts, and clean fallbacks.
The app should NEVER crash due to an external service being down.

Covers:
  - Grok AI API calls (retry + fallback message)
  - Supabase calls (already wrapped in persistence.py)
  - CSV parsing errors
  - VAT validation service errors
"""

import time
import requests
import streamlit as st
from functools import wraps


# ── Retry decorator ───────────────────────────────────────────────────────────

def with_retry(max_attempts: int = 3, delay_seconds: float = 1.5, fallback=None):
    """
    Decorator that retries a function up to max_attempts times.
    Returns fallback value if all attempts fail.

    Usage:
        @with_retry(max_attempts=3, fallback="Service unavailable")
        def call_api(): ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay_seconds * (attempt + 1))  # backoff
            # All attempts failed
            if fallback is not None:
                return fallback
            raise last_error
        return wrapper
    return decorator


# ── Safe Grok call with retry ─────────────────────────────────────────────────

@with_retry(
    max_attempts=3,
    delay_seconds=2,
    fallback="⚠️ Cognivis AI is temporarily unavailable. Please try again in a moment."
)
def safe_grok_call(system_prompt: str, user_message: str) -> str:
    """
    Calls Grok API with retry logic and clean error messages.
    Never crashes the app — always returns a string.
    """
    try:
        api_key = st.secrets.get("GROK_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return (
            "⚠️ Grok API key not configured. "
            "Add GROK_API_KEY to your Streamlit secrets to enable AI insights."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":    "grok-2-latest",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "max_tokens":  800,
        "temperature": 0.4,
    }

    resp = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )

    # Handle specific HTTP errors with clean messages
    if resp.status_code == 401:
        return "⚠️ Invalid Grok API key. Please check your GROK_API_KEY in Streamlit secrets."
    if resp.status_code == 429:
        return "⚠️ Grok rate limit reached. Please wait a moment and try again."
    if resp.status_code == 400:
        return "⚠️ Request error. Please try rephrasing your question."

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Safe CSV parser ───────────────────────────────────────────────────────────

def safe_parse_csv(uploaded_file) -> tuple:
    """
    Safely parses an uploaded CSV file.
    Returns (dataframe, error_message).
    error_message is empty string on success.
    """
    import pandas as pd

    if uploaded_file is None:
        return None, "No file provided."

    try:
        df = pd.read_csv(uploaded_file)

        if len(df) == 0:
            return None, "The uploaded file is empty. Please check your export."

        if len(df.columns) < 2:
            return None, "File has fewer than 2 columns. Please use a proper POS export."

        if len(df) > 10000:
            return None, "File has more than 10,000 rows. Please upload a monthly export (max 10,000 invoices)."

        return df, ""

    except UnicodeDecodeError:
        # Try with different encoding
        try:
            import io
            uploaded_file.seek(0)
            df = pd.read_csv(io.TextIOWrapper(uploaded_file, encoding='latin-1'))
            return df, ""
        except Exception:
            return None, "Could not read file encoding. Please save your CSV as UTF-8 and try again."

    except Exception as e:
        return None, f"Could not parse file: {str(e)}. Please ensure it is a valid CSV export from your POS."


# ── Safe VAT validator ────────────────────────────────────────────────────────

def safe_validate_vat(vat_id: str) -> tuple:
    """
    Validates VAT ID with a clear, user-friendly error message.
    Returns (is_valid: bool, message: str).
    """
    vat = str(vat_id).strip()

    if not vat:
        return False, "Please enter the customer's VAT number."
    if not vat.isdigit():
        return False, f"VAT ID should only contain numbers. You entered: '{vat}'"
    if len(vat) != 15:
        return False, f"Saudi VAT IDs are exactly 15 digits. Yours has {len(vat)} digits."
    if not vat.startswith("3"):
        return False, "Saudi VAT IDs always start with '3'. Please double-check with the customer."

    return True, "Valid"


# ── UI error renderer ─────────────────────────────────────────────────────────

def show_error(message: str, suggestion: str = ""):
    """Renders a consistent error card in the UI."""
    sugg_html = f"<div style='font-size:12px;color:#64748b;margin-top:6px;'>💡 {suggestion}</div>" if suggestion else ""
    st.markdown(f"""
    <div style='background:#1a0a0a;border:1px solid #ff4d4d55;border-left:3px solid #ff4d4d;
                border-radius:10px;padding:14px 16px;margin-bottom:12px;'>
        <div style='font-size:13px;color:#ff8080;'>{message}</div>
        {sugg_html}
    </div>
    """, unsafe_allow_html=True)


def show_warning(message: str):
    """Renders a consistent warning card."""
    st.markdown(f"""
    <div style='background:#1a1200;border:1px solid #ffc10755;border-left:3px solid #ffc107;
                border-radius:10px;padding:14px 16px;margin-bottom:12px;'>
        <div style='font-size:13px;color:#ffc107;'>{message}</div>
    </div>
    """, unsafe_allow_html=True)


def show_success(message: str):
    """Renders a consistent success card."""
    st.markdown(f"""
    <div style='background:#041a0f;border:1px solid #00ff8755;border-left:3px solid #00ff87;
                border-radius:10px;padding:14px 16px;margin-bottom:12px;'>
        <div style='font-size:13px;color:#00ff87;'>{message}</div>
    </div>
    """, unsafe_allow_html=True)
