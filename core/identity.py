"""
Cognivis OS — Identity Layer
=============================
Handles per-business identity, login, and session isolation.
Each business gets its own data, trust score, and audit log.

For pilots: uses a simple PIN-based login stored in Streamlit secrets.
For production: swap with Supabase Auth (one-day migration).

Structure in secrets.toml:
    [businesses]
    "AlBaik_Riyadh"   = "1234"
    "CafeNajd_Jeddah" = "5678"
"""

import streamlit as st
import hashlib
from datetime import datetime


# ── Business registry ─────────────────────────────────────────────────────────

def get_registered_businesses() -> dict:
    """
    Returns dict of { business_name: pin_hash } from Streamlit secrets.
    Falls back to a demo business if secrets not configured.
    """
    try:
        biz = dict(st.secrets.get("businesses", {}))
        if biz:
            return biz
    except Exception:
        pass
    # Demo fallback — always available
    return {"Demo Business": "0000"}


def _hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.strip().encode()).hexdigest()


def verify_login(business_name: str, pin: str) -> bool:
    """Verifies business name + PIN. Returns True if valid."""
    businesses = get_registered_businesses()
    if business_name not in businesses:
        return False
    stored = businesses[business_name]
    # Support both plain PINs and hashed PINs in secrets
    return stored == pin.strip() or stored == _hash_pin(pin)


# ── Session management ────────────────────────────────────────────────────────

def init_identity_state():
    """Initialises identity-related session state keys."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'business_name' not in st.session_state:
        st.session_state.business_name = None
    if 'login_error' not in st.session_state:
        st.session_state.login_error = ""


def is_authenticated() -> bool:
    return st.session_state.get('authenticated', False)


def get_current_business() -> str:
    return st.session_state.get('business_name', 'Unknown')


def logout():
    """Clears all session state and returns to login screen."""
    keys_to_clear = [
        'authenticated', 'business_name', 'raw_data', 'pos_source',
        'audit_ledger', 'review_mode', 'ai_messages', 'behavior_memory',
        'investor_step', 'cashier_state', 'cashier_result',
        'cashier_amount', 'cashier_vat', 'cashier_items',
        'last_uploaded_csv', 'login_error'
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]


# ── Login screen ──────────────────────────────────────────────────────────────

def render_login_screen():
    """
    Renders the full-page login screen.
    Called in app.py before anything else if not authenticated.
    """
    st.markdown("""
    <div style='max-width:420px; margin:80px auto 0;'>
        <div style='text-align:center; margin-bottom:32px;'>
            <div style='font-size:40px;'>🧠</div>
            <div style='font-size:24px; font-weight:700; color:#e2e8f0; margin-top:8px;'>Cognivis OS</div>
            <div style='font-size:13px; color:#00ff87; letter-spacing:.1em; margin-top:4px;'>INTELLIGENCE LAYER</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    businesses = list(get_registered_businesses().keys())

    with st.form("login_form"):
        st.markdown("<div style='max-width:420px; margin:0 auto;'>", unsafe_allow_html=True)

        business = st.selectbox(
            "Select your business",
            options=businesses,
            help="Contact your Cognivis admin to add your business."
        )
        pin = st.text_input(
            "Access PIN",
            type="password",
            placeholder="Enter your 4-digit PIN"
        )

        submitted = st.form_submit_button(
            "Access Cognivis OS",
            use_container_width=True,
            type="primary"
        )

        if submitted:
            if verify_login(business, pin):
                st.session_state.authenticated  = True
                st.session_state.business_name  = business
                st.session_state.login_error    = ""
                st.session_state.behavior_memory = {'b2b_conversions': 0}
                st.rerun()
            else:
                st.session_state.login_error = "Incorrect PIN. Please try again."

        if st.session_state.get('login_error'):
            st.error(st.session_state.login_error)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:24px;'>
        <span style='font-size:12px; color:#1a2744;'>
        New business? Contact us to get access.
        </span>
    </div>
    """, unsafe_allow_html=True)
