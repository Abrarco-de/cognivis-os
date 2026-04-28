"""
Cognivis OS V2 — Main Application
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json

from services.compliance_engine import audit_dataframe, check_transaction, get_compliance_summary, validate_transaction
from services.decision_engine import generate_all_recommendations, generate_all_insights
from services.trust_score import calculate_trust_score
from services.simulation_engine import simulate_price_change, simulate_vat_enforcement
from data.mock_data_generator import generate_realistic_invoices, get_foodics_mock, get_salla_mock
from utils.validators import normalize_dataframe, validate_vat_id, log_audit


# ── Page Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognivis OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { background-color: #020617 !important; color: #e2e8f0; font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] { background-color: #0a1628 !important; border-right: 1px solid #1a2744; }
.stButton > button { font-family: 'Inter', sans-serif !important; }

.c-shield  { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #00ff8733; border-left:3px solid #00ff87; margin-bottom:14px; }
.c-brain   { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #00d4ff33; border-left:3px solid #00d4ff; margin-bottom:14px; }
.c-danger  { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #ff4d4d55; border-left:3px solid #ff4d4d; margin-bottom:14px; }
.c-neutral { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #1a2744; margin-bottom:14px; }
.c-trust   { background:linear-gradient(135deg,#0d1b2a,#091221); border:1px solid #00ff8755; border-radius:14px; padding:24px; text-align:center; }
.c-warning { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #ffc10755; border-left:3px solid #ffc107; margin-bottom:14px; }

.metric-tile { background:#0d1b2a; border:1px solid #1a2744; border-radius:10px; padding:18px 16px; text-align:center; }
.metric-tile .label { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
.metric-tile .value { font-size:26px; font-weight:700; line-height:1; }
.metric-tile .sub   { font-size:11px; color:#64748b; margin-top:4px; }

.pos-terminal { background:#070f1a; border:1px solid #1a2744; border-top:3px solid #00ff87; border-radius:14px; padding:24px; }
.pos-blocked  { background:#1a0a0a; border:2px solid #ff4d4d; border-radius:12px; padding:20px; text-align:center; }
.pos-cleared  { background:#041a0f; border:2px solid #00ff87; border-radius:12px; padding:20px; text-align:center; }
.pos-receipt  { background:#f8f9fa; color:#1a1a2e; border-radius:8px; padding:20px; font-family:'Courier New',monospace; font-size:12px; line-height:1.8; border:1px dashed #ccc; }

.wa-bubble { background:#0d1b2a; color:#e2e8f0; padding:16px 18px; border-radius:12px; margin-bottom:12px; border-left:3px solid #00d4ff; font-size:14px; line-height:1.7; }
.ai-response { background:#070f1a; color:#e2e8f0; padding:20px 22px; border-radius:12px; margin-bottom:12px; border:1px solid #00d4ff33; font-size:14px; line-height:1.8; }

.badge-red    { background:rgba(255,77,77,.15);  color:#ff4d4d; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-green  { background:rgba(0,255,135,.12);  color:#00ff87; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-blue   { background:rgba(0,212,255,.12);  color:#00d4ff; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-yellow { background:rgba(255,193,7,.12);  color:#ffc107; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }

.trust-factor { background:#091221; border:1px solid #1a2744; border-radius:8px; padding:12px 16px; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }

.saas-header { background:#091221; border:1px solid #1a2744; border-radius:10px; padding:12px 14px; margin-bottom:16px; font-size:12px; color:#94a3b8; line-height:1.8; }
.step-pill   { text-align:center; padding:8px 4px; border-radius:8px; font-size:11px; font-weight:600; margin-bottom:2px; }

.stMetric { background:#0d1b2a; border-radius:10px; padding:12px; border:1px solid #1a2744; }
div[data-testid="stMetricValue"] { color:#e2e8f0 !important; }
.stTabs [data-baseweb="tab"] { color:#94a3b8; }
.stTabs [aria-selected="true"] { color:#00d4ff !important; border-bottom-color:#00d4ff !important; }
hr { border-color:#1a2744 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State ────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        'raw_data': None,
        'pos_source': None,
        'audit_ledger': [],
        'review_mode': {},
        'demo_mode': True,
        'behavior_memory': {'b2b_conversions': 0},
        'investor_step': 0,
        'cashier_state': 'idle',
        'cashier_result': None,
        'cashier_amount': 0,
        'cashier_vat': '',
        'cashier_items': '',
        'ai_messages': [],       # Grok conversation history
        'ai_loading': False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Safe dataframe helper ─────────────────────────────────────────────────────────
def safe_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    return df[[c for c in cols if c in df.columns]]


# ── Grok AI call ─────────────────────────────────────────────────────────────────
def call_grok(system_prompt: str, user_message: str) -> str:
    """
    Calls Grok (xAI) API. API key stored in Streamlit secrets as GROK_API_KEY.
    Falls back gracefully if key is missing.
    """
    try:
        api_key = st.secrets.get("GROK_API_KEY", "")
        if not api_key:
            return "⚠️ Grok API key not configured. Add GROK_API_KEY to your Streamlit secrets."

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok-3-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ],
            "max_tokens": 800,
            "temperature": 0.4
        }
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        return f"⚠️ API error: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"


def build_grok_system_prompt(df: pd.DataFrame, insights: dict) -> str:
    """
    Builds a tight, data-grounded system prompt so Grok responds
    as a business advisor who knows THIS business's actual numbers.
    """
    rev  = insights.get("revenue", {})
    risk = insights.get("risk", {})
    beh  = insights.get("behavior", {})

    return f"""You are Cognivis AI — a business intelligence advisor built specifically for Saudi SME owners.
You have direct access to this business's real transaction data. Speak like a smart, direct advisor — not a chatbot.
Never be generic. Every answer must reference the actual numbers below.
Use plain language. No bullet lists unless essential. No markdown headers. Keep answers under 150 words.
Always end with one concrete next action the owner can take today.

BUSINESS DATA SNAPSHOT:
- Total invoices: {rev.get('invoice_count', 0)}
- Total revenue: SAR {rev.get('total_revenue', 0):,.0f}
- Average order value: SAR {rev.get('avg_order_value', 0):,.0f}
- Top category: {rev.get('top_category', 'N/A')}
- Weakest category: {rev.get('bottom_category', 'N/A')}
- ZATCA violations: {risk.get('violation_count', 0)} ({risk.get('violation_rate_pct', 0)}% of invoices)
- Fine exposure: SAR {risk.get('capital_at_risk', 0):,.0f}
- Compliance patterns: {'; '.join(risk.get('patterns', ['No issues detected']))}
- Dominant resolution: {beh.get('dominant_pattern', 'None yet')}

Answer the owner's question based strictly on this data. Do not make up numbers not shown above."""


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:12px 0 8px;'>
        <span style='font-size:20px; font-weight:700; color:#e2e8f0;'>🧠 Cognivis OS</span><br>
        <span style='font-size:10px; color:#00ff87; letter-spacing:.1em;'>INTELLIGENCE LAYER</span>
    </div>
    """, unsafe_allow_html=True)

    mode = st.toggle("🎭 Investor Demo Mode", value=st.session_state.demo_mode)
    st.session_state.demo_mode = mode
    st.divider()

    org = "Al Baik Restaurant Group" if mode else "Your Business"
    st.markdown(f"""
    <div class="saas-header">
        <b style='color:#e2e8f0;'>🏢</b> {org}<br>
        <b style='color:#00ff87;'>●</b> {"12 POS Terminals · Demo" if mode else "Live"}<br>
        <b style='color:#64748b;'>👤</b> Admin · {datetime.now().strftime("%H:%M")}
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigation", [
        "📥 Integration Hub",
        "🛡️ ZATCA Shield",
        "🖥️ Cashier Terminal",
        "💡 AI Brain",
        "🔮 What-If Engine",
        "📓 Audit Ledger",
        "🎯 Investor Demo",
        "🏦 Financial Identity"
    ], label_visibility="collapsed")

    st.divider()
    if st.session_state.raw_data is not None:
        st.markdown(f"<span class='badge-green'>✓ {st.session_state.pos_source}</span>", unsafe_allow_html=True)
        if st.button("Disconnect", use_container_width=True):
            st.session_state.raw_data = None
            st.session_state.pos_source = None
            st.session_state.audit_ledger = []
            st.session_state.review_mode = {}
            st.session_state.ai_messages = []
            st.rerun()
    else:
        st.markdown("<span class='badge-yellow'>⚠ No POS Connected</span>", unsafe_allow_html=True)
    st.caption("v2.0 · April 2026")


# ── Page Header ──────────────────────────────────────────────────────────────────
page_meta = {
    "📥 Integration Hub":    ("📥", "Integration Hub",      "Connect your POS and load invoice data"),
    "🛡️ ZATCA Shield":       ("🛡️", "ZATCA Shield",         "Real-time compliance firewall"),
    "🖥️ Cashier Terminal":   ("🖥️", "Cashier Terminal",     "Live POS simulation — cashier point of view"),
    "💡 AI Brain":           ("💡", "AI Brain",              "Operational intelligence powered by Grok AI"),
    "🔮 What-If Engine":     ("🔮", "What-If Engine",        "Simulate decisions before you make them"),
    "📓 Audit Ledger":       ("📓", "Audit Ledger",          "Immutable compliance log"),
    "🎯 Investor Demo":      ("🎯", "Investor Demo",         "Guided 6-step product story"),
    "🏦 Financial Identity": ("🏦", "Financial Identity",    "Coming soon — Enterprise tier"),
}
icon, title, subtitle = page_meta.get(menu, ("", menu, ""))
st.markdown(f"""
<div style='margin-bottom:6px; padding-bottom:14px; border-bottom:1px solid #1a2744;'>
    <span style='font-size:20px; font-weight:700; color:#e2e8f0;'>{icon} {title}</span><br>
    <span style='font-size:13px; color:#64748b;'>{subtitle}</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════
# FINANCIAL IDENTITY (LOCKED)
# ══════════════════════════════════════════════════════════════════════════════════
if menu == "🏦 Financial Identity":
    st.markdown("""
    <div class="c-brain">
        <h4 style='color:#00d4ff; margin:0 0 8px;'>Coming in Enterprise Tier</h4>
        <p style='color:#94a3b8; margin:0; line-height:1.7;'>
            Every compliance action in Cognivis builds a verifiable financial identity.
            This module connects your Trust Score directly to Saudi lending partners —
            giving compliant SMEs faster access to working capital.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.info("🔒 Join the waitlist — launching with the Enterprise plan.")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 1 — INTEGRATION HUB  (FIX: CSV now reflected immediately)
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "📥 Integration Hub":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="c-shield">
            <div style='font-size:11px;color:#00ff87;font-weight:600;letter-spacing:.08em;'>SAUDI F&B</div>
            <div style='font-size:17px;font-weight:600;margin:6px 0;'>Foodics</div>
            <div style='font-size:12px;color:#64748b;'>POS leader · 10,000+ Saudi restaurants</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Connect Foodics", use_container_width=True):
            raw = normalize_dataframe(get_foodics_mock())
            st.session_state.raw_data = audit_dataframe(raw)
            st.session_state.pos_source = "Foodics"
            st.session_state.ai_messages = []
            log_audit("API Connect", "System", "Connected", "Foodics")
            st.rerun()

    with c2:
        st.markdown("""<div class="c-brain">
            <div style='font-size:11px;color:#00d4ff;font-weight:600;letter-spacing:.08em;'>E-COMMERCE</div>
            <div style='font-size:17px;font-weight:600;margin:6px 0;'>Salla</div>
            <div style='font-size:12px;color:#64748b;'>Saudi e-commerce · Retail & DTC</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Connect Salla", use_container_width=True):
            raw = normalize_dataframe(get_salla_mock())
            st.session_state.raw_data = audit_dataframe(raw)
            st.session_state.pos_source = "Salla"
            st.session_state.ai_messages = []
            log_audit("API Connect", "System", "Connected", "Salla")
            st.rerun()

    with c3:
        st.markdown("""<div class="c-neutral">
            <div style='font-size:11px;color:#ffc107;font-weight:600;letter-spacing:.08em;'>DEMO</div>
            <div style='font-size:17px;font-weight:600;margin:6px 0;'>Demo Dataset</div>
            <div style='font-size:12px;color:#64748b;'>50 realistic SME invoices with violations</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Load Demo Data", use_container_width=True):
            raw = generate_realistic_invoices(n=50, violation_rate=0.20)
            st.session_state.raw_data = audit_dataframe(raw)
            st.session_state.pos_source = "Demo Dataset"
            st.session_state.ai_messages = []
            log_audit("Demo Load", "System", "50 invoices loaded")
            st.rerun()

    st.divider()

    # ── CSV Upload — fixed: use session key to detect new upload ──────────────────
    st.markdown("<b style='color:#e2e8f0;'>Upload your own POS CSV</b>", unsafe_allow_html=True)
    st.caption("Columns accepted: invoice ID, amount, VAT number, category, date — any naming convention.")

    uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="csv_uploader")
    if uploaded is not None:
        # Only reprocess if it's a new file (different name or not yet loaded)
        if st.session_state.get("last_uploaded_csv") != uploaded.name:
            try:
                raw_csv = pd.read_csv(uploaded)
                if len(raw_csv) == 0:
                    st.error("❌ The uploaded file is empty.")
                else:
                    normalized = normalize_dataframe(raw_csv)
                    audited    = audit_dataframe(normalized)
                    st.session_state.raw_data           = audited
                    st.session_state.pos_source         = f"CSV: {uploaded.name}"
                    st.session_state.last_uploaded_csv  = uploaded.name
                    st.session_state.ai_messages        = []
                    log_audit("CSV Upload", "System", f"{len(audited)} rows from {uploaded.name}")
                    st.success(f"✅ Loaded {len(audited)} invoices from {uploaded.name}")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Could not read file: {str(e)}")
                st.caption("Make sure the file is a valid CSV with at least amount and invoice ID columns.")

    # ── Data preview ──────────────────────────────────────────────────────────────
    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        s  = get_compliance_summary(df)
        st.divider()
        st.markdown(f"<div style='font-size:13px;color:#64748b;margin-bottom:12px;'>Showing <b style='color:#e2e8f0;'>{s['total']} invoices</b> from <b style='color:#00ff87;'>{st.session_state.pos_source}</b></div>", unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Invoices", s['total'])
        m2.metric("Violations", s['violations'])
        m3.metric("Violation Rate", f"{s['violation_rate']}%")
        m4.metric("Fine Exposure", f"SAR {s['capital_at_risk']:,}")

        display_cols = ['invoice_id','date','category','amount_sar','customer_vat_id','doc_type','ai_risk_score','violations','status']
        st.dataframe(safe_df(df, display_cols), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ZATCA SHIELD  (FIX: shows exact error + trust score explanation)
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🛡️ ZATCA Shield":
    if st.session_state.raw_data is None:
        st.markdown("""<div class="c-neutral"><p style='color:#64748b;margin:0;'>Connect a POS source in Integration Hub first.</p></div>""", unsafe_allow_html=True)
    else:
        df    = st.session_state.raw_data
        trust = calculate_trust_score(df)
        s     = get_compliance_summary(df)

        # ── KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class="metric-tile">
                <div class="label">VIOLATIONS</div>
                <div class="value" style='color:#ff4d4d;'>{s['violations']}</div>
                <div class="sub">of {s['total']} invoices</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="metric-tile">
                <div class="label">FINE EXPOSURE</div>
                <div class="value" style='color:#ffc107;'>SAR {s['capital_at_risk']:,}</div>
                <div class="sub">at SAR 5,000 / violation</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            resolved = int((df.get('status', pd.Series(dtype=str)) == "Resolved").sum()) if 'status' in df.columns else 0
            st.markdown(f"""<div class="metric-tile">
                <div class="label">CAPITAL SAVED</div>
                <div class="value" style='color:#00ff87;'>SAR {resolved*5000:,}</div>
                <div class="sub">{resolved} violations resolved</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            sc = "#00ff87" if trust.score >= 80 else "#ffc107" if trust.score >= 60 else "#ff4d4d"
            st.markdown(f"""<div class="c-trust" style='padding:14px; cursor:pointer;'>
                <div style='font-size:10px;color:#64748b;letter-spacing:.08em;'>TRUST SCORE™</div>
                <div style='font-size:28px;font-weight:700;color:{sc};'>{trust.score}/100</div>
                <div style='font-size:11px;color:#00ff87;'>{trust.label}</div>
            </div>""", unsafe_allow_html=True)

        # ── Trust Score Breakdown (NEW — explains why score is what it is)
        with st.expander("📊 Why is my Trust Score this number?", expanded=(trust.score < 80)):
            st.markdown(f"""
            <div style='font-size:13px;color:#94a3b8;margin-bottom:14px;line-height:1.7;'>
                Your Trust Score is calculated from 3 factors. Here's exactly what's affecting yours:
            </div>
            """, unsafe_allow_html=True)

            # Factor 1: Violation Rate
            v_color = "#00ff87" if trust.violation_rate < 5 else "#ffc107" if trust.violation_rate < 20 else "#ff4d4d"
            v_status = "Good" if trust.violation_rate < 5 else "Needs work" if trust.violation_rate < 20 else "Critical"
            st.markdown(f"""
            <div class="trust-factor">
                <div>
                    <div style='font-size:13px;font-weight:600;color:#e2e8f0;'>Violation Rate</div>
                    <div style='font-size:12px;color:#64748b;margin-top:2px;'>
                        {trust.violation_rate}% of invoices have ZATCA violations
                        {"— this is healthy" if trust.violation_rate < 5 else "— this is dragging your score down" if trust.violation_rate >= 20 else ""}
                    </div>
                </div>
                <span style='color:{v_color};font-weight:600;font-size:13px;'>{v_status}</span>
            </div>
            """, unsafe_allow_html=True)

            # Factor 2: Resolution activity
            r_color = "#00ff87" if resolved > 0 else "#ff4d4d"
            r_status = f"{resolved} resolved" if resolved > 0 else "None resolved yet"
            st.markdown(f"""
            <div class="trust-factor">
                <div>
                    <div style='font-size:13px;font-weight:600;color:#e2e8f0;'>Resolution Activity</div>
                    <div style='font-size:12px;color:#64748b;margin-top:2px;'>
                        How quickly you fix violations when found
                        {"— keep resolving to boost this" if resolved == 0 else "— good, keep going"}
                    </div>
                </div>
                <span style='color:{r_color};font-weight:600;font-size:13px;'>{r_status}</span>
            </div>
            """, unsafe_allow_html=True)

            # Factor 3: Clean invoice consistency
            c_color = "#00ff87" if trust.consistency_score > 80 else "#ffc107" if trust.consistency_score > 50 else "#ff4d4d"
            st.markdown(f"""
            <div class="trust-factor">
                <div>
                    <div style='font-size:13px;font-weight:600;color:#e2e8f0;'>Invoice Consistency</div>
                    <div style='font-size:12px;color:#64748b;margin-top:2px;'>
                        {trust.consistency_score}% of invoices are fully clean
                    </div>
                </div>
                <span style='color:{c_color};font-weight:600;font-size:13px;'>{trust.consistency_score}%</span>
            </div>
            """, unsafe_allow_html=True)

            # What to do
            if trust.score < 80:
                fixes = []
                if s['violations'] > 0:
                    fixes.append(f"Resolve the {s['violations']} violations below — this alone will raise your score significantly.")
                if resolved == 0:
                    fixes.append("Start fixing violations to build your resolution track record.")
                if fixes:
                    st.markdown(f"""
                    <div class="c-brain" style='margin-top:12px;'>
                        <b style='color:#00d4ff;'>How to improve your score</b>
                        <p style='color:#94a3b8;font-size:13px;margin:6px 0 0;'>{"<br>".join(fixes)}</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # ── Resolution Queue
        violations = df[df['ai_risk_score'] >= 45]
        active_violations = violations[violations.get('status', pd.Series(dtype=str)) != 'Resolved'] if 'status' in violations.columns else violations

        if active_violations.empty:
            st.markdown("""<div class="c-shield">
                <b style='color:#00ff87;'>✓ Fully Compliant</b>
                <p style='color:#94a3b8;margin:4px 0 0;'>No active violations. Your business is ZATCA-clean.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"<b style='color:#ff4d4d;'>🚨 {len(active_violations)} violations require action</b>", unsafe_allow_html=True)
            st.caption("Each violation below explains exactly what is wrong and how to fix it.")

            for idx, row in active_violations.iterrows():
                # Get exact error messages for this invoice from the engine
                validation = validate_transaction(row.to_dict())
                fired_rules = validation.get("violations", [])
                messages    = validation.get("messages", [])

                with st.expander(
                    f"📄 {row['invoice_id']} · SAR {row['amount_sar']:,.2f} · "
                    f"{'  '.join(fired_rules) if fired_rules else 'Unknown'}",
                    expanded=False
                ):
                    # Show exactly what is wrong — one card per rule that fired
                    for rule_id, msg in zip(fired_rules, messages):
                        st.markdown(f"""
                        <div class="c-danger" style='padding:14px; margin-bottom:8px;'>
                            <div style='display:flex;gap:8px;align-items:center;margin-bottom:6px;'>
                                <span class='badge-red'>{rule_id}</span>
                                <span style='color:#64748b;font-size:11px;'>{row.get('category','')} · {row.get('date','')}</span>
                            </div>
                            <div style='font-size:13px;color:#e2e8f0;'>{msg}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Fix UI
                    if not st.session_state.review_mode.get(idx, False):
                        if st.button("🔧 Fix This Violation", key=f"open_{idx}", type="primary"):
                            st.session_state.review_mode[idx] = True
                            st.rerun()
                    else:
                        # Show the right fix based on which rules fired
                        if "BR-KSA-14" in fired_rules or "INVOICE_TYPE_CHECK" in fired_rules:
                            st.markdown("""<div class="c-brain">
                                <b style='color:#00d4ff;'>How to fix this</b>
                                <p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>
                                Add the customer's VAT registration number below.
                                This converts the invoice to a valid B2B Tax Invoice and removes the violation.</p>
                            </div>""", unsafe_allow_html=True)
                            vat = st.text_input(
                                "Customer VAT ID (15 digits):",
                                key=f"vat_{idx}",
                                placeholder="e.g. 310122393500003"
                            )
                            if st.button("✅ Save & Mark Resolved", key=f"approve_{idx}", type="primary"):
                                ok, err = validate_vat_id(vat)
                                if ok:
                                    st.session_state.raw_data.at[idx, 'customer_vat_id'] = vat
                                    st.session_state.raw_data.at[idx, 'doc_type']        = "Tax Invoice (388)"
                                    st.session_state.raw_data.at[idx, 'ai_risk_score']   = 0
                                    st.session_state.raw_data.at[idx, 'status']          = "Resolved"
                                    st.session_state.raw_data.at[idx, 'violations']      = "None"
                                    st.session_state.review_mode[idx] = False
                                    st.session_state.behavior_memory['b2b_conversions'] += 1
                                    log_audit("B2B Conversion", row['invoice_id'], "Resolved", "Admin")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {err}")

                        elif "VAT_FORMAT" in fired_rules:
                            st.markdown("""<div class="c-warning">
                                <b style='color:#ffc107;'>How to fix this</b>
                                <p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>
                                The VAT ID on this invoice has the wrong format.
                                Saudi VAT IDs are exactly 15 digits and start with 3.
                                Enter the correct number below.</p>
                            </div>""", unsafe_allow_html=True)
                            vat = st.text_input("Correct VAT ID:", key=f"vat_{idx}", placeholder="310122393500003")
                            if st.button("✅ Update VAT & Resolve", key=f"approve_{idx}", type="primary"):
                                ok, err = validate_vat_id(vat)
                                if ok:
                                    st.session_state.raw_data.at[idx, 'customer_vat_id'] = vat
                                    st.session_state.raw_data.at[idx, 'ai_risk_score']   = 0
                                    st.session_state.raw_data.at[idx, 'status']          = "Resolved"
                                    st.session_state.raw_data.at[idx, 'violations']      = "None"
                                    st.session_state.review_mode[idx] = False
                                    log_audit("VAT Format Fix", row['invoice_id'], "Resolved", "Admin")
                                    st.rerun()
                                else:
                                    st.error(f"❌ {err}")

                        elif "MISSING_FIELDS" in fired_rules:
                            st.markdown("""<div class="c-danger">
                                <b style='color:#ff4d4d;'>How to fix this</b>
                                <p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>
                                This invoice is missing required fields (ID or amount).
                                These should be corrected in your POS system directly.
                                Mark as acknowledged once reviewed.</p>
                            </div>""", unsafe_allow_html=True)
                            if st.button("✅ Mark as Reviewed", key=f"approve_{idx}"):
                                st.session_state.raw_data.at[idx, 'status'] = "Resolved"
                                st.session_state.review_mode[idx] = False
                                log_audit("Manual Review", row['invoice_id'], "Acknowledged", "Admin")
                                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CASHIER TERMINAL
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🖥️ Cashier Terminal":
    st.markdown("""
    <div class="c-neutral" style='margin-bottom:20px;'>
        <b style='color:#e2e8f0;'>Cashier point of view.</b>
        <span style='color:#64748b;font-size:13px;'> Cognivis runs silently in the background.
        When the cashier hits Print, Cognivis intercepts BEFORE the invoice reaches ZATCA.</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""<div class="pos-terminal">
            <div style='color:#00ff87;font-size:11px;letter-spacing:.1em;margin-bottom:16px;'>
            ● COGNIVIS SHIELD ACTIVE &nbsp;·&nbsp; TERMINAL 04</div>
            <div style='font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:18px;'>New Sale</div>
        </div>""", unsafe_allow_html=True)

        items  = st.text_area("Items (one per line)", value="Catering Package - Corporate\nExtra Setup Fee\nDelivery Charge", height=100)
        amount = st.number_input("Total Amount (SAR)", min_value=1.0, value=1850.0, step=50.0)
        vat_id = st.text_input("Customer VAT ID", placeholder="Leave blank for walk-in customer")

        if st.button("🖨️ Print Invoice", type="primary", use_container_width=True):
            result = check_transaction(amount, vat_id.strip())
            st.session_state.cashier_state  = 'blocked' if not result['allowed'] else 'cleared'
            st.session_state.cashier_result = result
            st.session_state.cashier_amount = amount
            st.session_state.cashier_vat    = vat_id.strip()
            st.session_state.cashier_items  = items
            st.rerun()

    with right:
        state = st.session_state.cashier_state

        if state == 'idle':
            st.markdown("""
            <div style='height:320px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1a2744;border-radius:12px;'>
                <div style='text-align:center;'>
                    <div style='font-size:40px;'>🖨️</div>
                    <div style='font-size:13px;margin-top:8px;color:#64748b;'>Invoice preview appears here</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif state == 'blocked':
            result = st.session_state.cashier_result
            amt    = st.session_state.cashier_amount
            msg    = result.get('first_message', 'This invoice cannot be processed.')
            rule   = result.get('first_rule_id', '')

            st.markdown(f"""
            <div class="pos-blocked">
                <div style='font-size:32px;'>🛑</div>
                <div style='font-size:18px;font-weight:700;color:#ff4d4d;margin:10px 0 6px;'>Cannot Print Invoice</div>
                <div style='font-size:12px;color:#ff8080;margin-bottom:16px;'>Cognivis blocked this before it was sent</div>
                <div style='background:#1a0a0a;border-radius:8px;padding:14px;text-align:left;margin-bottom:12px;'>
                    <div style='font-size:11px;color:#ff4d4d;font-weight:600;margin-bottom:6px;'>WHAT WENT WRONG</div>
                    <div style='font-size:13px;color:#e2e8f0;line-height:1.6;'>{msg}</div>
                </div>
                <div style='font-size:11px;color:#64748b;'>Rule: {rule} · Fine risk: SAR 5,000+</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<b style='color:#e2e8f0;'>Enter customer VAT number to continue</b>", unsafe_allow_html=True)
            st.caption("Every registered Saudi business has a 15-digit VAT number starting with 3.")

            fix_vat = st.text_input("Customer VAT Number:", key="cashier_fix_vat", placeholder="310122393500003")
            if st.button("✅ Submit & Print Invoice", type="primary", use_container_width=True):
                ok, err = validate_vat_id(fix_vat)
                if ok:
                    st.session_state.cashier_state = 'cleared'
                    st.session_state.cashier_vat   = fix_vat
                    log_audit("Cashier VAT Fix", f"POS-{int(st.session_state.cashier_amount)}", "Resolved", "Cashier")
                    st.session_state.behavior_memory['b2b_conversions'] += 1
                    st.rerun()
                else:
                    st.error(f"❌ {err} — Please check the number with the customer.")

        elif state == 'cleared':
            amt        = st.session_state.cashier_amount
            vat        = st.session_state.cashier_vat
            items_text = st.session_state.cashier_items
            now        = datetime.now().strftime("%Y-%m-%d %H:%M")
            inv_id     = f"INV-{datetime.now().strftime('%H%M%S')}"
            vat_amt    = round(amt * 0.15, 2)
            total      = round(amt + vat_amt, 2)
            items_list = [i.strip() for i in items_text.split('\n') if i.strip()]
            per_item   = amt / max(len(items_list), 1)
            lines      = "\n".join([f"  {item:<28} {per_item:>8.2f}" for item in items_list])

            st.markdown("""<div class="pos-cleared">
                <div style='font-size:22px;'>✅</div>
                <div style='font-size:15px;font-weight:600;color:#00ff87;margin:6px 0;'>Invoice Approved</div>
                <div style='font-size:12px;color:#00ff87aa;'>ZATCA-Compliant · Sending now</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="pos-receipt">
<pre style='font-family:Courier New,monospace;font-size:12px;color:#1a1a2e;margin:0;line-height:1.8;'>
================================
      COGNIVIS TAX INVOICE
================================
Invoice : {inv_id}
Date    : {now}
--------------------------------
ITEMS:
{lines}
--------------------------------
Subtotal:        SAR {amt:>9.2f}
VAT (15%):       SAR {vat_amt:>9.2f}
================================
TOTAL:           SAR {total:>9.2f}
================================
Customer VAT: {vat}

  [✓] ZATCA COMPLIANT
  Powered by Cognivis OS Shield
================================</pre>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 New Transaction", use_container_width=True):
                st.session_state.cashier_state = 'idle'
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 4 — AI BRAIN  (NEW: Grok-powered, no graphs, human-level insights)
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "💡 AI Brain":
    if st.session_state.raw_data is None:
        st.markdown("""<div class="c-neutral"><p style='color:#64748b;margin:0;'>
        Connect a POS source in Integration Hub to activate the AI Brain.</p></div>""", unsafe_allow_html=True)
    else:
        df      = st.session_state.raw_data
        memory  = st.session_state.behavior_memory
        insights = generate_all_insights(df, memory)

        # ── Status bar
        rev  = insights['revenue']
        risk = insights['risk']
        st.markdown(f"""
        <div class="c-brain">
            <b style='color:#00d4ff;'>AI Brain · Powered by Grok · Live Data</b>
            <p style='color:#94a3b8;font-size:13px;margin:4px 0 0;'>
                Analysing {rev.get('invoice_count',0)} invoices · 
                SAR {rev.get('total_revenue',0):,.0f} revenue · 
                {risk.get('violation_count',0)} compliance issues
            </p>
        </div>
        """, unsafe_allow_html=True)

        system_prompt = build_grok_system_prompt(df, insights)

        # ── Auto-generated insights (shown on load)
        actions = insights.get("actions", [])
        if actions:
            st.markdown("<b style='color:#e2e8f0;'>What Cognivis sees in your business right now</b>", unsafe_allow_html=True)
            st.caption("Generated from your actual transaction data — not generic advice.")
            for act in actions:
                icon  = "🛡️" if act['category'] == "COMPLIANCE" else "📈" if act['category'] == "REVENUE" else "⚙️"
                color = "#ff4d4d" if act['category'] == "COMPLIANCE" else "#00d4ff" if act['category'] == "REVENUE" else "#ffc107"
                priority_label = {"HIGH": "Urgent", "MEDIUM": "Recommended", "LOW": "Note"}.get(act['priority'], "")
                st.markdown(f"""
                <div style='background:#0d1b2a;border:1px solid #1a2744;border-radius:12px;padding:18px;margin-bottom:14px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                        <span style='font-weight:600;color:#e2e8f0;'>{icon} {act['title']}</span>
                        <span style='font-size:11px;color:{color};background:{color}22;padding:3px 8px;border-radius:20px;'>{priority_label}</span>
                    </div>
                    <div style='font-size:14px;color:#94a3b8;line-height:1.7;margin-bottom:8px;'>{act['description']}</div>
                    <div style='font-size:12px;color:#64748b;'>💡 {act['system_action']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # ── Grok Chat — ask anything about your business
        st.markdown("<b style='color:#00d4ff;'>Ask Cognivis AI anything about your business</b>", unsafe_allow_html=True)
        st.caption("Powered by Grok — answers are based on your actual data, not generic advice.")

        # Show conversation history
        for msg in st.session_state.ai_messages:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div style='background:#091221;border:1px solid #1a2744;border-radius:10px;
                            padding:12px 16px;margin-bottom:8px;font-size:14px;color:#94a3b8;'>
                    <b style='color:#e2e8f0;'>You:</b> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ai-response'>
                    <b style='color:#00d4ff;'>🧠 Cognivis AI:</b><br><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)

        # Quick question buttons
        st.markdown("<div style='margin-bottom:8px;font-size:12px;color:#64748b;'>Quick questions:</div>", unsafe_allow_html=True)
        quick_cols = st.columns(3)
        quick_questions = [
            "What is my biggest risk right now?",
            "Which category should I focus on?",
            "How do I improve my trust score?",
            "What should I do about my violations?",
            "Where am I losing the most money?",
            "What should I do this week?",
        ]
        for i, (col, q) in enumerate(zip(quick_cols * 2, quick_questions)):
            with col:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.ai_messages.append({"role": "user", "content": q})
                    with st.spinner("Cognivis is thinking..."):
                        answer = call_grok(system_prompt, q)
                    st.session_state.ai_messages.append({"role": "assistant", "content": answer})
                    st.rerun()

        # Free-text input
        st.markdown("<br>", unsafe_allow_html=True)
        user_q = st.text_input(
            "Or type your own question:",
            placeholder="e.g. Why are my margins low? What should I order less of next week?",
            key="ai_input"
        )
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            if st.button("Ask Cognivis AI", type="primary", use_container_width=True):
                if user_q.strip():
                    st.session_state.ai_messages.append({"role": "user", "content": user_q})
                    with st.spinner("Cognivis is thinking..."):
                        answer = call_grok(system_prompt, user_q)
                    st.session_state.ai_messages.append({"role": "assistant", "content": answer})
                    st.rerun()
        with col_clear:
            if st.button("Clear chat", use_container_width=True):
                st.session_state.ai_messages = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 5 — WHAT-IF ENGINE
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🔮 What-If Engine":
    if st.session_state.raw_data is None:
        st.markdown("""<div class="c-neutral"><p style='color:#64748b;margin:0;'>Connect POS data in Integration Hub first.</p></div>""", unsafe_allow_html=True)
    else:
        df = st.session_state.raw_data
        tab1, tab2 = st.tabs(["📈 Price Change", "🛡️ VAT Capture Rate"])

        with tab1:
            pct = st.slider("Simulate price change (%)", -20, 40, 15)
            r   = simulate_price_change(df, pct)
            c1, c2, c3 = st.columns(3)
            c1.metric("Revenue Change", f"SAR {r['revenue_delta']:+,.0f}", f"{pct:+}%")
            c2.metric("New Risks", r['new_risk_count'],
                      delta=f"+SAR {r['fine_exposure_delta']:,}" if r['new_risk_count'] > 0 else "None",
                      delta_color="inverse")
            c3.metric("Projected Revenue", f"SAR {r['sim_revenue']:,.0f}")
            if r['new_risk_count'] > 0:
                st.markdown(f"""<div class="c-danger">
                    <b style='color:#ff4d4d;'>⚠ Compliance Risk Alert</b>
                    <p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>
                    A {pct}% price increase pushes <b>{r['new_risk_count']} transactions</b> over the SAR 1,000 threshold.
                    Activate VAT capture on the Cashier Terminal before rolling this out.
                    Fine exposure: <b>SAR {r['fine_exposure_delta']:,}</b>.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="c-shield"><b style='color:#00ff87;'>✓ No new compliance risks from this change</b></div>""", unsafe_allow_html=True)

        with tab2:
            cap = st.slider("VAT capture rate on invoices ≥ SAR 1,000 (%)", 0, 100, 70)
            r2  = simulate_vat_enforcement(df, cap / 100)
            c1, c2, c3 = st.columns(3)
            c1.metric("At Risk", r2['total_at_risk'])
            c2.metric("Resolved by Capture", r2['resolved_by_capture'], f"SAR {r2['fine_saved']:,} saved")
            c3.metric("Remaining Exposure", f"SAR {r2['fine_remaining']:,}",
                      delta=f"{r2['remaining_violations']} unresolved", delta_color="inverse")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 6 — AUDIT LEDGER
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "📓 Audit Ledger":
    if not st.session_state.audit_ledger:
        st.markdown("""<div class="c-neutral"><p style='color:#64748b;margin:0;'>No events yet. Actions appear here as you use the system.</p></div>""", unsafe_allow_html=True)
    else:
        ledger = pd.DataFrame(st.session_state.audit_ledger)
        st.markdown(f"<span class='badge-green'>{len(ledger)} entries</span>", unsafe_allow_html=True)
        st.dataframe(ledger, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 7 — INVESTOR DEMO
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🎯 Investor Demo":
    if st.session_state.raw_data is None:
        raw = generate_realistic_invoices(n=50, violation_rate=0.22)
        st.session_state.raw_data   = audit_dataframe(raw)
        st.session_state.pos_source = "Demo · Al Baik Group"

    df    = st.session_state.raw_data
    s     = get_compliance_summary(df)
    trust = calculate_trust_score(df)
    insights = generate_all_insights(df, st.session_state.behavior_memory)

    STEPS = ["Connect POS","Scan Violations","Intercept Live","Money Saved","Trust Score","AI Intelligence"]
    step  = st.session_state.investor_step

    cols = st.columns(len(STEPS))
    for i, (col, label) in enumerate(zip(cols, STEPS)):
        with col:
            if i < step:
                bg, border, txt = "#00ff8722", "#00ff87", "#00ff87"
            elif i == step:
                bg, border, txt = "#00d4ff11", "#00d4ff", "#00d4ff"
            else:
                bg, border, txt = "#0d1b2a", "#1a2744", "#1a2744"
            st.markdown(f"""<div class='step-pill' style='background:{bg};color:{txt};border:1px solid {border};'>
                {"✓ " if i < step else ""}{i+1}. {label}
            </div>""", unsafe_allow_html=True)

    st.divider()

    if step == 0:
        st.markdown("""<div class="c-shield">
            <h4 style='color:#00ff87;margin:0 0 8px;'>Step 1 — Connect your POS</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>Cognivis connects to Foodics, Salla, or any POS in under 60 seconds.</p>
        </div>""", unsafe_allow_html=True)
        st.success(f"✅ {st.session_state.pos_source} — {s['total']} invoices loaded and scored.")

    elif step == 1:
        st.markdown(f"""<div class="c-danger">
            <h4 style='color:#ff4d4d;margin:0 0 8px;'>Step 2 — Violations Detected Instantly</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>AI rule engine scanned {s['total']} invoices in milliseconds.</p>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Violations Found", s['violations'])
        c2.metric("Violation Rate", f"{s['violation_rate']}%")
        c3.metric("Fine Exposure", f"SAR {s['capital_at_risk']:,}")

    elif step == 2:
        st.markdown("""<div class="c-shield">
            <h4 style='color:#00ff87;margin:0 0 8px;'>Step 3 — Real-Time Interception</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>Cognivis fires BEFORE the invoice is created. Plain English — no jargon.</p>
        </div>""", unsafe_allow_html=True)
        result = check_transaction(1850.00, "")
        st.markdown(f"""
        <div class="pos-blocked" style='max-width:420px;'>
            <div style='font-size:28px;'>🛑</div>
            <div style='font-size:16px;font-weight:700;color:#ff4d4d;margin:8px 0 4px;'>Cannot Print Invoice</div>
            <div style='font-size:13px;color:#e2e8f0;margin:8px 0;line-height:1.6;'>{result.get('first_message','')}</div>
            <div style='font-size:11px;color:#ff4d4d;'>{result.get('first_rule_id','')} · Fine risk: SAR 5,000</div>
        </div>
        """, unsafe_allow_html=True)

    elif step == 3:
        st.markdown("""<div class="c-shield">
            <h4 style='color:#00ff87;margin:0 0 8px;'>Step 4 — Real Money Saved</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>SAR 5,000 minimum fine per violation. Cognivis blocks every single one.</p>
        </div>""", unsafe_allow_html=True)
        saved = s['violations'] * 5000
        st.metric("Potential Savings", f"SAR {saved:,}", f"{s['violations']} violations blocked")
        st.progress(min(1.0, s['violations'] / 20))

    elif step == 4:
        sc = "#00ff87" if trust.score >= 80 else "#ffc107"
        st.markdown(f"""<div class="c-trust">
            <div style='font-size:11px;color:#64748b;letter-spacing:.08em;margin-bottom:8px;'>COGNIVIS TRUST SCORE™</div>
            <div style='font-size:52px;font-weight:700;color:{sc};'>{trust.score}</div>
            <div style='font-size:14px;color:#00ff87;margin:4px 0 8px;'>{trust.label}</div>
            <div style='font-size:12px;color:#64748b;'>Violation Rate: {trust.violation_rate}% · Consistency: {trust.consistency_score}%</div>
            <div style='font-size:11px;color:#94a3b8;margin-top:10px;'>
            This score will connect to Saudi lending partners — giving compliant SMEs faster credit access.</div>
        </div>""", unsafe_allow_html=True)

    elif step == 5:
        st.markdown("""<div class="c-brain">
            <h4 style='color:#00d4ff;margin:0 0 8px;'>Step 6 — The AI Brain</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>Beyond compliance — operational intelligence powered by Grok AI.</p>
        </div>""", unsafe_allow_html=True)
        acts = insights.get("actions", [])
        if acts:
            act = acts[0]
            st.markdown(f"""<div class="ai-response">
                🧠 <b style='color:#00d4ff;'>Cognivis AI</b><br><br>
                {act['description']}<br><br>
                <i style='color:#64748b;font-size:12px;'>Next step: {act['system_action']}</i>
            </div>""", unsafe_allow_html=True)
        st.success("🎉 This is Cognivis OS — compliance firewall + AI intelligence, in one system.")

    st.divider()
    nav1, _, nav2 = st.columns([1, 3, 1])
    with nav1:
        if step > 0 and st.button("← Back"):
            st.session_state.investor_step -= 1
            st.rerun()
    with nav2:
        if step < len(STEPS) - 1:
            if st.button("Next →", type="primary"):
                st.session_state.investor_step += 1
                st.rerun()
        else:
            if st.button("↺ Restart"):
                st.session_state.investor_step = 0
                st.rerun()
