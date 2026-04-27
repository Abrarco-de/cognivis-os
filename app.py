"""
Cognivis OS V2 — Main Application
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from services.compliance_engine import audit_dataframe, check_transaction, get_compliance_summary
from services.decision_engine import generate_all_recommendations
from services.trust_score import calculate_trust_score
from services.simulation_engine import simulate_price_change, simulate_vat_enforcement
from data.mock_data_generator import generate_realistic_invoices, generate_revenue_trend, get_foodics_mock, get_salla_mock
from utils.validators import normalize_dataframe, validate_vat_id, log_audit


# ── Page Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognivis OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Brand Theme ───────────────────────────────────────────────────────────────────
# Neon Green  #00ff87  → ZATCA / Compliance / Shield
# Neon Blue   #00d4ff  → AI / Intelligence / Brain
# Background  #020617
# Surface     #0d1b2a
# Border      #1a2744

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

.metric-tile { background:#0d1b2a; border:1px solid #1a2744; border-radius:10px; padding:18px 16px; text-align:center; }
.metric-tile .label { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
.metric-tile .value { font-size:26px; font-weight:700; line-height:1; }
.metric-tile .sub   { font-size:11px; color:#64748b; margin-top:4px; }

.pos-terminal { background:#070f1a; border:1px solid #1a2744; border-top:3px solid #00ff87; border-radius:14px; padding:24px; }
.pos-blocked  { background:#1a0a0a; border:2px solid #ff4d4d; border-radius:12px; padding:20px; text-align:center; }
.pos-cleared  { background:#041a0f; border:2px solid #00ff87; border-radius:12px; padding:20px; text-align:center; }
.pos-receipt  { background:#f8f9fa; color:#1a1a2e; border-radius:8px; padding:20px; font-family:'Courier New',monospace; font-size:12px; line-height:1.8; border:1px dashed #ccc; }

.wa-bubble { background:#0d1b2a; color:#e2e8f0; padding:16px 18px; border-radius:12px; margin-bottom:12px; border-left:3px solid #00d4ff; font-size:14px; line-height:1.7; }

.badge-red    { background:rgba(255,77,77,.15);  color:#ff4d4d; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-green  { background:rgba(0,255,135,.12);  color:#00ff87; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-blue   { background:rgba(0,212,255,.12);  color:#00d4ff; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
.badge-yellow { background:rgba(255,193,7,.12);  color:#ffc107; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helper: safe dataframe (fixes duplicate columns bug) ─────────────────────────
def safe_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    existing = [c for c in cols if c in df.columns]
    return df[existing]


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
            st.rerun()
    else:
        st.markdown("<span class='badge-yellow'>⚠ No POS Connected</span>", unsafe_allow_html=True)
    st.caption("v2.0 · April 2026")


# ── Page Header ──────────────────────────────────────────────────────────────────
page_meta = {
    "📥 Integration Hub":    ("📥", "Integration Hub",      "Connect your POS and load invoice data"),
    "🛡️ ZATCA Shield":       ("🛡️", "ZATCA Shield",         "Real-time compliance firewall"),
    "🖥️ Cashier Terminal":   ("🖥️", "Cashier Terminal",     "Live POS simulation — cashier point of view"),
    "💡 AI Brain":           ("💡", "AI Brain",              "Data-driven business intelligence"),
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
# MODULE 1 — INTEGRATION HUB
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
            log_audit("Demo Load", "System", "50 invoices loaded")
            st.rerun()

    st.divider()
    uploaded = st.file_uploader("Or upload your own POS CSV", type=["csv"])
    if uploaded:
        raw = normalize_dataframe(pd.read_csv(uploaded))
        st.session_state.raw_data = audit_dataframe(raw)
        st.session_state.pos_source = "CSV Upload"
        log_audit("CSV Upload", "System", f"{len(st.session_state.raw_data)} rows")
        st.rerun()

    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        s  = get_compliance_summary(df)
        st.divider()
        st.markdown(f"<div style='font-size:13px;color:#64748b;margin-bottom:12px;'>Showing {s['total']} invoices from <b style='color:#e2e8f0;'>{st.session_state.pos_source}</b></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Invoices", s['total'])
        m2.metric("Violations", s['violations'])
        m3.metric("Violation Rate", f"{s['violation_rate']}%")
        m4.metric("Fine Exposure", f"SAR {s['capital_at_risk']:,}")
        display_cols = ['invoice_id','date','category','amount_sar','customer_vat_id','doc_type','ai_risk_score','violations']
        st.dataframe(safe_df(df, display_cols), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ZATCA SHIELD
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🛡️ ZATCA Shield":
    if st.session_state.raw_data is None:
        st.markdown("""<div class="c-neutral"><p style='color:#64748b;margin:0;'>Connect a POS source in Integration Hub first.</p></div>""", unsafe_allow_html=True)
    else:
        df    = st.session_state.raw_data
        trust = calculate_trust_score(df)
        s     = get_compliance_summary(df)

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
            resolved = len(df[df['status'] == "Resolved"]) if 'status' in df.columns else 0
            st.markdown(f"""<div class="metric-tile">
                <div class="label">CAPITAL SAVED</div>
                <div class="value" style='color:#00ff87;'>SAR {resolved*5000:,}</div>
                <div class="sub">{resolved} violations resolved</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            sc = "#00ff87" if trust.score >= 80 else "#ffc107" if trust.score >= 60 else "#ff4d4d"
            st.markdown(f"""<div class="c-trust" style='padding:14px;'>
                <div style='font-size:10px;color:#64748b;letter-spacing:.08em;'>TRUST SCORE™</div>
                <div style='font-size:28px;font-weight:700;color:{sc};'>{trust.score}/100</div>
                <div style='font-size:11px;color:#00ff87;'>{trust.label}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        violations = df[df['ai_risk_score'] >= 80]
        if violations.empty:
            st.markdown("""<div class="c-shield">
                <b style='color:#00ff87;'>✓ Fully Compliant</b>
                <p style='color:#94a3b8;margin:4px 0 0;'>No violations detected. Your business is ZATCA-clean.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"<b style='color:#ff4d4d;'>🚨 {len(violations)} violations require action</b>", unsafe_allow_html=True)
            for idx, row in violations.iterrows():
                with st.expander(f"Invoice {row['invoice_id']} · SAR {row['amount_sar']:,.2f} · Risk {row['ai_risk_score']}/100"):
                    st.markdown(f"""<div style='display:flex;gap:8px;margin-bottom:12px;'>
                        <span class='badge-red'>HIGH RISK</span>
                        <span class='badge-yellow'>{row.get('violations','')}</span>
                        <span style='color:#64748b;font-size:12px;'>{row.get('category','')} · {row.get('date','')}</span>
                    </div>""", unsafe_allow_html=True)
                    if not st.session_state.review_mode.get(idx, False):
                        if st.button("Resolve Violation", key=f"open_{idx}", type="primary"):
                            st.session_state.review_mode[idx] = True
                            st.rerun()
                    else:
                        st.markdown("""<div class="c-brain">
                            <b style='color:#00d4ff;'>AI Recommendation</b>
                            <p style='color:#94a3b8;margin:4px 0 0;font-size:13px;'>
                            Convert to a B2B Tax Invoice by capturing the customer VAT ID.
                            This eliminates the violation and avoids the SAR 5,000 fine.</p>
                        </div>""", unsafe_allow_html=True)
                        vat = st.text_input("Customer VAT ID (15 digits):", key=f"vat_{idx}", placeholder="e.g. 310122393500003")
                        if st.button("✅ Confirm & Convert to B2B", key=f"approve_{idx}", type="primary"):
                            ok, err = validate_vat_id(vat)
                            if ok:
                                st.session_state.raw_data.at[idx, 'customer_vat_id'] = vat
                                st.session_state.raw_data.at[idx, 'doc_type']        = "Tax Invoice (388)"
                                st.session_state.raw_data.at[idx, 'ai_risk_score']   = 0
                                st.session_state.raw_data.at[idx, 'status']          = "Resolved"
                                st.session_state.review_mode[idx] = False
                                st.session_state.behavior_memory['b2b_conversions'] += 1
                                log_audit("B2B Conversion", row['invoice_id'], "Resolved", "Admin")
                                st.rerun()
                            else:
                                st.error(f"❌ {err}")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CASHIER TERMINAL
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🖥️ Cashier Terminal":
    st.markdown("""
    <div class="c-neutral" style='margin-bottom:20px;'>
        <b style='color:#e2e8f0;'>Cashier point of view.</b>
        <span style='color:#64748b;font-size:13px;'> Cognivis runs silently in the background.
        When the cashier hits Print, Cognivis intercepts BEFORE the invoice reaches ZATCA.
        If there is a violation, the cashier is asked to correct it — no technical jargon.</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""
        <div class="pos-terminal">
            <div style='color:#00ff87;font-size:11px;letter-spacing:.1em;margin-bottom:16px;'>
            ● COGNIVIS SHIELD ACTIVE &nbsp;·&nbsp; TERMINAL 04
            </div>
            <div style='font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:18px;'>New Sale</div>
        </div>
        """, unsafe_allow_html=True)

        items = st.text_area("Items (one per line)", value="Catering Package - Corporate\nExtra Setup Fee\nDelivery Charge", height=100)
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
                <div style='text-align:center;color:#1a2744;'>
                    <div style='font-size:40px;'>🖨️</div>
                    <div style='font-size:13px;margin-top:8px;color:#64748b;'>Invoice preview appears here</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif state == 'blocked':
            rule = st.session_state.cashier_result['blocking_rule']
            amt  = st.session_state.cashier_amount
            st.markdown(f"""
            <div class="pos-blocked">
                <div style='font-size:32px;'>🛑</div>
                <div style='font-size:18px;font-weight:700;color:#ff4d4d;margin:10px 0 6px;'>Cannot Print Invoice</div>
                <div style='font-size:12px;color:#ff8080;margin-bottom:16px;'>Cognivis blocked this before it was sent</div>
                <div style='background:#1a0a0a;border-radius:8px;padding:14px;text-align:left;margin-bottom:16px;'>
                    <div style='font-size:11px;color:#ff4d4d;font-weight:600;margin-bottom:6px;'>WHAT WENT WRONG</div>
                    <div style='font-size:13px;color:#e2e8f0;'>This invoice is for SAR {amt:,.0f}</div>
                    <div style='font-size:12px;color:#94a3b8;margin-top:6px;'>
                        Saudi tax law requires a customer VAT number for any invoice above SAR 1,000.
                        Without it, this business faces a fine of at least SAR 5,000.
                    </div>
                </div>
                <div style='font-size:12px;color:#64748b;'>Rule: {rule.rule_id}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<b style='color:#e2e8f0;'>Ask the customer for their VAT number to continue</b>", unsafe_allow_html=True)
            st.caption("This is a registered business purchase — they will have a 15-digit VAT number.")

            fix_vat = st.text_input("Customer VAT Number:", key="cashier_fix_vat", placeholder="e.g. 310122393500003")
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

            st.markdown("""
            <div class="pos-cleared">
                <div style='font-size:22px;'>✅</div>
                <div style='font-size:15px;font-weight:600;color:#00ff87;margin:6px 0;'>Invoice Approved</div>
                <div style='font-size:12px;color:#00ff87aa;'>ZATCA-Compliant · Sending now</div>
            </div>
            """, unsafe_allow_html=True)

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
# MODULE 4 — AI BRAIN
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "💡 AI Brain":
    if st.session_state.raw_data is None:
        st.markdown("""<div class="c-neutral"><p style='color:#64748b;margin:0;'>Connect a POS source in Integration Hub to activate the AI Brain.</p></div>""", unsafe_allow_html=True)
    else:
        df          = st.session_state.raw_data
        conversions = st.session_state.behavior_memory['b2b_conversions']
        st.markdown(f"""
        <div class="c-brain">
            <b style='color:#00d4ff;'>AI Brain · Behavioral Memory Active</b>
            <p style='color:#94a3b8;font-size:13px;margin:4px 0 0;'>
                Tracked {conversions} B2B conversions this session.
                Every action you take trains the model on your specific business patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

        recs = generate_all_recommendations(df)
        if not recs:
            st.info("Load more invoice data to generate insights.")
        else:
            for rec in recs:
                icon  = "🛡️" if rec.category == "COMPLIANCE" else "📈" if rec.category == "REVENUE" else "⚙️"
                color = "#ff4d4d" if rec.category == "COMPLIANCE" else "#00d4ff" if rec.category == "REVENUE" else "#ffc107"
                st.markdown(f"""
                <div style='background:#0d1b2a;border:1px solid #1a2744;border-radius:12px;padding:18px;margin-bottom:14px;'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:10px;'>
                        <span style='font-weight:600;color:#e2e8f0;'>{icon} {rec.title}</span>
                        <span style='font-size:11px;color:#64748b;background:#091221;padding:3px 8px;border-radius:20px;'>
                            {int(rec.confidence*100)}% confidence
                        </span>
                    </div>
                    <div class='wa-bubble' style='border-left-color:{color};'>{rec.body}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("<b style='color:#e2e8f0;'>Revenue — Last 30 Days</b>", unsafe_allow_html=True)
        trend = generate_revenue_trend(30)
        st.line_chart(trend.set_index('date')['revenue'], color="#00d4ff")

        st.markdown("<b style='color:#e2e8f0;'>Revenue by Category</b>", unsafe_allow_html=True)
        cat_rev = df[df['amount_sar'] > 0].groupby('category')['amount_sar'].sum().reset_index()
        cat_rev.columns = ['Category', 'Revenue']
        st.bar_chart(cat_rev.set_index('Category'), color="#00ff87")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 5 — WHAT-IF ENGINE
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🔮 What-If Engine":
    if st.session_state.raw_data is None:
        st.markdown("""<div class="c-neutral"><p style='color:#64748b;margin:0;'>Connect POS data in Integration Hub to run simulations.</p></div>""", unsafe_allow_html=True)
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
        st.session_state.raw_data  = audit_dataframe(raw)
        st.session_state.pos_source = "Demo · Al Baik Group"

    df    = st.session_state.raw_data
    s     = get_compliance_summary(df)
    trust = calculate_trust_score(df)
    recs  = generate_all_recommendations(df)

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
            <p style='color:#94a3b8;margin:0;font-size:14px;'>Cognivis connects to Foodics, Salla, or any POS in under 60 seconds. Data is normalized and risk-scored locally — nothing leaves your environment.</p>
        </div>""", unsafe_allow_html=True)
        st.success(f"✅ {st.session_state.pos_source} — {s['total']} invoices loaded and scored.")

    elif step == 1:
        st.markdown(f"""<div class="c-danger">
            <h4 style='color:#ff4d4d;margin:0 0 8px;'>Step 2 — Violations Detected Instantly</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>AI rule engine scanned {s['total']} invoices. Results in milliseconds.</p>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Violations Found", s['violations'])
        c2.metric("Violation Rate", f"{s['violation_rate']}%")
        c3.metric("Fine Exposure", f"SAR {s['capital_at_risk']:,}")

    elif step == 2:
        st.markdown("""<div class="c-shield">
            <h4 style='color:#00ff87;margin:0 0 8px;'>Step 3 — Real-Time Interception</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>Cognivis fires BEFORE the invoice is created. The cashier sees a clear message — no technical jargon.</p>
        </div>""", unsafe_allow_html=True)
        result = check_transaction(1850.00, "")
        rule   = result['blocking_rule']
        st.markdown(f"""
        <div class="pos-blocked" style='max-width:420px;'>
            <div style='font-size:28px;'>🛑</div>
            <div style='font-size:16px;font-weight:700;color:#ff4d4d;margin:8px 0 4px;'>Cannot Print Invoice</div>
            <div style='font-size:12px;color:#94a3b8;margin-bottom:10px;'>
                This invoice is for SAR 1,850 with no customer VAT number.<br>
                Saudi tax law requires one for invoices over SAR 1,000.
            </div>
            <div style='font-size:11px;color:#ff4d4d;'>{rule.rule_id} · Fine risk: SAR 5,000</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("The cashier is prompted to ask for the VAT number. No fine. No paperwork. Done.")

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
            <div style='font-size:11px;color:#94a3b8;margin-top:10px;'>This score will connect to Saudi lending partners — giving compliant SMEs faster credit access.</div>
        </div>""", unsafe_allow_html=True)

    elif step == 5:
        st.markdown("""<div class="c-brain">
            <h4 style='color:#00d4ff;margin:0 0 8px;'>Step 6 — The AI Brain</h4>
            <p style='color:#94a3b8;margin:0;font-size:14px;'>Beyond compliance — every transaction becomes a growth insight.</p>
        </div>""", unsafe_allow_html=True)
        if recs:
            rec = recs[0]
            st.markdown(f"""<div class="wa-bubble">🧠 <b style='color:#00d4ff;'>Cognivis AI</b><br><br>{rec.body}</div>""", unsafe_allow_html=True)
        st.success("🎉 This is Cognivis OS — compliance firewall + business intelligence, in one system.")

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
