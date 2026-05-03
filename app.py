"""
Cognivis OS V3 — Main Application
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
from datetime import datetime

# ── Services ──────────────────────────────────────────────────────────────────
from services.compliance_engine  import audit_dataframe, check_transaction, get_compliance_summary, validate_transaction
from services.decision_engine    import generate_all_insights
from services.trust_score        import calculate_trust_score
from services.simulation_engine  import simulate_price_change, simulate_vat_enforcement
from data.mock_data_generator    import generate_realistic_invoices
from utils.validators            import normalize_dataframe, validate_vat_id

# ── Core infrastructure (new) ─────────────────────────────────────────────────
from core.identity        import init_identity_state, is_authenticated, render_login_screen, get_current_business, logout
from core.persistence     import save_audit_entry, load_audit_log, get_persistence_mode
from core.failure_handler import safe_grok_call, safe_parse_csv, safe_validate_vat, show_error, show_warning, show_success


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cognivis OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp { background-color:#020617 !important; color:#e2e8f0; font-family:'Inter',sans-serif; }
section[data-testid="stSidebar"] { background-color:#0a1628 !important; border-right:1px solid #1a2744; }
.stButton>button { font-family:'Inter',sans-serif !important; }

.c-shield  { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #00ff8733; border-left:3px solid #00ff87; margin-bottom:14px; }
.c-brain   { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #00d4ff33; border-left:3px solid #00d4ff; margin-bottom:14px; }
.c-danger  { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #ff4d4d55; border-left:3px solid #ff4d4d; margin-bottom:14px; }
.c-neutral { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #1a2744; margin-bottom:14px; }
.c-warning { background:#0d1b2a; border-radius:12px; padding:20px; border:1px solid #ffc10755; border-left:3px solid #ffc107; margin-bottom:14px; }
.c-trust   { background:linear-gradient(135deg,#0d1b2a,#091221); border:1px solid #00ff8755; border-radius:14px; padding:24px; text-align:center; }

.metric-tile { background:#0d1b2a; border:1px solid #1a2744; border-radius:10px; padding:18px 16px; text-align:center; }
.metric-tile .label { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.06em; margin-bottom:6px; }
.metric-tile .value { font-size:26px; font-weight:700; line-height:1; }
.metric-tile .sub   { font-size:11px; color:#64748b; margin-top:4px; }

.pos-terminal { background:#070f1a; border:1px solid #1a2744; border-top:3px solid #00ff87; border-radius:14px; padding:24px; }
.pos-blocked  { background:#1a0a0a; border:2px solid #ff4d4d; border-radius:12px; padding:20px; text-align:center; }
.pos-cleared  { background:#041a0f; border:2px solid #00ff87; border-radius:12px; padding:20px; text-align:center; }
.pos-receipt  { background:#f8f9fa; color:#1a1a2e; border-radius:8px; padding:20px; font-family:'Courier New',monospace; font-size:12px; line-height:1.8; border:1px dashed #ccc; }

.ai-response { background:#070f1a; color:#e2e8f0; padding:20px 22px; border-radius:12px; margin-bottom:12px; border:1px solid #00d4ff33; font-size:14px; line-height:1.8; }
.trust-factor { background:#091221; border:1px solid #1a2744; border-radius:8px; padding:12px 16px; margin-bottom:8px; }

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


# ══════════════════════════════════════════════════════════════════════════════
# IDENTITY CHECK — show login if not authenticated
# ══════════════════════════════════════════════════════════════════════════════
init_identity_state()

if not is_authenticated():
    render_login_screen()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        'raw_data':          None,
        'pos_source':        None,
        'review_mode':       {},
        'behavior_memory':   {'b2b_conversions': 0},
        'cashier_state':     'idle',
        'cashier_result':    None,
        'cashier_amount':    0,
        'cashier_vat':       '',
        'cashier_items':     '',
        'ai_messages':       [],
        'last_uploaded_csv': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

business = get_current_business()
persist_mode = get_persistence_mode()


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    return df[[c for c in cols if c in df.columns]]


def log(action: str, invoice_id: str, status: str, user: str = "System"):
    """Saves to both session state (UI) and persistence layer (permanent)."""
    entry = {
        "Timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Invoice ID":   invoice_id,
        "Action":       action,
        "Status":       status,
        "Authorized By": user,
    }
    if 'audit_ledger' not in st.session_state:
        st.session_state.audit_ledger = []
    st.session_state.audit_ledger.append(entry)
    save_audit_entry(business, invoice_id, action, status, user)


def build_grok_prompt(df: pd.DataFrame, insights: dict) -> str:
    rev  = insights.get("revenue", {})
    risk = insights.get("risk", {})
    beh  = insights.get("behavior", {})
    return f"""You are Cognivis AI — a business intelligence advisor for Saudi SME owners.
You have direct access to this business's real transaction data.
Speak like a smart, direct advisor — not a chatbot. Never be generic.
Every answer must reference the actual numbers below.
Use plain language. No bullet lists unless essential. Keep answers under 150 words.
Always end with one concrete next action the owner can take today.

BUSINESS: {business}
- Invoices: {rev.get('invoice_count', 0)}
- Revenue: SAR {rev.get('total_revenue', 0):,.0f}
- Average order: SAR {rev.get('avg_order_value', 0):,.0f}
- Top category: {rev.get('top_category', 'N/A')}
- Weakest category: {rev.get('bottom_category', 'N/A')}
- ZATCA violations: {risk.get('violation_count', 0)} ({risk.get('violation_rate_pct', 0)}%)
- Fine exposure: SAR {risk.get('capital_at_risk', 0):,.0f}
- Patterns: {'; '.join(risk.get('patterns', ['No issues']))}
- Resolution pattern: {beh.get('dominant_pattern', 'None yet')}

Answer strictly from this data. Do not invent numbers."""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:12px 0 8px;'>
        <span style='font-size:18px; font-weight:700; color:#e2e8f0;'>🧠 Cognivis OS</span><br>
        <span style='font-size:10px; color:#00ff87; letter-spacing:.1em;'>INTELLIGENCE LAYER</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown(f"""
    <div class="saas-header">
        <b style='color:#e2e8f0;'>🏢</b> {business}<br>
        <b style='color:#00ff87;'>●</b> Active<br>
        <b style='color:#64748b;'>💾</b> {persist_mode.upper()} · {datetime.now().strftime("%H:%M")}
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio("Navigation", [
        "📥 Data Hub",
        "🛡️ ZATCA Shield",
        "🖥️ Cashier Terminal",
        "💡 AI Brain",
        "🔮 What-If Engine",
        "📓 Audit Ledger",
    ], label_visibility="collapsed")

    st.divider()

    if st.session_state.raw_data is not None:
        st.markdown(f"<span class='badge-green'>✓ {st.session_state.pos_source}</span>", unsafe_allow_html=True)
        if st.button("Clear Data", use_container_width=True):
            st.session_state.raw_data        = None
            st.session_state.pos_source      = None
            st.session_state.review_mode     = {}
            st.session_state.ai_messages     = []
            st.session_state.last_uploaded_csv = None
            st.rerun()
    else:
        st.markdown("<span class='badge-yellow'>⚠ No data loaded</span>", unsafe_allow_html=True)

    st.divider()
    if st.button("🔓 Logout", use_container_width=True):
        logout()
        st.rerun()

    st.caption(f"v3.0 · {business}")


# ── Page header ───────────────────────────────────────────────────────────────
page_meta = {
    "📥 Data Hub":          ("📥", "Data Hub",           "Load your invoice data to get started"),
    "🛡️ ZATCA Shield":      ("🛡️", "ZATCA Shield",       "Compliance firewall — detect and fix violations"),
    "🖥️ Cashier Terminal":  ("🖥️", "Cashier Terminal",   "Live POS — cashier point of view"),
    "💡 AI Brain":          ("💡", "AI Brain",            "Operational intelligence powered by Grok AI"),
    "🔮 What-If Engine":    ("🔮", "What-If Engine",      "Simulate decisions before making them"),
    "📓 Audit Ledger":      ("📓", "Audit Ledger",        "Permanent compliance log"),
}
icon, title, subtitle = page_meta.get(menu, ("", menu, ""))
st.markdown(f"""
<div style='padding-bottom:14px; border-bottom:1px solid #1a2744; margin-bottom:20px;'>
    <span style='font-size:20px; font-weight:700; color:#e2e8f0;'>{icon} {title}</span><br>
    <span style='font-size:13px; color:#64748b;'>{subtitle}</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — DATA HUB (clean — no POS logos, just demo + upload)
# ══════════════════════════════════════════════════════════════════════════════
if menu == "📥 Data Hub":

    col_demo, col_upload = st.columns(2, gap="large")

    with col_demo:
        st.markdown("""
        <div class="c-neutral">
            <div style='font-size:11px;color:#ffc107;font-weight:600;letter-spacing:.08em;margin-bottom:6px;'>QUICK START</div>
            <div style='font-size:17px;font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Load Demo Data</div>
            <div style='font-size:13px;color:#64748b;line-height:1.6;'>
                50 realistic invoices with ZATCA violations pre-loaded.
                Use this to explore all features before uploading your own data.
            </div>
        </div>
        """, unsafe_allow_html=True)

        n_invoices = st.slider("Number of invoices", 20, 100, 50, key="demo_n")
        v_rate     = st.slider("Violation rate (%)", 5, 40, 20, key="demo_v")

        if st.button("Load Demo Dataset", use_container_width=True, type="primary"):
            raw = generate_realistic_invoices(n=n_invoices, violation_rate=v_rate/100)
            st.session_state.raw_data   = audit_dataframe(normalize_dataframe(raw))
            st.session_state.pos_source = "Demo Dataset"
            st.session_state.ai_messages = []
            log("Demo Load", "System", f"{n_invoices} invoices loaded")
            st.rerun()

    with col_upload:
        st.markdown("""
        <div class="c-shield">
            <div style='font-size:11px;color:#00ff87;font-weight:600;letter-spacing:.08em;margin-bottom:6px;'>YOUR DATA</div>
            <div style='font-size:17px;font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Upload POS Export</div>
            <div style='font-size:13px;color:#64748b;line-height:1.6;'>
                Upload a CSV export from any POS system — Foodics, Salla, or custom.
                Column names are auto-detected.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption("Accepted columns (any naming): Invoice ID, Amount, VAT number, Category, Date")
        uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="csv_uploader", label_visibility="collapsed")

        if uploaded is not None and st.session_state.last_uploaded_csv != uploaded.name:
            df_raw, err = safe_parse_csv(uploaded)
            if err:
                show_error(err, "Export a fresh CSV from your POS system and try again.")
            else:
                try:
                    normalized = normalize_dataframe(df_raw)
                    audited    = audit_dataframe(normalized)
                    st.session_state.raw_data           = audited
                    st.session_state.pos_source         = uploaded.name
                    st.session_state.last_uploaded_csv  = uploaded.name
                    st.session_state.ai_messages        = []
                    log("CSV Upload", "System", f"{len(audited)} rows", "Admin")
                    show_success(f"✓ Loaded {len(audited)} invoices from {uploaded.name}")
                    st.rerun()
                except Exception as e:
                    show_error(f"Error processing file: {str(e)}", "Ensure the file has amount and invoice columns.")

    # ── Data preview
    if st.session_state.raw_data is not None:
        df = st.session_state.raw_data
        s  = get_compliance_summary(df)
        st.divider()
        st.markdown(f"""
        <div style='font-size:13px;color:#64748b;margin-bottom:14px;'>
            <b style='color:#e2e8f0;'>{s['total']} invoices</b> loaded from
            <b style='color:#00ff87;'>{st.session_state.pos_source}</b>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Invoices", s['total'])
        m2.metric("Violations", s['violations'])
        m3.metric("Violation Rate", f"{s['violation_rate']}%")
        m4.metric("Fine Exposure", f"SAR {s['capital_at_risk']:,}")

        with st.expander("View invoice table"):
            cols = ['invoice_id','date','category','amount_sar','customer_vat_id','doc_type','ai_risk_score','violations','status']
            st.dataframe(safe_df(df, cols), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — ZATCA SHIELD
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🛡️ ZATCA Shield":
    if st.session_state.raw_data is None:
        show_warning("Load data in the Data Hub first.")
    else:
        df    = st.session_state.raw_data
        trust = calculate_trust_score(df)
        s     = get_compliance_summary(df)

        # KPI row
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
                <div class="sub">SAR 5,000 per violation</div>
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
            st.markdown(f"""<div class="c-trust" style='padding:14px;'>
                <div style='font-size:10px;color:#64748b;letter-spacing:.08em;'>TRUST SCORE™</div>
                <div style='font-size:28px;font-weight:700;color:{sc};'>{trust.score}/100</div>
                <div style='font-size:11px;color:#00ff87;'>{trust.label}</div>
            </div>""", unsafe_allow_html=True)

        # Trust score breakdown
        with st.expander("📊 Why is my Trust Score this number?", expanded=(trust.score < 75)):
            factors = [
                {
                    "label":  "Violation Rate",
                    "detail": f"{trust.violation_rate}% of invoices have ZATCA violations",
                    "status": "Good" if trust.violation_rate < 5 else "Needs Work" if trust.violation_rate < 20 else "Critical",
                    "color":  "#00ff87" if trust.violation_rate < 5 else "#ffc107" if trust.violation_rate < 20 else "#ff4d4d",
                },
                {
                    "label":  "Resolution Activity",
                    "detail": f"{resolved} violations resolved so far",
                    "status": "Active" if resolved > 0 else "Not Started",
                    "color":  "#00ff87" if resolved > 0 else "#ff4d4d",
                },
                {
                    "label":  "Invoice Consistency",
                    "detail": f"{trust.consistency_score}% of invoices are fully clean",
                    "status": "Good" if trust.consistency_score > 80 else "Fair" if trust.consistency_score > 50 else "Poor",
                    "color":  "#00ff87" if trust.consistency_score > 80 else "#ffc107" if trust.consistency_score > 50 else "#ff4d4d",
                },
            ]
            for f in factors:
                st.markdown(f"""
                <div class="trust-factor" style='display:flex;justify-content:space-between;align-items:center;'>
                    <div>
                        <div style='font-size:13px;font-weight:600;color:#e2e8f0;'>{f['label']}</div>
                        <div style='font-size:12px;color:#64748b;margin-top:2px;'>{f['detail']}</div>
                    </div>
                    <span style='color:{f["color"]};font-weight:600;font-size:12px;white-space:nowrap;margin-left:12px;'>{f['status']}</span>
                </div>
                """, unsafe_allow_html=True)

            if trust.score < 80 and s['violations'] > 0:
                st.markdown(f"""
                <div class="c-brain" style='margin-top:12px;'>
                    <b style='color:#00d4ff;'>How to improve</b>
                    <p style='color:#94a3b8;font-size:13px;margin:6px 0 0;'>
                    Resolve the {s['violations']} violations below. Each one resolved raises your score.
                    Your score directly affects your future credit eligibility.</p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Violation queue
        violations = df[df['ai_risk_score'] >= 45]
        active = violations[violations['status'] != 'Resolved'] if 'status' in violations.columns else violations

        if active.empty:
            st.markdown("""<div class="c-shield">
                <b style='color:#00ff87;'>✓ Fully Compliant</b>
                <p style='color:#94a3b8;margin:4px 0 0;'>No active violations. Your business is ZATCA-clean.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"<b style='color:#ff4d4d;'>🚨 {len(active)} violations need attention</b><br>", unsafe_allow_html=True)

            for idx, row in active.iterrows():
                v = validate_transaction(row.to_dict())
                rules    = v.get("violations", [])
                messages = v.get("messages", [])

                with st.expander(f"📄 {row['invoice_id']}  ·  SAR {row['amount_sar']:,.2f}  ·  {', '.join(rules) if rules else 'Unknown'}"):
                    for rule_id, msg in zip(rules, messages):
                        st.markdown(f"""
                        <div class="c-danger" style='padding:14px;margin-bottom:8px;'>
                            <div style='display:flex;gap:8px;align-items:center;margin-bottom:6px;'>
                                <span class='badge-red'>{rule_id}</span>
                                <span style='color:#64748b;font-size:11px;'>{row.get('category','')} · {row.get('date','')}</span>
                            </div>
                            <div style='font-size:13px;color:#e2e8f0;line-height:1.6;'>{msg}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    if not st.session_state.review_mode.get(idx, False):
                        if st.button("🔧 Fix This", key=f"open_{idx}", type="primary"):
                            st.session_state.review_mode[idx] = True
                            st.rerun()
                    else:
                        if "BR-KSA-14" in rules or "INVOICE_TYPE_CHECK" in rules:
                            st.markdown("""<div class="c-brain">
                                <b style='color:#00d4ff;'>How to fix</b>
                                <p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>
                                Add the customer's 15-digit VAT number. This converts the invoice
                                to a valid B2B Tax Invoice and removes the violation.</p>
                            </div>""", unsafe_allow_html=True)
                            vat_input = st.text_input("Customer VAT ID:", key=f"vat_{idx}", placeholder="310122393500003")
                            if st.button("✅ Save & Resolve", key=f"approve_{idx}", type="primary"):
                                ok, err = safe_validate_vat(vat_input)
                                if ok:
                                    st.session_state.raw_data.at[idx, 'customer_vat_id'] = vat_input
                                    st.session_state.raw_data.at[idx, 'doc_type']        = "Tax Invoice (388)"
                                    st.session_state.raw_data.at[idx, 'ai_risk_score']   = 0
                                    st.session_state.raw_data.at[idx, 'status']          = "Resolved"
                                    st.session_state.raw_data.at[idx, 'violations']      = "None"
                                    st.session_state.review_mode[idx] = False
                                    st.session_state.behavior_memory['b2b_conversions'] += 1
                                    log("B2B Conversion", row['invoice_id'], "Resolved", "Admin")
                                    st.rerun()
                                else:
                                    show_error(err)

                        elif "VAT_FORMAT" in rules:
                            st.markdown("""<div class="c-warning">
                                <b style='color:#ffc107;'>How to fix</b>
                                <p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>
                                The VAT ID format is wrong. Must be exactly 15 digits starting with 3.</p>
                            </div>""", unsafe_allow_html=True)
                            vat_input = st.text_input("Correct VAT ID:", key=f"vat_{idx}", placeholder="310122393500003")
                            if st.button("✅ Update & Resolve", key=f"approve_{idx}", type="primary"):
                                ok, err = safe_validate_vat(vat_input)
                                if ok:
                                    st.session_state.raw_data.at[idx, 'customer_vat_id'] = vat_input
                                    st.session_state.raw_data.at[idx, 'ai_risk_score']   = 0
                                    st.session_state.raw_data.at[idx, 'status']          = "Resolved"
                                    st.session_state.raw_data.at[idx, 'violations']      = "None"
                                    st.session_state.review_mode[idx] = False
                                    log("VAT Format Fix", row['invoice_id'], "Resolved", "Admin")
                                    st.rerun()
                                else:
                                    show_error(err)

                        elif "MISSING_FIELDS" in rules:
                            st.markdown("""<div class="c-danger">
                                <b style='color:#ff4d4d;'>How to fix</b>
                                <p style='color:#94a3b8;margin:6px 0 0;font-size:13px;'>
                                This invoice is missing required fields. Fix in your POS system directly,
                                then re-export and reload.</p>
                            </div>""", unsafe_allow_html=True)
                            if st.button("✅ Mark as Reviewed", key=f"approve_{idx}"):
                                st.session_state.raw_data.at[idx, 'status'] = "Resolved"
                                st.session_state.review_mode[idx] = False
                                log("Manual Review", row['invoice_id'], "Acknowledged", "Admin")
                                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — CASHIER TERMINAL
# ══════════════════════════════════════════════════════════════════════════════
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

        items  = st.text_area("Items", value="Catering Package\nSetup Fee\nDelivery", height=90)
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
                <div style='font-size:11px;color:#64748b;'>{rule} · Fine risk: SAR 5,000+</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<b style='color:#e2e8f0;'>Enter customer VAT number to continue</b>", unsafe_allow_html=True)
            st.caption("Every registered Saudi business has a 15-digit VAT number starting with 3.")

            fix_vat = st.text_input("Customer VAT Number:", key="cashier_fix_vat", placeholder="310122393500003")
            if st.button("✅ Submit & Print Invoice", type="primary", use_container_width=True):
                ok, err = safe_validate_vat(fix_vat)
                if ok:
                    st.session_state.cashier_state = 'cleared'
                    st.session_state.cashier_vat   = fix_vat
                    log("Cashier VAT Fix", f"POS-{int(st.session_state.cashier_amount)}", "Resolved", "Cashier")
                    st.session_state.behavior_memory['b2b_conversions'] += 1
                    st.rerun()
                else:
                    show_error(err, "Check the number directly with the customer.")

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
            lines      = "\n".join([f"  {item:<26} {per_item:>8.2f}" for item in items_list])

            st.markdown("""<div class="pos-cleared">
                <div style='font-size:22px;'>✅</div>
                <div style='font-size:15px;font-weight:600;color:#00ff87;margin:6px 0;'>Invoice Approved</div>
                <div style='font-size:12px;color:#00ff87aa;'>ZATCA-Compliant · Sending now</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <br>
            <div class="pos-receipt">
<pre style='font-family:Courier New,monospace;font-size:12px;color:#1a1a2e;margin:0;line-height:1.8;'>
================================
      COGNIVIS TAX INVOICE
================================
Invoice : {inv_id}
Date    : {now}
--------------------------------
{lines}
--------------------------------
Subtotal:      SAR {amt:>10.2f}
VAT (15%):     SAR {vat_amt:>10.2f}
================================
TOTAL:         SAR {total:>10.2f}
================================
VAT: {vat}
[✓] ZATCA COMPLIANT
================================</pre>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🔄 New Transaction", use_container_width=True):
                st.session_state.cashier_state = 'idle'
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — AI BRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "💡 AI Brain":
    if st.session_state.raw_data is None:
        show_warning("Load data in the Data Hub first.")
    else:
        df       = st.session_state.raw_data
        memory   = st.session_state.behavior_memory
        insights = generate_all_insights(df, memory)
        rev      = insights['revenue']
        risk     = insights['risk']

        st.markdown(f"""
        <div class="c-brain">
            <b style='color:#00d4ff;'>AI Brain · Powered by Grok · {business}</b>
            <p style='color:#94a3b8;font-size:13px;margin:4px 0 0;'>
                {rev.get('invoice_count',0)} invoices · SAR {rev.get('total_revenue',0):,.0f} revenue ·
                {risk.get('violation_count',0)} compliance issues
            </p>
        </div>
        """, unsafe_allow_html=True)

        system_prompt = build_grok_prompt(df, insights)

        # Auto insights
        actions = insights.get("actions", [])
        if actions:
            st.markdown("<b style='color:#e2e8f0;'>What Cognivis sees in your business right now</b>", unsafe_allow_html=True)
            for act in actions:
                icon  = "🛡️" if act['category'] == "COMPLIANCE" else "📈" if act['category'] == "REVENUE" else "⚙️"
                color = "#ff4d4d" if act['category'] == "COMPLIANCE" else "#00d4ff" if act['category'] == "REVENUE" else "#ffc107"
                pl    = {"HIGH": "Urgent", "MEDIUM": "Recommended", "LOW": "Note"}.get(act['priority'], "")
                st.markdown(f"""
                <div style='background:#0d1b2a;border:1px solid #1a2744;border-radius:12px;padding:18px;margin-bottom:14px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
                        <span style='font-weight:600;color:#e2e8f0;'>{icon} {act['title']}</span>
                        <span style='font-size:11px;color:{color};background:{color}22;padding:3px 8px;border-radius:20px;'>{pl}</span>
                    </div>
                    <div style='font-size:14px;color:#94a3b8;line-height:1.7;'>{act['description']}</div>
                    <div style='font-size:12px;color:#64748b;margin-top:8px;'>💡 {act['system_action']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        # Grok chat
        st.markdown("<b style='color:#00d4ff;'>Ask Cognivis AI about your business</b>", unsafe_allow_html=True)
        st.caption("Answers are based on your actual data — not generic advice.")

        for msg in st.session_state.ai_messages:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div style='background:#091221;border:1px solid #1a2744;border-radius:10px;
                            padding:12px 16px;margin-bottom:8px;font-size:14px;'>
                    <b style='color:#e2e8f0;'>You:</b>
                    <span style='color:#94a3b8;margin-left:8px;'>{msg['content']}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='ai-response'>
                    <b style='color:#00d4ff;'>🧠 Cognivis AI</b><br><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)

        # Quick questions
        quick = [
            "What is my biggest risk right now?",
            "Which category should I focus on?",
            "How do I improve my trust score?",
            "Where am I losing the most money?",
            "What should I do this week?",
            "How do I reduce my violation rate?",
        ]
        q_cols = st.columns(3)
        for i, q in enumerate(quick):
            with q_cols[i % 3]:
                if st.button(q, key=f"q_{i}", use_container_width=True):
                    st.session_state.ai_messages.append({"role": "user", "content": q})
                    with st.spinner("Thinking..."):
                        ans = safe_grok_call(system_prompt, q)
                    st.session_state.ai_messages.append({"role": "assistant", "content": ans})
                    st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        user_q = st.text_input("Or ask your own question:", placeholder="Why are my margins low? What should I order less of?", key="ai_q")
        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("Ask Cognivis AI", type="primary", use_container_width=True):
                if user_q.strip():
                    st.session_state.ai_messages.append({"role": "user", "content": user_q})
                    with st.spinner("Thinking..."):
                        ans = safe_grok_call(system_prompt, user_q)
                    st.session_state.ai_messages.append({"role": "assistant", "content": ans})
                    st.rerun()
        with c2:
            if st.button("Clear", use_container_width=True):
                st.session_state.ai_messages = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 5 — WHAT-IF ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "🔮 What-If Engine":
    if st.session_state.raw_data is None:
        show_warning("Load data in the Data Hub first.")
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
                show_warning(f"A {pct}% increase pushes {r['new_risk_count']} transactions over SAR 1,000. Fine exposure: SAR {r['fine_exposure_delta']:,}. Enable VAT capture on the Cashier Terminal first.")
            else:
                show_success("No new compliance risks from this price change.")

        with tab2:
            cap = st.slider("VAT capture rate on invoices ≥ SAR 1,000 (%)", 0, 100, 70)
            r2  = simulate_vat_enforcement(df, cap / 100)
            c1, c2, c3 = st.columns(3)
            c1.metric("At Risk", r2['total_at_risk'])
            c2.metric("Resolved", r2['resolved_by_capture'], f"SAR {r2['fine_saved']:,} saved")
            c3.metric("Remaining", f"SAR {r2['fine_remaining']:,}",
                      f"{r2['remaining_violations']} unresolved", delta_color="inverse")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 6 — AUDIT LEDGER (permanent — loads from persistence layer)
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "📓 Audit Ledger":
    st.markdown(f"""
    <div class="c-neutral">
        <b style='color:#e2e8f0;'>Permanent compliance log for {business}</b>
        <p style='color:#64748b;font-size:13px;margin:4px 0 0;'>
        Storage: <b style='color:#00ff87;'>{persist_mode.upper()}</b> —
        {"data survives app restarts" if persist_mode == "supabase" else "data saved to local file · add Supabase to make it permanent across deployments"}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load from persistence layer (not just session)
    persistent_log = load_audit_log(business)
    session_log    = st.session_state.get('audit_ledger', [])

    # Merge: persistent log is source of truth, session log shows current-session entries
    if persistent_log:
        ledger = pd.DataFrame(persistent_log)
    elif session_log:
        ledger = pd.DataFrame(session_log)
    else:
        ledger = None

    if ledger is None or len(ledger) == 0:
        show_warning("No audit events yet. Actions appear here as you use the system.")
    else:
        st.markdown(f"<span class='badge-green'>{len(ledger)} entries</span>", unsafe_allow_html=True)
        st.dataframe(ledger, use_container_width=True, hide_index=True)
