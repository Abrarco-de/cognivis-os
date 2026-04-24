"""
Cognivis OS V2 — Main Application
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# ── Service Imports ─────────────────────────────────────────────────────────────
from services.compliance_engine import audit_dataframe, check_transaction, get_compliance_summary
from services.decision_engine import generate_all_recommendations
from services.trust_score import calculate_trust_score
from services.simulation_engine import simulate_price_change, simulate_vat_enforcement
from data.mock_data_generator import generate_realistic_invoices, generate_revenue_trend, get_foodics_mock, get_salla_mock
from utils.validators import normalize_dataframe, validate_vat_id, log_audit


# ── Page Config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognivis OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Styles ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; }
.cognivis-card {
    background: #0f172a; border-radius: 12px; padding: 20px;
    border: 1px solid rgba(255,255,255,0.07); margin-bottom: 16px;
}
.shield-card { border-left: 4px solid #22c55e; background: #0f172a; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
.brain-card  { border-left: 4px solid #3b82f6; background: #0f172a; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
.risk-card   { border-left: 4px solid #ef4444; background: #0f172a; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
.trust-card  { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #eab308; padding: 20px; border-radius: 12px; text-align: center; }
.metric-card { background: #0f172a; border: 1px solid rgba(255,255,255,0.07); padding: 16px; border-radius: 10px; text-align: center; }
.wa-bubble   { background: #1e293b; color: #f8fafc; padding: 14px 18px; border-radius: 10px; margin-bottom: 10px; border-left: 3px solid #3b82f6; font-size: 14px; line-height: 1.6; }
.badge-high  { background: rgba(239,68,68,.2); color: #ef4444; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.badge-warn  { background: rgba(245,158,11,.2); color: #f59e0b; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.badge-ok    { background: rgba(34,197,94,.2); color: #22c55e; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.saas-header { background: #0f172a; padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 18px; font-size: 12px; color: #94a3b8; }
.step-block  { background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────────
def init_state():
    defaults = {
        'raw_data': None,
        'pos_source': None,
        'audit_ledger': [],
        'review_mode': {},
        'demo_mode': True,
        'behavior_memory': {'b2b_conversions': 0, 'auto_pref': False},
        'investor_step': 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Mode toggle
    mode = st.toggle("🎭 Demo Mode", value=st.session_state.demo_mode)
    st.session_state.demo_mode = mode
    st.caption("Demo mode: full investor flow with realistic data")
    st.divider()

    # Org header
    org_name = "Al Baik Restaurant Group" if mode else "Live Business"
    st.markdown(f"""
    <div class="saas-header">
        <b>🏢 Org:</b> {org_name}<br>
        <b>🟢 Status:</b> {"12 Simulated POS Terminals" if mode else "Live POS"}<br>
        <b>👤 User:</b> Admin<br>
        <b>⏱️ Last Sync:</b> {datetime.now().strftime("%H:%M:%S")}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    menu = st.radio("", [
        "📥 Integration Hub",
        "🛡️ ZATCA Shield",
        "💡 AI Brain",
        "🔮 What-If Engine",
        "📓 Audit Ledger",
        "🎯 Investor Story",
        "🏦 Financial Identity (Locked)"
    ], label_visibility="collapsed")

    st.divider()
    if st.session_state.raw_data is not None:
        st.success(f"✅ Connected: {st.session_state.pos_source}")
        if st.button("⏹ Disconnect POS"):
            for key in ['raw_data', 'pos_source', 'audit_ledger', 'review_mode']:
                st.session_state[key] = None if key in ['raw_data', 'pos_source'] else []
            st.session_state.review_mode = {}
            st.rerun()
    else:
        st.warning("No POS connected")

    st.caption("Cognivis OS v2.0 — Build April 2026")


# ── Header ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:8px;'>
    <span style='font-size:22px; font-weight:700; color:#f8fafc;'>🧠 Cognivis OS</span>
    <span style='font-size:13px; color:#64748b; margin-left:12px;'>The Intelligence Layer for Saudi SME Growth</span>
</div>
""", unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE: LOCKED
# ══════════════════════════════════════════════════════════════════════════════════
if "Locked" in menu:
    st.title("🏦 Cognivis Financial Identity")
    st.markdown("""
    <div class="brain-card">
        <h4 style='color:#3b82f6; margin-top:0;'>Coming in Enterprise Tier</h4>
        <p style='color:#94a3b8;'>
            Every compliance action you take in Cognivis builds a verifiable financial identity.
            This module connects your Trust Score directly to Saudi lending partners and BNPL providers —
            giving compliant SMEs faster access to working capital.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.info("🔒 Available in the Enterprise roadmap. Contact us to join the waitlist.")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 1: INTEGRATION HUB
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "📥 Integration Hub":
    st.title("Unified POS Integration Hub")
    st.write("Connect your point of sale provider. Data is mapped and scored locally.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class="cognivis-card"><h4 style='color:#22c55e; margin:0;'>🟢 Foodics</h4>
        <p style='color:#94a3b8; font-size:13px;'>Saudi F&B POS leader</p></div>""", unsafe_allow_html=True)
        if st.button("Connect Foodics API", use_container_width=True):
            raw = normalize_dataframe(get_foodics_mock())
            df = audit_dataframe(raw)
            st.session_state.raw_data = df
            st.session_state.pos_source = "Foodics"
            log_audit("API Handshake", "System", "Connected", "Foodics API")
            st.success("Connected to Foodics sandbox!")
            st.rerun()

    with col2:
        st.markdown("""<div class="cognivis-card"><h4 style='color:#3b82f6; margin:0;'>🔵 Salla</h4>
        <p style='color:#94a3b8; font-size:13px;'>Saudi e-commerce platform</p></div>""", unsafe_allow_html=True)
        if st.button("Connect Salla API", use_container_width=True):
            raw = normalize_dataframe(get_salla_mock())
            df = audit_dataframe(raw)
            st.session_state.raw_data = df
            st.session_state.pos_source = "Salla"
            log_audit("API Handshake", "System", "Connected", "Salla API")
            st.success("Connected to Salla sandbox!")
            st.rerun()

    with col3:
        st.markdown("""<div class="cognivis-card"><h4 style='color:#eab308; margin:0;'>📊 Demo Dataset</h4>
        <p style='color:#94a3b8; font-size:13px;'>50 realistic SME invoices</p></div>""", unsafe_allow_html=True)
        if st.button("Load Demo Dataset", use_container_width=True):
            raw = generate_realistic_invoices(n=50, violation_rate=0.20)
            df = audit_dataframe(raw)
            st.session_state.raw_data = df
            st.session_state.pos_source = "Demo Dataset"
            log_audit("Demo Load", "System", "50 invoices loaded")
            st.success("Demo data loaded — 50 invoices with realistic violations.")
            st.rerun()

    st.divider()

    # CSV Upload
    st.subheader("Upload Your Own Data")
    uploaded = st.file_uploader("Upload a POS CSV export", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded)
        raw = normalize_dataframe(raw)
        df = audit_dataframe(raw)
        st.session_state.raw_data = df
        st.session_state.pos_source = "CSV Upload"
        log_audit("CSV Upload", "System", f"{len(df)} rows imported")
        st.success(f"Imported {len(df)} invoices.")
        st.rerun()

    # Data preview
    if st.session_state.raw_data is not None:
        st.subheader(f"Live Data: {st.session_state.pos_source}")
        df = st.session_state.raw_data
        summary = get_compliance_summary(df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Invoices", summary['total'])
        m2.metric("Violations", summary['violations'], delta=f"-SAR {summary['capital_at_risk']:,} risk", delta_color="inverse")
        m3.metric("Violation Rate", f"{summary['violation_rate']}%")
        m4.metric("Capital at Risk", f"SAR {summary['capital_at_risk']:,}")

        st.dataframe(
            df[['invoice_id', 'date', 'category', 'amount_sar', 'customer_vat_id', 'doc_type', 'ai_risk_score', 'violations']],
            use_container_width=True,
            hide_index=True
        )


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 2: ZATCA SHIELD
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🛡️ ZATCA Shield":
    st.title("ZATCA Shield — Real-Time Compliance Firewall")
    st.caption("Intercepts transactions BEFORE they become violations. Fires inside the POS, not after.")

    # ── Live POS Simulator
    st.markdown("### ⚡ Live POS Terminal Simulator")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            checkout_amt = st.number_input("Transaction Amount (SAR)", min_value=1.0, value=1200.0, step=50.0)
        with c2:
            checkout_vat = st.text_input("Customer VAT ID", placeholder="Leave blank to simulate B2C")

        if st.button("🛒 Process Transaction", type="primary"):
            result = check_transaction(checkout_amt, checkout_vat.strip())

            if not result['allowed']:
                rule = result['blocking_rule']
                st.error(f"🛑 **TRANSACTION BLOCKED** — {rule.rule_id}")
                st.markdown(f"""
                <div class="risk-card">
                    <b>Rule:</b> {rule.description}<br>
                    <b>Severity:</b> {rule.severity}<br>
                    <b>Fix:</b> Add a valid 15-digit customer VAT ID to convert to B2B invoice.
                </div>
                """, unsafe_allow_html=True)

                # Memory-aware suggestion
                if st.session_state.behavior_memory['b2b_conversions'] >= 2:
                    st.info("🧠 **Cognivis Memory:** You've handled this before. Enter VAT ID above to auto-resolve.")

                col_fix1, col_fix2 = st.columns(2)
                if col_fix1.button("📝 Convert to B2B"):
                    st.warning("Enter a valid VAT ID above and click Process Transaction again.")
                if col_fix2.button("✂️ Split Transaction"):
                    half = checkout_amt / 2
                    st.success(f"✅ Split approved: 2× SAR {half:,.2f} — both below SAR 1,000 threshold.")
                    log_audit("Transaction Split", f"POS-LIVE", "Cleared", "Cashier")
            else:
                st.success("✅ Transaction Compliant — Cleared for ZATCA submission.")
                log_audit("POS Transaction", f"POS-{int(checkout_amt)}", "Cleared")
                if result['warnings']:
                    for w in result['warnings']:
                        st.warning(f"⚠️ Warning: {w.description}")

    st.divider()

    # ── Backlog Resolution
    if st.session_state.raw_data is None:
        st.info("Connect a POS source in Integration Hub to see your violation backlog.")
    else:
        df = st.session_state.raw_data
        trust = calculate_trust_score(df)
        summary = get_compliance_summary(df)

        # KPI Row
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""<div class="metric-card">
                <p style='color:#ef4444; font-size:12px; margin:0;'>VIOLATIONS</p>
                <h2 style='margin:8px 0; color:#f8fafc;'>{summary['violations']}</h2>
                <p style='color:#64748b; font-size:12px; margin:0;'>SAR {summary['capital_at_risk']:,} exposure</p>
            </div>""", unsafe_allow_html=True)
        with k2:
            protected = (summary.get('initial', summary['violations']) - summary['violations']) * 5000
            st.markdown(f"""<div class="metric-card">
                <p style='color:#22c55e; font-size:12px; margin:0;'>CAPITAL PROTECTED</p>
                <h2 style='margin:8px 0; color:#f8fafc;'>SAR {protected:,}</h2>
                <p style='color:#64748b; font-size:12px; margin:0;'>Violations resolved</p>
            </div>""", unsafe_allow_html=True)
        with k3:
            score_color = "#22c55e" if trust.score >= 80 else "#eab308" if trust.score >= 60 else "#ef4444"
            st.markdown(f"""<div class="trust-card">
                <p style='color:#94a3b8; font-size:12px; margin:0;'>COGNIVIS TRUST SCORE™</p>
                <h2 style='margin:8px 0; color:{score_color};'>{trust.score} / 100</h2>
                <p style='color:#eab308; font-size:12px; margin:0;'>{trust.label}</p>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Resolution Queue
        violations = df[df['ai_risk_score'] >= 80]
        if violations.empty:
            st.success("🎉 No violations detected. Your business is fully ZATCA compliant.")
        else:
            st.markdown(f"### 🚨 Resolution Queue: {len(violations)} violations pending")
            for idx, row in violations.iterrows():
                with st.expander(f"📄 {row['invoice_id']} | SAR {row['amount_sar']:,.2f} | Risk: {row['ai_risk_score']}/100 | Rules: {row['violations']}"):
                    if not st.session_state.review_mode.get(idx, False):
                        st.markdown(f"<span class='badge-high'>ACTION REQUIRED</span>", unsafe_allow_html=True)
                        st.write(f"**Category:** {row['category']} | **Date:** {row['date']}")
                        if st.button("👨‍💻 Resolve This Violation", key=f"open_{idx}"):
                            st.session_state.review_mode[idx] = True
                            st.rerun()
                    else:
                        st.info("💡 **AI Suggestion:** Convert to B2B Tax Invoice by capturing customer VAT ID.")
                        vat_input = st.text_input("Enter Customer VAT ID (15 digits):", key=f"vat_{idx}")
                        if st.button("✅ Approve & Convert to B2B", key=f"approve_{idx}"):
                            valid, err = validate_vat_id(vat_input)
                            if valid:
                                st.session_state.raw_data.at[idx, 'customer_vat_id'] = vat_input
                                st.session_state.raw_data.at[idx, 'doc_type'] = "Tax Invoice (388)"
                                st.session_state.raw_data.at[idx, 'ai_risk_score'] = 0
                                st.session_state.raw_data.at[idx, 'status'] = "Resolved"
                                st.session_state.review_mode[idx] = False
                                st.session_state.behavior_memory['b2b_conversions'] += 1
                                log_audit("B2B Conversion", row['invoice_id'], "Resolved", "Admin")
                                st.rerun()
                            else:
                                st.error(f"❌ {err}")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 3: AI BRAIN
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "💡 AI Brain":
    st.title("AI Brain — Decision Intelligence Engine")
    st.caption("Turns your transaction data into actionable business strategy. Every insight is derived from your real data.")

    if st.session_state.raw_data is None:
        st.info("Connect a POS source in Integration Hub to activate the AI Brain.")
    else:
        df = st.session_state.raw_data

        # Memory status
        conversions = st.session_state.behavior_memory['b2b_conversions']
        st.markdown(f"""
        <div class="brain-card">
            <b style='color:#3b82f6;'>Behavioral Memory Active</b><br>
            <span style='color:#94a3b8; font-size:13px;'>
                Tracked {conversions} manual B2B conversions this session.
                Model confidence adapts with each action you take.
            </span>
        </div>
        """, unsafe_allow_html=True)

        recommendations = generate_all_recommendations(df)

        if not recommendations:
            st.info("Not enough data to generate recommendations. Load a larger dataset.")
        else:
            st.markdown("### Actionable Intelligence")
            for rec in recommendations:
                icon = "🛡️" if rec.category == "COMPLIANCE" else "📈" if rec.category == "REVENUE" else "⚙️"
                border = "#ef4444" if rec.category == "COMPLIANCE" else "#3b82f6" if rec.category == "REVENUE" else "#eab308"

                st.markdown(f"""
                <div class="wa-container" style="background:#0f172a; padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.07); margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span style="font-weight:600; color:#f8fafc;">{icon} {rec.title}</span>
                        <span style="font-size:11px; color:#64748b;">{int(rec.confidence * 100)}% confidence</span>
                    </div>
                    <div class="wa-bubble" style="border-left-color:{border};">{rec.body}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"↗ {rec.action_label}", key=f"rec_{rec.title[:20]}"):
                    st.session_state['pending_prompt'] = rec.action_prompt

        # Revenue chart
        st.divider()
        st.subheader("Revenue Trend (Last 30 Days)")
        trend = generate_revenue_trend(30)
        st.line_chart(trend.set_index('date')['revenue'], color="#3b82f6")

        # Category breakdown
        st.subheader("Revenue by Category")
        cat_rev = df[df['amount_sar'] > 0].groupby('category')['amount_sar'].sum().reset_index()
        cat_rev.columns = ['Category', 'Revenue (SAR)']
        st.bar_chart(cat_rev.set_index('Category'), color="#22c55e")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 4: WHAT-IF ENGINE
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🔮 What-If Engine":
    st.title("Predictive What-If Engine")
    st.caption("Simulate business decisions before you make them. See revenue impact AND compliance risk change.")

    if st.session_state.raw_data is None:
        st.warning("Connect POS data in Integration Hub to run simulations.")
    else:
        df = st.session_state.raw_data
        tab1, tab2 = st.tabs(["📈 Price Change Simulator", "🛡️ VAT Enforcement Simulator"])

        with tab1:
            st.markdown("### Simulate a Price Change")
            price_pct = st.slider("Price adjustment (%)", -20, 40, 15)
            result = simulate_price_change(df, price_pct)

            c1, c2, c3 = st.columns(3)
            c1.metric("Revenue Impact", f"SAR {result['revenue_delta']:+,.0f}", delta=f"{price_pct:+}%")
            c2.metric("New Compliance Risks", f"{result['new_risk_count']} transactions", delta_color="inverse",
                      delta=f"+SAR {result['fine_exposure_delta']:,} fine exposure" if result['new_risk_count'] > 0 else "No new risks")
            c3.metric("Projected Monthly Revenue", f"SAR {result['sim_revenue']:,.0f}")

            if result['new_risk_count'] > 0:
                st.warning(
                    f"⚠️ A {price_pct}% increase pushes **{result['new_risk_count']} transactions** over the SAR 1,000 compliance threshold. "
                    f"Activate Proactive POS Intercept before deploying this change to avoid SAR {result['fine_exposure_delta']:,} in potential fines."
                )
            else:
                st.success("✅ This price adjustment introduces no new compliance risks.")

        with tab2:
            st.markdown("### Simulate VAT Capture Enforcement")
            capture_pct = st.slider("VAT capture rate on transactions ≥ SAR 1,000 (%)", 0, 100, 70)
            result2 = simulate_vat_enforcement(df, capture_pct / 100)

            c1, c2, c3 = st.columns(3)
            c1.metric("Violations at Risk", result2['total_at_risk'])
            c2.metric("Violations Resolved", result2['resolved_by_capture'],
                      delta=f"SAR {result2['fine_saved']:,} saved")
            c3.metric("Remaining Exposure", f"SAR {result2['fine_remaining']:,}", delta_color="inverse",
                      delta=f"{result2['remaining_violations']} unresolved")

            if result2['total_at_risk'] == 0:
                st.success("No high-value transactions with missing VAT. You're clean.")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 5: AUDIT LEDGER
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "📓 Audit Ledger":
    st.title("Immutable Compliance Audit Ledger")
    st.caption("Every action taken in Cognivis OS is permanently logged. This log is append-only.")

    if not st.session_state.audit_ledger:
        st.info("No audit events yet. Connect a POS source and start resolving violations.")
    else:
        ledger_df = pd.DataFrame(st.session_state.audit_ledger)
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)
        st.caption(f"Total entries: {len(ledger_df)}")


# ══════════════════════════════════════════════════════════════════════════════════
# MODULE 6: INVESTOR STORY MODE
# ══════════════════════════════════════════════════════════════════════════════════
elif menu == "🎯 Investor Story":
    st.title("Investor Demo — Live Guided Flow")
    st.caption("A 6-step walkthrough showing exactly how Cognivis OS works end-to-end.")

    # Load demo data if not loaded
    if st.session_state.raw_data is None:
        raw = generate_realistic_invoices(n=50, violation_rate=0.22)
        df = audit_dataframe(raw)
        st.session_state.raw_data = df
        st.session_state.pos_source = "Investor Demo"

    df = st.session_state.raw_data
    summary = get_compliance_summary(df)
    trust = calculate_trust_score(df)
    recs = generate_all_recommendations(df)

    steps = [
        "Connect POS",
        "Detect Violations",
        "Intercept in Real-Time",
        "Money Saved",
        "Trust Score",
        "AI Intelligence"
    ]

    step = st.session_state.investor_step
    cols = st.columns(len(steps))
    for i, (col, label) in enumerate(zip(cols, steps)):
        with col:
            color = "#22c55e" if i < step else "#3b82f6" if i == step else "#1e293b"
            st.markdown(f"""<div style="text-align:center; padding:8px; background:{color}; border-radius:8px; font-size:12px; font-weight:600;">
            {'✓ ' if i < step else ''}{i+1}. {label}</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Step content
    if step == 0:
        st.markdown("""<div class='step-block'>
        <h3 style='color:#3b82f6; margin:0;'>Step 1: Connect your POS</h3>
        <p style='color:#94a3b8;'>Cognivis connects to Foodics, Salla, or any POS via API in under 60 seconds. 
        All data is normalized and scored locally — nothing leaves your environment.</p></div>""", unsafe_allow_html=True)
        st.success(f"✅ Connected: {st.session_state.pos_source} — {summary['total']} invoices loaded.")

    elif step == 1:
        st.markdown(f"""<div class='step-block'>
        <h3 style='color:#ef4444; margin:0;'>Step 2: Violations Detected Instantly</h3>
        <p style='color:#94a3b8;'>Our AI rule engine scanned all {summary['total']} invoices in milliseconds.</p></div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Violations Found", summary['violations'])
        c2.metric("Violation Rate", f"{summary['violation_rate']}%")
        c3.metric("Fine Exposure", f"SAR {summary['capital_at_risk']:,}")

    elif step == 2:
        st.markdown("""<div class='step-block'>
        <h3 style='color:#22c55e; margin:0;'>Step 3: Real-Time POS Interception</h3>
        <p style='color:#94a3b8;'>This is the core moat. Cognivis fires INSIDE the POS — before the invoice is created.</p></div>""", unsafe_allow_html=True)
        result = check_transaction(1450.00, "")
        st.error(f"🛑 BLOCKED — {result['blocking_rule'].rule_id}: {result['blocking_rule'].description}")
        st.caption("Transaction for SAR 1,450 with no VAT ID — intercepted before ZATCA submission.")

    elif step == 3:
        st.markdown("""<div class='step-block'>
        <h3 style='color:#22c55e; margin:0;'>Step 4: Money Saved</h3>
        <p style='color:#94a3b8;'>Each resolved violation = SAR 5,000 minimum fine avoided.</p></div>""", unsafe_allow_html=True)
        saved = summary['violations'] * 5000
        st.metric("Potential Savings", f"SAR {saved:,}", delta=f"{summary['violations']} violations blocked")
        st.progress(min(1.0, summary['violations'] / 20))

    elif step == 4:
        st.markdown("""<div class='step-block'>
        <h3 style='color:#eab308; margin:0;'>Step 5: Cognivis Trust Score™</h3>
        <p style='color:#94a3b8;'>Compliance activity builds a verifiable financial identity — the foundation for future SME credit access.</p></div>""", unsafe_allow_html=True)
        sc_color = "#22c55e" if trust.score >= 80 else "#eab308"
        st.markdown(f"""<div class='trust-card'>
            <h1 style='color:{sc_color}; margin:0;'>{trust.score} / 100</h1>
            <h3 style='color:#eab308; margin:4px 0;'>{trust.label}</h3>
            <p style='color:#94a3b8; font-size:12px;'>Violation Rate: {trust.violation_rate}% | Consistency: {trust.consistency_score}%</p>
        </div>""", unsafe_allow_html=True)

    elif step == 5:
        st.markdown("""<div class='step-block'>
        <h3 style='color:#3b82f6; margin:0;'>Step 6: AI Intelligence (The Brain)</h3>
        <p style='color:#94a3b8;'>Beyond compliance — Cognivis turns every transaction into a growth insight.</p></div>""", unsafe_allow_html=True)
        if recs:
            rec = recs[0]
            st.markdown(f"""<div class="wa-bubble">🧠 <b>Cognivis OS</b><br><br>{rec.body}</div>""", unsafe_allow_html=True)

    st.divider()
    nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
    with nav_c1:
        if step > 0 and st.button("← Previous"):
            st.session_state.investor_step -= 1
            st.rerun()
    with nav_c3:
        if step < len(steps) - 1:
            if st.button("Next →", type="primary"):
                st.session_state.investor_step += 1
                st.rerun()
        else:
            st.success("🎉 Demo complete. This is Cognivis OS.")
