"""
Cognivis OS — Decision Engine (AI Brain)
Generates real, data-driven recommendations from invoice data.
No random insights — every recommendation derives from actual patterns.
"""
import pandas as pd
import numpy as np
from data.schemas import DecisionRecommendation


def _confidence(df: pd.DataFrame) -> float:
    """
    Confidence scales with data volume.
    <10 rows = low confidence, 50+ rows = high confidence.
    """
    n = len(df)
    if n < 5: return 0.40
    if n < 20: return 0.65
    if n < 50: return 0.80
    return 0.94


def analyze_category_performance(df: pd.DataFrame) -> DecisionRecommendation | None:
    """Finds the highest and lowest revenue-generating categories."""
    if 'category' not in df.columns or len(df) < 5:
        return None

    perf = df[df['amount_sar'] > 0].groupby('category')['amount_sar'].agg(['sum', 'mean', 'count'])
    if len(perf) < 2:
        return None

    top = perf['sum'].idxmax()
    bottom = perf['sum'].idxmin()
    top_rev = perf.loc[top, 'sum']
    bottom_rev = perf.loc[bottom, 'sum']
    top_avg = perf.loc[top, 'mean']

    return DecisionRecommendation(
        title="Category Revenue Gap Detected",
        body=(
            f"**{top}** is your top-performing category generating SAR {top_rev:,.0f} "
            f"(avg SAR {top_avg:,.0f}/invoice). "
            f"**{bottom}** contributes only SAR {bottom_rev:,.0f}. "
            f"Consider reallocating floor space or marketing spend toward {top}."
        ),
        confidence=_confidence(df),
        category="REVENUE",
        action_label="Deep dive into category split",
        action_prompt=f"Give me a detailed strategy to grow {top} revenue in a Saudi SME context"
    )


def analyze_average_order_value(df: pd.DataFrame) -> DecisionRecommendation | None:
    """Detects if AOV is below a healthy threshold for the category mix."""
    if 'amount_sar' not in df.columns or len(df) < 5:
        return None

    aov = df[df['amount_sar'] > 0]['amount_sar'].mean()
    below_1000 = df[(df['amount_sar'] > 0) & (df['amount_sar'] < 1000)]
    pct_below = round(len(below_1000) / len(df[df['amount_sar'] > 0]) * 100, 1)

    if aov < 500:
        insight = (
            f"Your average order value is SAR {aov:,.0f}. "
            f"{pct_below}% of invoices are under SAR 1,000. "
            f"Bundling or upselling could push AOV into the SAR 1,000+ tier, "
            f"which also triggers stronger B2B incentives."
        )
    else:
        insight = (
            f"Your average order value is a healthy SAR {aov:,.0f}. "
            f"{pct_below}% of invoices are below SAR 1,000. "
            f"Watch for margin dilution from smaller transactions."
        )

    return DecisionRecommendation(
        title="Average Order Value Analysis",
        body=insight,
        confidence=_confidence(df),
        category="REVENUE",
        action_label="How to increase AOV",
        action_prompt="What are the best strategies to increase average order value for a Saudi SME in F&B or retail?"
    )


def detect_anomalies(df: pd.DataFrame) -> DecisionRecommendation | None:
    """Flags statistical outliers in transaction amounts."""
    if 'amount_sar' not in df.columns or len(df) < 10:
        return None

    amounts = df[df['amount_sar'] > 0]['amount_sar']
    mean = amounts.mean()
    std = amounts.std()
    outliers = df[(df['amount_sar'] > mean + 2.5 * std)]

    if len(outliers) == 0:
        return None

    outlier_ids = ", ".join(outliers['invoice_id'].head(3).tolist())
    outlier_total = outliers['amount_sar'].sum()

    return DecisionRecommendation(
        title=f"{len(outliers)} Anomalous Transactions Detected",
        body=(
            f"Transactions {outlier_ids} are statistical outliers "
            f"(>2.5 standard deviations above your mean SAR {mean:,.0f}). "
            f"Combined value: SAR {outlier_total:,.0f}. "
            f"Verify these are legitimate bulk or B2B orders — not data entry errors."
        ),
        confidence=min(0.92, _confidence(df)),
        category="OPERATIONS",
        action_label="Review anomalous invoices",
        action_prompt="What are common causes of transaction anomalies in Saudi SME POS systems and how should I investigate them?"
    )


def analyze_compliance_cost(df: pd.DataFrame) -> DecisionRecommendation | None:
    """Calculates the real financial cost of pending violations."""
    if 'ai_risk_score' not in df.columns:
        return None

    violations = df[df['ai_risk_score'] >= 80]
    if len(violations) == 0:
        return None

    min_fine = len(violations) * 5000
    violation_revenue = violations['amount_sar'].sum()

    return DecisionRecommendation(
        title=f"SAR {min_fine:,} at Risk from {len(violations)} Violations",
        body=(
            f"You have {len(violations)} invoices with critical compliance risk. "
            f"At SAR 5,000 minimum fine per violation, your exposure is SAR {min_fine:,}. "
            f"These invoices represent SAR {violation_revenue:,.0f} in revenue — "
            f"resolving them protects both the revenue and eliminates the fine risk."
        ),
        confidence=0.99,
        category="COMPLIANCE",
        action_label="Resolve all violations now",
        action_prompt="Walk me through how to resolve ZATCA B2B invoice violations for a Saudi restaurant group"
    )


def generate_all_recommendations(df: pd.DataFrame) -> list[DecisionRecommendation]:
    """
    Runs all analysis functions and returns a ranked list of recommendations.
    Compliance issues always surface first.
    """
    generators = [
        analyze_compliance_cost,
        analyze_category_performance,
        detect_anomalies,
        analyze_average_order_value,
    ]

    results = []
    for fn in generators:
        try:
            rec = fn(df)
            if rec is not None:
                results.append(rec)
        except Exception:
            pass

    return results
