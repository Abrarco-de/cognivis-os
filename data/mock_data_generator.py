"""
Cognivis OS — Mock Data Generator
Generates realistic Saudi SME invoice datasets for demo and testing.
"""
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import random

CATEGORIES = ["F&B", "Retail", "Catering", "Pharmacy", "Electronics", "Clothing"]
DOC_TYPES = ["Tax Invoice (388)", "Simplified Invoice (388-B2C)"]

# Realistic Saudi VAT ID pool (15-digit format)
VALID_VAT_IDS = [
    "310122393500003", "311234567800003", "312987654300003",
    "314001122300003", "315678901200003", "316543210900003",
]


def generate_invoice_id(prefix="INV") -> str:
    return f"{prefix}-{random.randint(10000, 99999)}"


def generate_realistic_invoices(n: int = 50, violation_rate: float = 0.20) -> pd.DataFrame:
    """
    Generates a realistic SME invoice dataset with a controlled violation rate.
    Violations: amount >= 1000 SAR with no VAT ID (triggers ZATCA BR-KSA-14).
    """
    records = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(n):
        category = random.choice(CATEGORIES)
        is_b2b = random.random() > 0.4  # 60% B2B, 40% B2C (realistic for F&B/retail)

        # Amount distributions by category
        if category in ["Electronics", "Catering"]:
            amount = round(random.uniform(500, 5000), 2)
        elif category == "F&B":
            amount = round(random.uniform(30, 800), 2)
        else:
            amount = round(random.uniform(50, 2000), 2)

        # Inject violations: high-value transactions with missing VAT
        is_violation = (amount >= 1000) and (random.random() < violation_rate) and not is_b2b

        vat_id = random.choice(VALID_VAT_IDS) if (is_b2b and not is_violation) else ""
        doc_type = "Tax Invoice (388)" if is_b2b else "Simplified Invoice (388-B2C)"
        date = (base_date + timedelta(days=random.randint(0, 29))).strftime("%Y-%m-%d")

        records.append({
            "invoice_id": generate_invoice_id(),
            "amount_sar": amount,
            "customer_vat_id": vat_id,
            "category": category,
            "doc_type": doc_type,
            "date": date,
            "ai_risk_score": 0,
            "status": "Pending"
        })

    df = pd.DataFrame(records)
    return df


def generate_revenue_trend(days: int = 30) -> pd.DataFrame:
    """Generates realistic daily revenue with weekend dips and growth trend."""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    base = 8000
    trend = np.linspace(0, 2000, days)  # Growth over time

    revenue = []
    for i, d in enumerate(dates):
        day_factor = 0.6 if d.weekday() == 4 else 1.0  # Friday dip (Saudi market)
        noise = random.uniform(-800, 800)
        revenue.append(max(0, base + trend[i] + noise) * day_factor)

    return pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates], "revenue": revenue})


def get_foodics_mock() -> pd.DataFrame:
    """Simulate a Foodics API response (realistic column names)."""
    return pd.DataFrame({
        "Bill No": [generate_invoice_id("FDC") for _ in range(8)],
        "Item Group": random.choices(CATEGORIES, k=8),
        "Total (SAR)": [round(random.uniform(50, 4500), 2) for _ in range(8)],
        "Tax Number": [random.choice(VALID_VAT_IDS) if random.random() > 0.35 else "" for _ in range(8)],
        "Bill Date": [(datetime.now() - timedelta(days=random.randint(0, 7))).strftime("%Y-%m-%d") for _ in range(8)]
    })


def get_salla_mock() -> pd.DataFrame:
    """Simulate a Salla API response (e-commerce column names)."""
    return pd.DataFrame({
        "order_id": [generate_invoice_id("SLA") for _ in range(10)],
        "product_category": random.choices(["Clothing", "Electronics", "Home"], k=10),
        "order_total": [round(random.uniform(100, 3000), 2) for _ in range(10)],
        "buyer_vat": [random.choice(VALID_VAT_IDS) if random.random() > 0.5 else "" for _ in range(10)],
        "created_at": [(datetime.now() - timedelta(days=random.randint(0, 14))).strftime("%Y-%m-%d") for _ in range(10)]
    })
