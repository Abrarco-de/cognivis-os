"""
Cognivis OS — Data Schemas
Central definitions for all data structures.
"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Invoice:
    invoice_id: str
    amount_sar: float
    customer_vat_id: str
    category: str
    doc_type: str
    date: str
    ai_risk_score: int = 0
    status: str = "Pending"  # Pending | Cleared | Violated | Resolved


@dataclass
class ComplianceRule:
    rule_id: str
    description: str
    severity: str       # HIGH | MEDIUM | LOW
    action: str         # BLOCK | WARN | LOG
    condition: object   # callable: (invoice) -> bool


@dataclass
class AuditEntry:
    timestamp: str
    invoice_id: str
    action: str
    status: str
    authorized_by: str = "System Engine"


@dataclass
class TrustScore:
    score: int
    label: str          # PRIME | GOOD | FAIR | AT_RISK
    violation_rate: float
    resolution_speed_avg: float
    consistency_score: float


@dataclass
class DecisionRecommendation:
    title: str
    body: str
    confidence: float
    category: str       # COMPLIANCE | REVENUE | OPERATIONS
    action_label: str
    action_prompt: str
