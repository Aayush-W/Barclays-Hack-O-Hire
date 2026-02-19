from __future__ import annotations

from config.settings import PipelineSettings
from core.evidence_mapper import map_case_evidence
from core.feature_extractor import extract_signals
from core.reasoning_engine import build_reasoning


def test_clean_payroll_like_case_not_escalated() -> None:
    case = {
        "case_id": "CASE_CLEAN_PAYROLL",
        "typology": "GATHER-SCATTER",
        "transactions": [
            {
                "timestamp": "2022/10/01 09:00",
                "from_bank": "B001",
                "from_account": "EMPLOYER01",
                "to_bank": "B001",
                "to_account": "EMP1001",
                "amount_received": 75000.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 75000.0,
                "payment_currency": "US Dollar",
                "payment_format": "ACH",
            },
            {
                "timestamp": "2022/10/01 09:02",
                "from_bank": "B001",
                "from_account": "EMPLOYER01",
                "to_bank": "B001",
                "to_account": "EMP1002",
                "amount_received": 72000.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 72000.0,
                "payment_currency": "US Dollar",
                "payment_format": "ACH",
            },
            {
                "timestamp": "2022/10/02 11:15",
                "from_bank": "B001",
                "from_account": "EMP1001",
                "to_bank": "B002",
                "to_account": "LANDLORD01",
                "amount_received": 1500.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 1500.0,
                "payment_currency": "US Dollar",
                "payment_format": "ACH",
            },
            {
                "timestamp": "2022/10/07 16:10",
                "from_bank": "B001",
                "from_account": "EMP1002",
                "to_bank": "B005",
                "to_account": "GROCERY01",
                "amount_received": 180.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 180.0,
                "payment_currency": "US Dollar",
                "payment_format": "CARD",
            },
        ],
    }

    signals = extract_signals(case)
    reasoning = build_reasoning(case, signals, settings=PipelineSettings())
    evidence_map = map_case_evidence(case, signals, reasoning, retriever=None)
    risk = evidence_map["risk_assessment"]

    assert risk["escalation_recommended"] is False
    assert risk["score"] < 40


def test_suspicious_rapid_multi_hop_case_escalated() -> None:
    case = {
        "case_id": "CASE_SUSPICIOUS",
        "typology": "CYCLE",
        "transactions": [
            {
                "timestamp": "2022/10/01 09:00",
                "from_bank": "B001",
                "from_account": "A1",
                "to_bank": "B002",
                "to_account": "B1",
                "amount_received": 250000.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 250000.0,
                "payment_currency": "US Dollar",
                "payment_format": "WIRE",
            },
            {
                "timestamp": "2022/10/01 09:05",
                "from_bank": "B002",
                "from_account": "B1",
                "to_bank": "B003",
                "to_account": "C1",
                "amount_received": 249000.0,
                "receiving_currency": "Euro",
                "amount_paid": 249000.0,
                "payment_currency": "Euro",
                "payment_format": "WIRE",
            },
            {
                "timestamp": "2022/10/01 09:10",
                "from_bank": "B003",
                "from_account": "C1",
                "to_bank": "B004",
                "to_account": "D1",
                "amount_received": 248000.0,
                "receiving_currency": "Euro",
                "amount_paid": 248000.0,
                "payment_currency": "US Dollar",
                "payment_format": "WIRE",
            },
            {
                "timestamp": "2022/10/01 09:15",
                "from_bank": "B004",
                "from_account": "D1",
                "to_bank": "B005",
                "to_account": "E1",
                "amount_received": 247000.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 247000.0,
                "payment_currency": "US Dollar",
                "payment_format": "WIRE",
            },
        ],
    }

    signals = extract_signals(case)
    reasoning = build_reasoning(case, signals, settings=PipelineSettings())
    evidence_map = map_case_evidence(case, signals, reasoning, retriever=None)
    risk = evidence_map["risk_assessment"]

    assert risk["escalation_recommended"] is True
    assert risk["score"] >= 55
