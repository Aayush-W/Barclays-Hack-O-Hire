from __future__ import annotations

from core.sar_report_formatter import build_sar_report_json


def test_build_sar_report_json_shape() -> None:
    case = {
        "case_id": "CASE_ABC",
        "typology": "STACK",
        "transactions": [
            {
                "timestamp": "2022/09/01 05:14",
                "from_bank": "001",
                "from_account": "A1",
                "to_bank": "002",
                "to_account": "B1",
                "amount_received": 12000.0,
                "receiving_currency": "USD",
                "amount_paid": 12000.0,
                "payment_currency": "USD",
                "payment_format": "ACH",
            }
        ],
    }
    evidence_map = {
        "case_id": "CASE_ABC",
        "typology": "STACK",
        "risk_assessment": {"score": 78, "band": "high", "escalation_recommended": True},
        "reasons": [{"claim": "Rapid pass-through activity observed.", "category": "rapid_pass_through"}],
    }

    report = build_sar_report_json(
        case=case,
        evidence_map=evidence_map,
        narrative_text="Test narrative",
        filing_institution_id="BANK_1",
        associated_sar_references=["SAR-001"],
    )

    assert "sar_report" in report
    root = report["sar_report"]
    assert root["report_meta"]["filing_institution_id"] == "BANK_1"
    assert root["report_meta"]["case_reference_number"] == "CASE_ABC"
    assert "part_1_introduction" in root["narrative"]
    assert "part_2_body" in root["narrative"]
    assert "part_3_conclusion" in root["narrative"]
    assert isinstance(root["supporting_documents"], list)
