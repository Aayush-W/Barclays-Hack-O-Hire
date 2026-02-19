from __future__ import annotations

from core.audit_input_parser import parse_audit_trail_text


def test_parse_csv_audit_trail_lines() -> None:
    raw = (
        "2022/09/01 05:14,001,A1,002,B1,12000,USD,12000,USD,ACH\n"
        "2022/09/01 06:00,002,B1,003,C1,11800,USD,11800,USD,ACH\n"
    )
    txs = parse_audit_trail_text(raw)
    assert len(txs) == 2
    assert txs[0]["from_bank"] == "001"
    assert float(txs[0]["amount_received"]) == 12000.0


def test_parse_json_audit_trail_payload() -> None:
    raw = (
        '{"transactions":[{"timestamp":"2022/09/01 05:14","from_bank":"001","from_account":"A1",'
        '"to_bank":"002","to_account":"B1","amount_received":12000,"receiving_currency":"USD",'
        '"amount_paid":12000,"payment_currency":"USD","payment_format":"ACH"}]}'
    )
    txs = parse_audit_trail_text(raw)
    assert len(txs) == 1
    assert txs[0]["to_account"] == "B1"
