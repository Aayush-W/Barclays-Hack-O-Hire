from __future__ import annotations

import json
from pathlib import Path

from core.typology_classifier import predict_typology_for_case, train_typology_classifier


def _tx(ts: str, from_bank: str, from_acct: str, to_bank: str, to_acct: str, amount: float) -> dict:
    return {
        "timestamp": ts,
        "from_bank": from_bank,
        "from_account": from_acct,
        "to_bank": to_bank,
        "to_account": to_acct,
        "amount_received": amount,
        "receiving_currency": "USD",
        "amount_paid": amount,
        "payment_currency": "USD",
        "payment_format": "ACH",
    }


def _write_case(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_train_and_predict_typology_classifier(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    patterns_path = tmp_path / "patterns.txt"
    model_path = tmp_path / "models" / "typology.joblib"

    pattern_lines = []
    case_index = 1

    # Class A: FAN-IN style (many senders to one receiver).
    for i in range(12):
        case_id = f"CASE_{case_index:03d}"
        txs = [
            _tx("2022/09/01 05:14", f"{100 + k}", f"A{i}{k}", "999", "CENTRAL", 1000 + 10 * k)
            for k in range(1, 6)
        ]
        _write_case(cases_dir / f"{case_id}.json", {"case_id": case_id, "typology": "In", "transactions": txs})
        pattern_lines.extend(
            [
                "BEGIN LAUNDERING ATTEMPT - FAN-IN:  Max 5-degree Fan-In",
                "END LAUNDERING ATTEMPT - FAN-IN",
            ]
        )
        case_index += 1

    # Class B: CYCLE style (chained hops).
    for i in range(12):
        case_id = f"CASE_{case_index:03d}"
        txs = [
            _tx("2022/09/01 05:14", "001", f"C{i}1", "002", f"C{i}2", 2100),
            _tx("2022/09/01 05:20", "002", f"C{i}2", "003", f"C{i}3", 2095),
            _tx("2022/09/01 05:30", "003", f"C{i}3", "004", f"C{i}4", 2090),
            _tx("2022/09/01 05:40", "004", f"C{i}4", "001", f"C{i}1", 2085),
        ]
        _write_case(cases_dir / f"{case_id}.json", {"case_id": case_id, "typology": "CYCLE", "transactions": txs})
        pattern_lines.extend(
            [
                "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 4 hops",
                "END LAUNDERING ATTEMPT - CYCLE",
            ]
        )
        case_index += 1

    patterns_path.write_text("\n".join(pattern_lines), encoding="utf-8")

    metrics = train_typology_classifier(
        cases_dir=cases_dir,
        model_output_path=model_path,
        patterns_path=patterns_path,
        expected_types=2,
        test_size=0.25,
        random_state=7,
        min_transactions=1,
    )
    assert metrics["class_count"] == 2
    assert set(metrics["classes"]) == {"CYCLE", "FAN-IN"}
    assert model_path.exists()

    sample_case = json.loads((cases_dir / "CASE_001.json").read_text(encoding="utf-8"))
    prediction = predict_typology_for_case(sample_case, model_path=model_path, top_k=2)
    assert prediction["predicted_typology"] in {"CYCLE", "FAN-IN"}
    assert len(prediction["top_predictions"]) >= 1
