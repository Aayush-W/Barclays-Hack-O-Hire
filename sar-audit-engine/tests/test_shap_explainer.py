from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shap_explainer import TransactionExplainer


def test_transaction_explainer_runs_on_typology_artifact(tmp_path: Path) -> None:
    feature_names = [
        "tx_count",
        "total_inflow",
        "total_outflow",
        "abs_net_flow_ratio",
        "unique_senders",
        "unique_receivers",
        "hop_count",
        "cross_bank_ratio",
        "multi_currency_flag",
        "rapid_pass_through_flag",
        "rapid_pass_through_min_hours",
        "duration_hours",
        "tx_per_day",
        "avg_interarrival_minutes",
        "median_interarrival_minutes",
        "max_tx_in_1h",
        "payment_format_count",
        "currency_count",
        "avg_amount_received",
        "max_amount_received",
    ]
    x = np.random.RandomState(7).randn(40, len(feature_names))
    y = np.array(["STACK"] * 20 + ["CYCLE"] * 20)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    model.fit(x, y)

    artifact_path = tmp_path / "typology.joblib"
    joblib.dump({"model": model, "feature_names": feature_names}, artifact_path)

    explainer = TransactionExplainer(model_path=artifact_path)
    case_payload = {
        "case_id": "CASE_TEST",
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
    output = explainer.explain_case(case_payload)

    assert "prediction" in output
    assert "contributions" in output
    assert len(output["contributions"]) == len(feature_names)
