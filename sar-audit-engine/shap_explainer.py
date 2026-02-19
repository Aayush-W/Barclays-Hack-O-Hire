from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from config.settings import PROCESSED_DIR
from core.typology_classifier import FEATURE_NAMES, build_typology_features


class TransactionExplainer:
    """
    Explains multiclass typology predictions with linear feature attributions.
    """

    def __init__(self, model_path: str | Path | None = None):
        default_model = PROCESSED_DIR / "models" / "audit_typology_classifier.joblib"
        self.model_path = Path(model_path) if model_path else default_model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Typology model not found at: {self.model_path}")

        loaded = joblib.load(self.model_path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.pipeline = loaded["model"]
            self.feature_names = list(loaded.get("feature_names") or FEATURE_NAMES)
        else:
            self.pipeline = loaded
            self.feature_names = list(FEATURE_NAMES)

    def _prepare_row(self, case_payload: Dict[str, Any]) -> np.ndarray:
        feature_map = build_typology_features(case_payload)
        row = np.array([[float(feature_map.get(name, 0.0)) for name in self.feature_names]], dtype=float)
        return row

    def explain_case(self, case_payload: Dict[str, Any]) -> Dict[str, Any]:
        row = self._prepare_row(case_payload)
        probabilities = self.pipeline.predict_proba(row)[0]
        classes = list(self.pipeline.classes_)

        pred_idx = int(np.argmax(probabilities))
        predicted_label = str(classes[pred_idx])
        predicted_probability = float(probabilities[pred_idx])

        imputer = self.pipeline.named_steps.get("imputer")
        scaler = self.pipeline.named_steps.get("scaler")
        clf = self.pipeline.named_steps.get("clf")

        transformed = row
        if imputer is not None:
            transformed = imputer.transform(transformed)
        if scaler is not None:
            transformed = scaler.transform(transformed)

        contributions: List[Dict[str, Any]] = []
        coef = None
        if clf is not None and hasattr(clf, "coef_"):
            coef_matrix = np.array(clf.coef_)
            if coef_matrix.ndim == 1:
                coef = coef_matrix
            else:
                coef = coef_matrix[pred_idx]

        values_by_feature = row[0].tolist()
        scaled_values = transformed[0].tolist()

        for idx, feature in enumerate(self.feature_names):
            feature_value = float(values_by_feature[idx])
            scaled_value = float(scaled_values[idx])
            weight = float(coef[idx]) if coef is not None else 0.0
            score = float(weight * scaled_value)
            contributions.append(
                {
                    "feature": feature,
                    "value": feature_value,
                    "scaled_value": scaled_value,
                    "coefficient": weight,
                    "shap_value": score,
                    "impact": "increases likelihood" if score > 0 else "decreases likelihood",
                    "magnitude": abs(score),
                }
            )

        contributions.sort(key=lambda item: item["magnitude"], reverse=True)
        top_predictions = [
            {"label": str(label), "score": float(score)}
            for label, score in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)[:5]
        ]

        return {
            "prediction": predicted_label,
            "probability": predicted_probability,
            "top_predictions": top_predictions,
            "contributions": contributions,
            "feature_values": {name: float(val) for name, val in zip(self.feature_names, values_by_feature)},
            "method": "linear_attribution_proxy",
        }
