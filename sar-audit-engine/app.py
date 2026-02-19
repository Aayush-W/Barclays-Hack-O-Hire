from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from config.settings import PROCESSED_DIR, PipelineSettings
from core.audit_input_parser import parse_audit_trail_text
from core.evidence_mapper import map_case_evidence
from core.feature_extractor import extract_signals
from core.reasoning_engine import build_reasoning
from core.typology_classifier import predict_typology_for_case
from langchain_narrator import SARNarrativeGenerator
from shap_explainer import TransactionExplainer


SAMPLE_AUDIT_TRAIL = """2022/09/01 05:14,001,A1,002,B1,12000,US Dollar,12000,US Dollar,ACH
2022/09/01 06:00,002,B1,003,C1,11800,US Dollar,11800,US Dollar,ACH
2022/09/01 06:40,003,C1,004,D1,11700,US Dollar,11700,US Dollar,ACH
"""


def _default_paths() -> Dict[str, str]:
    model_path = PROCESSED_DIR / "models" / "audit_typology_classifier.joblib"
    adapter_candidates = [
        PROCESSED_DIR / "models" / "sar_lora_adapter",
        PROCESSED_DIR / "models" / "sar_lora_smoke_v2",
        PROCESSED_DIR / "models" / "sar_lora_smoke",
    ]
    adapter_path = ""
    for candidate in adapter_candidates:
        if candidate.exists():
            adapter_path = str(candidate)
            break
    return {
        "typology_model_path": str(model_path),
        "narrative_backend": "ollama",
        "narrative_model_name": "mistral:latest",
        "ollama_base_url": "http://127.0.0.1:11434",
        "speed_profile": "fast",
        "adapter_path": adapter_path,
    }


@st.cache_resource(show_spinner=False)
def _load_explainer(model_path: str) -> TransactionExplainer:
    return TransactionExplainer(model_path=model_path)


@st.cache_resource(show_spinner=False)
def _load_narrator(
    narrative_model_name: str,
    narrative_backend: str,
    ollama_base_url: str,
    speed_profile: str,
    adapter_path: str,
    max_new_tokens: int,
    temperature: float,
) -> SARNarrativeGenerator:
    return SARNarrativeGenerator(
        model_name=narrative_model_name,
        provider=narrative_backend,
        ollama_base_url=ollama_base_url,
        speed_profile=speed_profile,
        adapter_path=adapter_path or None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def _build_case(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return {
        "case_id": f"MANUAL_{timestamp}",
        "typology": None,
        "transactions": transactions,
    }


def _show_top_class_chart(top_predictions: List[Dict[str, Any]]) -> None:
    labels = [item["label"] for item in top_predictions]
    scores = [float(item["score"]) for item in top_predictions]
    frame = pd.DataFrame({"typology": labels, "probability": scores}).set_index("typology")
    st.bar_chart(frame)


def _show_contrib_chart(contributions: List[Dict[str, Any]], limit: int = 10) -> None:
    top = contributions[: max(1, int(limit))]
    frame = pd.DataFrame(
        {
            "feature": [item["feature"] for item in top],
            "attribution": [float(item["shap_value"]) for item in top],
        }
    ).set_index("feature")
    st.bar_chart(frame)


def _pipeline_from_text(
    audit_text: str,
    model_path: str,
    narrator: SARNarrativeGenerator,
    style: str,
) -> Dict[str, Any]:
    transactions = parse_audit_trail_text(audit_text)
    case = _build_case(transactions)

    prediction = predict_typology_for_case(case_payload=case, model_path=Path(model_path), top_k=5)
    case["typology"] = prediction["predicted_typology"]

    signals_payload = extract_signals(case)
    reasoning_payload = build_reasoning(case, signals_payload, settings=PipelineSettings())
    evidence_map = map_case_evidence(
        case=case,
        signals_payload=signals_payload,
        reasoning_payload=reasoning_payload,
        retriever=None,
        top_k_per_reason=3,
    )
    risk = evidence_map.get("risk_assessment", {}) or {}
    if bool(risk.get("escalation_recommended")):
        narrative_result = narrator.generate_narrative(evidence_map=evidence_map, style=style)
    else:
        narrative_result = {
            "backend": "rule_based_non_sar",
            "narrative": (
                "No SAR filing is recommended for this case at current thresholds. "
                "The observed activity is classified as lower-risk based on weak laundering signals "
                "and benign pattern adjustments. Continue routine monitoring and periodic review."
            ),
            "audit_trail": [
                {
                    "event": "sar_suppressed_low_risk",
                    "details": {
                        "risk_score": risk.get("score"),
                        "risk_band": risk.get("band"),
                        "escalation_recommended": risk.get("escalation_recommended"),
                    },
                }
            ],
        }
    return {
        "case": case,
        "prediction": prediction,
        "signals": signals_payload,
        "reasoning": reasoning_payload,
        "evidence_map": evidence_map,
        "narrative_result": narrative_result,
    }


def main() -> None:
    defaults = _default_paths()
    st.set_page_config(page_title="SAR Typology & Narrative Dashboard", layout="wide")
    st.title("SAR Typology Classification + Mistral Narrative")
    st.caption("Paste transaction audit trail, classify laundering typology, and generate SAR narrative.")

    st.sidebar.header("Model Configuration")
    model_path = st.sidebar.text_input("Typology Model Path", value=defaults["typology_model_path"])
    narrative_backend = st.sidebar.selectbox(
        "Narrative Backend",
        options=["ollama", "huggingface"],
        index=0 if defaults["narrative_backend"] == "ollama" else 1,
    )
    if narrative_backend == "ollama":
        narrative_model_name = st.sidebar.text_input(
            "Ollama Model Name",
            value=defaults["narrative_model_name"],
        )
        ollama_base_url = st.sidebar.text_input(
            "Ollama Base URL",
            value=defaults["ollama_base_url"],
        )
        adapter_path = ""
    else:
        narrative_model_name = st.sidebar.text_input(
            "Hugging Face Model Name",
            value="mistralai/Mistral-7B-Instruct-v0.2",
        )
        ollama_base_url = defaults["ollama_base_url"]
        adapter_path = st.sidebar.text_input("LoRA Adapter Path (optional)", value=defaults["adapter_path"])
    speed_profile = st.sidebar.selectbox(
        "Response Speed",
        options=["fast", "balanced", "detailed"],
        index=["fast", "balanced", "detailed"].index(defaults["speed_profile"]),
    )
    style = st.sidebar.selectbox("Narrative Style", options=["standard", "condensed"], index=1)
    max_new_tokens = st.sidebar.slider("Narrative Max Tokens", min_value=64, max_value=1024, value=192, step=32)
    temperature = st.sidebar.slider("Narrative Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    if st.sidebar.button("Load Sample Audit Trail"):
        st.session_state["audit_text"] = SAMPLE_AUDIT_TRAIL

    audit_text = st.text_area(
        "Paste Audit Trail (CSV lines or JSON)",
        value=st.session_state.get("audit_text", SAMPLE_AUDIT_TRAIL),
        height=220,
    )
    st.caption(
        "CSV format: timestamp,from_bank,from_account,to_bank,to_account,amount_received,receiving_currency,amount_paid,payment_currency,payment_format"
    )

    run_clicked = st.button("Analyze And Generate SAR", type="primary", use_container_width=True)
    if not run_clicked:
        return

    if not Path(model_path).exists():
        st.error(f"Typology model not found: {model_path}")
        return

    try:
        with st.spinner("Loading model explainer and narrator..."):
            explainer = _load_explainer(model_path=model_path)
            narrator = _load_narrator(
                narrative_model_name=narrative_model_name,
                narrative_backend=narrative_backend,
                ollama_base_url=ollama_base_url,
                speed_profile=speed_profile,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        with st.spinner("Running classification and narrative generation..."):
            result = _pipeline_from_text(
                audit_text=audit_text,
                model_path=model_path,
                narrator=narrator,
                style=style,
            )

        explanation = explainer.explain_case(result["case"])
        prediction = result["prediction"]
        evidence_map = result["evidence_map"]
        narrative_result = result["narrative_result"]
        risk = evidence_map.get("risk_assessment", {})
        escalation_recommended = bool(risk.get("escalation_recommended"))
        display_typology = (
            prediction["predicted_typology"] if escalation_recommended else "NO_CLEAR_LAUNDERING_PATTERN"
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Predicted Typology", display_typology)
        col2.metric("Top Probability", f"{prediction['top_predictions'][0]['score'] * 100:.2f}%")
        col3.metric("Risk Band", str(risk.get("band", "unknown")).upper())
        col4.metric("Risk Score", str(risk.get("score", "n/a")))

        st.subheader("Class Probabilities")
        _show_top_class_chart(prediction["top_predictions"])

        st.subheader("Feature Contributions")
        _show_contrib_chart(explanation["contributions"], limit=10)

        st.subheader("Generated SAR Narrative")
        if escalation_recommended:
            st.info(narrative_result["narrative"])
        else:
            st.success(narrative_result["narrative"])
        st.caption(f"Narrative backend: {narrative_result.get('backend')}")

        with st.expander("Reasoning Steps"):
            st.json(result["reasoning"])

        with st.expander("Evidence Map"):
            st.json(evidence_map)

        with st.expander("Generation Audit Trail"):
            st.json(narrative_result.get("audit_trail", []))

        output_payload = {
            "prediction": prediction,
            "risk_assessment": risk,
            "narrative_backend": narrative_result.get("backend"),
            "narrative": narrative_result.get("narrative"),
            "audit_trail": narrative_result.get("audit_trail"),
            "reasoning": result["reasoning"],
            "signals": result["signals"],
        }
        st.download_button(
            label="Download Result JSON",
            data=json.dumps(output_payload, indent=2),
            file_name=f"{result['case']['case_id']}_sar_result.json",
            mime="application/json",
        )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")


if __name__ == "__main__":
    main()
