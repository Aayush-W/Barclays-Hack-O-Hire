from __future__ import annotations

from llm.functional_eval import (
    build_template_baseline,
    compute_aml_adherence,
    compute_rouge,
    parse_prompt_fields,
)


def _sample_user_prompt() -> str:
    return (
        "Style instructions: Write in full paragraphs with clear timeline.\n\n"
        "Case ID: CASE_123\n"
        "Typology: FUNNEL\n"
        "Risk score: 88\n"
        "Risk band: high\n\n"
        "Reasons:\n"
        "- R01: Aggregate transaction value exceeds high-value threshold. "
        "(category=high_value_activity, tx_count=5)\n"
        "- R02: Majority of transfers are cross-bank. "
        "(category=cross_bank_movement, tx_count=5)\n"
    )


def test_parse_prompt_fields_extracts_core_fields() -> None:
    fields = parse_prompt_fields(_sample_user_prompt())
    assert fields.case_id == "CASE_123"
    assert fields.typology == "FUNNEL"
    assert fields.risk_score == "88"
    assert fields.risk_band == "high"
    assert len(fields.reasons) == 2
    assert fields.reasons[0]["reason_id"] == "R01"


def test_template_baseline_contains_required_sections() -> None:
    narrative = build_template_baseline(
        system_text="You are an AML investigator assistant.",
        user_text=_sample_user_prompt(),
    )
    assert "Activity Summary:" in narrative
    assert "Key Suspicious Indicators:" in narrative
    assert "Evidence and Rationale:" in narrative
    assert "Recommended Action:" in narrative
    assert "CASE_123" in narrative


def test_aml_adherence_penalizes_missing_sections() -> None:
    user_text = _sample_user_prompt()
    complete_narrative = build_template_baseline(system_text="", user_text=user_text)
    incomplete_narrative = "Case CASE_123 has suspicious activity."

    complete_score = compute_aml_adherence(user_text=user_text, narrative=complete_narrative)
    incomplete_score = compute_aml_adherence(user_text=user_text, narrative=incomplete_narrative)

    assert complete_score["aml_adherence_score"] > incomplete_score["aml_adherence_score"]
    assert complete_score["section_compliance"] > incomplete_score["section_compliance"]


def test_compute_rouge_returns_expected_shape() -> None:
    reference = build_template_baseline(system_text="", user_text=_sample_user_prompt())
    candidate = reference.replace("high risk", "elevated risk")
    scores = compute_rouge(predictions=[candidate], references=[reference])

    assert scores["available"] is True
    assert "rougeL_f1" in scores
    assert 0.0 <= float(scores["rougeL_f1"]) <= 1.0
