from __future__ import annotations

from typing import Any, Dict, List

SYSTEM_PROMPT = """You are an AML investigator assistant.
Write a factual, audit-ready SAR narrative.
Use only provided evidence and retrieved knowledge context.
Do not invent transactions, regulations, or account entities."""

STYLE_GUIDANCE = {
    "standard": (
        "Write in full paragraphs with clear timeline, suspicious pattern rationale, and "
        "a final recommendation section."
    ),
    "condensed": (
        "Write concise sections suitable for analyst review with minimal repetition."
    ),
}


def _reason_lines(reasons: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for reason in reasons:
        lines.append(
            f"- {reason.get('reason_id')}: {reason.get('claim')} "
            f"(category={reason.get('category')}, tx_count={len(reason.get('transaction_ids', []))})"
        )
    return lines


def build_user_prompt(evidence_map: Dict[str, Any], style: str = "standard") -> str:
    style_instructions = STYLE_GUIDANCE.get(style, STYLE_GUIDANCE["standard"])
    reasons = evidence_map.get("reasons", [])
    reason_block = "\n".join(_reason_lines(reasons)) if reasons else "- No explicit reasons available."

    return (
        f"Style instructions: {style_instructions}\n\n"
        f"Case ID: {evidence_map.get('case_id')}\n"
        f"Typology: {evidence_map.get('typology')}\n"
        f"Risk score: {evidence_map.get('risk_assessment', {}).get('score')}\n"
        f"Risk band: {evidence_map.get('risk_assessment', {}).get('band')}\n\n"
        f"Reasons:\n{reason_block}\n\n"
        "Generate a SAR narrative with sections:\n"
        "1) Activity Summary\n2) Key Suspicious Indicators\n3) Evidence and Rationale\n4) Recommended Action"
    )


def build_prompt_payload(evidence_map: Dict[str, Any], style: str = "standard") -> Dict[str, str]:
    return {"system": SYSTEM_PROMPT, "user": build_user_prompt(evidence_map=evidence_map, style=style)}
