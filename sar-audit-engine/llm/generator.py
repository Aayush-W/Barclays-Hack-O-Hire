from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .prompt_templates import build_prompt_payload


def _top_sources(reasons: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    seen = set()
    output = []
    for reason in reasons:
        for context in reason.get("knowledge_support", []):
            source = context.get("source_file")
            if source and source not in seen:
                seen.add(source)
                output.append(source)
                if len(output) >= limit:
                    return output
    return output


def _build_reason_bullets(reasons: List[Dict[str, Any]]) -> str:
    if not reasons:
        return "- No explicit suspicious indicators were generated."

    lines = []
    for reason in reasons:
        claim = reason.get("claim")
        category = reason.get("category")
        tx_count = len(reason.get("transaction_ids", []))
        lines.append(f"- {claim} (category: {category}, linked_tx: {tx_count})")
    return "\n".join(lines)


def generate_sar_narrative(evidence_map: Dict[str, Any], style: str = "standard") -> Dict[str, Any]:
    reasons = evidence_map.get("reasons", [])
    risk = evidence_map.get("risk_assessment", {})
    case_id = evidence_map.get("case_id")
    typology = evidence_map.get("typology")

    prompt_payload = build_prompt_payload(evidence_map=evidence_map, style=style)
    sources = _top_sources(reasons)
    reason_bullets = _build_reason_bullets(reasons)

    narrative = (
        f"Activity Summary:\n"
        f"Case {case_id} was reviewed under typology {typology}. "
        f"The case is currently assessed as {risk.get('band', 'unknown')} risk "
        f"with score {risk.get('score', 'n/a')}.\n\n"
        "Key Suspicious Indicators:\n"
        f"{reason_bullets}\n\n"
        "Evidence and Rationale:\n"
        "The risk assessment combines transactional signals with explicit reasoning steps. "
        "Each reason is linked to transaction identifiers and supporting knowledge chunks "
        "retrieved from SAR typology/regulatory reference material.\n\n"
        "Recommended Action:\n"
        "Escalate for SAR filing review, preserve an audit trail of supporting evidence, "
        "and request analyst sign-off for final narrative submission."
    )

    return {
        "case_id": case_id,
        "style": style,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_payload": prompt_payload,
        "narrative": narrative,
        "citations": sources,
    }
