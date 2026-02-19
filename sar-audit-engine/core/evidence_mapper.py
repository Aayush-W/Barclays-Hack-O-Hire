from __future__ import annotations

from collections import Counter
from datetime import datetime
import re
from statistics import mean
from typing import Any, Dict, List, Optional, Protocol

from .tx_normalizer import normalize_transactions


class Retriever(Protocol):
    def retrieve(
        self, query: str, top_k: int = 5, min_score: float = 0.0, source_type: Optional[str] = None
    ) -> List[Dict]:
        ...


def _reason_category(step_text: str) -> str:
    text = step_text.lower()
    if "high-value" in text:
        return "high_value_activity"
    if "pass-through" in text:
        return "rapid_pass_through"
    if "cross-bank" in text:
        return "cross_bank_movement"
    if "hop" in text:
        return "multi_hop_layering"
    if "velocity" in text:
        return "high_velocity"
    if "currenc" in text:
        return "multi_currency_activity"
    if "originators" in text or "beneficiaries" in text:
        return "multi_party_network"
    if "typology" in text:
        return "typology_classification"
    return "general_suspicion"


def _metric_tokens(signals: Dict[str, Any]) -> str:
    velocity = signals.get("transaction_velocity_metrics", {})
    metric_parts = [
        f"cross_bank_ratio {round(float(signals.get('cross_bank_ratio') or 0.0), 3)}",
        f"hop_count {int(signals.get('hop_count') or 0)}",
        f"multi_currency {bool(signals.get('multi_currency_flag') or False)}",
        f"rapid_pass_through {bool(signals.get('rapid_pass_through_flag') or False)}",
        f"max_tx_1h {int(velocity.get('max_tx_in_1h') or 0)}",
        f"tx_per_day {round(float(velocity.get('tx_per_day') or 0.0), 3)}",
    ]
    return " ".join(metric_parts)


def _build_reason_query(typology: Optional[str], step: Dict[str, Any], signals: Dict[str, Any]) -> str:
    step_text = step.get("step", "")
    evidence = step.get("evidence", {})
    reason_category = _reason_category(step_text)
    evidence_tokens = " ".join(f"{key} {value}" for key, value in evidence.items())
    metrics = _metric_tokens(signals)
    return f"SAR {typology or ''} category {reason_category} {step_text} {evidence_tokens} {metrics}".strip()


PREFERRED_SOURCE_BY_CATEGORY = {
    "typology_classification": "typologies",
    "rapid_pass_through": "typologies",
    "cross_bank_movement": "typologies",
    "multi_hop_layering": "typologies",
    "multi_party_network": "typologies",
    "multi_currency_activity": "typologies",
    "high_velocity": "typologies",
    "high_value_activity": "regulations",
    "general_suspicion": "regulations",
}


def _needs_regulatory_context(reason_category: str) -> bool:
    return reason_category in {"high_value_activity", "general_suspicion", "rapid_pass_through"}


def _merge_contexts(primary: List[Dict], fallback: List[Dict], top_k: int) -> List[Dict]:
    merged = []
    seen = set()
    for context in primary + fallback:
        chunk_id = context.get("chunk_id")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(context)
        if len(merged) >= top_k:
            break
    return merged


def _ensure_source_in_contexts(
    contexts: List[Dict], fallback_contexts: List[Dict], required_source: str, top_k: int
) -> List[Dict]:
    if any(context.get("metadata", {}).get("source_type") == required_source for context in contexts):
        return contexts[:top_k]

    if not fallback_contexts:
        return contexts[:top_k]

    chosen_fallback = fallback_contexts[0]
    if top_k <= 1:
        return [chosen_fallback]

    retained = contexts[: max(top_k - 1, 0)]
    merged = _merge_contexts(retained, [chosen_fallback], top_k=top_k)
    if len(merged) < top_k:
        merged = _merge_contexts(merged, contexts, top_k=top_k)
    return merged


def _select_primary_source_type(reason_category: str, positives: List[Dict[str, Any]]) -> Optional[str]:
    if not positives:
        return None
    preferred = PREFERRED_SOURCE_BY_CATEGORY.get(reason_category)
    if preferred and any(context.get("source_type") == preferred for context in positives):
        return preferred
    return positives[0].get("source_type")


def _transaction_snapshot(tx: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = tx.get("timestamp")
    timestamp_text = timestamp.isoformat(sep=" ") if isinstance(timestamp, datetime) else tx.get("timestamp_raw")
    return {
        "tx_id": tx.get("tx_id"),
        "timestamp": timestamp_text,
        "from_bank": tx.get("from_bank"),
        "from_account": tx.get("from_account"),
        "to_bank": tx.get("to_bank"),
        "to_account": tx.get("to_account"),
        "amount_received": tx.get("amount_received"),
        "amount_paid": tx.get("amount_paid"),
        "receiving_currency": tx.get("receiving_currency"),
        "payment_currency": tx.get("payment_currency"),
        "payment_format": tx.get("payment_format"),
    }


def _contains_any(text: str, patterns: List[str]) -> bool:
    normalized = (text or "").upper()
    return any(pattern in normalized for pattern in patterns)


def _benign_pattern_score(transactions: List[Dict[str, Any]], signals: Dict[str, Any]) -> float:
    if not transactions:
        return 0.0

    from_accounts = [str(tx.get("from_account") or "") for tx in transactions]
    to_accounts = [str(tx.get("to_account") or "") for tx in transactions]
    from_counter = Counter(from_accounts)
    dominant_from_ratio = max(from_counter.values()) / max(len(transactions), 1)

    employer_like = any(_contains_any(account, ["EMPLOYER", "PAYROLL", "SALARY"]) for account in from_accounts)
    employee_like_targets = sum(1 for account in to_accounts if re.match(r"^EMP\d+$", account.upper()))

    consumer_keywords = [
        "GROCERY",
        "UTILITY",
        "AMAZON",
        "RETAIL",
        "INSURANCE",
        "LANDLORD",
        "RENT",
        "LOAN",
        "TAX",
    ]
    consumer_counterparties = sum(1 for account in to_accounts if _contains_any(account, consumer_keywords))

    velocity = signals.get("transaction_velocity_metrics", {}) or {}
    tx_per_day = float(velocity.get("tx_per_day") or 0.0)
    max_tx_in_1h = int(velocity.get("max_tx_in_1h") or 0)

    benign_score = 0.0
    if employer_like and employee_like_targets >= 3:
        benign_score += 18.0
    elif dominant_from_ratio >= 0.55 and employee_like_targets >= 3:
        benign_score += 10.0

    if consumer_counterparties >= 3:
        benign_score += 8.0
    if (
        not bool(signals.get("rapid_pass_through_flag"))
        and int(signals.get("hop_count") or 0) <= 2
        and not bool(signals.get("multi_currency_flag"))
    ):
        benign_score += 10.0
    if tx_per_day <= 3.0 and max_tx_in_1h <= 2:
        benign_score += 6.0

    return benign_score


def _risk_score(signals: Dict[str, Any], reasons: List[Dict[str, Any]], transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    reason_weights = {
        "rapid_pass_through": 16.0,
        "multi_hop_layering": 16.0,
        "high_velocity": 10.0,
        "multi_currency_activity": 10.0,
        "cross_bank_movement": 6.0,
        "multi_party_network": 5.0,
        "high_value_activity": 4.0,
        "general_suspicion": 4.0,
        "typology_classification": 0.0,
    }
    score = sum(reason_weights.get(reason.get("category", ""), 3.0) for reason in reasons)

    rapid_flag = bool(signals.get("rapid_pass_through_flag"))
    hop_count = int(signals.get("hop_count") or 0)
    cross_bank_ratio = float(signals.get("cross_bank_ratio") or 0.0)
    multi_currency_flag = bool(signals.get("multi_currency_flag"))

    if rapid_flag:
        score += 20.0
    if hop_count >= 4:
        score += 14.0
    elif hop_count >= 3:
        score += 8.0
    if multi_currency_flag:
        score += 8.0

    velocity = signals.get("transaction_velocity_metrics", {})
    tx_per_day = float(velocity.get("tx_per_day") or 0.0)
    max_tx_in_1h = int(velocity.get("max_tx_in_1h") or 0)
    velocity_flag = tx_per_day >= 8.0 or max_tx_in_1h >= 4
    if velocity_flag:
        score += 8.0

    if cross_bank_ratio >= 0.8 and (rapid_flag or hop_count >= 3 or multi_currency_flag):
        score += 7.0

    strong_signal_present = rapid_flag or hop_count >= 3 or multi_currency_flag or velocity_flag
    if not strong_signal_present:
        score = min(score, 35.0)

    benign_adjustment = _benign_pattern_score(transactions=transactions, signals=signals)
    score -= benign_adjustment
    score = min(max(score, 0.0), 100.0)

    if score >= 70:
        band = "high"
    elif score >= 40:
        band = "medium"
    else:
        band = "low"

    escalation_recommended = bool(score >= 70 or (score >= 55 and strong_signal_present))
    return {
        "score": int(round(score)),
        "band": band,
        "escalation_recommended": escalation_recommended,
        "strong_signal_present": strong_signal_present,
        "benign_adjustment": round(float(benign_adjustment), 3),
    }


def map_case_evidence(
    case: Dict[str, Any],
    signals_payload: Dict[str, Any],
    reasoning_payload: Dict[str, Any],
    retriever: Optional[Retriever] = None,
    top_k_per_reason: int = 3,
) -> Dict[str, Any]:
    normalized_transactions = normalize_transactions(case)
    tx_by_id = {tx["tx_id"]: tx for tx in normalized_transactions}
    signals = signals_payload.get("signals", signals_payload)
    reasoning_steps = reasoning_payload.get("reasoning", [])

    mapped_reasons = []
    retrieval_scores = []

    for step in reasoning_steps:
        reason_category = _reason_category(step.get("step", ""))
        tx_ids = [str(tx_id) for tx_id in step.get("transaction_ids", [])]
        tx_snippets = []
        for tx_id in tx_ids:
            tx = tx_by_id.get(tx_id)
            if tx:
                tx_snippets.append(_transaction_snapshot(tx))

        knowledge_support = []
        query = _build_reason_query(case.get("typology"), step, signals=signals)
        if retriever:
            contexts = retriever.retrieve(query=query, top_k=top_k_per_reason, min_score=0.0)
            if _needs_regulatory_context(reason_category) and not any(
                context.get("metadata", {}).get("source_type") == "regulations" for context in contexts
            ):
                regulatory_query = f"{query} reporting requirements suspicious activity narrative guidelines"
                regulatory_contexts = retriever.retrieve(
                    query=regulatory_query,
                    top_k=1,
                    min_score=0.0,
                    source_type="regulations",
                )
                contexts = _ensure_source_in_contexts(
                    contexts=contexts,
                    fallback_contexts=regulatory_contexts,
                    required_source="regulations",
                    top_k=top_k_per_reason,
                )

            for context in contexts:
                retrieval_scores.append(float(context.get("score", 0.0)))
                knowledge_support.append(
                    {
                        "chunk_id": context.get("chunk_id"),
                        "score": context.get("score"),
                        "base_score": context.get("base_score"),
                        "source_model_probability": context.get("source_model_probability"),
                        "source_file": context.get("metadata", {}).get("source_file"),
                        "source_type": context.get("metadata", {}).get("source_type"),
                        "excerpt": context.get("text"),
                    }
                )

        mapped_reasons.append(
            {
                "reason_id": step.get("step_id"),
                "category": reason_category,
                "claim": step.get("step"),
                "retrieval_query": query,
                "evidence_metrics": step.get("evidence", {}),
                "transaction_ids": tx_ids,
                "transaction_examples": tx_snippets,
                "knowledge_support": knowledge_support,
            }
        )

    risk = _risk_score(signals=signals, reasons=mapped_reasons, transactions=normalized_transactions)

    return {
        "case_id": case.get("case_id"),
        "typology": case.get("typology"),
        "risk_assessment": {
            **risk,
            "retrieval_confidence": round(mean(retrieval_scores), 6) if retrieval_scores else 0.0,
        },
        "signals": signals,
        "reasons": mapped_reasons,
        "summary": {
            "transaction_count": len(normalized_transactions),
            "reason_count": len(mapped_reasons),
        },
    }


def training_examples_from_evidence(
    evidence_map: Dict[str, Any], target_narrative: str = "", max_contexts: int = 3
) -> List[Dict[str, Any]]:
    examples = []
    typology = evidence_map.get("typology")
    case_id = evidence_map.get("case_id")
    signals = evidence_map.get("signals", {})
    risk = evidence_map.get("risk_assessment", {})
    velocity = signals.get("transaction_velocity_metrics", {})

    for reason in evidence_map.get("reasons", []):
        reason_category = reason.get("category") or "general_suspicion"
        contexts = reason.get("knowledge_support", [])
        if not contexts:
            continue

        positives = []
        for context in contexts[:max_contexts]:
            positives.append(
                {
                    "chunk_id": context.get("chunk_id"),
                    "source_file": context.get("source_file"),
                    "source_type": context.get("source_type"),
                    "score": context.get("score"),
                    "base_score": context.get("base_score"),
                    "source_model_probability": context.get("source_model_probability"),
                    "text": context.get("excerpt"),
                }
            )

        primary_source_type = _select_primary_source_type(reason_category=reason_category, positives=positives)

        examples.append(
            {
                "case_id": case_id,
                "typology": typology,
                "reason_id": reason.get("reason_id"),
                "reason_category": reason.get("category"),
                "query": reason.get("retrieval_query")
                or f"SAR rationale for {typology}: {reason.get('claim')}",
                "primary_source_type": primary_source_type,
                "metrics": {
                    "risk_score": risk.get("score"),
                    "risk_band": risk.get("band"),
                    "cross_bank_ratio": signals.get("cross_bank_ratio"),
                    "hop_count": signals.get("hop_count"),
                    "multi_currency_flag": signals.get("multi_currency_flag"),
                    "rapid_pass_through_flag": signals.get("rapid_pass_through_flag"),
                    "max_tx_in_1h": velocity.get("max_tx_in_1h"),
                    "tx_per_day": velocity.get("tx_per_day"),
                },
                "positive_contexts": positives,
                "target_narrative": target_narrative,
            }
        )

    return examples
