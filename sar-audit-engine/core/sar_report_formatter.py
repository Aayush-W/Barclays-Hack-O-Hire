from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .tx_normalizer import normalize_transactions


def _fmt_date(value: Optional[datetime]) -> str:
    if not isinstance(value, datetime):
        return ""
    return value.strftime("%Y-%m-%d")


def _date_range(transactions: List[Dict[str, Any]]) -> tuple[str, str]:
    stamps = [tx.get("timestamp") for tx in transactions if isinstance(tx.get("timestamp"), datetime)]
    if not stamps:
        return "", ""
    stamps = sorted(stamps)
    return _fmt_date(stamps[0]), _fmt_date(stamps[-1])


def _dominant_currency(transactions: List[Dict[str, Any]]) -> str:
    currencies: List[str] = []
    for tx in transactions:
        rc = str(tx.get("receiving_currency") or "").strip()
        pc = str(tx.get("payment_currency") or "").strip()
        if rc:
            currencies.append(rc)
        if pc:
            currencies.append(pc)
    if not currencies:
        return "USD"
    return Counter(currencies).most_common(1)[0][0]


def _subjects(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    source_accounts = Counter()
    destination_accounts = Counter()
    account_banks: Dict[str, set[str]] = {}

    for tx in transactions:
        from_account = str(tx.get("from_account") or "").strip()
        to_account = str(tx.get("to_account") or "").strip()
        from_bank = str(tx.get("from_bank") or "").strip()
        to_bank = str(tx.get("to_bank") or "").strip()

        if from_account:
            source_accounts[from_account] += 1
            account_banks.setdefault(from_account, set()).add(from_bank)
        if to_account:
            destination_accounts[to_account] += 1
            account_banks.setdefault(to_account, set()).add(to_bank)

    all_accounts = sorted(set(source_accounts) | set(destination_accounts))
    subjects: List[Dict[str, Any]] = []
    for account in all_accounts[:25]:
        in_sources = source_accounts.get(account, 0)
        in_dests = destination_accounts.get(account, 0)
        if in_sources > 0 and in_dests > 0:
            role = "Intermediary Account"
        elif in_sources > 0:
            role = "Sender / Originator"
        else:
            role = "Receiver / Beneficiary"

        subjects.append(
            {
                "name": account,
                "address": "Not provided",
                "account_numbers": [account],
                "role": role,
                "institution_banks": sorted([b for b in account_banks.get(account, set()) if b]),
            }
        )
    return subjects


def _flow_of_funds(transactions: List[Dict[str, Any]]) -> str:
    flow_counter = Counter()
    for tx in transactions:
        src = str(tx.get("from_account") or "UNKNOWN")
        dst = str(tx.get("to_account") or "UNKNOWN")
        flow_counter[(src, dst)] += 1
    top_flows = flow_counter.most_common(5)
    if not top_flows:
        return "No clear fund flow could be derived from provided records."
    flow_parts = [f"{src} -> {dst} (count={count})" for (src, dst), count in top_flows]
    return "; ".join(flow_parts)


def _activity_description(transactions: List[Dict[str, Any]]) -> str:
    if not transactions:
        return "No transaction activity provided."
    sorted_txs = sorted(
        transactions,
        key=lambda tx: tx.get("timestamp") or datetime.min,
    )
    lines = []
    for tx in sorted_txs[:20]:
        ts = tx.get("timestamp")
        ts_text = ts.strftime("%Y-%m-%d %H:%M") if isinstance(ts, datetime) else str(tx.get("timestamp_raw") or "")
        lines.append(
            f"{ts_text}: {tx.get('from_account')} ({tx.get('from_bank')}) -> "
            f"{tx.get('to_account')} ({tx.get('to_bank')}), amount={tx.get('amount_paid')} "
            f"{tx.get('payment_currency')}, channel={tx.get('payment_format')}."
        )
    return " ".join(lines)


def _how_occurred(transactions: List[Dict[str, Any]], reasons: List[Dict[str, Any]]) -> str:
    channels = sorted({str(tx.get("payment_format") or "").strip() for tx in transactions if tx.get("payment_format")})
    categories = sorted({str(reason.get("category") or "").strip() for reason in reasons if reason.get("category")})
    channel_text = ", ".join(channels) if channels else "unspecified channels"
    category_text = ", ".join(categories) if categories else "no flagged typology methods"
    return f"Observed through channels: {channel_text}. Reason categories: {category_text}."


def _why_suspicious(reasons: List[Dict[str, Any]], risk: Dict[str, Any]) -> str:
    if not reasons:
        return "Insufficient suspicious indicators in current review window."
    claims = [str(reason.get("claim") or "").strip() for reason in reasons if reason.get("claim")]
    claim_text = "; ".join(claims[:6])
    return (
        f"Risk score {risk.get('score')} ({risk.get('band')}) based on flagged indicators. "
        f"Key rationale: {claim_text}."
    )


def build_sar_report_json(
    case: Dict[str, Any],
    evidence_map: Dict[str, Any],
    narrative_text: str,
    filing_institution_id: str = "INSTITUTION_UNKNOWN",
    associated_sar_references: Optional[List[str]] = None,
) -> Dict[str, Any]:
    transactions = normalize_transactions(case)
    risk = evidence_map.get("risk_assessment", {}) or {}
    reasons = evidence_map.get("reasons", []) or []

    start_date, end_date = _date_range(transactions)
    filing_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    total_amount = round(
        sum(float(tx.get("amount_paid") or 0.0) for tx in transactions),
        2,
    )
    branches = sorted(
        {
            str(tx.get("from_bank") or "").strip()
            for tx in transactions
            if str(tx.get("from_bank") or "").strip()
        }
        | {
            str(tx.get("to_bank") or "").strip()
            for tx in transactions
            if str(tx.get("to_bank") or "").strip()
        }
    )
    red_flags = [str(reason.get("claim") or "").strip() for reason in reasons if reason.get("claim")]
    if not red_flags:
        red_flags = ["No explicit suspicious indicators met configured thresholds."]

    escalation_recommended = bool(risk.get("escalation_recommended"))
    suspicion_type = str(evidence_map.get("typology") or "NO_CLEAR_LAUNDERING_PATTERN")
    if not escalation_recommended:
        suspicion_type = "NO_CLEAR_LAUNDERING_PATTERN"

    purpose = (
        "Filing submitted due to suspicious activity indicators consistent with potential money laundering."
        if escalation_recommended
        else "Internal monitoring record: activity reviewed, but SAR escalation not recommended at current thresholds."
    )

    report = {
        "sar_report": {
            "report_meta": {
                "filing_institution_id": str(filing_institution_id or "INSTITUTION_UNKNOWN"),
                "date_filed": filing_date,
                "case_reference_number": str(case.get("case_id") or ""),
                "associated_sar_references": associated_sar_references or [],
            },
            "narrative": {
                "part_1_introduction": {
                    "purpose": purpose,
                    "subject_summary": (
                        f"Case includes {len(_subjects(transactions))} identified subjects/accounts linked to the institution."
                    ),
                    "date_of_initial_detection": start_date,
                    "suspicion_type": suspicion_type,
                },
                "part_2_body": {
                    "subjects": _subjects(transactions),
                    "transaction_details": {
                        "total_amount": total_amount,
                        "currency": _dominant_currency(transactions),
                        "date_range": f"{start_date} to {end_date}" if start_date and end_date else "",
                        "activity_description": _activity_description(transactions),
                        "flow_of_funds": _flow_of_funds(transactions),
                        "branches_involved": branches,
                    },
                    "red_flags": red_flags,
                    "how_occurred": _how_occurred(transactions, reasons),
                    "why_suspicious": _why_suspicious(reasons, risk),
                },
                "part_3_conclusion": {
                    "summary": narrative_text,
                    "follow_up_action": (
                        "Escalate for SAR filing and enhanced due diligence."
                        if escalation_recommended
                        else "No SAR filing at this time; continue routine monitoring."
                    ),
                    "investigator_notes": (
                        f"Risk score={risk.get('score')}, band={risk.get('band')}, "
                        f"escalation_recommended={escalation_recommended}."
                    ),
                },
            },
            "supporting_documents": [
                {
                    "document_type": "evidence_map",
                    "file_reference": f"data/processed/evidence_maps/{case.get('case_id')}.json",
                },
                {
                    "document_type": "reasoning_trace",
                    "file_reference": f"data/processed/reasoning/{case.get('case_id')}.json",
                },
            ],
        }
    }
    return report
