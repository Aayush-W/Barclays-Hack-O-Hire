from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .tx_normalizer import normalize_transactions
from config.settings import PipelineSettings


def _account_key(bank: Optional[str], account: Optional[str]) -> Optional[str]:
    if bank and account:
        return f"{bank}|{account}"
    if account:
        return str(account)
    if bank:
        return str(bank)
    return None


def _tx_ids(transactions: List[Dict[str, Any]]) -> List[str]:
    return [tx["tx_id"] for tx in transactions]


def _tx_ids_cross_bank(transactions: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for tx in transactions:
        if tx.get("from_bank") and tx.get("to_bank") and tx.get("from_bank") != tx.get("to_bank"):
            ids.append(tx["tx_id"])
    return ids


def _tx_ids_multi_currency(transactions: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for tx in transactions:
        rc = tx.get("receiving_currency")
        pc = tx.get("payment_currency")
        if rc and pc and rc != pc:
            ids.append(tx["tx_id"])
    return ids


def _rapid_pass_through_accounts(
    transactions: List[Dict[str, Any]], threshold_hours: int
) -> Tuple[bool, Optional[float], List[str]]:
    incoming: Dict[str, List[datetime]] = {}
    outgoing: Dict[str, List[datetime]] = {}

    for tx in transactions:
        ts = tx.get("timestamp")
        if not isinstance(ts, datetime):
            continue

        from_key = _account_key(tx.get("from_bank"), tx.get("from_account"))
        to_key = _account_key(tx.get("to_bank"), tx.get("to_account"))
        if to_key:
            incoming.setdefault(to_key, []).append(ts)
        if from_key:
            outgoing.setdefault(from_key, []).append(ts)

    min_delta_hours: Optional[float] = None
    flagged_accounts = set()

    for account in set(incoming).intersection(outgoing):
        ins = sorted(incoming[account])
        outs = sorted(outgoing[account])
        i = 0
        for out_ts in outs:
            while i < len(ins) and ins[i] <= out_ts:
                i += 1
            if i == 0:
                continue
            in_ts = ins[i - 1]
            delta_hours = (out_ts - in_ts).total_seconds() / 3600
            if delta_hours < 0:
                continue
            if delta_hours <= threshold_hours:
                flagged_accounts.add(account)
            if min_delta_hours is None or delta_hours < min_delta_hours:
                min_delta_hours = delta_hours

    if min_delta_hours is None:
        return False, None, []

    tx_ids = []
    if flagged_accounts:
        for tx in transactions:
            from_key = _account_key(tx.get("from_bank"), tx.get("from_account"))
            to_key = _account_key(tx.get("to_bank"), tx.get("to_account"))
            if from_key in flagged_accounts or to_key in flagged_accounts:
                tx_ids.append(tx["tx_id"])

    return min_delta_hours <= threshold_hours, round(min_delta_hours, 2), tx_ids


def build_reasoning(
    case: Dict[str, Any],
    signals_payload: Dict[str, Any],
    settings: PipelineSettings,
) -> Dict[str, Any]:
    transactions = normalize_transactions(case)
    tx_ids_all = _tx_ids(transactions)

    signals = signals_payload.get("signals", signals_payload)
    typology = case.get("typology") or signals_payload.get("typology")

    reasoning: List[Dict[str, Any]] = []
    step_counter = 1

    def add_step(step: str, evidence: Dict[str, Any], tx_ids: List[str]) -> None:
        nonlocal step_counter
        reasoning.append(
            {
                "step_id": f"R{step_counter:02}",
                "step": step,
                "evidence": evidence,
                "transaction_ids": tx_ids,
            }
        )
        step_counter += 1

    if typology:
        add_step(
            f"Case typology labeled as {typology} based on pattern parser classification.",
            {"typology": typology, "source": "pattern_parser"},
            tx_ids_all,
        )

    total_inflow = signals.get("total_inflow") or 0.0
    total_outflow = signals.get("total_outflow") or 0.0
    net_flow = signals.get("net_flow") or 0.0

    if total_inflow >= settings.high_value_threshold or total_outflow >= settings.high_value_threshold:
        add_step(
            "Aggregate transaction value exceeds high-value threshold.",
            {
                "total_inflow": total_inflow,
                "total_outflow": total_outflow,
                "threshold": settings.high_value_threshold,
            },
            tx_ids_all,
        )

    if total_inflow:
        net_ratio = abs(net_flow) / total_inflow
        if net_ratio <= settings.net_flow_ratio_threshold:
            add_step(
                "Net flow is near zero relative to total inflow, indicating potential pass-through.",
                {
                    "net_flow": net_flow,
                    "total_inflow": total_inflow,
                    "ratio": round(net_ratio, 4),
                    "threshold": settings.net_flow_ratio_threshold,
                },
                tx_ids_all,
            )

    hop_count = signals.get("hop_count") or 0
    if hop_count >= settings.multi_hop_threshold:
        add_step(
            f"Funds traversed {hop_count} hops across accounts.",
            {"hop_count": hop_count, "threshold": settings.multi_hop_threshold},
            tx_ids_all,
        )

    cross_bank_ratio = signals.get("cross_bank_ratio") or 0.0
    if cross_bank_ratio >= settings.cross_bank_ratio_threshold:
        add_step(
            "Majority of transfers are cross-bank.",
            {"cross_bank_ratio": cross_bank_ratio, "threshold": settings.cross_bank_ratio_threshold},
            _tx_ids_cross_bank(transactions) or tx_ids_all,
        )

    unique_senders = signals.get("unique_senders") or 0
    unique_receivers = signals.get("unique_receivers") or 0
    if unique_senders >= settings.multi_party_threshold or unique_receivers >= settings.multi_party_threshold:
        add_step(
            "Multiple originators or beneficiaries are involved.",
            {
                "unique_senders": unique_senders,
                "unique_receivers": unique_receivers,
                "threshold": settings.multi_party_threshold,
            },
            tx_ids_all,
        )

    if signals.get("multi_currency_flag"):
        add_step(
            "Multiple currencies appear within the transaction set.",
            {"multi_currency_flag": True},
            _tx_ids_multi_currency(transactions) or tx_ids_all,
        )

    pt_hours = settings.pass_through_hours_default
    rapid_flag, min_hours, rapid_tx_ids = _rapid_pass_through_accounts(transactions, pt_hours)
    if rapid_flag:
        add_step(
            "Rapid pass-through activity observed.",
            {"min_dwell_hours": min_hours, "threshold_hours": pt_hours},
            rapid_tx_ids or tx_ids_all,
        )

    velocity = signals.get("transaction_velocity_metrics") or {}
    tx_per_day = velocity.get("tx_per_day")
    max_tx_in_1h = velocity.get("max_tx_in_1h")
    if (tx_per_day is not None and tx_per_day >= settings.velocity_tx_per_day_threshold) or (
        max_tx_in_1h is not None and max_tx_in_1h >= settings.velocity_max_tx_1h_threshold
    ):
        add_step(
            "Transaction velocity exceeds monitoring thresholds.",
            {
                "tx_per_day": tx_per_day,
                "max_tx_in_1h": max_tx_in_1h,
                "tx_per_day_threshold": settings.velocity_tx_per_day_threshold,
                "max_tx_in_1h_threshold": settings.velocity_max_tx_1h_threshold,
            },
            tx_ids_all,
        )

    return {
        "case_id": case.get("case_id"),
        "typology": typology,
        "reasoning": reasoning,
    }
