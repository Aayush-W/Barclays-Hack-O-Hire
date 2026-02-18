from __future__ import annotations

from datetime import datetime, timedelta
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from .tx_normalizer import normalize_transactions

PASS_THROUGH_HOURS_DEFAULT = 24
MAX_PATH_NODES = 80


def _account_key(bank: Optional[str], account: Optional[str]) -> Optional[str]:
    if bank and account:
        return f"{bank}|{account}"
    if account:
        return str(account)
    if bank:
        return str(bank)
    return None


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(value, digits)


def _estimate_hop_count(transactions: List[Dict[str, Any]]) -> int:
    edges: Dict[str, List[str]] = {}
    nodes = set()

    for tx in transactions:
        from_key = _account_key(tx.get("from_bank"), tx.get("from_account"))
        to_key = _account_key(tx.get("to_bank"), tx.get("to_account"))
        if not from_key or not to_key:
            continue
        nodes.add(from_key)
        nodes.add(to_key)
        edges.setdefault(from_key, []).append(to_key)

    if not nodes:
        return 0

    if len(nodes) > MAX_PATH_NODES:
        return len(transactions)

    def dfs(node: str, visited: set) -> int:
        max_len = 0
        for nxt in edges.get(node, []):
            if nxt in visited:
                continue
            max_len = max(max_len, 1 + dfs(nxt, visited | {nxt}))
        return max_len

    return max(dfs(node, {node}) for node in nodes)


def _velocity_metrics(timestamps: List[datetime], tx_count: int) -> Dict[str, Optional[float]]:
    if not timestamps:
        return {
            "tx_count": tx_count,
            "start_time": None,
            "end_time": None,
            "duration_hours": None,
            "tx_per_day": None,
            "avg_interarrival_minutes": None,
            "median_interarrival_minutes": None,
            "max_tx_in_1h": None,
        }

    timestamps = sorted(timestamps)
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
    duration_hours = duration_seconds / 3600 if duration_seconds else 0.0
    duration_days = duration_seconds / 86400 if duration_seconds else 0.0

    interarrivals = []
    for prev, curr in zip(timestamps, timestamps[1:]):
        interarrivals.append((curr - prev).total_seconds() / 60)

    max_tx_in_1h = 1
    left = 0
    for right in range(len(timestamps)):
        while (timestamps[right] - timestamps[left]).total_seconds() > 3600:
            left += 1
        max_tx_in_1h = max(max_tx_in_1h, right - left + 1)

    return {
        "tx_count": tx_count,
        "start_time": start_time.isoformat(sep=" "),
        "end_time": end_time.isoformat(sep=" "),
        "duration_hours": _round(duration_hours, 2),
        "tx_per_day": _round(tx_count / duration_days, 4) if duration_days else float(tx_count),
        "avg_interarrival_minutes": _round(mean(interarrivals), 2) if interarrivals else None,
        "median_interarrival_minutes": _round(median(interarrivals), 2) if interarrivals else None,
        "max_tx_in_1h": max_tx_in_1h,
    }


def _rapid_pass_through(
    transactions: List[Dict[str, Any]], threshold_hours: int
) -> Tuple[bool, Optional[float]]:
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
            if min_delta_hours is None or delta_hours < min_delta_hours:
                min_delta_hours = delta_hours

    if min_delta_hours is None:
        return False, None

    return min_delta_hours <= threshold_hours, _round(min_delta_hours, 2)


def extract_signals(case: Dict[str, Any], pass_through_hours: int = PASS_THROUGH_HOURS_DEFAULT) -> Dict[str, Any]:
    transactions = normalize_transactions(case)

    tx_count = len(transactions)
    total_inflow = sum(tx.get("amount_received") or 0.0 for tx in transactions)
    total_outflow = sum(tx.get("amount_paid") or 0.0 for tx in transactions)
    net_flow = total_inflow - total_outflow

    senders = set()
    receivers = set()
    cross_bank_count = 0
    currencies = set()
    timestamps: List[datetime] = []

    for tx in transactions:
        from_key = _account_key(tx.get("from_bank"), tx.get("from_account"))
        to_key = _account_key(tx.get("to_bank"), tx.get("to_account"))
        if from_key:
            senders.add(from_key)
        if to_key:
            receivers.add(to_key)

        if tx.get("from_bank") and tx.get("to_bank") and tx.get("from_bank") != tx.get("to_bank"):
            cross_bank_count += 1

        if tx.get("receiving_currency"):
            currencies.add(str(tx.get("receiving_currency")))
        if tx.get("payment_currency"):
            currencies.add(str(tx.get("payment_currency")))

        ts = tx.get("timestamp")
        if isinstance(ts, datetime):
            timestamps.append(ts)

    cross_bank_ratio = cross_bank_count / tx_count if tx_count else 0.0
    multi_currency_flag = len(currencies) > 1
    hop_count = _estimate_hop_count(transactions)
    rapid_flag, rapid_min_hours = _rapid_pass_through(transactions, pass_through_hours)

    velocity = _velocity_metrics(timestamps, tx_count)

    signals = {
        "total_inflow": _round(total_inflow, 2),
        "total_outflow": _round(total_outflow, 2),
        "net_flow": _round(net_flow, 2),
        "unique_senders": len(senders),
        "unique_receivers": len(receivers),
        "hop_count": hop_count,
        "cross_bank_ratio": _round(cross_bank_ratio, 4),
        "multi_currency_flag": multi_currency_flag,
        "rapid_pass_through_flag": rapid_flag,
        "rapid_pass_through_min_hours": rapid_min_hours,
        "transaction_velocity_metrics": velocity,
    }

    return {
        "case_id": case.get("case_id"),
        "typology": case.get("typology"),
        "signals": signals,
    }
