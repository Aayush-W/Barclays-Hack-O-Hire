from __future__ import annotations

import csv
import io
import json
import re
from typing import Any, Dict, List


_EXPECTED_FIELDS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
]

_ALIASES = {
    "timestamp": {"timestamp", "time", "datetime"},
    "from_bank": {"from_bank", "from bank", "sender_bank", "origin_bank"},
    "from_account": {"from_account", "from account", "sender_account", "origin_account"},
    "to_bank": {"to_bank", "to bank", "receiver_bank", "beneficiary_bank"},
    "to_account": {"to_account", "to account", "receiver_account", "beneficiary_account"},
    "amount_received": {"amount_received", "amount received", "amount_in"},
    "receiving_currency": {"receiving_currency", "receiving currency", "currency_in"},
    "amount_paid": {"amount_paid", "amount paid", "amount_out"},
    "payment_currency": {"payment_currency", "payment currency", "currency_out"},
    "payment_format": {"payment_format", "payment format", "channel"},
}


def _normalize_header(header: str) -> str:
    return re.sub(r"\s+", " ", str(header or "").strip().lower())


def _coerce_float(value: Any) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _canonical_key(raw_key: str) -> str | None:
    normalized = _normalize_header(raw_key)
    for canonical, aliases in _ALIASES.items():
        if normalized in aliases:
            return canonical
    return None


def _normalize_tx(raw_tx: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in (raw_tx or {}).items():
        key = _canonical_key(str(raw_key))
        if not key:
            continue
        normalized[key] = value

    tx = {field: normalized.get(field, "") for field in _EXPECTED_FIELDS}
    tx["amount_received"] = _coerce_float(tx.get("amount_received"))
    tx["amount_paid"] = _coerce_float(tx.get("amount_paid"))
    return tx


def _parse_json_payload(text: str) -> List[Dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, dict):
        transactions = payload.get("transactions", [])
    elif isinstance(payload, list):
        transactions = payload
    else:
        transactions = []
    if not isinstance(transactions, list):
        return []
    return [_normalize_tx(item) for item in transactions if isinstance(item, dict)]


def _parse_csv_rows(text: str) -> List[Dict[str, Any]]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("BEGIN LAUNDERING ATTEMPT"):
            continue
        if stripped.upper().startswith("END LAUNDERING ATTEMPT"):
            continue
        lines.append(stripped)
    if not lines:
        return []

    has_header = any(_canonical_key(part) for part in re.split(r"[,|\t]", lines[0]))
    delimiter = "\t" if ("\t" in lines[0] and "," not in lines[0]) else ","
    joined = "\n".join(lines)

    if has_header:
        reader = csv.DictReader(io.StringIO(joined), delimiter=delimiter)
        return [_normalize_tx(row) for row in reader]

    reader = csv.reader(io.StringIO(joined), delimiter=delimiter)
    records: List[Dict[str, Any]] = []
    for row in reader:
        if len(row) < len(_EXPECTED_FIELDS):
            continue
        row = row[: len(_EXPECTED_FIELDS)]
        mapped = {field: value for field, value in zip(_EXPECTED_FIELDS, row)}
        records.append(_normalize_tx(mapped))
    return records


def parse_audit_trail_text(text: str) -> List[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Audit trail input is empty.")

    transactions: List[Dict[str, Any]] = []
    try:
        transactions = _parse_json_payload(raw)
    except json.JSONDecodeError:
        transactions = _parse_csv_rows(raw)

    cleaned = [tx for tx in transactions if tx.get("timestamp")]
    if not cleaned:
        raise ValueError(
            "Could not parse transactions. Provide JSON with `transactions` list or CSV lines with 10 fields."
        )
    return cleaned
