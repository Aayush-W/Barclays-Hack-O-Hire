from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

_TIMESTAMP_FORMATS = (
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)

_FIELD_ALIASES = {
    "timestamp": ("Timestamp", "timestamp", "time", "datetime"),
    "from_bank": ("From Bank", "from_bank", "sender_bank", "origin_bank"),
    "from_account": ("Account", "from_account", "sender_account", "origin_account"),
    "to_bank": ("To Bank", "to_bank", "receiver_bank", "beneficiary_bank"),
    "to_account": ("Account.1", "to_account", "receiver_account", "beneficiary_account"),
    "amount_received": ("Amount Received", "amount_received", "amount_in"),
    "amount_paid": ("Amount Paid", "amount_paid", "amount_out"),
    "receiving_currency": ("Receiving Currency", "receiving_currency", "currency_in"),
    "payment_currency": ("Payment Currency", "payment_currency", "currency_out"),
    "payment_format": ("Payment Format", "payment_format", "channel"),
}

_TX_ID_ALIASES = (
    "Transaction ID",
    "transaction_id",
    "transaction id",
    "transactionId",
    "tx_id",
    "TxID",
)


def _first_present(tx: Dict[str, Any], aliases: Iterable[str]) -> Optional[Any]:
    for key in aliases:
        if key in tx and tx[key] not in (None, ""):
            return tx[key]
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def normalize_transactions(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_transactions = case.get("enriched_transactions") or case.get("transactions") or []
    normalized: List[Dict[str, Any]] = []

    for idx, tx in enumerate(raw_transactions, start=1):
        tx_id = _first_present(tx, _TX_ID_ALIASES)
        if tx_id is None:
            tx_id = f"TX_{idx:03}"
        else:
            tx_id = str(tx_id)

        normalized.append(
            {
                "tx_id": tx_id,
                "timestamp": parse_timestamp(_first_present(tx, _FIELD_ALIASES["timestamp"])),
                "timestamp_raw": _first_present(tx, _FIELD_ALIASES["timestamp"]),
                "from_bank": _first_present(tx, _FIELD_ALIASES["from_bank"]),
                "from_account": _first_present(tx, _FIELD_ALIASES["from_account"]),
                "to_bank": _first_present(tx, _FIELD_ALIASES["to_bank"]),
                "to_account": _first_present(tx, _FIELD_ALIASES["to_account"]),
                "amount_received": _coerce_float(_first_present(tx, _FIELD_ALIASES["amount_received"])),
                "amount_paid": _coerce_float(_first_present(tx, _FIELD_ALIASES["amount_paid"])),
                "receiving_currency": _first_present(tx, _FIELD_ALIASES["receiving_currency"]),
                "payment_currency": _first_present(tx, _FIELD_ALIASES["payment_currency"]),
                "payment_format": _first_present(tx, _FIELD_ALIASES["payment_format"]),
            }
        )

    return normalized
