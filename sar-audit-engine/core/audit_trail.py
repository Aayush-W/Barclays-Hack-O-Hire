from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Tuple


def build_run_id(prefix: str = "sar_run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


def _write_json_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


def append_audit_logs(log_path: Path, new_records: List[Dict[str, Any]], max_records: int = 20000) -> None:
    if not new_records:
        return
    records = _load_json_records(log_path)
    records.extend(new_records)
    if len(records) > max_records:
        records = records[-max_records:]
    _write_json_records(log_path, records)


def append_audit_log(log_path: Path, record: Dict[str, Any], max_records: int = 20000) -> None:
    append_audit_logs(log_path=log_path, new_records=[record], max_records=max_records)


def create_case_audit_record(
    run_id: str,
    case_id: str,
    status: str,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "case_id": case_id,
        "status": status,
        "details": details,
    }


def log_case_audit(
    log_path: Path,
    run_id: str,
    case_id: str,
    status: str,
    details: Dict[str, Any],
) -> None:
    record = create_case_audit_record(run_id=run_id, case_id=case_id, status=status, details=details)
    append_audit_log(log_path=log_path, record=record)


def create_reasoning_version_record(
    run_id: str,
    case_id: str,
    reasoning_payload: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    canonical = json.dumps(reasoning_payload, sort_keys=True, separators=(",", ":"))
    digest = sha256(canonical.encode("utf-8")).hexdigest()
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "case_id": case_id,
        "version_hash": digest,
        "reasoning_step_count": len(reasoning_payload.get("reasoning", [])),
    }
    return digest, record


def register_reasoning_version(
    version_path: Path,
    run_id: str,
    case_id: str,
    reasoning_payload: Dict[str, Any],
) -> str:
    digest, record = create_reasoning_version_record(
        run_id=run_id,
        case_id=case_id,
        reasoning_payload=reasoning_payload,
    )
    append_audit_log(log_path=version_path, record=record)
    return digest
