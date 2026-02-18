from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from config.settings import PROCESSED_DIR

from .prompt_templates import build_prompt_payload


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_case_ids(evidence_dir: Path, sar_dir: Path) -> List[str]:
    evidence_ids = {path.stem for path in evidence_dir.glob("*.json")}
    sar_ids = {path.stem for path in sar_dir.glob("*.json")}
    return sorted(evidence_ids.intersection(sar_ids))


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _extract_prompt_payload(evidence_map: Dict, sar_payload: Dict, style: str) -> Dict[str, str]:
    payload = sar_payload.get("prompt_payload")
    if isinstance(payload, dict) and payload.get("system") and payload.get("user"):
        return {"system": _safe_text(payload.get("system")), "user": _safe_text(payload.get("user"))}
    return build_prompt_payload(evidence_map=evidence_map, style=style)


def _record_from_case(case_id: str, evidence_map: Dict, sar_payload: Dict, style: str) -> Optional[Dict]:
    narrative = _safe_text(sar_payload.get("narrative"))
    if not narrative:
        return None

    prompt_payload = _extract_prompt_payload(evidence_map=evidence_map, sar_payload=sar_payload, style=style)
    system_text = _safe_text(prompt_payload.get("system"))
    user_text = _safe_text(prompt_payload.get("user"))
    if not system_text or not user_text:
        return None

    return {
        "case_id": case_id,
        "typology": evidence_map.get("typology"),
        "risk_band": evidence_map.get("risk_assessment", {}).get("band"),
        "risk_score": evidence_map.get("risk_assessment", {}).get("score"),
        "style": style,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": narrative},
        ],
        "prompt": f"System:\n{system_text}\n\nUser:\n{user_text}\n\nAssistant:\n",
        "completion": narrative,
    }


def _split_records(
    records: Sequence[Dict], val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if not records:
        return [], [], []

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    if total >= 3:
        if test_ratio > 0 and test_count == 0:
            test_count = 1
        if val_ratio > 0 and val_count == 0:
            val_count = 1
    if test_count + val_count >= total:
        overflow = test_count + val_count - total + 1
        val_count = max(0, val_count - overflow)

    test_rows = shuffled[:test_count]
    val_rows = shuffled[test_count : test_count + val_count]
    train_rows = shuffled[test_count + val_count :]
    return train_rows, val_rows, test_rows


def _write_jsonl(rows: Sequence[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_sft_dataset(
    evidence_dir: Path,
    sar_dir: Path,
    output_dir: Path,
    style: str = "standard",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    min_narrative_words: int = 40,
    seed: int = 42,
) -> Dict:
    evidence_dir = Path(evidence_dir)
    sar_dir = Path(sar_dir)
    output_dir = Path(output_dir)

    if not evidence_dir.exists():
        raise FileNotFoundError(f"Evidence directory not found: {evidence_dir}")
    if not sar_dir.exists():
        raise FileNotFoundError(f"SAR drafts directory not found: {sar_dir}")
    if not (0.0 <= val_ratio < 1.0 and 0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must each be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    case_ids = _iter_case_ids(evidence_dir=evidence_dir, sar_dir=sar_dir)
    rows: List[Dict] = []
    skipped_empty = 0
    skipped_short = 0
    seen = set()

    for case_id in case_ids:
        evidence_map = _load_json(evidence_dir / f"{case_id}.json")
        sar_payload = _load_json(sar_dir / f"{case_id}.json")
        record = _record_from_case(case_id=case_id, evidence_map=evidence_map, sar_payload=sar_payload, style=style)
        if record is None:
            skipped_empty += 1
            continue
        narrative_words = len(_safe_text(record.get("completion")).split())
        if narrative_words < int(min_narrative_words):
            skipped_short += 1
            continue

        dedupe_key = (record.get("prompt"), record.get("completion"))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(record)

    train_rows, val_rows, test_rows = _split_records(
        records=rows,
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        seed=int(seed),
    )

    all_path = output_dir / "sft_all.jsonl"
    train_path = output_dir / "sft_train.jsonl"
    val_path = output_dir / "sft_val.jsonl"
    test_path = output_dir / "sft_test.jsonl"

    _write_jsonl(rows, all_path)
    _write_jsonl(train_rows, train_path)
    _write_jsonl(val_rows, val_path)
    _write_jsonl(test_rows, test_path)

    manifest = {
        "input": {
            "evidence_dir": str(evidence_dir),
            "sar_dir": str(sar_dir),
        },
        "output": {
            "all": str(all_path),
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "stats": {
            "cases_with_both_files": len(case_ids),
            "records_total": len(rows),
            "train_records": len(train_rows),
            "val_records": len(val_rows),
            "test_records": len(test_rows),
            "skipped_empty": skipped_empty,
            "skipped_short": skipped_short,
            "min_narrative_words": int(min_narrative_words),
        },
        "params": {
            "style": style,
            "val_ratio": float(val_ratio),
            "test_ratio": float(test_ratio),
            "seed": int(seed),
        },
    }
    manifest_path = output_dir / "sft_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM SFT dataset from evidence maps and SAR drafts.")
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=pipeline_settings.evidence_map_dir,
    )
    parser.add_argument(
        "--sar-dir",
        type=Path,
        default=pipeline_settings.sar_drafts_dir,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR / "training" / "llm_sft",
    )
    parser.add_argument("--style", choices=("standard", "condensed"), default="standard")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--min-narrative-words", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_sft_dataset(
        evidence_dir=args.evidence_dir,
        sar_dir=args.sar_dir,
        output_dir=args.output_dir,
        style=args.style,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_narrative_words=args.min_narrative_words,
        seed=args.seed,
    )
    stats = manifest.get("stats", {})
    print(
        "SFT dataset built:",
        f"total={stats.get('records_total')}",
        f"train={stats.get('train_records')}",
        f"val={stats.get('val_records')}",
        f"test={stats.get('test_records')}",
    )


if __name__ == "__main__":
    main()
