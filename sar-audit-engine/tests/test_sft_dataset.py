from __future__ import annotations

import json
from pathlib import Path

from llm.sft_dataset import build_sft_dataset


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_sft_dataset_creates_splits(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence_maps"
    sar_dir = tmp_path / "sar_drafts"
    out_dir = tmp_path / "llm_sft"

    for idx in range(1, 6):
        case_id = f"CASE_{idx:03d}"
        _write_json(
            evidence_dir / f"{case_id}.json",
            {
                "case_id": case_id,
                "typology": "STACK",
                "risk_assessment": {"band": "high", "score": 80},
                "reasons": [],
            },
        )
        _write_json(
            sar_dir / f"{case_id}.json",
            {
                "case_id": case_id,
                "narrative": (
                    "Activity Summary: Test narrative for SAR quality and supervision. "
                    "Key Suspicious Indicators: synthetic test indicator list. "
                    "Evidence and Rationale: linked to evidence map. "
                    "Recommended Action: escalate."
                ),
                "prompt_payload": {
                    "system": "You are an AML investigator assistant.",
                    "user": f"Generate SAR for {case_id}",
                },
            },
        )

    manifest = build_sft_dataset(
        evidence_dir=evidence_dir,
        sar_dir=sar_dir,
        output_dir=out_dir,
        val_ratio=0.2,
        test_ratio=0.2,
        min_narrative_words=5,
        seed=7,
    )

    stats = manifest["stats"]
    assert stats["records_total"] == 5
    assert stats["train_records"] + stats["val_records"] + stats["test_records"] == 5

    train_lines = (out_dir / "sft_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert train_lines
    sample = json.loads(train_lines[0])
    assert "messages" in sample
    assert len(sample["messages"]) == 3
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["role"] == "user"
    assert sample["messages"][2]["role"] == "assistant"
