from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import PipelineSettings, RAGSettings, ensure_project_directories
from core.audit_trail import (
    append_audit_logs,
    build_run_id,
    create_case_audit_record,
    create_reasoning_version_record,
)
from core.case_enricher import enrich_case_with_transactions
from core.evidence_mapper import map_case_evidence, training_examples_from_evidence
from core.feature_extractor import extract_signals
from core.pattern_parser import parse_patterns_file
from core.reasoning_engine import build_reasoning
from core.transaction_loader import build_transaction_index, load_transactions
from llm.generator import generate_sar_narrative
from llm.sft_dataset import build_sft_dataset
from vector_db.build_vector_db import build_and_save_index
from vector_db.retrieval import build_default_retriever


def _extract_case_number(case_id_or_name: str) -> int:
    match = re.search(r"(\d+)", case_id_or_name or "")
    return int(match.group(1)) if match else 0


def _select_case_files(cases_dir: Path, start_case: int, max_cases: Optional[int]) -> List[Path]:
    selected: List[Path] = []
    case_files = sorted(cases_dir.glob("*.json"), key=lambda path: _extract_case_number(path.stem))
    for case_file in case_files:
        if _extract_case_number(case_file.stem) < start_case:
            continue
        selected.append(case_file)
        if max_cases is not None and len(selected) >= max_cases:
            break
    return selected


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs:02}:{mins:02}:{sec:02}"
    return f"{mins:02}:{sec:02}"


def run_pipeline(args: argparse.Namespace) -> None:
    ensure_project_directories()
    pipeline_settings = PipelineSettings()
    rag_settings = RAGSettings()

    patterns_path = Path(args.patterns_path or pipeline_settings.patterns_path)
    transactions_path = Path(args.transactions_path or pipeline_settings.transactions_path)
    cases_dir = Path(args.cases_dir or pipeline_settings.cases_dir)
    output_root = Path(args.output_root or pipeline_settings.enriched_cases_dir.parent)

    enriched_cases_dir = output_root / "enriched_cases"
    signals_dir = output_root / "signals"
    reasoning_dir = output_root / "reasoning"
    evidence_map_dir = output_root / "evidence_maps"
    sar_drafts_dir = output_root / "sar_drafts"
    training_output = Path(args.training_output or pipeline_settings.training_output_path)
    llm_sft_output_dir = Path(args.llm_sft_output_dir)

    index_path = Path(args.index_path or rag_settings.index_path)
    knowledge_dir = Path(args.knowledge_dir or rag_settings.knowledge_dir)
    model_output = Path(args.model_output)

    if args.train_model_before_run and training_output.exists():
        from core.model_training import train_retrieval_source_model

        bootstrap_metrics = train_retrieval_source_model(
            training_jsonl_path=training_output,
            model_output_path=model_output,
        )
        print(
            "Bootstrap model trained before run:",
            f"accuracy={bootstrap_metrics['accuracy']}",
            f"macro_f1={bootstrap_metrics.get('macro_f1')}",
            flush=True,
        )

    if args.refresh_cases or not any(cases_dir.glob("*.json")):
        print(f"Parsing pattern file into case JSONs: {patterns_path}")
        parse_patterns_file(str(patterns_path), output_folder=str(cases_dir))

    if args.build_rag_index or not index_path.exists():
        print(f"Building RAG index from: {knowledge_dir}")
        build_and_save_index(
            knowledge_dir=knowledge_dir,
            output_path=index_path,
            chunk_size_words=args.chunk_size_words,
            chunk_overlap_words=args.chunk_overlap_words,
        )

    source_model_path = None if args.disable_model_guided_rag else model_output
    retriever = build_default_retriever(
        index_path=index_path,
        source_model_path=source_model_path,
        source_boost_weight=args.source_boost_weight,
    )
    run_id = build_run_id()
    print(f"Run ID: {run_id}", flush=True)
    print(
        "RAG retriever mode:",
        type(retriever).__name__,
        flush=True,
    )

    transactions_index = None
    if args.enrich_from_raw:
        print(f"Loading transactions for enrichment: {transactions_path}", flush=True)
        tx_df = load_transactions(str(transactions_path))
        transactions_index = build_transaction_index(tx_df)

    case_files = _select_case_files(cases_dir=cases_dir, start_case=args.start_case, max_cases=args.max_cases)
    total_cases = len(case_files)
    print(
        f"Starting case processing: total_cases={total_cases} start_case={args.start_case}",
        flush=True,
    )
    if total_cases == 0:
        print("No cases found for selected range. Exiting.", flush=True)
        return

    if training_output.exists() and not args.append_training_data:
        training_output.unlink()
    training_output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    failures = 0
    started = time.time()
    audit_buffer: List[Dict] = []
    reasoning_version_buffer: List[Dict] = []

    with training_output.open("a", encoding="utf-8") as training_handle:
        for idx, case_file in enumerate(case_files, start=1):
            with case_file.open("r", encoding="utf-8") as handle:
                case = json.load(handle)

            case_id = case.get("case_id", case_file.stem)
            case_started = time.time()

            try:
                if transactions_index is not None:
                    enriched_case = enrich_case_with_transactions(case, transactions_index)
                else:
                    enriched_case = case
                    enriched_case["enriched_transactions"] = case.get("transactions", [])
                    enriched_case["transaction_count"] = len(enriched_case["enriched_transactions"])

                signals_payload = extract_signals(enriched_case, pass_through_hours=args.pass_through_hours)
                reasoning_payload = build_reasoning(
                    enriched_case,
                    signals_payload,
                    settings=pipeline_settings,
                )

                evidence_map = map_case_evidence(
                    case=enriched_case,
                    signals_payload=signals_payload,
                    reasoning_payload=reasoning_payload,
                    retriever=retriever,
                    top_k_per_reason=args.top_k_per_reason,
                )
                sar_payload = generate_sar_narrative(evidence_map=evidence_map, style=args.sar_style)
                version_hash, version_record = create_reasoning_version_record(
                    run_id=run_id,
                    case_id=case_id,
                    reasoning_payload=reasoning_payload,
                )
                reasoning_version_buffer.append(version_record)

                _write_json(enriched_cases_dir / f"{case_id}.json", enriched_case)
                _write_json(signals_dir / f"{case_id}.json", signals_payload)
                _write_json(reasoning_dir / f"{case_id}.json", reasoning_payload)
                _write_json(evidence_map_dir / f"{case_id}.json", evidence_map)
                _write_json(sar_drafts_dir / f"{case_id}.json", sar_payload)

                training_examples = training_examples_from_evidence(
                    evidence_map=evidence_map,
                    target_narrative=sar_payload.get("narrative", ""),
                )
                for record in training_examples:
                    training_handle.write(json.dumps(record) + "\n")

                elapsed = round(time.time() - case_started, 3)
                audit_buffer.append(
                    create_case_audit_record(
                        run_id=run_id,
                        case_id=case_id,
                        status="completed",
                        details={
                            "duration_seconds": elapsed,
                            "reasoning_version_hash": version_hash,
                            "reason_count": len(reasoning_payload.get("reasoning", [])),
                            "training_examples": len(training_examples),
                        },
                    )
                )
                processed += 1
            except Exception as exc:
                failures += 1
                audit_buffer.append(
                    create_case_audit_record(
                        run_id=run_id,
                        case_id=case_id,
                        status="failed",
                        details={"error": repr(exc)},
                    )
                )

            if idx % args.log_flush_every == 0:
                append_audit_logs(
                    log_path=pipeline_settings.audit_log_path,
                    new_records=audit_buffer,
                )
                append_audit_logs(
                    log_path=pipeline_settings.reasoning_versions_path,
                    new_records=reasoning_version_buffer,
                )
                audit_buffer.clear()
                reasoning_version_buffer.clear()

            if idx == 1 or idx % args.progress_every == 0 or idx == total_cases:
                elapsed_total = max(time.time() - started, 1e-9)
                rate = idx / elapsed_total
                eta_seconds = (total_cases - idx) / rate if rate > 0 else 0
                print(
                    f"[progress] {idx}/{total_cases} done | ok={processed} fail={failures} "
                    f"| rate={rate:.2f} cases/s | eta={_format_duration(eta_seconds)}",
                    flush=True,
                )

    append_audit_logs(
        log_path=pipeline_settings.audit_log_path,
        new_records=audit_buffer,
    )
    append_audit_logs(
        log_path=pipeline_settings.reasoning_versions_path,
        new_records=reasoning_version_buffer,
    )

    total_time = round(time.time() - started, 2)
    print(
        f"Pipeline complete. processed={processed} failures={failures} "
        f"runtime_seconds={total_time} training_output={training_output}",
        flush=True,
    )

    if args.train_ml_model:
        from core.model_training import train_retrieval_source_model

        metrics = train_retrieval_source_model(
            training_jsonl_path=training_output,
            model_output_path=model_output,
        )
        print(
            "ML model trained:",
            f"accuracy={metrics['accuracy']}",
            f"macro_f1={metrics.get('macro_f1')}",
            f"model_path={metrics['model_path']}",
            flush=True,
        )

    if args.export_llm_sft_data:
        sft_manifest = build_sft_dataset(
            evidence_dir=evidence_map_dir,
            sar_dir=sar_drafts_dir,
            output_dir=llm_sft_output_dir,
            style=args.sar_style,
            val_ratio=args.llm_sft_val_ratio,
            test_ratio=args.llm_sft_test_ratio,
            min_narrative_words=args.llm_sft_min_narrative_words,
            seed=args.llm_sft_seed,
        )
        sft_stats = sft_manifest.get("stats", {})
        print(
            "LLM SFT dataset exported:",
            f"total={sft_stats.get('records_total')}",
            f"train={sft_stats.get('train_records')}",
            f"val={sft_stats.get('val_records')}",
            f"test={sft_stats.get('test_records')}",
            f"output_dir={llm_sft_output_dir}",
            flush=True,
        )

    if hasattr(retriever, "close"):
        retriever.close()


def parse_args() -> argparse.Namespace:
    pipeline_settings = PipelineSettings()
    rag_settings = RAGSettings()

    parser = argparse.ArgumentParser(description="SAR RAG orchestration pipeline")
    parser.add_argument("--patterns-path", type=Path, default=pipeline_settings.patterns_path)
    parser.add_argument("--transactions-path", type=Path, default=pipeline_settings.transactions_path)
    parser.add_argument("--cases-dir", type=Path, default=pipeline_settings.cases_dir)
    parser.add_argument("--output-root", type=Path, default=pipeline_settings.enriched_cases_dir.parent)
    parser.add_argument("--training-output", type=Path, default=pipeline_settings.training_output_path)
    parser.add_argument("--knowledge-dir", type=Path, default=rag_settings.knowledge_dir)
    parser.add_argument("--index-path", type=Path, default=rag_settings.index_path)
    parser.add_argument("--build-rag-index", action="store_true")
    parser.add_argument("--refresh-cases", action="store_true")
    parser.add_argument("--enrich-from-raw", action="store_true")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--start-case", type=int, default=1)
    parser.add_argument("--top-k-per-reason", type=int, default=3)
    parser.add_argument("--chunk-size-words", type=int, default=rag_settings.chunk_size_words)
    parser.add_argument("--chunk-overlap-words", type=int, default=rag_settings.chunk_overlap_words)
    parser.add_argument("--pass-through-hours", type=int, default=24)
    parser.add_argument("--sar-style", choices=("standard", "condensed"), default="standard")
    parser.add_argument("--train-ml-model", action="store_true")
    parser.add_argument("--train-model-before-run", action="store_true")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=rag_settings.source_router_model_path,
    )
    parser.add_argument("--disable-model-guided-rag", action="store_true")
    parser.add_argument("--source-boost-weight", type=float, default=rag_settings.source_boost_weight)
    parser.add_argument("--export-llm-sft-data", action="store_true")
    parser.add_argument(
        "--llm-sft-output-dir",
        type=Path,
        default=pipeline_settings.training_output_path.parent / "llm_sft",
    )
    parser.add_argument("--llm-sft-val-ratio", type=float, default=0.1)
    parser.add_argument("--llm-sft-test-ratio", type=float, default=0.1)
    parser.add_argument("--llm-sft-min-narrative-words", type=int, default=40)
    parser.add_argument("--llm-sft-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--log-flush-every", type=int, default=100)
    parser.add_argument("--append-training-data", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
