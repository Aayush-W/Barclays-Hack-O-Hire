# SAR Audit Engine

This project generates SAR-ready evidence packages and draft narratives from transaction pattern cases.

## Pipeline

1. Parse laundering attempts into case JSON files.
2. Optionally enrich case transactions from raw ledger data.
3. Extract signals and build reasoning steps.
4. Build and query Weaviate embedding index (sentence-transformers vectors).
5. Map reasons to transaction evidence and retrieved context.
6. Generate SAR draft narrative and export retrieval training data.

## Run

```bash
python main.py --build-rag-index --max-cases 100
```

Useful flags:
- `--refresh-cases` to re-parse pattern text.
- `--enrich-from-raw` to match against the full raw transaction file.
- `--top-k-per-reason 3` to control retrieval depth.
- `--sar-style standard|condensed` for narrative format.
- `--train-ml-model` to train baseline retrieval helper model from generated training JSONL.
- `--train-model-before-run` to train router from existing training data first, then use it during retrieval.
- `--disable-model-guided-rag` to disable source-model reranking (Weaviate vector retrieval still used).
- `--source-boost-weight 0.45` to control how strongly the trained model influences retrieval ranking.
- `--export-llm-sft-data` to export chat-format SFT files from generated evidence maps + SAR drafts.

## Audit Typology Classification

Train a lightweight multiclass classifier that predicts laundering typology directly from transaction features:

```bash
python -m core.typology_classifier train \
  --cases-dir data/processed/cases \
  --patterns-path data/raw/HI-Medium_Patterns.txt \
  --model-output data/processed/models/audit_typology_classifier.joblib \
  --expected-types 10
```

Predict on a single audit case JSON:

```bash
python -m core.typology_classifier predict \
  --case-json data/processed/cases/CASE_001.json \
  --model-path data/processed/models/audit_typology_classifier.joblib \
  --top-k 3
```

Notes:
- `core/pattern_parser.py` now preserves hyphenated typology names (for example `FAN-IN`, `GATHER-SCATTER`).
- If `--expected-types` is larger than the number of classes present in the dataset, training still runs and reports a warning in model metadata.

## LLM Fine-Tuning (LoRA)

1) Export SFT dataset from your generated SAR artifacts:

```bash
python -m llm.sft_dataset \
  --evidence-dir data/processed/evidence_maps \
  --sar-dir data/processed/sar_drafts \
  --output-dir data/processed/training/llm_sft \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

2) Train a LoRA adapter on the exported train/val files:

```bash
python -m llm.train_lora \
  --train-jsonl data/processed/training/llm_sft/sft_train.jsonl \
  --val-jsonl data/processed/training/llm_sft/sft_val.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir data/processed/models/sar_lora_adapter \
  --epochs 2 \
  --max-length 1024
```

Functional evaluation runs after training (unless `--skip-functional-eval`) and compares:
- `fine_tuned` adapter output
- `base_model` output
- `template_baseline` deterministic output from prompt facts

Metrics include ROUGE-1/2/L, optional BERTScore, and AML adherence checks (section coverage, reason/fact grounding).  
Optional judge mode:

```bash
python -m llm.train_lora \
  ... \
  --judge-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --judge-max-samples 8
```

Output artifacts:
- LoRA adapter weights and tokenizer in `--output-dir`.
- Training summary in `training_metrics.json`.
- Functional evaluation summary in `functional_eval.json`.
- Per-sample generations in `functional_eval_samples.jsonl`.

Accuracy improvement notes:
- Training now uses one primary source label per reason row to avoid contradictory labels.
- Retrieval uses source priors by reason category (typology/risk reasons reduce template dominance).

Weaviate notes:
- The index manifest is stored at `vector_db/weaviate_store/index_manifest.json`.
- The embedding model defaults to `sentence-transformers/all-MiniLM-L6-v2`.
- The pipeline connects to a running Weaviate server (default `localhost:8080`, gRPC `50051`).
- On Windows, embedded Weaviate is not supported by `weaviate-client`; run Weaviate separately (Docker/WSL/cloud).
- Start local Weaviate with Docker: `docker compose -f docker-compose.weaviate.yml up -d`.
