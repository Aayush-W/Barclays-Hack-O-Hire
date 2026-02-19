# 🏦 SAR Audit Engine  
### Explainable AI Platform for Suspicious Activity Report Generation

---

## 🚀 Overview

SAR Audit Engine is an end-to-end compliance intelligence system that transforms transaction patterns into regulator-ready Suspicious Activity Reports (SARs) with full explainability, audit traceability, and human oversight.

Unlike black-box AI tools, this platform combines:

- Structured risk signal extraction  
- SHAP-driven explainability  
- Metadata-filtered regulatory RAG  
- Sentence-to-evidence traceability  
- AI-assisted human review  
- Immutable audit logging  

Designed for enterprise-grade defensibility and scalability.

---

# 🧠 System Architecture

```
Input Data
    ↓
Data Normalization & Privacy Layer
    ↓
Feature & Signal Extraction
    ↓
Regulatory-Aware RAG (Weaviate Vector DB)
    ↓
Structured SAR Reasoning Engine
    ↓
Constrained LLM Narrative Generator
    ↓
Sentence-to-Evidence Mapping
    ↓
Human Review & Workflow
    ↓
Immutable Audit Ledger
```

---

# 🔍 Core Capabilities

## 1️⃣ Case Parsing & Normalization
- Parse laundering attempts into structured case JSON
- Optional enrichment from full ledger data
- Schema validation and case-level aggregation
- Preservation of structured typology naming

## 2️⃣ Risk Signal & Typology Detection
- Threshold breaches
- Structuring patterns
- Velocity anomalies
- Geo-risk indicators
- Behavioral deviation analysis
- Multiclass typology classifier

## 3️⃣ SHAP Explainability Layer
- Per-case feature contribution scoring
- Transparent risk attribution
- Feature-to-reason mapping
- Risk factor ranking

## 4️⃣ Regulatory-Aware RAG
- Weaviate vector database
- Sentence-Transformers embeddings
- Metadata-filtered retrieval
- Jurisdiction-aware grounding
- Typology-specific regulatory context

## 5️⃣ Structured SAR Reasoning Engine
- Builds deterministic reasoning graph
- Links evidence to transactions
- Maps typology to regulation
- Produces regulator-readable reasoning JSON

## 6️⃣ Constrained LLM Narrative Generator
- Uses reasoning JSON + retrieved regulation
- Template-controlled section generation
- No free hallucination
- Escalation gating for low-risk cases

## 7️⃣ Sentence-to-Evidence Mapper
- Each narrative sentence linked to:
  - Transaction ID
  - Risk feature
  - Regulatory source
- Transparent justification trail

## 8️⃣ AI-Assisted Editable Workspace (Dashboard)
- Interactive SAR editor
- Compliance-preserving rephrasing
- Legal-style JSON structure display
- Version comparison and diff tracking

## 9️⃣ Immutable Audit Ledger
- Prompt version logging
- Model version tracking
- Retrieved context storage
- Analyst edits history
- Append-only audit record

---

# ⚙️ Installation & Setup

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Pipeline

```bash
python main.py --build-rag-index --max-cases 100
```

### Useful Flags

- `--refresh-cases`
- `--enrich-from-raw`
- `--top-k-per-reason 3`
- `--sar-style standard|condensed`
- `--train-ml-model`
- `--train-model-before-run`
- `--disable-model-guided-rag`
- `--source-boost-weight 0.45`
- `--export-llm-sft-data`

---

# 📊 Streamlit SAR Dashboard

Run interactive dashboard:

```bash
streamlit run app.py
```

### Dashboard Features

- Paste audit trail (CSV / JSON)
- Typology classification
- SHAP feature contribution display
- Escalation gating
- LLM-powered SAR generation (Mistral via Ollama)
- Legal-style SAR JSON structure view
- Downloadable SAR report package

Check local Ollama:

```bash
ollama list
```

---

# 🧮 Audit Typology Classification

### Train model

```bash
python -m core.typology_classifier train \
  --cases-dir data/processed/cases \
  --patterns-path data/raw/HI-Medium_Patterns.txt \
  --model-output data/processed/models/audit_typology_classifier.joblib \
  --expected-types 10
```

### Predict

```bash
python -m core.typology_classifier predict \
  --case-json data/processed/cases/CASE_001.json \
  --model-path data/processed/models/audit_typology_classifier.joblib \
  --top-k 3
```

---

# 🤖 LLM Fine-Tuning (LoRA)

## 1️⃣ Export SFT dataset

```bash
python -m llm.sft_dataset \
  --evidence-dir data/processed/evidence_maps \
  --sar-dir data/processed/sar_drafts \
  --output-dir data/processed/training/llm_sft \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

## 2️⃣ Train LoRA Adapter

```bash
python -m llm.train_lora \
  --train-jsonl data/processed/training/llm_sft/sft_train.jsonl \
  --val-jsonl data/processed/training/llm_sft/sft_val.jsonl \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir data/processed/models/sar_lora_adapter \
  --epochs 2 \
  --max-length 1024
```

Includes:
- ROUGE metrics
- Optional BERTScore
- AML adherence checks
- Functional evaluation reports

---

# 🧠 Weaviate Setup

### Start locally

```bash
docker compose -f docker-compose.weaviate.yml up -d
```

Default:
- REST: `localhost:8080`
- gRPC: `localhost:50051`

Embedding model:
```
sentence-transformers/all-MiniLM-L6-v2
```

---

# 🛡 Enterprise-Grade Controls

- Case-level data isolation
- Metadata-filtered retrieval
- Prompt version logging
- Model version traceability
- Source-prior re-ranking
- Immutable audit manifest
- Confidence-aware escalation gating

---

# 🏆 Innovation Highlights

- Deterministic reasoning layer before generation  
- SHAP-to-narrative explainability bridge  
- Sentence-to-evidence traceability  
- Jurisdiction-aware RAG filtering  
- AI-assisted compliance editor  
- Full decision lineage logging  

---

# 📈 Impact

- Up to 70% reduction in drafting time  
- Improved narrative consistency  
- Reduced regulatory risk  
- Transparent, defensible AI outputs  
- Scalable microservice architecture  

---

# 🤝 Built For

Financial institutions, compliance teams, and regulatory technology innovators seeking explainable AI-driven reporting.
