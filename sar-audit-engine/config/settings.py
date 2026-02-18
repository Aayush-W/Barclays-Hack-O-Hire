from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
VECTOR_STORE_DIR = BASE_DIR / "vector_db" / "weaviate_store"


@dataclass(frozen=True)
class RAGSettings:
    backend: str = "weaviate"
    knowledge_dir: Path = KNOWLEDGE_DIR
    index_path: Path = VECTOR_STORE_DIR / "index_manifest.json"
    weaviate_persistence_path: Path = VECTOR_STORE_DIR / "data"
    weaviate_http_host: str = "localhost"
    weaviate_http_port: int = 8080
    weaviate_grpc_host: str = "localhost"
    weaviate_grpc_port: int = 50051
    weaviate_secure: bool = False
    weaviate_collection_name: str = "SarKnowledgeChunk"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    source_router_model_path: Path = PROCESSED_DIR / "models" / "retrieval_source_classifier.joblib"
    chunk_size_words: int = 140
    chunk_overlap_words: int = 25
    top_k: int = 5
    source_boost_weight: float = 0.45


@dataclass(frozen=True)
class PipelineSettings:
    patterns_path: Path = RAW_DIR / "HI-Medium_Patterns.txt"
    transactions_path: Path = RAW_DIR / "HI-Medium_Trans.csv"
    cases_dir: Path = PROCESSED_DIR / "cases"
    enriched_cases_dir: Path = PROCESSED_DIR / "enriched_cases"
    signals_dir: Path = PROCESSED_DIR / "signals"
    reasoning_dir: Path = PROCESSED_DIR / "reasoning"
    evidence_map_dir: Path = PROCESSED_DIR / "evidence_maps"
    sar_drafts_dir: Path = PROCESSED_DIR / "sar_drafts"
    training_output_path: Path = PROCESSED_DIR / "training" / "rag_training.jsonl"
    audit_log_path: Path = LOGS_DIR / "audit_logs.json"
    reasoning_versions_path: Path = LOGS_DIR / "reasoning_versions.json"
    high_value_threshold: float = 10000.0
    cross_bank_ratio_threshold: float = 0.6
    multi_hop_threshold: int = 3
    multi_party_threshold: int = 3
    velocity_tx_per_day_threshold: float = 5.0
    velocity_max_tx_1h_threshold: int = 3
    net_flow_ratio_threshold: float = 0.1
    pass_through_hours_default: int = 24


def ensure_project_directories() -> None:
    for path in (PROCESSED_DIR, LOGS_DIR, VECTOR_STORE_DIR):
        path.mkdir(parents=True, exist_ok=True)
