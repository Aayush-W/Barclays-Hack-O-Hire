from pathlib import Path

from core.evidence_mapper import map_case_evidence, training_examples_from_evidence
from core.feature_extractor import extract_signals
from core.reasoning_engine import build_reasoning
from vector_db.build_vector_db import build_sparse_index
from vector_db.retrieval import SparseRetriever


def test_evidence_mapping_generates_reason_links(tmp_path: Path) -> None:
    case = {
        "case_id": "CASE_TEST_001",
        "typology": "FUNNEL",
        "transactions": [
            {
                "timestamp": "2022/09/01 05:14",
                "from_bank": "001",
                "from_account": "A1",
                "to_bank": "002",
                "to_account": "B1",
                "amount_received": 12000.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 12000.0,
                "payment_currency": "US Dollar",
                "payment_format": "ACH",
            },
            {
                "timestamp": "2022/09/01 06:00",
                "from_bank": "002",
                "from_account": "B1",
                "to_bank": "003",
                "to_account": "C1",
                "amount_received": 11800.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 11800.0,
                "payment_currency": "US Dollar",
                "payment_format": "ACH",
            },
            {
                "timestamp": "2022/09/01 06:40",
                "from_bank": "003",
                "from_account": "C1",
                "to_bank": "004",
                "to_account": "D1",
                "amount_received": 11700.0,
                "receiving_currency": "US Dollar",
                "amount_paid": 11700.0,
                "payment_currency": "US Dollar",
                "payment_format": "ACH",
            },
        ],
    }

    knowledge_dir = tmp_path / "knowledge"
    (knowledge_dir / "typologies").mkdir(parents=True)
    (knowledge_dir / "typologies" / "funnel_accounts.txt").write_text(
        "Funnel account risk includes rapid pass-through and near-zero net flow.",
        encoding="utf-8",
    )

    index = build_sparse_index(knowledge_dir=knowledge_dir, chunk_size_words=50, chunk_overlap_words=5)
    retriever = SparseRetriever(index)

    signals = extract_signals(case)
    reasoning = build_reasoning(case, signals)
    evidence_map = map_case_evidence(case, signals, reasoning, retriever=retriever, top_k_per_reason=2)

    assert evidence_map["reasons"]
    assert evidence_map["risk_assessment"]["score"] > 0

    training_examples = training_examples_from_evidence(evidence_map, target_narrative="test narrative")
    assert training_examples
