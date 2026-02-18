from pathlib import Path

from vector_db.build_vector_db import build_sparse_index
from vector_db.retrieval import SparseRetriever


def test_sparse_retriever_returns_relevant_chunk(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    typologies_dir = knowledge_dir / "typologies"
    regulations_dir = knowledge_dir / "regulations"
    typologies_dir.mkdir(parents=True)
    regulations_dir.mkdir(parents=True)

    (typologies_dir / "funnel_accounts.txt").write_text(
        "Funnel account behavior includes rapid pass-through and many inbound transfers.",
        encoding="utf-8",
    )
    (regulations_dir / "narrative_guidelines.txt").write_text(
        "Narratives should be factual and include transaction identifiers.",
        encoding="utf-8",
    )

    index = build_sparse_index(knowledge_dir=knowledge_dir, chunk_size_words=40, chunk_overlap_words=5)
    retriever = SparseRetriever(index)
    results = retriever.retrieve("rapid pass-through funnel account", top_k=2)

    assert results
    assert results[0]["metadata"]["source_file"].endswith("funnel_accounts.txt")
