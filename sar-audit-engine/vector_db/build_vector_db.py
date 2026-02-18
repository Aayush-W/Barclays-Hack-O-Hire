from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from urllib.parse import urlparse

from config.settings import RAGSettings

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _chunk_text_by_words(
    text: str, chunk_size_words: int, chunk_overlap_words: int
) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(chunk_size_words - chunk_overlap_words, 1)
    chunks: List[str] = []

    for start in range(0, len(words), step):
        end = start + chunk_size_words
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return chunks


def _iter_knowledge_files(knowledge_dir: Path) -> Iterable[Path]:
    for pattern in ("*.txt", "*.md"):
        for path in sorted(knowledge_dir.rglob(pattern)):
            if path.is_file():
                yield path


def _build_knowledge_chunks(
    knowledge_dir: Path,
    chunk_size_words: int = 140,
    chunk_overlap_words: int = 25,
) -> List[Dict]:
    chunks: List[Dict] = []
    for source_path in _iter_knowledge_files(knowledge_dir):
        text = source_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        source_type = source_path.parent.name
        source_file = str(source_path.relative_to(knowledge_dir).as_posix())
        text_chunks = _chunk_text_by_words(text, chunk_size_words, chunk_overlap_words)

        for chunk_idx, chunk_text in enumerate(text_chunks, start=1):
            chunk_id_seed = f"{source_file}:{chunk_idx}:{chunk_text[:40]}"
            chunk_id = sha1(chunk_id_seed.encode("utf-8")).hexdigest()[:16]
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source_file": source_file,
                        "source_type": source_type,
                        "title": source_path.stem.replace("_", " ").title(),
                        "chunk_index": chunk_idx,
                    },
                }
            )
    return chunks


def _load_embedder(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedding index build. "
            "Install with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def _encode_texts(embedder, texts: Sequence[str], batch_size: int = 64) -> List[List[float]]:
    vectors = embedder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def _connect_weaviate(settings: RAGSettings):
    try:
        import weaviate
    except ImportError as exc:
        raise RuntimeError(
            "weaviate-client is required for Weaviate index build. "
            "Install with: pip install weaviate-client"
        ) from exc

    url = os.getenv("WEAVIATE_URL")
    if url:
        parsed = urlparse(url)
        http_host = parsed.hostname or settings.weaviate_http_host
        http_port = parsed.port or settings.weaviate_http_port
        secure = parsed.scheme == "https"
        grpc_host = os.getenv("WEAVIATE_GRPC_HOST", settings.weaviate_grpc_host or http_host)
        grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", str(settings.weaviate_grpc_port)))
        try:
            return weaviate.connect_to_custom(
                http_host=http_host,
                http_port=http_port,
                http_secure=secure,
                grpc_host=grpc_host,
                grpc_port=grpc_port,
                grpc_secure=secure,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(f"Failed to connect to Weaviate at {url}: {exc}") from exc

    try:
        return weaviate.connect_to_local(
            host=settings.weaviate_http_host,
            port=settings.weaviate_http_port,
            grpc_port=settings.weaviate_grpc_port,
            skip_init_checks=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to connect to local Weaviate. Start a Weaviate instance (e.g. Docker) "
            f"on {settings.weaviate_http_host}:{settings.weaviate_http_port} and "
            f"gRPC {settings.weaviate_grpc_port}. Error: {exc}"
        ) from exc


def _ensure_weaviate_collection(client, collection_name: str) -> None:
    from weaviate.classes.config import Configure, DataType, Property

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="chunk_id", data_type=DataType.TEXT),
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source_file", data_type=DataType.TEXT),
            Property(name="source_type", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )


def build_weaviate_index(
    knowledge_dir: Path,
    rag_settings: RAGSettings,
    chunk_size_words: int = 140,
    chunk_overlap_words: int = 25,
) -> Dict:
    knowledge_dir = Path(knowledge_dir)
    chunks = _build_knowledge_chunks(
        knowledge_dir=knowledge_dir,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
    )

    embedder = _load_embedder(rag_settings.embedding_model_name)
    texts = [chunk["text"] for chunk in chunks]
    vectors = _encode_texts(embedder, texts=texts) if texts else []
    vector_dim = len(vectors[0]) if vectors else 0

    client = _connect_weaviate(rag_settings)
    try:
        _ensure_weaviate_collection(client, collection_name=rag_settings.weaviate_collection_name)
        collection = client.collections.get(rag_settings.weaviate_collection_name)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                metadata = chunk["metadata"]
                batch.add_object(
                    properties={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "source_file": metadata.get("source_file"),
                        "source_type": metadata.get("source_type"),
                        "title": metadata.get("title"),
                        "chunk_index": int(metadata.get("chunk_index") or 0),
                    },
                    vector=vector,
                )
    finally:
        client.close()

    source_counts = Counter(chunk["metadata"]["source_type"] for chunk in chunks)
    configured_url = os.getenv("WEAVIATE_URL")
    return {
        "backend": "weaviate",
        "version": "weaviate_embedding_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_dir": str(knowledge_dir),
        "connection": {
            "weaviate_url": configured_url,
            "http_host": rag_settings.weaviate_http_host,
            "http_port": rag_settings.weaviate_http_port,
            "grpc_host": rag_settings.weaviate_grpc_host,
            "grpc_port": rag_settings.weaviate_grpc_port,
            "secure": bool(rag_settings.weaviate_secure),
        },
        "collection_name": rag_settings.weaviate_collection_name,
        "embedding_model_name": rag_settings.embedding_model_name,
        "stats": {
            "num_chunks": len(chunks),
            "vector_dim": vector_dim,
            "source_type_counts": dict(source_counts),
        },
    }


def save_index(index: Dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    return output_path


def build_and_save_index(
    knowledge_dir: Path,
    output_path: Path,
    chunk_size_words: int = 140,
    chunk_overlap_words: int = 25,
) -> Dict:
    rag_settings = RAGSettings()
    index = build_weaviate_index(
        knowledge_dir=knowledge_dir,
        rag_settings=rag_settings,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    save_index(index, output_path)
    return index


def build_sparse_index(
    knowledge_dir: Path,
    chunk_size_words: int = 140,
    chunk_overlap_words: int = 25,
) -> Dict:
    """Legacy sparse index kept for tests/backward compatibility."""
    knowledge_dir = Path(knowledge_dir)
    chunks = _build_knowledge_chunks(
        knowledge_dir=knowledge_dir,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    document_frequency = Counter()
    sparse_chunks = []
    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        if not tokens:
            continue
        tf = Counter(tokens)
        for token in tf.keys():
            document_frequency[token] += 1
        sparse_chunks.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "tf": dict(tf),
                "length": sum(tf.values()),
                "metadata": chunk["metadata"],
            }
        )

    total_chunks = len(sparse_chunks)
    idf: Dict[str, float] = {}
    for token, df in document_frequency.items():
        idf[token] = math.log((1 + total_chunks) / (1 + df)) + 1.0

    for chunk in sparse_chunks:
        length = max(chunk["length"], 1)
        norm_sq = 0.0
        for token, count in chunk["tf"].items():
            weight = (count / length) * idf.get(token, 0.0)
            norm_sq += weight * weight
        chunk["norm"] = math.sqrt(norm_sq) if norm_sq else 1e-12

    return {
        "version": "sparse_tfidf_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "knowledge_dir": str(knowledge_dir),
        "stats": {
            "num_chunks": total_chunks,
            "vocab_size": len(idf),
        },
        "idf": idf,
        "chunks": sparse_chunks,
    }


def main() -> None:
    rag_settings = RAGSettings()
    parser = argparse.ArgumentParser(description="Build Weaviate embedding index for SAR knowledge.")
    parser.add_argument("--knowledge-dir", type=Path, default=rag_settings.knowledge_dir)
    parser.add_argument("--output-path", type=Path, default=rag_settings.index_path)
    parser.add_argument("--chunk-size-words", type=int, default=rag_settings.chunk_size_words)
    parser.add_argument("--chunk-overlap-words", type=int, default=rag_settings.chunk_overlap_words)
    args = parser.parse_args()

    index = build_and_save_index(
        knowledge_dir=args.knowledge_dir,
        output_path=args.output_path,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
    )
    stats = index.get("stats", {})
    print(
        "Weaviate index built:",
        f"chunks={stats.get('num_chunks')}",
        f"vector_dim={stats.get('vector_dim')}",
        f"collection={index.get('collection_name')}",
    )


if __name__ == "__main__":
    main()
