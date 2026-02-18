from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from config.settings import RAGSettings

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
REASON_CATEGORY_PATTERN = re.compile(r"\bcategory\s+([a-z_]+)\b", re.IGNORECASE)

SOURCE_TYPE_BASE_WEIGHTS = {
    "typologies": 1.0,
    "regulations": 0.95,
    "templates": 0.72,
}

REASON_CATEGORY_SOURCE_PRIORS = {
    "typology_classification": {"typologies": 1.25, "templates": 0.8, "regulations": 0.8},
    "high_value_activity": {"regulations": 1.25, "typologies": 1.0, "templates": 0.8},
    "rapid_pass_through": {"typologies": 1.2, "regulations": 1.0, "templates": 0.8},
    "cross_bank_movement": {"typologies": 1.2, "regulations": 1.0, "templates": 0.8},
    "multi_hop_layering": {"typologies": 1.25, "regulations": 1.0, "templates": 0.78},
    "high_velocity": {"typologies": 1.2, "regulations": 1.0, "templates": 0.8},
    "multi_currency_activity": {"typologies": 1.2, "regulations": 1.0, "templates": 0.82},
    "multi_party_network": {"typologies": 1.18, "regulations": 1.0, "templates": 0.82},
    "general_suspicion": {"regulations": 1.1, "typologies": 1.0, "templates": 0.82},
}


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _extract_reason_category(query: str) -> Optional[str]:
    match = REASON_CATEGORY_PATTERN.search(query or "")
    if not match:
        return None
    return match.group(1).lower().strip()


def load_index(index_path: Optional[Path] = None) -> Dict:
    if index_path is None:
        index_path = RAGSettings().index_path

    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"RAG index manifest not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_source_model(source_model_path: Optional[Path]) -> Optional[Any]:
    if source_model_path is None:
        return None
    model_path = Path(source_model_path)
    if not model_path.exists():
        return None
    try:
        import joblib
    except ImportError:
        return None
    return joblib.load(model_path)


def _predict_source_distribution(source_model: Optional[Any], query: str) -> Dict[str, float]:
    if source_model is None:
        return {}
    if not hasattr(source_model, "predict_proba") or not hasattr(source_model, "classes_"):
        return {}
    probabilities = source_model.predict_proba([query])[0]
    classes = list(source_model.classes_)
    return {str(cls_name): float(prob) for cls_name, prob in zip(classes, probabilities)}


def _adjusted_score(
    base_score: float,
    source_type: Optional[str],
    source_prob: float,
    reason_category: Optional[str],
    source_boost_weight: float,
) -> float:
    source_base_weight = SOURCE_TYPE_BASE_WEIGHTS.get(source_type, 1.0)
    category_priors = REASON_CATEGORY_SOURCE_PRIORS.get(reason_category, {})
    category_weight = category_priors.get(source_type, 1.0)
    return (
        float(base_score)
        * float(source_base_weight)
        * float(category_weight)
        * (1.0 + float(source_boost_weight) * float(source_prob))
    )


class SparseRetriever:
    """Legacy sparse retriever retained for compatibility/tests."""

    def __init__(
        self,
        index: Dict,
        source_model: Optional[Any] = None,
        source_boost_weight: float = 0.45,
    ):
        self.index = index
        self.idf: Dict[str, float] = index.get("idf", {})
        self.chunks: List[Dict] = index.get("chunks", [])
        self._inverted = self._build_inverted_index(self.chunks, self.idf)
        self.source_model = source_model
        self.source_boost_weight = max(float(source_boost_weight), 0.0)

    @staticmethod
    def _build_inverted_index(chunks: List[Dict], idf: Dict[str, float]) -> Dict[str, List[List[float]]]:
        inverted = defaultdict(list)
        for chunk_idx, chunk in enumerate(chunks):
            tf = chunk.get("tf", {})
            length = max(chunk.get("length", 0), 1)
            norm = max(float(chunk.get("norm", 1e-12)), 1e-12)
            for term, count in tf.items():
                weight = ((count / length) * idf.get(term, 0.0)) / norm
                if weight:
                    inverted[term].append([chunk_idx, weight])
        return inverted

    @classmethod
    def from_path(
        cls,
        index_path: Optional[Path] = None,
        source_model_path: Optional[Path] = None,
        source_boost_weight: float = 0.45,
    ) -> "SparseRetriever":
        if source_model_path is None:
            source_model_path = RAGSettings().source_router_model_path
        source_model = _load_source_model(source_model_path)
        return cls(
            load_index(index_path=index_path),
            source_model=source_model,
            source_boost_weight=source_boost_weight,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        tf_query = Counter(query_tokens)
        query_length = max(sum(tf_query.values()), 1)

        query_weights: Dict[str, float] = {}
        norm_sq = 0.0
        for term, count in tf_query.items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            weight = (count / query_length) * idf
            query_weights[term] = weight
            norm_sq += weight * weight

        if not query_weights:
            return []

        query_norm = math.sqrt(norm_sq) if norm_sq else 1e-12
        for term in list(query_weights.keys()):
            query_weights[term] /= query_norm

        scores = defaultdict(float)
        for term, query_weight in query_weights.items():
            for chunk_idx, doc_weight in self._inverted.get(term, []):
                scores[chunk_idx] += query_weight * doc_weight

        source_distribution = _predict_source_distribution(self.source_model, query=query)
        reason_category = _extract_reason_category(query)
        reranked = []
        for chunk_idx, base_score in scores.items():
            chunk = self.chunks[chunk_idx]
            metadata = chunk.get("metadata", {})
            chunk_source = metadata.get("source_type")
            source_prob = source_distribution.get(chunk_source, 0.0)
            adjusted_score = _adjusted_score(
                base_score=base_score,
                source_type=chunk_source,
                source_prob=source_prob,
                reason_category=reason_category,
                source_boost_weight=self.source_boost_weight,
            )
            reranked.append((chunk_idx, float(base_score), float(adjusted_score), float(source_prob)))

        ranked = sorted(reranked, key=lambda item: item[2], reverse=True)
        results: List[Dict] = []
        for chunk_idx, base_score, adjusted_score, source_prob in ranked:
            if adjusted_score < min_score:
                continue
            chunk = self.chunks[chunk_idx]
            metadata = chunk.get("metadata", {})
            if source_type and metadata.get("source_type") != source_type:
                continue
            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "score": round(float(adjusted_score), 6),
                    "base_score": round(float(base_score), 6),
                    "source_model_probability": round(float(source_prob), 6),
                    "text": chunk.get("text", ""),
                    "metadata": metadata,
                }
            )
            if len(results) >= top_k:
                break
        return results

    def close(self) -> None:
        return None


class WeaviateRetriever:
    def __init__(
        self,
        collection_name: str,
        embedding_model_name: str,
        weaviate_url: Optional[str],
        http_host: str,
        http_port: int,
        grpc_host: str,
        grpc_port: int,
        secure: bool,
        source_model: Optional[Any] = None,
        source_boost_weight: float = 0.45,
    ):
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.weaviate_url = (weaviate_url or "").strip() or None
        self.http_host = http_host
        self.http_port = int(http_port)
        self.grpc_host = grpc_host
        self.grpc_port = int(grpc_port)
        self.secure = bool(secure)
        self.source_model = source_model
        self.source_boost_weight = max(float(source_boost_weight), 0.0)
        self._client = self._connect_weaviate()
        self._collection = self._client.collections.get(collection_name)
        self._embedder = self._load_embedder(embedding_model_name)

    @staticmethod
    def _load_embedder(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for Weaviate retrieval. "
                "Install with: pip install sentence-transformers"
            ) from exc
        return SentenceTransformer(model_name)

    def _connect_weaviate(self):
        try:
            import weaviate
        except ImportError as exc:
            raise RuntimeError(
                "weaviate-client is required for Weaviate retrieval. "
                "Install with: pip install weaviate-client"
            ) from exc

        url = self.weaviate_url or os.getenv("WEAVIATE_URL")
        if url:
            parsed = urlparse(url)
            http_host = parsed.hostname or self.http_host
            http_port = parsed.port or self.http_port
            secure = parsed.scheme == "https"
            grpc_host = os.getenv("WEAVIATE_GRPC_HOST", self.grpc_host or http_host)
            grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", str(self.grpc_port)))
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
                host=self.http_host,
                port=self.http_port,
                grpc_port=self.grpc_port,
                skip_init_checks=False,
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Failed to connect to local Weaviate. Start Weaviate server "
                f"on {self.http_host}:{self.http_port} and gRPC {self.grpc_port}. Error: {exc}"
            ) from exc

    @classmethod
    def from_manifest(
        cls,
        index_manifest_path: Optional[Path] = None,
        source_model_path: Optional[Path] = None,
        source_boost_weight: float = 0.45,
    ) -> "WeaviateRetriever":
        manifest = load_index(index_manifest_path)
        settings = RAGSettings()
        if source_model_path is None:
            source_model_path = settings.source_router_model_path
        source_model = _load_source_model(source_model_path)
        connection = manifest.get("connection", {})
        return cls(
            collection_name=manifest.get("collection_name", settings.weaviate_collection_name),
            embedding_model_name=manifest.get("embedding_model_name", settings.embedding_model_name),
            weaviate_url=connection.get("weaviate_url"),
            http_host=connection.get("http_host", settings.weaviate_http_host),
            http_port=int(connection.get("http_port", settings.weaviate_http_port)),
            grpc_host=connection.get("grpc_host", settings.weaviate_grpc_host),
            grpc_port=int(connection.get("grpc_port", settings.weaviate_grpc_port)),
            secure=bool(connection.get("secure", settings.weaviate_secure)),
            source_model=source_model,
            source_boost_weight=source_boost_weight,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        source_type: Optional[str] = None,
    ) -> List[Dict]:
        if not query.strip():
            return []

        from weaviate.classes.query import Filter, MetadataQuery

        query_vector = self._embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

        filters = None
        if source_type:
            filters = Filter.by_property("source_type").equal(source_type)

        raw_limit = max(top_k * 6, top_k)
        response = self._collection.query.near_vector(
            near_vector=query_vector,
            limit=raw_limit,
            filters=filters,
            return_metadata=MetadataQuery(distance=True),
        )

        source_distribution = _predict_source_distribution(self.source_model, query=query)
        reason_category = _extract_reason_category(query)

        reranked = []
        for item in response.objects:
            metadata = {
                "source_file": item.properties.get("source_file"),
                "source_type": item.properties.get("source_type"),
                "title": item.properties.get("title"),
                "chunk_index": item.properties.get("chunk_index"),
            }
            chunk_source = metadata.get("source_type")
            distance = float(getattr(item.metadata, "distance", 1.0) or 1.0)
            base_score = max(0.0, 1.0 - distance)
            source_prob = source_distribution.get(chunk_source, 0.0)
            adjusted = _adjusted_score(
                base_score=base_score,
                source_type=chunk_source,
                source_prob=source_prob,
                reason_category=reason_category,
                source_boost_weight=self.source_boost_weight,
            )
            reranked.append(
                {
                    "chunk_id": item.properties.get("chunk_id"),
                    "score": round(float(adjusted), 6),
                    "base_score": round(float(base_score), 6),
                    "source_model_probability": round(float(source_prob), 6),
                    "text": item.properties.get("text", ""),
                    "metadata": metadata,
                }
            )

        reranked.sort(key=lambda obj: obj["score"], reverse=True)
        output = []
        for obj in reranked:
            if obj["score"] < min_score:
                continue
            output.append(obj)
            if len(output) >= top_k:
                break
        return output

    def close(self) -> None:
        self._client.close()


def build_default_retriever(
    index_path: Optional[Path] = None,
    source_model_path: Optional[Path] = None,
    source_boost_weight: Optional[float] = None,
):
    rag_settings = RAGSettings()
    if source_boost_weight is None:
        source_boost_weight = rag_settings.source_boost_weight

    backend = (rag_settings.backend or "weaviate").lower().strip()
    if backend == "weaviate":
        return WeaviateRetriever.from_manifest(
            index_manifest_path=index_path,
            source_model_path=source_model_path,
            source_boost_weight=source_boost_weight,
        )

    return SparseRetriever.from_path(
        index_path=index_path,
        source_model_path=source_model_path,
        source_boost_weight=source_boost_weight,
    )
