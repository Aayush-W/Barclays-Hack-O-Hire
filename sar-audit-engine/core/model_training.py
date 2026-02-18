from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _load_training_rows(training_jsonl_path: Path) -> List[Dict]:
    rows = []
    with training_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_feature_text(row: Dict) -> str:
    query = row.get("query", "")
    metrics = row.get("metrics", {}) or {}
    reason_category = row.get("reason_category", "")
    typology = row.get("typology", "")
    metric_tokens = []
    for key in sorted(metrics.keys()):
        value = metrics.get(key)
        if value is None:
            continue
        metric_tokens.append(f"{key}_{value}")
    return (
        f"{query} reason_category {reason_category} typology {typology} {' '.join(metric_tokens)}"
    ).strip()


def _primary_label_from_contexts(row: Dict) -> str:
    primary = row.get("primary_source_type")
    if primary:
        return str(primary)

    score_by_source = defaultdict(float)
    for rank, context in enumerate(row.get("positive_contexts", []), start=1):
        source_type = context.get("source_type")
        if not source_type:
            continue
        context_score = context.get("score")
        if context_score is None:
            context_score = 1.0 / rank
        try:
            score = float(context_score)
        except (TypeError, ValueError):
            score = 1.0 / rank
        score_by_source[str(source_type)] += score

    if not score_by_source:
        return ""
    return max(score_by_source.items(), key=lambda item: item[1])[0]


def _build_source_classification_samples(rows: List[Dict]) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []

    for row in rows:
        enriched_query = _build_feature_text(row)
        label = _primary_label_from_contexts(row)
        if enriched_query and label:
            texts.append(enriched_query)
            labels.append(label)

    return texts, labels


def train_retrieval_source_model(
    training_jsonl_path: Path,
    model_output_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn and joblib are required for model training. "
            "Install with: pip install scikit-learn joblib"
        ) from exc

    rows = _load_training_rows(training_jsonl_path=Path(training_jsonl_path))
    texts, labels = _build_source_classification_samples(rows)

    if len(texts) < 10:
        raise ValueError("Not enough training samples. Generate more cases before training.")
    if len(set(labels)) < 2:
        raise ValueError("Need at least two source_type classes to train this model.")

    label_distribution = dict(sorted(Counter(labels).items()))
    if len(label_distribution) < 2:
        raise ValueError(f"Need at least two classes after label cleanup. Got: {label_distribution}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=700,
                    C=3.0,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))

    model_output_path = Path(model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output_path)

    macro_f1 = float(f1_score(y_test, predictions, average="macro"))

    metadata = {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "classes": sorted(set(labels)),
        "label_distribution": label_distribution,
        "model_path": str(model_output_path),
    }
    metadata_path = model_output_path.with_suffix(model_output_path.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline retrieval helper model for SAR RAG.")
    parser.add_argument("--training-jsonl", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()

    metrics = train_retrieval_source_model(
        training_jsonl_path=args.training_jsonl,
        model_output_path=args.model_output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
