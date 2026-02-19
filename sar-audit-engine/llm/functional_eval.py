from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple


REQUIRED_SECTIONS = [
    "Activity Summary:",
    "Key Suspicious Indicators:",
    "Evidence and Rationale:",
    "Recommended Action:",
]


@dataclass
class PromptFields:
    case_id: str
    typology: str
    risk_score: str
    risk_band: str
    reasons: List[Dict[str, str]]


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def _f1_from_counters(pred: Counter, ref: Counter) -> float:
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(sum(pred.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    if precision + recall == 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _lcs_length(lhs: Sequence[str], rhs: Sequence[str]) -> int:
    if not lhs or not rhs:
        return 0
    previous = [0] * (len(rhs) + 1)
    current = [0] * (len(rhs) + 1)
    for left_tok in lhs:
        for idx, right_tok in enumerate(rhs, start=1):
            if left_tok == right_tok:
                current[idx] = previous[idx - 1] + 1
            else:
                current[idx] = max(current[idx - 1], previous[idx])
        previous, current = current, [0] * (len(rhs) + 1)
    return previous[-1]


def _simple_rouge(prediction: str, reference: str) -> Dict[str, float]:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    rouge1 = _f1_from_counters(_ngram_counts(pred_tokens, 1), _ngram_counts(ref_tokens, 1))
    rouge2 = _f1_from_counters(_ngram_counts(pred_tokens, 2), _ngram_counts(ref_tokens, 2))

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / max(len(pred_tokens), 1)
    recall = lcs / max(len(ref_tokens), 1)
    rouge_l = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rouge_l}


def compute_rouge(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, object]:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length for ROUGE computation.")
    if not predictions:
        return {"available": False, "reason": "No predictions available for scoring."}

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        aggregate = {"rouge1": [], "rouge2": [], "rougeL": []}
        for prediction, reference in zip(predictions, references):
            local = _simple_rouge(prediction, reference)
            for key, value in local.items():
                aggregate[key].append(value)
        return {
            "available": True,
            "backend": "fallback_simple",
            "rouge1_f1": float(mean(aggregate["rouge1"])),
            "rouge2_f1": float(mean(aggregate["rouge2"])),
            "rougeL_f1": float(mean(aggregate["rougeL"])),
        }

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregate = {"rouge1": [], "rouge2": [], "rougeL": []}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        aggregate["rouge1"].append(scores["rouge1"].fmeasure)
        aggregate["rouge2"].append(scores["rouge2"].fmeasure)
        aggregate["rougeL"].append(scores["rougeL"].fmeasure)

    return {
        "available": True,
        "backend": "rouge_score",
        "rouge1_f1": float(mean(aggregate["rouge1"])),
        "rouge2_f1": float(mean(aggregate["rouge2"])),
        "rougeL_f1": float(mean(aggregate["rougeL"])),
    }


def compute_bertscore(
    predictions: Sequence[str],
    references: Sequence[str],
    enabled: bool,
    model_type: str,
    device: str,
) -> Dict[str, object]:
    if not enabled:
        return {"available": False, "reason": "Disabled by configuration."}
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have equal length for BERTScore computation.")
    if not predictions:
        return {"available": False, "reason": "No predictions available for scoring."}

    try:
        from bert_score import score as bert_score
    except ImportError:
        return {"available": False, "reason": "Missing optional dependency: bert-score."}

    precision, recall, f1 = bert_score(
        cands=list(predictions),
        refs=list(references),
        model_type=model_type,
        lang="en",
        verbose=False,
        device=device,
    )
    return {
        "available": True,
        "model_type": model_type,
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
    }


def parse_prompt_fields(user_text: str) -> PromptFields:
    case_id = ""
    typology = ""
    risk_score = ""
    risk_band = ""

    reasons: List[Dict[str, str]] = []
    reason_line_pattern = re.compile(
        r"^\s*-\s*(R\d+):\s*(.*?)\s+\(category=([^,]+),\s*tx_count=(\d+)\)\s*$"
    )

    for raw_line in user_text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("case id:"):
            case_id = line.split(":", maxsplit=1)[1].strip()
            continue
        if line.lower().startswith("typology:"):
            typology = line.split(":", maxsplit=1)[1].strip()
            continue
        if line.lower().startswith("risk score:"):
            risk_score = line.split(":", maxsplit=1)[1].strip()
            continue
        if line.lower().startswith("risk band:"):
            risk_band = line.split(":", maxsplit=1)[1].strip()
            continue

        match = reason_line_pattern.match(line)
        if match:
            reasons.append(
                {
                    "reason_id": match.group(1).strip(),
                    "claim": match.group(2).strip(),
                    "category": match.group(3).strip(),
                    "tx_count": match.group(4).strip(),
                }
            )

    return PromptFields(
        case_id=case_id,
        typology=typology,
        risk_score=risk_score,
        risk_band=risk_band,
        reasons=reasons,
    )


def build_template_baseline(system_text: str, user_text: str) -> str:
    _ = system_text
    fields = parse_prompt_fields(user_text)

    reason_lines = []
    for reason in fields.reasons:
        reason_lines.append(
            f"- {reason['claim']} (category: {reason['category']}, linked_tx: {reason['tx_count']})"
        )
    reasons_block = (
        "\n".join(reason_lines)
        if reason_lines
        else "- No explicit suspicious indicators were generated."
    )

    return (
        "Activity Summary:\n"
        f"Case {fields.case_id or 'unknown'} was reviewed under typology {fields.typology or 'unknown'}. "
        f"The case is currently assessed as {fields.risk_band or 'unknown'} risk "
        f"with score {fields.risk_score or 'n/a'}.\n\n"
        "Key Suspicious Indicators:\n"
        f"{reasons_block}\n\n"
        "Evidence and Rationale:\n"
        "The risk assessment combines transactional signals with explicit reasoning steps. "
        "Each reason is linked to transaction identifiers and supporting knowledge chunks "
        "retrieved from SAR typology/regulatory reference material.\n\n"
        "Recommended Action:\n"
        "Escalate for SAR filing review, preserve an audit trail of supporting evidence, "
        "and request analyst sign-off for final narrative submission."
    )


def compute_aml_adherence(user_text: str, narrative: str) -> Dict[str, float]:
    fields = parse_prompt_fields(user_text)
    lowered_narrative = narrative.lower()

    section_hits = 0
    for section in REQUIRED_SECTIONS:
        if section.lower() in lowered_narrative:
            section_hits += 1
    section_compliance = section_hits / len(REQUIRED_SECTIONS)

    required_fact_tokens = [
        fields.case_id.lower(),
        fields.typology.lower(),
        fields.risk_score.lower(),
        fields.risk_band.lower(),
    ]
    required_fact_tokens = [tok for tok in required_fact_tokens if tok]
    if required_fact_tokens:
        fact_hits = sum(1 for tok in required_fact_tokens if tok in lowered_narrative)
        fact_grounding = fact_hits / len(required_fact_tokens)
    else:
        fact_grounding = 0.0

    if fields.reasons:
        reason_hits = 0
        tx_hits = 0
        for reason in fields.reasons:
            claim = reason["claim"].lower()
            if claim and claim in lowered_narrative:
                reason_hits += 1
            tx_count = reason["tx_count"]
            if tx_count and tx_count in lowered_narrative:
                tx_hits += 1
        reason_coverage = reason_hits / len(fields.reasons)
        tx_count_grounding = tx_hits / len(fields.reasons)
    else:
        reason_coverage = 0.0
        tx_count_grounding = 0.0

    adherence = (
        (0.35 * section_compliance)
        + (0.35 * reason_coverage)
        + (0.20 * fact_grounding)
        + (0.10 * tx_count_grounding)
    )
    return {
        "section_compliance": float(section_compliance),
        "reason_coverage": float(reason_coverage),
        "fact_grounding": float(fact_grounding),
        "tx_count_grounding": float(tx_count_grounding),
        "aml_adherence_score": float(adherence),
    }


def _extract_json_object(text: str) -> Optional[Dict]:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            continue
    return None


def _judge_prompt(user_text: str, reference: str, candidate: str) -> str:
    return (
        "You are an AML SAR narrative quality judge.\n"
        "Score the candidate narrative against prompt requirements and reference narrative.\n"
        "Return JSON only with keys:\n"
        "{"
        '"factual_consistency": float, '
        '"regulatory_adherence": float, '
        '"completeness": float, '
        '"overall": float, '
        '"rationale": string'
        "}\n"
        "Use score range [0,1].\n\n"
        f"Prompt:\n{user_text}\n\n"
        f"Reference Narrative:\n{reference}\n\n"
        f"Candidate Narrative:\n{candidate}\n"
    )


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    import torch

    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    max_positions = int(getattr(model.config, "max_position_embeddings", 2048))
    max_input_tokens = max(max_positions - int(max_new_tokens), 32)
    input_ids = inputs["input_ids"][:, -max_input_tokens:]
    attention_mask = inputs["attention_mask"][:, -max_input_tokens:]

    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _aggregate_aml(scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return {
            "section_compliance": 0.0,
            "reason_coverage": 0.0,
            "fact_grounding": 0.0,
            "tx_count_grounding": 0.0,
            "aml_adherence_score": 0.0,
        }
    keys = list(scores[0].keys())
    return {key: float(mean(score[key] for score in scores)) for key in keys}


def _overall_score(rouge: Dict[str, object], bertscore: Dict[str, object], aml: Dict[str, float]) -> float:
    components: List[float] = []
    if rouge.get("available"):
        components.append(float(rouge.get("rougeL_f1", 0.0)))
    if bertscore.get("available"):
        components.append(float(bertscore.get("f1", 0.0)))
    components.append(float(aml.get("aml_adherence_score", 0.0)))
    if not components:
        return 0.0
    return float(mean(components))


def _delta(candidate_score: float, baseline_score: float) -> float:
    return float(candidate_score - baseline_score)


def evaluate_model_variants(
    eval_rows: Sequence[Dict],
    tokenizer,
    fine_tuned_model,
    base_model_name: str,
    max_samples: int,
    generation_max_new_tokens: int,
    run_bertscore: bool,
    bertscore_model: str,
    judge_model_name: Optional[str],
    judge_max_samples: int,
    judge_max_new_tokens: int,
) -> Dict[str, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation dependencies are missing. Install with: pip install transformers peft accelerate"
        ) from exc

    sampled_rows = list(eval_rows[: max(0, int(max_samples))])
    if not sampled_rows:
        return {"available": False, "reason": "No validation rows available for functional evaluation."}

    model_variants = {
        "fine_tuned": [],
        "base_model": [],
        "template_baseline": [],
    }
    references: List[str] = []
    aml_scores: Dict[str, List[Dict[str, float]]] = {
        "fine_tuned": [],
        "base_model": [],
        "template_baseline": [],
    }
    sample_outputs: List[Dict[str, object]] = []

    base_model = None
    can_disable_adapter = hasattr(fine_tuned_model, "disable_adapter")

    for row in sampled_rows:
        messages = row.get("messages", []) or []
        system_text = ""
        user_text = ""
        assistant_text = ""
        for msg in messages:
            role = _safe_text(msg.get("role"))
            content = _safe_text(msg.get("content"))
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content
        if not assistant_text:
            assistant_text = _safe_text(row.get("completion"))

        prompt = "<|system|>\n" + system_text + "\n<|user|>\n" + user_text + "\n<|assistant|>\n"
        references.append(assistant_text)

        fine_tuned_prediction = _generate_text(
            model=fine_tuned_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=generation_max_new_tokens,
        )

        base_prediction = ""
        if can_disable_adapter:
            with fine_tuned_model.disable_adapter():
                base_prediction = _generate_text(
                    model=fine_tuned_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=generation_max_new_tokens,
                )
        else:
            if base_model is None:
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                base_model.config.pad_token_id = tokenizer.pad_token_id
                base_model.to(next(fine_tuned_model.parameters()).device)
            base_prediction = _generate_text(
                model=base_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=generation_max_new_tokens,
            )

        template_prediction = build_template_baseline(system_text=system_text, user_text=user_text)

        model_variants["fine_tuned"].append(fine_tuned_prediction)
        model_variants["base_model"].append(base_prediction)
        model_variants["template_baseline"].append(template_prediction)

        aml_scores["fine_tuned"].append(compute_aml_adherence(user_text=user_text, narrative=fine_tuned_prediction))
        aml_scores["base_model"].append(compute_aml_adherence(user_text=user_text, narrative=base_prediction))
        aml_scores["template_baseline"].append(
            compute_aml_adherence(user_text=user_text, narrative=template_prediction)
        )

        sample_outputs.append(
            {
                "case_id": row.get("case_id"),
                "prompt_user": user_text,
                "reference": assistant_text,
                "predictions": {
                    "fine_tuned": fine_tuned_prediction,
                    "base_model": base_prediction,
                    "template_baseline": template_prediction,
                },
                "aml_scores": {
                    "fine_tuned": aml_scores["fine_tuned"][-1],
                    "base_model": aml_scores["base_model"][-1],
                    "template_baseline": aml_scores["template_baseline"][-1],
                },
            }
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics: Dict[str, Dict[str, object]] = {}
    for variant_name, predictions in model_variants.items():
        rouge_scores = compute_rouge(predictions=predictions, references=references)
        bert_scores = compute_bertscore(
            predictions=predictions,
            references=references,
            enabled=run_bertscore,
            model_type=bertscore_model,
            device=device,
        )
        aml_aggregate = _aggregate_aml(aml_scores[variant_name])
        overall = _overall_score(rouge=rouge_scores, bertscore=bert_scores, aml=aml_aggregate)

        metrics[variant_name] = {
            "samples": len(predictions),
            "rouge": rouge_scores,
            "bertscore": bert_scores,
            "aml_adherence": aml_aggregate,
            "overall_score": overall,
        }

    comparison = {
        "fine_tuned_vs_base_model": {
            "overall_score_delta": _delta(
                metrics["fine_tuned"]["overall_score"],
                metrics["base_model"]["overall_score"],
            ),
            "rougeL_delta": _delta(
                float(metrics["fine_tuned"]["rouge"].get("rougeL_f1", 0.0)),
                float(metrics["base_model"]["rouge"].get("rougeL_f1", 0.0)),
            ),
            "aml_adherence_delta": _delta(
                float(metrics["fine_tuned"]["aml_adherence"].get("aml_adherence_score", 0.0)),
                float(metrics["base_model"]["aml_adherence"].get("aml_adherence_score", 0.0)),
            ),
        },
        "fine_tuned_vs_template_baseline": {
            "overall_score_delta": _delta(
                metrics["fine_tuned"]["overall_score"],
                metrics["template_baseline"]["overall_score"],
            ),
            "rougeL_delta": _delta(
                float(metrics["fine_tuned"]["rouge"].get("rougeL_f1", 0.0)),
                float(metrics["template_baseline"]["rouge"].get("rougeL_f1", 0.0)),
            ),
            "aml_adherence_delta": _delta(
                float(metrics["fine_tuned"]["aml_adherence"].get("aml_adherence_score", 0.0)),
                float(metrics["template_baseline"]["aml_adherence"].get("aml_adherence_score", 0.0)),
            ),
        },
    }

    judge_summary: Dict[str, object] = {"available": False, "reason": "Judge model not configured."}
    if judge_model_name:
        judge_summary = _run_llm_judge(
            judge_model_name=judge_model_name,
            sample_outputs=sample_outputs,
            max_samples=judge_max_samples,
            max_new_tokens=judge_max_new_tokens,
        )

    return {
        "available": True,
        "sample_count": len(sampled_rows),
        "metrics": metrics,
        "comparison": comparison,
        "judge": judge_summary,
        "samples": sample_outputs,
    }


def _run_llm_judge(
    judge_model_name: str,
    sample_outputs: Sequence[Dict[str, object]],
    max_samples: int,
    max_new_tokens: int,
) -> Dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return {"available": False, "reason": "transformers is required for judge evaluation."}

    subset = list(sample_outputs[: max(0, int(max_samples))])
    if not subset:
        return {"available": False, "reason": "No samples available for judge evaluation."}

    try:
        tokenizer = AutoTokenizer.from_pretrained(judge_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(judge_model_name)
    except Exception as exc:
        return {"available": False, "reason": f"Unable to load judge model: {exc}"}

    variant_scores: Dict[str, List[float]] = {
        "fine_tuned": [],
        "base_model": [],
        "template_baseline": [],
    }
    parser_failures = 0
    raw_responses: List[Dict[str, object]] = []

    for sample in subset:
        user_text = _safe_text(sample.get("prompt_user"))
        reference = _safe_text(sample.get("reference"))
        predictions = sample.get("predictions", {}) or {}
        case_id = sample.get("case_id")

        per_case_judgement: Dict[str, object] = {"case_id": case_id, "variants": {}}
        for variant in ("fine_tuned", "base_model", "template_baseline"):
            candidate = _safe_text(predictions.get(variant))
            prompt = _judge_prompt(user_text=user_text, reference=reference, candidate=candidate)
            response = _generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )
            payload = _extract_json_object(response)
            if payload is None:
                parser_failures += 1
                per_case_judgement["variants"][variant] = {
                    "parse_success": False,
                    "raw_response": response,
                }
                continue

            overall = payload.get("overall")
            try:
                numeric_overall = float(overall)
            except (TypeError, ValueError):
                parser_failures += 1
                per_case_judgement["variants"][variant] = {
                    "parse_success": False,
                    "raw_response": response,
                }
                continue

            numeric_overall = min(max(numeric_overall, 0.0), 1.0)
            variant_scores[variant].append(numeric_overall)
            per_case_judgement["variants"][variant] = {
                "parse_success": True,
                "overall": numeric_overall,
                "factual_consistency": payload.get("factual_consistency"),
                "regulatory_adherence": payload.get("regulatory_adherence"),
                "completeness": payload.get("completeness"),
                "rationale": payload.get("rationale"),
            }

        raw_responses.append(per_case_judgement)

    aggregate = {}
    for variant, scores in variant_scores.items():
        aggregate[variant] = float(mean(scores)) if scores else None

    return {
        "available": True,
        "judge_model": judge_model_name,
        "samples": len(subset),
        "parse_failures": parser_failures,
        "aggregate_overall": aggregate,
        "raw_results": raw_responses,
    }


def write_functional_eval_artifacts(
    output_dir: Path,
    functional_eval: Dict[str, object],
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "functional_eval.json"
    summary_path.write_text(json.dumps(functional_eval, indent=2), encoding="utf-8")

    samples_path = output_dir / "functional_eval_samples.jsonl"
    samples = functional_eval.get("samples", []) or []
    with samples_path.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    return {
        "functional_eval_json": str(summary_path),
        "functional_eval_samples_jsonl": str(samples_path),
    }
