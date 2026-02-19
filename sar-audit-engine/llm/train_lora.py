from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from .functional_eval import evaluate_model_variants, write_functional_eval_artifacts


def _load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _message_content(messages: List[Dict], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            return str(msg.get("content") or "").strip()
    return ""


def _render_prompt_and_target(row: Dict) -> Tuple[str, str]:
    messages = row.get("messages", []) or []
    system_text = _message_content(messages, "system")
    user_text = _message_content(messages, "user")
    assistant_text = _message_content(messages, "assistant")
    if not assistant_text:
        assistant_text = str(row.get("completion") or "").strip()

    prompt = (
        "<|system|>\n"
        f"{system_text}\n"
        "<|user|>\n"
        f"{user_text}\n"
        "<|assistant|>\n"
    )
    return prompt, assistant_text


def _build_features(rows: List[Dict], tokenizer, max_length: int) -> List[Dict]:
    encoded_rows: List[Dict] = []
    eos_id = tokenizer.eos_token_id

    for row in rows:
        prompt_text, answer_text = _render_prompt_and_target(row)
        if not answer_text:
            continue

        prompt_ids_full = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        answer_ids = tokenizer(
            answer_text,
            add_special_tokens=False,
        )["input_ids"]

        eos_budget = 1 if eos_id is not None else 0
        max_answer_len = max(int(max_length) - eos_budget, 1)
        if len(answer_ids) > max_answer_len:
            answer_ids = answer_ids[:max_answer_len]

        remaining_for_prompt = max(int(max_length) - len(answer_ids) - eos_budget, 0)
        if remaining_for_prompt > 0:
            prompt_ids = prompt_ids_full[-remaining_for_prompt:]
        else:
            prompt_ids = []

        full_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids.copy()

        if eos_id is not None and len(full_ids) < int(max_length):
            full_ids.append(eos_id)
            labels.append(eos_id)

        if not any(label != -100 for label in labels):
            continue

        encoded_rows.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            }
        )
    return encoded_rows


class _ListDataset:
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        return self.rows[idx]


class _PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict]):
        import torch

        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for row in batch:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(row["attention_mask"] + [0] * pad_len)
            labels.append(row["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _infer_lora_targets(model) -> List[str]:
    import torch

    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
    }
    found = set()
    all_suffixes = set()
    for module_name, module in model.named_modules():
        suffix = module_name.split(".")[-1]
        all_suffixes.add(suffix)
        if isinstance(module, torch.nn.Linear):
            if suffix in candidates:
                found.add(suffix)

    if found:
        return sorted(found)

    # Fallback for architectures that use non-Linear projection modules (for example GPT2 Conv1D).
    fallback_order = ["q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]
    fallback = [name for name in fallback_order if name in all_suffixes]
    if fallback:
        return fallback

    raise ValueError("Unable to infer LoRA target modules for the selected base model.")


def train_lora(
    train_jsonl: Path,
    output_dir: Path,
    base_model: str,
    val_jsonl: Path | None = None,
    max_length: int = 1024,
    epochs: float = 2.0,
    learning_rate: float = 2e-4,
    train_batch_size: int = 1,
    eval_batch_size: int = 1,
    grad_accum_steps: int = 8,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    run_functional_eval: bool = True,
    functional_eval_max_samples: int = 64,
    functional_eval_max_new_tokens: int = 384,
    run_bertscore: bool = True,
    bertscore_model: str = "distilbert-base-uncased",
    judge_model: str | None = None,
    judge_max_samples: int = 8,
    judge_max_new_tokens: int = 192,
) -> Dict:
    try:
        import inspect
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Fine-tuning dependencies are missing. Install with: "
            "pip install transformers peft accelerate wandb"
        ) from exc

    train_rows = _load_jsonl(Path(train_jsonl))
    val_rows = _load_jsonl(Path(val_jsonl)) if val_jsonl else []
    if not train_rows:
        raise ValueError(f"No training rows found in {train_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_features = _build_features(train_rows, tokenizer=tokenizer, max_length=max_length)
    if not train_features:
        raise ValueError("No valid train rows after prompt/label build.")
    val_features = _build_features(val_rows, tokenizer=tokenizer, max_length=max_length) if val_rows else []

    train_dataset = _ListDataset(train_features)
    eval_dataset = _ListDataset(val_features) if val_features else None

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.config.pad_token_id = tokenizer.pad_token_id

    target_modules = _infer_lora_targets(model)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp16 = bool(torch.cuda.is_available())
    bf16 = False

    # Initialize W&B tracking
    use_wandb = eval_dataset is not None
    if use_wandb:
        wandb.init(
            project="sar-lora-training",
            name=f"lora-{base_model.split('/')[-1]}-{epochs}ep-lr{learning_rate}",
            config={
                "base_model": base_model,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "train_samples": len(train_features),
                "val_samples": len(val_features),
                "max_length": max_length,
            },
            tags=["lora", "sar", "fine-tuning"]
        )

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(epochs),
        "learning_rate": float(learning_rate),
        "per_device_train_batch_size": int(train_batch_size),
        "per_device_eval_batch_size": int(eval_batch_size),
        "gradient_accumulation_steps": int(grad_accum_steps),
        "weight_decay": 0.0,
        "report_to": "wandb" if use_wandb else "none",
        "logging_strategy": "steps",
        "logging_steps": 20,
        "save_strategy": "steps",
        "save_steps": 100,
        "eval_steps": 100,
        "load_best_model_at_end": True if eval_dataset else False,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3,
        "fp16": fp16,
        "bf16": bf16,
    }
    eval_value = "steps" if eval_dataset is not None else "no"
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_args_kwargs["evaluation_strategy"] = eval_value
    else:
        training_args_kwargs["eval_strategy"] = eval_value

    args = TrainingArguments(**training_args_kwargs)

    callbacks = []
    if eval_dataset is not None:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=3
        )
        callbacks.append(early_stopping)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=_PadCollator(tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    trainer.train()

    if use_wandb:
        wandb.finish()

    train_metrics = trainer.evaluate(eval_dataset=train_dataset)
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset) if eval_dataset is not None else {}

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    functional_eval_summary: Dict[str, object] = {
        "available": False,
        "reason": "Functional evaluation disabled.",
    }
    if run_functional_eval:
        if not val_rows:
            functional_eval_summary = {
                "available": False,
                "reason": "Validation rows are required for functional evaluation.",
            }
        else:
            try:
                functional_eval = evaluate_model_variants(
                    eval_rows=val_rows,
                    tokenizer=tokenizer,
                    fine_tuned_model=model,
                    base_model_name=base_model,
                    max_samples=int(functional_eval_max_samples),
                    generation_max_new_tokens=int(functional_eval_max_new_tokens),
                    run_bertscore=bool(run_bertscore),
                    bertscore_model=bertscore_model,
                    judge_model_name=judge_model,
                    judge_max_samples=int(judge_max_samples),
                    judge_max_new_tokens=int(judge_max_new_tokens),
                )
                artifacts = write_functional_eval_artifacts(output_dir=output_dir, functional_eval=functional_eval)
                functional_eval_summary = {
                    "available": bool(functional_eval.get("available")),
                    "sample_count": functional_eval.get("sample_count"),
                    "metrics": functional_eval.get("metrics"),
                    "comparison": functional_eval.get("comparison"),
                    "judge": functional_eval.get("judge"),
                    "artifacts": artifacts,
                }
            except Exception as exc:
                functional_eval_summary = {
                    "available": False,
                    "reason": f"Functional evaluation failed: {exc}",
                }

    summary = {
        "base_model": base_model,
        "output_dir": str(output_dir),
        "train_rows": len(train_features),
        "val_rows": len(val_features),
        "max_length": int(max_length),
        "epochs": float(epochs),
        "learning_rate": float(learning_rate),
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_target_modules": target_modules,
        "train_loss": train_metrics.get("eval_loss"),
        "train_perplexity": (
            float(math.exp(train_metrics["eval_loss"])) if train_metrics.get("eval_loss") is not None else None
        ),
        "val_loss": eval_metrics.get("eval_loss"),
        "val_perplexity": (
            float(math.exp(eval_metrics["eval_loss"])) if eval_metrics.get("eval_loss") is not None else None
        ),
        "functional_eval": functional_eval_summary,
    }
    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA SFT model for SAR narrative generation.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--skip-functional-eval", action="store_true")
    parser.add_argument("--functional-eval-max-samples", type=int, default=64)
    parser.add_argument("--functional-eval-max-new-tokens", type=int, default=384)
    parser.add_argument("--disable-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--judge-max-samples", type=int, default=8)
    parser.add_argument("--judge-max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_lora(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        base_model=args.base_model,
        output_dir=args.output_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        run_functional_eval=not args.skip_functional_eval,
        functional_eval_max_samples=args.functional_eval_max_samples,
        functional_eval_max_new_tokens=args.functional_eval_max_new_tokens,
        run_bertscore=not args.disable_bertscore,
        bertscore_model=args.bertscore_model,
        judge_model=args.judge_model,
        judge_max_samples=args.judge_max_samples,
        judge_max_new_tokens=args.judge_max_new_tokens,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
