from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from llm.generator import generate_sar_narrative
from llm.prompt_templates import build_prompt_payload


class DetailedAuditCallback:
    def __init__(self):
        self.audit_log = []
        self.current_step = 0

    def log(self, event: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.current_step += 1
        self.audit_log.append(
            {
                "step": self.current_step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "details": details or {},
            }
        )

    def get_audit_trail(self):
        return list(self.audit_log)

    def save_audit_trail(self, filename: str = "audit_trail.json") -> None:
        with open(filename, "w", encoding="utf-8") as handle:
            json.dump(self.audit_log, handle, indent=2)


class SARNarrativeGenerator:
    """
    Mistral/LLM SAR narrative generator with deterministic fallback.
    """

    def __init__(
        self,
        model_name: str = "mistral:latest",
        provider: str = "ollama",
        ollama_base_url: str = "http://127.0.0.1:11434",
        adapter_path: str | Path | None = None,
        max_new_tokens: int = 384,
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.provider = str(provider or "ollama").strip().lower()
        self.ollama_base_url = str(ollama_base_url or "http://127.0.0.1:11434").rstrip("/")
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.callback = DetailedAuditCallback()

        self.tokenizer = None
        self.model = None

    def _ensure_model_loaded(self) -> None:
        if self.model is not None and self.tokenizer is not None:
            return

        self.callback.log("llm_load_start", {"model_name": self.model_name, "adapter_path": self.adapter_path})
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for narrative generation. Install with: pip install transformers"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        if self.adapter_path:
            adapter_dir = Path(self.adapter_path)
            if adapter_dir.exists():
                try:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, str(adapter_dir))
                    self.callback.log("adapter_loaded", {"adapter_path": str(adapter_dir)})
                except Exception as exc:
                    self.callback.log("adapter_load_failed", {"error": str(exc), "adapter_path": str(adapter_dir)})
            else:
                self.callback.log("adapter_not_found", {"adapter_path": str(adapter_dir)})

        model_device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(model_device)

        self.tokenizer = tokenizer
        self.model = model
        self.callback.log("llm_load_complete", {"device": model_device})

    def _generate_with_ollama(self, system_text: str, user_text: str) -> str:
        self.callback.log(
            "ollama_generate_start",
            {
                "model_name": self.model_name,
                "base_url": self.ollama_base_url,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
            },
        )
        payload = {
            "model": self.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
        }
        req = urlrequest.Request(
            url=f"{self.ollama_base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlrequest.urlopen(req, timeout=300) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urlerror.URLError as exc:
            raise RuntimeError(
                f"Failed to connect to Ollama at {self.ollama_base_url}. "
                "Ensure Ollama is running (`ollama serve`) and model is pulled."
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON response.") from exc

        message = response_payload.get("message", {}) or {}
        content = str(message.get("content") or "").strip()
        if not content:
            content = str(response_payload.get("response") or "").strip()
        self.callback.log("ollama_generate_complete", {"generated_chars": len(content)})
        return content

    def _render_chat_prompt(self, evidence_map: Dict[str, Any], style: str) -> str:
        payload = build_prompt_payload(evidence_map=evidence_map, style=style)
        system_text = payload.get("system", "")
        user_text = payload.get("user", "")
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                self.callback.log("chat_template_failed", {"error": str(exc)})
        return "System:\n" + system_text + "\n\nUser:\n" + user_text + "\n\nAssistant:\n"

    def _generate_with_model(self, prompt: str) -> str:
        import torch

        assert self.model is not None and self.tokenizer is not None
        self.model.eval()

        self.callback.log(
            "llm_generate_start",
            {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_device = next(self.model.parameters()).device
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        do_sample = self.temperature > 0.0
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = 0.95
        with torch.no_grad():
            output = self.model.generate(**generate_kwargs)

        completion_ids = output[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        self.callback.log("llm_generate_complete", {"generated_chars": len(text)})
        return text

    def generate_narrative(self, evidence_map: Dict[str, Any], style: str = "standard") -> Dict[str, Any]:
        try:
            prompt_payload = build_prompt_payload(evidence_map=evidence_map, style=style)
            system_text = prompt_payload.get("system", "")
            user_text = prompt_payload.get("user", "")

            if self.provider == "ollama":
                narrative = self._generate_with_ollama(system_text=system_text, user_text=user_text)
                backend_name = "ollama"
            else:
                self._ensure_model_loaded()
                prompt = self._render_chat_prompt(evidence_map=evidence_map, style=style)
                narrative = self._generate_with_model(prompt)
                backend_name = "huggingface_llm"

            if not narrative:
                raise RuntimeError("Model returned empty output.")

            return {
                "narrative": narrative,
                "audit_trail": self.callback.get_audit_trail(),
                "backend": backend_name,
            }
        except Exception as exc:
            self.callback.log("llm_failed_fallback", {"error": str(exc)})
            fallback = generate_sar_narrative(evidence_map=evidence_map, style=style)
            return {
                "narrative": fallback.get("narrative", ""),
                "audit_trail": self.callback.get_audit_trail(),
                "backend": "template_fallback",
                "fallback_reason": str(exc),
            }
