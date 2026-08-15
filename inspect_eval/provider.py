"""Inspect model provider that reproduces the audit's four-label next-token scoring.

The built-in Inspect Hugging Face provider is chat-oriented. The historical MMLU
regeneration used a raw completion prompt and selected the highest next-token
logit among A/B/C/D only. This provider keeps that exact scoring boundary while
letting Inspect own task execution, logging, scoring, and aggregation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from inspect_ai.model import ChatMessage, GenerateConfig, ModelAPI, ModelOutput, modelapi
from inspect_ai.tool import ToolChoice, ToolInfo

LABELS = ("A", "B", "C", "D")
DEFAULT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def _top_diagnostics(probs: list[float]) -> dict[str, object]:
    arr = np.asarray(probs, dtype=float)
    order = np.argsort(-arr, kind="stable")
    top1 = float(arr[order[0]])
    top2 = float(arr[order[1]])
    ties = np.flatnonzero(arr == top1).astype(int).tolist()
    entropy = float(-(arr[arr > 0] * np.log(arr[arr > 0])).sum())
    return {
        "top1_probability": top1,
        "top2_probability": top2,
        "top1_top2_margin": float(top1 - top2),
        "top_tie_count": len(ties),
        "top_exact_tie": len(ties) > 1,
        "top_tie_indices": ties,
        "label_entropy": entropy,
    }


class MMLULabelLogitAPI(ModelAPI):
    """Local Hugging Face provider for normalized A/B/C/D next-token scores."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        revision = str(model_args.pop("revision", DEFAULT_REVISION))
        dtype_name = str(model_args.pop("dtype", "float32"))
        device_name = str(model_args.pop("device", "cpu"))
        if model_args:
            unknown = ", ".join(sorted(model_args))
            raise ValueError(f"Unsupported mmlu-labels model args: {unknown}")

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype {dtype_name!r}; choose {sorted(dtype_map)}")

        self.revision = revision
        self.dtype_name = dtype_name
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype_map[dtype_name],
        )
        self.model.to(self.device)
        self.model.eval()
        self.label_token_ids = self._label_token_ids()

    def _label_token_ids(self) -> list[int]:
        ids: list[int] = []
        for label in LABELS:
            plain = self.tokenizer.encode(label, add_special_tokens=False)
            spaced = self.tokenizer.encode(" " + label, add_special_tokens=False)
            token_ids = spaced if len(spaced) == 1 else plain
            if len(token_ids) != 1:
                raise ValueError(
                    f"Label {label} is not a single token for {self.model_name} tokenizer"
                )
            ids.append(int(token_ids[0]))
        return ids

    def max_connections(self) -> int:
        # One local model instance; avoid concurrent forwards on CPU/MPS.
        return 1

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools:
            raise ValueError("mmlu-labels provider does not support tools")
        if not input:
            raise ValueError("mmlu-labels provider received an empty prompt")

        import torch

        # Important: use the raw final message text exactly. Do not apply a chat
        # template or add role prefixes; the July/August audit used raw completion.
        prompt = input[-1].text
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits[0, -1]
        selected = logits[self.label_token_ids]
        probs_arr = torch.softmax(selected.float(), dim=0).cpu().numpy()
        probs = [float(x) for x in probs_arr]
        pred = int(np.argmax(probs_arr))
        diag = _top_diagnostics(probs)

        output = ModelOutput.from_content(model=self.model_name, content=LABELS[pred])
        output.metadata = {
            "protocol": "normalized next-token logits over A/B/C/D",
            "model_revision": self.revision,
            "requested_dtype": self.dtype_name,
            "resolved_model_dtype": str(self.model.dtype),
            "device": str(self.device),
            "prompt_format": "raw completion; no chat template",
            "pred_display": pred,
            "confidence": probs[pred],
            "four_label_probs": probs,
            **diag,
        }
        return output


@modelapi(name="mmlu-labels")
def mmlu_labels() -> type[ModelAPI]:
    """Register the exact four-label local Hugging Face provider with Inspect."""

    return MMLULabelLogitAPI
