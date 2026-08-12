#!/usr/bin/env python3
"""Reconstructed MMLU option-order robustness audit.

This script follows the documented July 2026 protocol. It is a transparent
reconstruction and is not claimed to be byte-for-byte identical to the
original local script.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from audit_utils import (
    inverse_map_prediction,
    rotate_answer_index,
    rotate_choices,
    sha256_file,
    top_diagnostics,
)

LABELS = ["A", "B", "C", "D"]
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_DATASET = "cais/mmlu"
DEFAULT_DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def format_prompt(question, choices):
    """Frozen raw-completion prompt used by the regenerated 2026-08-12 runs."""
    lines = [question.strip(), ""]
    for label, choice in zip(LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("Answer:")
    return "\n".join(lines)


def label_token_ids(tokenizer):
    ids = []
    for label in LABELS:
        candidates = tokenizer.encode(label, add_special_tokens=False)
        spaced = tokenizer.encode(" " + label, add_special_tokens=False)
        token_ids = spaced if len(spaced) == 1 else candidates
        if len(token_ids) != 1:
            raise ValueError(f"Label {label} is not a single token for this tokenizer")
        ids.append(token_ids[0])
    return ids


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(requested: str):
    mapping = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[requested]


def score_prompt(model, tokenizer, prompt, token_ids, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    selected = logits[token_ids]
    probs = torch.softmax(selected.float(), dim=0).cpu().numpy()
    pred = int(np.argmax(probs))
    diagnostics = top_diagnostics(probs)
    return pred, float(probs[pred]), probs.tolist(), diagnostics


def write_provenance(args, out: Path, device: torch.device, model) -> None:
    payload = {
        "schema_version": 1,
        "protocol": "MMLU all/test; 300 fixed-seed questions; four cyclic rotations; raw completion prompt",
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset": DEFAULT_DATASET,
        "dataset_revision": args.dataset_revision,
        "n": args.n,
        "seed": args.seed,
        "requested_dtype": args.dtype,
        "resolved_model_dtype": str(getattr(model, "dtype", "unknown")),
        "requested_device": args.device,
        "resolved_device": str(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "output": out.name,
        "output_sha256": sha256_file(out),
        "prompt_format": "question + blank line + A-D choices + blank line + 'Answer:'; no chat template",
    }
    path = out.with_suffix(out.suffix + ".provenance.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote provenance to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--out", default="audit_results.parquet")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset(
        DEFAULT_DATASET,
        "all",
        split="test",
        revision=args.dataset_revision,
    )
    if args.n > len(dataset):
        raise ValueError(f"Requested {args.n} questions but split has only {len(dataset)}")

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.n]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.model_revision,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=resolve_dtype(args.dtype),
    )
    device = resolve_device(args.device)
    model.to(device)
    model.eval()
    token_ids = label_token_ids(tokenizer)

    rows = []
    for q_rank, idx in enumerate(indices):
        item = dataset[idx]
        question = item["question"]
        choices = list(item["choices"])
        answer = int(item["answer"])
        subject = item.get("subject", "")

        for r in range(4):
            rotated = rotate_choices(choices, r)
            rotated_answer = rotate_answer_index(answer, r)
            prompt = format_prompt(question, rotated)
            pred_display, confidence, probs, diag = score_prompt(
                model, tokenizer, prompt, token_ids, device
            )
            pred_underlying = inverse_map_prediction(pred_display, r)
            rows.append(
                {
                    "question_rank": q_rank,
                    "dataset_index": idx,
                    "subject": subject,
                    "rotation": r,
                    "answer_underlying": answer,
                    "answer_display": rotated_answer,
                    "pred_display": pred_display,
                    "pred_underlying": pred_underlying,
                    "correct": int(pred_underlying == answer),
                    "confidence": confidence,
                    "prob_A": probs[0],
                    "prob_B": probs[1],
                    "prob_C": probs[2],
                    "prob_D": probs[3],
                    "top1_probability": diag["top1_probability"],
                    "top2_probability": diag["top2_probability"],
                    "top1_top2_margin": diag["top1_top2_margin"],
                    "top_tie_count": diag["top_tie_count"],
                    "top_exact_tie": diag["top_exact_tie"],
                    "top_tie_indices": ",".join(map(str, diag["top_tie_indices"])),
                    "label_entropy": diag["label_entropy"],
                    "model": args.model,
                    "model_revision": args.model_revision,
                    "dataset_revision": args.dataset_revision,
                    "requested_dtype": args.dtype,
                    "resolved_model_dtype": str(model.dtype),
                    "device": str(device),
                    "seed": args.seed,
                }
            )

    out = Path(args.out)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"Wrote {len(rows)} predictions to {out}")
    write_provenance(args, out, device, model)


if __name__ == "__main__":
    main()
