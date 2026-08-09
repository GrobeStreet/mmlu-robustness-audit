#!/usr/bin/env python3
"""Reconstructed MMLU option-order robustness audit.

This script follows the documented July 2026 protocol. It is a transparent
reconstruction and is not claimed to be byte-for-byte identical to the
original local script.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

LABELS = ["A", "B", "C", "D"]


def rotate_choices(choices, r):
    return choices[r:] + choices[:r]


def rotate_answer_index(answer_idx, r):
    return (answer_idx - r) % 4


def inverse_map_prediction(display_idx, r):
    return (display_idx + r) % 4


def format_prompt(question, choices):
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


def score_prompt(model, tokenizer, prompt, token_ids, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    selected = logits[token_ids]
    probs = torch.softmax(selected.float(), dim=0).cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, float(probs[pred]), probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="audit_results.parquet")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset("cais/mmlu", "all", split="test")
    if args.n > len(dataset):
        raise ValueError(f"Requested {args.n} questions but split has only {len(dataset)}")

    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[: args.n]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            pred_display, confidence, probs = score_prompt(
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
                    "model": args.model,
                    "seed": args.seed,
                }
            )

    out = Path(args.out)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"Wrote {len(rows)} predictions to {out}")


if __name__ == "__main__":
    main()
