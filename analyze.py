#!/usr/bin/env python3
"""Analyze an MMLU option-order audit parquet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ece(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(conf)
    out = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(correct[mask].mean())
        c = float(conf[mask].mean())
        gap = abs(acc - c)
        out += (n / total) * gap
        rows.append({"bin_low": lo, "bin_high": hi, "n": n, "accuracy": acc, "confidence": c, "gap": gap})
    return float(out), rows


def summarize(df):
    required = {
        "question_rank", "rotation", "answer_underlying", "pred_underlying",
        "pred_display", "correct", "confidence"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    grouped = df.groupby("question_rank", sort=True)
    qrows = []
    for qid, g in grouped:
        preds = g.sort_values("rotation")["pred_underlying"].tolist()
        stable = len(set(preds)) == 1
        qrows.append(
            {
                "question_rank": int(qid),
                "stable": stable,
                "question_correct": int(g["correct"].mean() == 1.0),
                "rotation0_correct": int(g.loc[g["rotation"] == 0, "correct"].iloc[0]),
            }
        )
    qdf = pd.DataFrame(qrows)

    all_rotation_accuracy = float(df["correct"].mean())
    rotation0_accuracy = float(df.loc[df["rotation"] == 0, "correct"].mean())
    stable_rate = float(qdf["stable"].mean())
    flip_rate = 1.0 - stable_rate

    stable_ids = set(qdf.loc[qdf["stable"], "question_rank"])
    flip_ids = set(qdf.loc[~qdf["stable"], "question_rank"])
    stable_acc = float(df[df["question_rank"].isin(stable_ids)]["correct"].mean()) if stable_ids else float("nan")
    flip_acc = float(df[df["question_rank"].isin(flip_ids)]["correct"].mean()) if flip_ids else float("nan")

    cal_ece, bins = ece(df["confidence"], df["correct"], n_bins=10)

    positional = (
        df.groupby("pred_display").size().reindex([0, 1, 2, 3], fill_value=0).astype(int).to_dict()
    )

    per_rotation = (
        df.groupby("rotation")["correct"].mean().reindex([0, 1, 2, 3]).to_dict()
    )

    return {
        "n_questions": int(df["question_rank"].nunique()),
        "n_predictions": int(len(df)),
        "headline_accuracy_rotation0": rotation0_accuracy,
        "all_rotation_accuracy": all_rotation_accuracy,
        "stable_rate": stable_rate,
        "flip_rate": flip_rate,
        "accuracy_on_stable_questions": stable_acc,
        "accuracy_on_flipping_questions": flip_acc,
        "mean_confidence": float(df["confidence"].mean()),
        "ece_10_bin": cal_ece,
        "prediction_letter_counts": {str(k): int(v) for k, v in positional.items()},
        "accuracy_by_rotation": {str(k): float(v) for k, v in per_rotation.items()},
        "calibration_bins": bins,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquet")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    result = summarize(df)

    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
