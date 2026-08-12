#!/usr/bin/env python3
"""Analyze an MMLU option-order audit parquet file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_utils import inverse_map_prediction


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


def _question_table(df, pred_col="pred_underlying"):
    qrows = []
    for qid, g in df.groupby("question_rank", sort=True):
        g = g.sort_values("rotation")
        preds = g[pred_col].tolist()
        stable = len(set(preds)) == 1
        qrows.append(
            {
                "question_rank": int(qid),
                "stable": stable,
                "rotation0_correct": int(g.loc[g["rotation"] == 0, "correct"].iloc[0]),
            }
        )
    return pd.DataFrame(qrows)


def _flip_rate_from_underlying(df, pred_col="pred_underlying"):
    return float(1.0 - _question_table(df, pred_col=pred_col)["stable"].mean())


def _random_tie_flip_rates(df, reseeds=200, seed=0):
    needed = {"top_exact_tie", "top_tie_indices", "rotation", "question_rank", "pred_display"}
    if not needed.issubset(df.columns):
        return None

    rates = []
    base = df.copy()
    for s in range(seed, seed + reseeds):
        rng = np.random.default_rng(s)
        preds = []
        for row in base.itertuples(index=False):
            if bool(row.top_exact_tie):
                tie_indices = [int(x) for x in str(row.top_tie_indices).split(",") if str(x) != ""]
                display = int(rng.choice(tie_indices))
            else:
                display = int(row.pred_display)
            preds.append(inverse_map_prediction(display, int(row.rotation)))
        tmp = base.assign(pred_underlying_random_tie=preds)
        rates.append(_flip_rate_from_underlying(tmp, pred_col="pred_underlying_random_tie"))

    arr = np.asarray(rates, dtype=float)
    return {
        "reseeds": reseeds,
        "mean_flip_rate": float(arr.mean()),
        "interval_95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "min_flip_rate": float(arr.min()),
        "max_flip_rate": float(arr.max()),
    }


def summarize(df, random_tie_reseeds=200, random_tie_seed=0):
    required = {
        "question_rank", "rotation", "answer_underlying", "pred_underlying",
        "pred_display", "correct", "confidence"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    qdf = _question_table(df)
    all_rotation_accuracy = float(df["correct"].mean())
    rotation0_accuracy = float(df.loc[df["rotation"] == 0, "correct"].mean())
    stable_rate = float(qdf["stable"].mean())
    flip_rate = 1.0 - stable_rate

    stable_ids = set(qdf.loc[qdf["stable"], "question_rank"])
    flip_ids = set(qdf.loc[~qdf["stable"], "question_rank"])
    stable_acc = float(df[df["question_rank"].isin(stable_ids)]["correct"].mean()) if stable_ids else float("nan")
    flip_acc = float(df[df["question_rank"].isin(flip_ids)]["correct"].mean()) if flip_ids else float("nan")

    cal_ece, bins = ece(df["confidence"], df["correct"], n_bins=10)
    positional = df.groupby("pred_display").size().reindex([0, 1, 2, 3], fill_value=0).astype(int).to_dict()
    underlying = df.groupby("pred_underlying").size().reindex([0, 1, 2, 3], fill_value=0).astype(int).to_dict()
    per_rotation = df.groupby("rotation")["correct"].mean().reindex([0, 1, 2, 3]).to_dict()

    result = {
        "n_questions": int(df["question_rank"].nunique()),
        "n_predictions": int(len(df)),
        "legacy": {
            "headline_accuracy_rotation0": rotation0_accuracy,
            "all_rotation_accuracy": all_rotation_accuracy,
            "stable_rate": stable_rate,
            "flip_rate": flip_rate,
            "accuracy_on_stable_questions": stable_acc,
            "accuracy_on_flipping_questions": flip_acc,
            "mean_four_label_confidence": float(df["confidence"].mean()),
            "ece_10_bin": cal_ece,
        },
        "prediction_display_counts": {str(k): int(v) for k, v in positional.items()},
        "prediction_underlying_counts": {str(k): int(v) for k, v in underlying.items()},
        "accuracy_by_rotation": {str(k): float(v) for k, v in per_rotation.items()},
        "calibration_bins": bins,
    }

    if "label_entropy" in df.columns:
        result["mean_label_entropy"] = float(df["label_entropy"].mean())

    if "top1_top2_margin" in df.columns:
        margin = df["top1_top2_margin"].astype(float)
        result["top_margin"] = {
            "lt_0_05_rate": float((margin < 0.05).mean()),
            "lt_0_01_rate": float((margin < 0.01).mean()),
        }

    if "top_exact_tie" in df.columns:
        tie_pred = df["top_exact_tie"].astype(bool)
        tie_questions = set(df.loc[tie_pred, "question_rank"].astype(int))
        no_tie = df[~df["question_rank"].isin(tie_questions)]
        result["tie_aware"] = {
            "exact_tie_predictions": int(tie_pred.sum()),
            "exact_tie_prediction_rate": float(tie_pred.mean()),
            "tie_affected_questions": len(tie_questions),
            "tie_affected_question_rate": float(len(tie_questions) / df["question_rank"].nunique()),
            "tie_excluded_flip_rate": _flip_rate_from_underlying(no_tie) if len(no_tie) else float("nan"),
            "random_tie_break": _random_tie_flip_rates(
                df,
                reseeds=random_tie_reseeds,
                seed=random_tie_seed,
            ),
        }

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("parquet")
    p.add_argument("--json-out", default=None)
    p.add_argument("--random-tie-reseeds", type=int, default=200)
    p.add_argument("--random-tie-seed", type=int, default=0)
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    result = summarize(
        df,
        random_tie_reseeds=args.random_tie_reseeds,
        random_tie_seed=args.random_tie_seed,
    )

    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
