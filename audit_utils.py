"""Pure helpers shared by the MMLU audit and its tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np


def rotate_choices(choices: Sequence[str], r: int) -> list[str]:
    choices = list(choices)
    return choices[r:] + choices[:r]


def rotate_answer_index(answer_idx: int, r: int) -> int:
    return (answer_idx - r) % 4


def inverse_map_prediction(display_idx: int, r: int) -> int:
    return (display_idx + r) % 4


def top_diagnostics(probs: Sequence[float]) -> dict[str, object]:
    """Return exact-top-tie and top-two-margin diagnostics for four probabilities."""
    arr = np.asarray(probs, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"Expected four probabilities, got shape {arr.shape}")

    order = np.argsort(-arr, kind="stable")
    top1 = float(arr[order[0]])
    top2 = float(arr[order[1]])
    tie_indices = np.flatnonzero(arr == top1).astype(int).tolist()
    entropy = float(-(arr[arr > 0] * np.log(arr[arr > 0])).sum())
    return {
        "top1_probability": top1,
        "top2_probability": top2,
        "top1_top2_margin": float(top1 - top2),
        "top_tie_count": len(tie_indices),
        "top_exact_tie": len(tie_indices) > 1,
        "top_tie_indices": tie_indices,
        "label_entropy": entropy,
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
