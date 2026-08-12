import math

import numpy as np
import pandas as pd

from analyze import ece, summarize
from audit_utils import inverse_map_prediction, rotate_answer_index, rotate_choices, top_diagnostics


def test_rotation_round_trip():
    choices = ["a", "b", "c", "d"]
    for r in range(4):
        rotated = rotate_choices(choices, r)
        for underlying in range(4):
            display = rotate_answer_index(underlying, r)
            assert inverse_map_prediction(display, r) == underlying
            assert rotated[display] == choices[underlying]


def test_top_diagnostics_exact_tie():
    d = top_diagnostics([0.4, 0.4, 0.1, 0.1])
    assert d["top_exact_tie"] is True
    assert d["top_tie_count"] == 2
    assert d["top_tie_indices"] == [0, 1]
    assert d["top1_top2_margin"] == 0.0


def test_ece_known_answer():
    value, _ = ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0], n_bins=10)
    assert math.isclose(value, 0.1, abs_tol=1e-12)


def synthetic_frame():
    rows = []
    # q0 is stable and correct across rotations.
    for r in range(4):
        rows.append({
            "question_rank": 0, "rotation": r, "answer_underlying": 0,
            "pred_underlying": 0, "pred_display": rotate_answer_index(0, r),
            "correct": 1, "confidence": 0.8, "top_exact_tie": False,
            "top_tie_indices": str(rotate_answer_index(0, r)),
            "top1_top2_margin": 0.2, "label_entropy": 0.8,
        })
    # q1 flips and contains an exact tie on rotation 0.
    preds = [0, 1, 0, 1]
    for r, pred in enumerate(preds):
        display = rotate_answer_index(pred, r)
        rows.append({
            "question_rank": 1, "rotation": r, "answer_underlying": 0,
            "pred_underlying": pred, "pred_display": display,
            "correct": int(pred == 0), "confidence": 0.6,
            "top_exact_tie": r == 0,
            "top_tie_indices": f"{display},{(display + 1) % 4}" if r == 0 else str(display),
            "top1_top2_margin": 0.0 if r == 0 else 0.1,
            "label_entropy": 1.0,
        })
    return pd.DataFrame(rows)


def test_summarize_legacy_and_tie_aware():
    result = summarize(synthetic_frame(), random_tie_reseeds=20, random_tie_seed=3)
    assert result["legacy"]["stable_rate"] == 0.5
    assert result["legacy"]["flip_rate"] == 0.5
    assert result["tie_aware"]["exact_tie_predictions"] == 1
    assert result["tie_aware"]["tie_affected_questions"] == 1
    assert result["tie_aware"]["tie_excluded_flip_rate"] == 0.0
    assert result["tie_aware"]["random_tie_break"]["reseeds"] == 20
    assert sum(result["prediction_display_counts"].values()) == 8
