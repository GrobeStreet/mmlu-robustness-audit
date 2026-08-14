"""Inspect AI task for MMLU option-order robustness.

This task preserves the audit's fixed-seed sample selection, four cyclic
rotations, raw-completion prompt, and underlying-answer remapping. With the
`mmlu-labels` provider from this package it also preserves the exact normalized
A/B/C/D next-token scoring rule used by the regenerated public harness.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

# Import also makes direct Python use register the custom provider. CLI use is
# registered earlier through the package's inspect_ai entry point.
import inspect_eval.provider  # noqa: F401
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    accuracy,
    metric,
    scorer,
    value_to_float,
)
from inspect_ai.solver import TaskState, generate

LABELS = ("A", "B", "C", "D")
DATASET_NAME = "cais/mmlu"
DATASET_CONFIG = "all"
DATASET_SPLIT = "test"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
EXPECTED_ROTATIONS = frozenset(range(4))
SCORE_TO_FLOAT = value_to_float()


def rotate_choices(choices: list[str], rotation: int) -> list[str]:
    rotation %= 4
    return choices[rotation:] + choices[:rotation]


def rotate_answer_index(answer_idx: int, rotation: int) -> int:
    return (answer_idx - rotation) % 4


def inverse_map_prediction(display_idx: int, rotation: int) -> int:
    return (display_idx + rotation) % 4


def format_prompt(question: str, choices: list[str]) -> str:
    """Exact raw-completion prompt used by the regenerated public harness."""

    lines = [question.strip(), ""]
    lines.extend(f"{label}. {choice}" for label, choice in zip(LABELS, choices))
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def records_to_samples(
    records: list[dict[str, Any]],
    indices: Iterable[int],
) -> list[Sample]:
    """Expand selected MMLU records into four cyclic-rotation Inspect samples."""

    samples: list[Sample] = []
    for question_rank, dataset_index in enumerate(indices):
        record = records[dataset_index]
        choices = list(record["choices"])
        answer = int(record["answer"])
        for rotation in range(4):
            rotated = rotate_choices(choices, rotation)
            display_answer = rotate_answer_index(answer, rotation)
            samples.append(
                Sample(
                    id=f"q{question_rank:04d}-r{rotation}",
                    input=format_prompt(str(record["question"]), rotated),
                    target=LABELS[display_answer],
                    metadata={
                        "question_rank": question_rank,
                        "dataset_index": int(dataset_index),
                        "rotation": rotation,
                        "answer_underlying": answer,
                        "answer_display": display_answer,
                        "subject": str(record.get("subject", "")),
                        "dataset_revision": DATASET_REVISION,
                    },
                )
            )
    return samples


def load_mmlu_samples(n: int = 300, seed: int = 0) -> list[Sample]:
    """Load the pinned MMLU test split and select the historical fixed-seed sample."""

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    if n > len(dataset):
        raise ValueError(f"Requested {n} questions but split contains {len(dataset)}")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = indices[:n]
    records = [dict(dataset[i]) for i in range(len(dataset))]
    return records_to_samples(records, selected)


def _score_rows(scores: list[SampleScore]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_score in scores:
        sample_meta = sample_score.sample_metadata or {}
        score_meta = sample_score.score.metadata or {}
        if "pred_underlying" not in score_meta:
            continue
        rows.append(
            {
                "question_rank": int(sample_meta["question_rank"]),
                "rotation": int(sample_meta["rotation"]),
                "pred_underlying": int(score_meta["pred_underlying"]),
                # Inspect's default mean reducer can convert C/I into numeric
                # 1/0 before custom metrics are called. value_to_float handles
                # both unreduced strings and reduced numeric values.
                "correct": bool(SCORE_TO_FLOAT(sample_score.score.value)),
                "confidence": float(score_meta.get("confidence", float("nan"))),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate rows safely during both partial and completed Inspect runs.

    Inspect may evaluate metrics incrementally before all samples have completed.
    Question-level robustness metrics therefore use only questions for which all
    four rotations are present. Sample-level rotation-0 accuracy and confidence
    use whatever valid rows are currently available. At final evaluation, every
    question is complete and these quantities match the frozen definitions.
    """

    empty = {
        "flip_rate": float("nan"),
        "rotation0_accuracy": float("nan"),
        "stable_accuracy": float("nan"),
        "flipping_accuracy": float("nan"),
        "mean_four_label_confidence": float("nan"),
    }
    if not rows:
        return empty

    by_question: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[int(row["question_rank"])].append(row)

    complete_questions = {
        qid: qrows
        for qid, qrows in by_question.items()
        if {int(row["rotation"]) for row in qrows} == EXPECTED_ROTATIONS
    }
    complete_ids = set(complete_questions)
    stable_ids = {
        qid
        for qid, qrows in complete_questions.items()
        if len({int(row["pred_underlying"]) for row in qrows}) == 1
    }
    flipping_ids = complete_ids - stable_ids

    def accuracy_for(question_ids: set[int]) -> float:
        selected = [
            bool(row["correct"])
            for row in rows
            if int(row["question_rank"]) in question_ids
        ]
        return sum(selected) / len(selected) if selected else float("nan")

    rotation0 = [bool(row["correct"]) for row in rows if int(row["rotation"]) == 0]
    confidences = [
        float(row["confidence"])
        for row in rows
        if math.isfinite(float(row["confidence"]))
    ]

    return {
        "flip_rate": (
            len(flipping_ids) / len(complete_ids)
            if complete_ids
            else float("nan")
        ),
        "rotation0_accuracy": (
            sum(rotation0) / len(rotation0) if rotation0 else float("nan")
        ),
        "stable_accuracy": accuracy_for(stable_ids),
        "flipping_accuracy": accuracy_for(flipping_ids),
        "mean_four_label_confidence": (
            sum(confidences) / len(confidences) if confidences else float("nan")
        ),
    }


@metric
def option_order_flip_rate() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return summarize_rows(_score_rows(scores))["flip_rate"]

    return compute


@metric
def rotation0_accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return summarize_rows(_score_rows(scores))["rotation0_accuracy"]

    return compute


@metric
def stable_question_accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return summarize_rows(_score_rows(scores))["stable_accuracy"]

    return compute


@metric
def flipping_question_accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return summarize_rows(_score_rows(scores))["flipping_accuracy"]

    return compute


@metric
def mean_four_label_confidence() -> Metric:
    def compute(scores: list[SampleScore]) -> Value:
        return summarize_rows(_score_rows(scores))["mean_four_label_confidence"]

    return compute


@scorer(
    metrics=[
        accuracy(),
        rotation0_accuracy(),
        option_order_flip_rate(),
        stable_question_accuracy(),
        flipping_question_accuracy(),
        mean_four_label_confidence(),
    ]
)
def option_order_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        raw = state.output.completion.strip().upper()
        if raw not in LABELS:
            return Score.unscored(
                answer=raw,
                explanation="Expected a single A/B/C/D label from the mmlu-labels provider.",
            )

        display_idx = LABELS.index(raw)
        rotation = int(state.metadata["rotation"])
        pred_underlying = inverse_map_prediction(display_idx, rotation)
        answer_underlying = int(state.metadata["answer_underlying"])
        model_meta = state.output.metadata or {}
        return Score(
            value=CORRECT if pred_underlying == answer_underlying else INCORRECT,
            answer=raw,
            metadata={
                "pred_display": display_idx,
                "pred_underlying": pred_underlying,
                "confidence": float(model_meta.get("confidence", float("nan"))),
                "four_label_probs": model_meta.get("four_label_probs"),
                "top_exact_tie": model_meta.get("top_exact_tie"),
                "top_tie_count": model_meta.get("top_tie_count"),
                "top_tie_indices": model_meta.get("top_tie_indices"),
                "top1_top2_margin": model_meta.get("top1_top2_margin"),
                "label_entropy": model_meta.get("label_entropy"),
                "model_revision": model_meta.get("model_revision"),
                "requested_dtype": model_meta.get("requested_dtype"),
                "resolved_model_dtype": model_meta.get("resolved_model_dtype"),
                "device": model_meta.get("device"),
                "prompt_format": model_meta.get("prompt_format"),
            },
        )

    return score


@task
def mmlu_option_order(n: int = 300, seed: int = 0) -> Task:
    """MMLU cyclic option-order audit in Inspect AI."""

    samples = load_mmlu_samples(n=n, seed=seed)
    return Task(
        dataset=MemoryDataset(
            samples=samples,
            name="mmlu-option-order",
            location=f"hf://{DATASET_NAME}@{DATASET_REVISION}/{DATASET_CONFIG}/{DATASET_SPLIT}",
        ),
        solver=[generate()],
        scorer=option_order_scorer(),
        version="1.0.0",
        metadata={
            "protocol": "four cyclic rotations; raw completion; normalized A/B/C/D next-token logits",
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
            "n_questions": n,
            "seed": seed,
            "rotations": 4,
            "comparability": "Designed for parity with the 2026-08-12 regenerated harness when used with mmlu-labels provider.",
        },
    )
