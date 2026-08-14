from inspect_eval.mmlu_option_order import (
    format_prompt,
    inverse_map_prediction,
    records_to_samples,
    rotate_answer_index,
    rotate_choices,
    summarize_rows,
)


def test_rotation_helpers_match_original_protocol():
    choices = ["a", "b", "c", "d"]
    assert rotate_choices(choices, 1) == ["b", "c", "d", "a"]
    assert rotate_answer_index(2, 1) == 1
    assert inverse_map_prediction(1, 1) == 2


def test_prompt_is_raw_completion_format():
    prompt = format_prompt("Question?", ["one", "two", "three", "four"])
    assert prompt == (
        "Question?\n\n"
        "A. one\n"
        "B. two\n"
        "C. three\n"
        "D. four\n\n"
        "Answer:"
    )
    assert "ANSWER:" not in prompt
    assert "user:" not in prompt


def test_records_expand_to_four_rotations_with_stable_underlying_target():
    records = [
        {
            "question": "Q",
            "choices": ["A0", "B0", "C0", "D0"],
            "answer": 2,
            "subject": "test",
        }
    ]
    samples = records_to_samples(records, [0])
    assert len(samples) == 4
    assert [sample.metadata["rotation"] for sample in samples] == [0, 1, 2, 3]
    assert [sample.metadata["answer_underlying"] for sample in samples] == [2, 2, 2, 2]
    assert [sample.target for sample in samples] == ["C", "B", "A", "D"]


def test_summary_matches_question_level_flip_definition():
    rows = [
        # q0 stable and correct in all rotations
        {"question_rank": 0, "rotation": r, "pred_underlying": 1, "correct": True, "confidence": 0.8}
        for r in range(4)
    ] + [
        # q1 flips, with 1/4 correct
        {"question_rank": 1, "rotation": 0, "pred_underlying": 0, "correct": True, "confidence": 0.5},
        {"question_rank": 1, "rotation": 1, "pred_underlying": 1, "correct": False, "confidence": 0.5},
        {"question_rank": 1, "rotation": 2, "pred_underlying": 2, "correct": False, "confidence": 0.5},
        {"question_rank": 1, "rotation": 3, "pred_underlying": 3, "correct": False, "confidence": 0.5},
    ]
    summary = summarize_rows(rows)
    assert summary["flip_rate"] == 0.5
    assert summary["rotation0_accuracy"] == 1.0
    assert summary["stable_accuracy"] == 1.0
    assert summary["flipping_accuracy"] == 0.25
    assert summary["mean_four_label_confidence"] == 0.65
