# MMLU Robustness & Calibration Audit

[![Source verification](https://github.com/GrobeStreet/mmlu-robustness-audit/actions/workflows/verification.yml/badge.svg)](https://github.com/GrobeStreet/mmlu-robustness-audit/actions/workflows/verification.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-78e6c4.svg)](LICENSE)
[![Citation metadata](https://img.shields.io/badge/citation-CFF-78e6c4.svg)](CITATION.cff)
[![Status: transparent reconstruction](https://img.shields.io/badge/status-transparent_reconstruction-2a3b55.svg)](#repository-status-and-provenance)

A small, reproducible stress test of multiple-choice benchmark validity.

This audit asks a simple question: **does a model give the same underlying answer when the answer choices are cyclically reordered but the question itself is unchanged?**

## Why this matters

Headline accuracy can look stable even when a model's answer changes under a meaning-preserving perturbation. This repository measures that instability directly and keeps it separate from calibration and accuracy.

## Frozen reported results

### Qwen2.5-0.5B-Instruct

300 fixed-seed MMLU questions, four cyclic option orderings per question (1,200 predictions total).

| Metric | Reported value |
|---|---:|
| Standard single-shot accuracy | 42.7% |
| Accuracy over all four rotations | 41.2% |
| Stable across all four rotations | 35.7% |
| Answer flips under reordering | 64.3% |
| Accuracy on stable questions | 56.1% |
| Accuracy on flipping questions | 35.2% |
| Expected calibration error | 0.28 |
| Mean stated confidence | 69.1% |

### Two-model follow-up

| Metric | Qwen2.5-0.5B | Llama-3.2-3B |
|---|---:|---:|
| Headline accuracy | 42.7% | 56.7% |
| Answer flips under reordering | 64.3% | 52.5% |
| Accuracy on flipping questions | 35.2% | 34.9% |
| Expected calibration error | 0.28 | 0.09 |

The larger model is more accurate and much better calibrated, yet it still changes answers under a trivial option reorder on a majority of questions. **Calibration and robustness are different properties.**

## Protocol

1. Load MMLU, configuration `all`, split `test`.
2. Select 300 questions using fixed seed 0.
3. For every question, create all four cyclic rotations of the answer choices.
4. Score only the model probabilities assigned to the next-token labels `A`, `B`, `C`, `D`.
5. Convert each displayed-letter prediction back to the underlying answer-option index.
6. A question is **stable** only if all four rotations select the same underlying option.
7. Compute accuracy, all-rotation accuracy, flip rate, stable-vs-flip accuracy, confidence and ECE.

## Reproduce

```bash
python -m pip install -r requirements.txt
python audit_full.py --model Qwen/Qwen2.5-0.5B-Instruct --out audit_results.parquet
python analyze.py audit_results.parquet
```

Swap `--model` for another open-weight causal language model to extend the audit.

## Repository status and provenance

This public package is a **transparent reconstruction of the documented July 2026 protocol and frozen reported results**. The original local audit scripts and raw parquet artifact were not present in the connected GitHub account when this repository was published, so this repository does not claim byte-for-byte identity with the original run. The Qwen figures above are the frozen reported values from that run. The Llama figures are the frozen values recorded in the later research handoff and have not yet been independently re-executed from the original raw artifact here.

That distinction is intentional: reported results and newly reproduced results should never be silently conflated.

## Interpretation limits

- Four cyclic rotations are tested, not all 24 permutations.
- This is an option-order robustness test, not a contamination test.
- Next-token letter scoring is one defensible multiple-choice scoring method, not the only one.
- The 300-question sample is deliberately small enough to be accessible; its estimates should not be treated as benchmark-wide constants.
- Prior work exists on multiple-choice option-order / selection bias. The contribution to test further is the **decoupling between calibration, accuracy, and stability**, not merely the observation that answers can flip.

## Next research step

Run the same frozen protocol across a larger panel of open-weight models to test whether order robustness improves with scale, accuracy, or calibration.

— Bobby Morong, independent researcher
