# Frozen results and provenance

This file records the results available in the July/August 2026 research record. It separates **reported historical results** from **newly reproduced outputs** so they are not silently conflated.

## Qwen2.5-0.5B-Instruct

Protocol: 300 fixed-seed MMLU test questions; four cyclic answer-option rotations per question; 1,200 predictions total.

| Metric | Frozen reported value |
|---|---:|
| Headline accuracy (rotation 0) | 42.7% |
| Accuracy across all rotations | 41.2% |
| Stable across all four rotations | 35.7% |
| Answer flips under reordering | 64.3% |
| Accuracy on stable questions | 56.1% |
| Accuracy on flipping questions | 35.2% |
| Expected calibration error | 0.28 |
| Mean stated confidence | 69.1% |

Interpretation: the aggregate score changes little, but the model's underlying answer changes on most questions. Stability is therefore not implied by benchmark accuracy.

## Two-model follow-up

| Metric | Qwen2.5-0.5B | Llama-3.2-3B |
|---|---:|---:|
| Headline accuracy | 42.7% | 56.7% |
| Answer flips under reordering | 64.3% | 52.5% |
| Accuracy on flipping questions | 35.2% | 34.9% |
| Expected calibration error | 0.28 | 0.09 |

Interpretation: the larger model is more accurate and much better calibrated, yet still flips a majority of answers under the same meaning-preserving option reorder. The working hypothesis is that calibration, accuracy, and order robustness are related but distinct properties.

## Provenance status

- The Qwen figures above are frozen values from the original July 2026 audit record.
- The Llama figures are frozen values recorded in the later research handoff.
- The original local scripts and raw prediction parquet files were not present in the connected GitHub account when this public repository was reconstructed.
- `audit_full.py` and `analyze.py` are transparent reconstructions of the documented protocol, not claimed to be byte-identical to the original local source.
- Until the reconstructed pipeline is re-run and compared against the frozen values, these tables should be described as **reported results**, not as a fresh reproduction from this repository.

## Planned verification

1. Re-run the Qwen audit from this repository.
2. Compare every reported metric to the frozen record.
3. Re-run the Llama checkpoint under the same protocol.
4. Extend the panel to additional open-weight models.
5. Report confidence intervals and model-level relationships between accuracy, ECE, and flip rate.

Any future reproduced values that differ from this file should be added explicitly with version/date rather than overwriting the historical record.
