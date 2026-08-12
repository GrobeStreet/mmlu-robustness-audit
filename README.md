# MMLU Robustness & Calibration Audit

[![Source verification](https://github.com/GrobeStreet/mmlu-robustness-audit/actions/workflows/verification.yml/badge.svg)](https://github.com/GrobeStreet/mmlu-robustness-audit/actions/workflows/verification.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-78e6c4.svg)](LICENSE)
[![Citation metadata](https://img.shields.io/badge/citation-CFF-78e6c4.svg)](CITATION.cff)
[![Status: regenerated — partial metric agreement](https://img.shields.io/badge/status-regenerated_%E2%80%94_partial_metric_agreement-2a3b55.svg)](regeneration/REGENERATION.md)

A small, reproducible stress test of multiple-choice benchmark validity.

This audit asks: **does a model give the same underlying answer when answer choices are cyclically reordered but the question itself is unchanged?**

## Current evidence status

The public harness has now been independently executed on `Qwen/Qwen2.5-0.5B-Instruct` under both bf16 and fp32. The central robustness finding regenerated; several historical calibration/stability quantities did not. The original July raw artifact is unavailable, so the repository preserves the historical table and the regenerated table separately rather than silently replacing either one.

See [`regeneration/REGENERATION.md`](regeneration/REGENERATION.md) and [`regeneration/PROVENANCE.json`](regeneration/PROVENANCE.json).

## Headline: frozen vs regenerated

| Metric | Frozen historical | bf16 regeneration | fp32 regeneration |
|---|---:|---:|---:|
| Standard single-shot accuracy | 42.7% | 43.7% | **44.0%** |
| Accuracy over all four rotations | 41.2% | 43.6% | **43.1%** |
| Stable across all four rotations | 35.7% | 21.3% | **21.7%** |
| Answer flips under reordering | 64.3% | 78.7% | **78.3%** |
| Accuracy on stable questions | 56.1% | 76.6% | **75.4%** |
| Accuracy on flipping questions | 35.2% | 34.6% | **34.1%** |
| 10-bin ECE | 0.28 | 0.132 | **0.137** |
| Mean four-label confidence | 69.1% | 56.8% | **56.8%** |

**What regenerated:** headline accuracy, near-chance performance on flipping questions, and the central finding that the underlying answer changes on a majority of questions under a meaning-preserving reorder. The regenerated flip rate is higher than the frozen value.

**What did not regenerate:** the historical stable rate, accuracy on stable questions, ECE, and mean confidence. The earlier two-model claim that the larger model was *much better calibrated* must therefore be treated as historical/unconfirmed until that arm is rerun and the provenance gap is resolved.

## Tie control

The bf16 run contained 100 exact top-score ties out of 1,200 predictions. A pre-committed concern was that positional `argmax` tie-breaking might manufacture flips. It did not:

| Tie policy | Flip rate |
|---|---:|
| Positional `argmax` | 78.7% |
| Random tie-break, 200 reseeds | **78.5%** [77.7, 79.3] |
| Exclude all tie-affected questions | **71.7%** |

The fp32 run eliminated every exact tie and still produced a 78.3% flip rate. The robustness finding therefore does not depend on bf16 ties or positional tie-breaking.

## Positional bias

The rotation design also exposes a strong displayed-label asymmetry:

| | A | B | C | D |
|---|---:|---:|---:|---:|
| Predicted display position, bf16 | 290 | 378 | 377 | **155** |
| Predicted display position, fp32 | 272 | 379 | 383 | **166** |
| Underlying answer chosen, bf16 | 311 | 294 | 306 | 289 |
| Underlying answer chosen, fp32 | 308 | 290 | 309 | 293 |

Displayed positions B/C are favored while D is strongly avoided, yet underlying selected answers remain near-uniform. That contrast localizes the effect to display position rather than answer content.

## Frozen protocol for future reruns

The hardened runner now pins the current immutable Hugging Face revisions and records prompt, dtype, device, environment, output hash, ties, margins, and entropy.

```bash
python -m pip install -r requirements.txt
python audit_full.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --dataset-revision c30699e8356da336a370243923dbaf21066bb9fe \
  --dtype float32 \
  --device cpu \
  --n 300 \
  --seed 0 \
  --out audit_results.parquet
python analyze.py audit_results.parquet --json-out audit_summary.json
```

Prompt format is explicitly frozen as raw completion: question, blank line, A–D choices, blank line, `Answer:`. No chat template is used.

## Interpretation limits

- Four cyclic rotations are tested, not all 24 permutations.
- This is an option-order robustness test, not a contamination test.
- The score is based on normalized next-token logits for `A/B/C/D`; "confidence" therefore means **four-label normalized confidence**, not a model-authored confidence statement.
- The 300-question sample is deliberately small; estimates should not be treated as benchmark-wide constants.
- The original July run's raw predictions and exact execution provenance are unavailable, so frozen historical values are not claimed to be reproduced where the new runs disagree.
- The Llama-3.2-3B arm remains unregenerated.
- No second human verifier has executed the package yet.

## Repository status and provenance

This repository began as a transparent reconstruction of a documented July 2026 protocol. It now contains an independent Qwen regeneration with **partial metric agreement**, explicit failed hypotheses, frozen provenance records, and a hardened runner for future exact reruns. Historical values remain in [`RESULTS.md`](RESULTS.md); new results are versioned beside them.

— Bobby Morong, independent researcher
