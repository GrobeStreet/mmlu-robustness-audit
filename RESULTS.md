# Frozen results and regenerated comparison

This file separates **historical reported results** from **independently regenerated outputs**. They must not be silently conflated.

## 1. Frozen historical record

### Qwen2.5-0.5B-Instruct

Protocol recorded in the July/August 2026 handoff: 300 fixed-seed MMLU test questions; four cyclic answer-option rotations per question; 1,200 predictions total.

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

### Frozen two-model follow-up

| Metric | Qwen2.5-0.5B | Llama-3.2-3B |
|---|---:|---:|
| Headline accuracy | 42.7% | 56.7% |
| Answer flips under reordering | 64.3% | 52.5% |
| Accuracy on flipping questions | 35.2% | 34.9% |
| Expected calibration error | 0.28 | 0.09 |

These figures remain preserved as historical reported values. The original local scripts/raw parquet artifact were not available when this public repository was reconstructed.

## 2. Independent Qwen regeneration — 2026-08-12

The reconstructed public harness was independently executed at two numerical precisions on CPU.

| Metric | Frozen | bf16 | fp32 |
|---|---:|---:|---:|
| Headline accuracy (rotation 0) | 42.7% | 43.7% | **44.0%** |
| Accuracy across all rotations | 41.2% | 43.6% | **43.1%** |
| Stable across all four rotations | 35.7% | 21.3% | **21.7%** |
| Answer flips under reordering | 64.3% | 78.7% | **78.3%** |
| Accuracy on stable questions | 56.1% | 76.6% | **75.4%** |
| Accuracy on flipping questions | 35.2% | 34.6% | **34.1%** |
| 10-bin ECE | 0.28 | 0.132 | **0.137** |
| Mean four-label normalized confidence | 69.1% | 56.8% | **56.8%** |

### Regeneration verdict

**Regenerated:** headline accuracy to within 1.3 percentage points, accuracy on flipping questions to within 1.1 points, and the qualitative/quantitative direction of the core robustness result. The public harness produces a majority flip rate at both precisions, and the regenerated rate is higher than the frozen historical value.

**Not regenerated:** the historical stable rate, accuracy on stable questions, ECE, and mean confidence. Because bf16 and fp32 agree closely with each other, dtype does not explain these discrepancies.

**Calibration language:** the historical statement that the larger model is "much better calibrated" is not currently supported by a regenerated Qwen baseline. The Llama arm has not yet been rerun. Treat the 0.28-vs-0.09 contrast as historical/unconfirmed, not as a currently reproduced result.

## 3. Tie control

The bf16 run produced 100 exact top-score ties (8.3% of predictions; 27.0% of questions touched by at least one tie).

| Policy | Flip rate |
|---|---:|
| Positional `argmax` | 78.7% |
| Uniform random tie-break, 200 reseeds | **78.5%** [77.7, 79.3] |
| Excluding all tie-affected questions | **71.7%** |
| fp32, where exact ties disappear | **78.3%** |

The tie-breaking hypothesis is therefore refuted: the robustness result does not depend on positional `argmax` behavior.

## 4. Positional-bias result

| | A | B | C | D |
|---|---:|---:|---:|---:|
| Predicted display position, bf16 | 290 | 378 | 377 | **155** |
| Predicted display position, fp32 | 272 | 379 | 383 | **166** |
| Underlying answer chosen, bf16 | 311 | 294 | 306 | 289 |
| Underlying answer chosen, fp32 | 308 | 290 | 309 | 293 |

The displayed-label distribution is strongly asymmetric while underlying chosen answers remain near-uniform. This supports a positional-bias interpretation rather than a simple content-frequency explanation.

## 5. Provenance status

- Full regeneration narrative: [`regeneration/REGENERATION.md`](regeneration/REGENERATION.md)
- Machine-readable record: [`regeneration/PROVENANCE.json`](regeneration/PROVENANCE.json)
- The 2026-08-12 independent run environment was Python 3.11.15, torch 2.13.0, transformers 4.57.6, datasets 3.6.0, CPU.
- The original independent run used the then-default Hugging Face revisions and did not record immutable commit hashes at execution time.
- Future runs are pinned to model revision `7ae557604adf67be50417f59c2c2f167def9a775` and dataset revision `c30699e8356da336a370243923dbaf21066bb9fe` and automatically write a SHA-256 provenance sidecar.
- No claim is made that the frozen historical values are wrong; only that several do not regenerate under the surviving public harness at either bf16 or fp32.
- Llama-3.2-3B-Instruct remains unregenerated.
- A second independent human execution remains outstanding.
