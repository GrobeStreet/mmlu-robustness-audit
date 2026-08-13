# MMLU Robustness & Calibration Audit

[![Source verification](https://github.com/GrobeStreet/mmlu-robustness-audit/actions/workflows/verification.yml/badge.svg)](https://github.com/GrobeStreet/mmlu-robustness-audit/actions/workflows/verification.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-78e6c4.svg)](LICENSE)
[![Citation metadata](https://img.shields.io/badge/citation-CFF-78e6c4.svg)](CITATION.cff)
[![Status: regenerated — partial metric agreement](https://img.shields.io/badge/status-regenerated_%E2%80%94_partial_metric_agreement-2a3b55.svg)](regeneration/REGENERATION.md)

**Question:** does a model keep the same underlying answer when multiple-choice options are cyclically reordered but the question itself is unchanged?

## Headline result

The separately regenerated `Qwen/Qwen2.5-0.5B-Instruct` harness changes its underlying answer on roughly **78% of sampled questions** under four cyclic reorderings in both bf16 and fp32. Accuracy on the questions that flip remains near chance.

The central robustness result regenerated. Several historical calibration/stability quantities did **not**, so the repository preserves both records instead of silently replacing the older values.

## Proof / receipts

- **Green CI:** [verification workflow](https://github.com/GrobeStreet/mmlu-robustness-audit/actions/workflows/verification.yml)
- **Regeneration record:** [`regeneration/REGENERATION.md`](regeneration/REGENERATION.md)
- **Machine-readable provenance:** [`regeneration/PROVENANCE.json`](regeneration/PROVENANCE.json)
- **Historical + regenerated tables:** [`RESULTS.md`](RESULTS.md)
- **Frozen future rerun inputs:** model + dataset revisions, dtype, device, prompt format, seed, and sample size are explicit below
- **Controls that failed to explain the effect:** random tie-breaking and fp32 rerun

**Verification status:** regeneration complete with **partial metric agreement**. No second human verifier has executed the package yet.

## Frozen vs regenerated

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

**What regenerated:** headline accuracy, near-chance performance on flipping questions, and the majority-flip robustness failure.

**What did not regenerate:** the historical stable rate, stable-question accuracy, ECE, and mean four-label confidence. The historical claim that the larger comparison model was *much better calibrated* remains unconfirmed until that arm is rerun.

## Tie and precision controls

The bf16 run contained 100 exact top-score ties out of 1,200 predictions. If positional `argmax` tie-breaking were manufacturing the effect, a randomized tie-break or fp32 rerun should substantially reduce it. They did not.

| Control | Flip rate |
|---|---:|
| bf16 positional `argmax` | 78.7% |
| bf16 random tie-break, 200 reseeds | **78.5%** [77.7, 79.3] |
| bf16 excluding tie-affected questions | **71.7%** |
| fp32, zero exact ties | **78.3%** |

These controls reject two plausible implementation explanations without claiming they explain the remaining historical discrepancy.

## Positional asymmetry

| | A | B | C | D |
|---|---:|---:|---:|---:|
| Predicted display position, bf16 | 290 | 378 | 377 | **155** |
| Predicted display position, fp32 | 272 | 379 | 383 | **166** |
| Underlying answer chosen, bf16 | 311 | 294 | 306 | 289 |
| Underlying answer chosen, fp32 | 308 | 290 | 309 | 293 |

Displayed positions B/C are favored and D is avoided while underlying selections remain much closer to uniform. This supports a positional-bias interpretation rather than a simple answer-content frequency explanation.

## Reproduce the frozen protocol

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

Prompt format is frozen as raw completion: question, blank line, A–D choices, blank line, `Answer:`. No chat template is used.

## What this does not claim

- Four cyclic rotations are tested, not all 24 permutations.
- This is an option-order robustness test, not a contamination test.
- “Confidence” means normalized next-token probability over `A/B/C/D`, not a model-authored confidence statement.
- The 300-question sample is intentionally small and should not be treated as a benchmark-wide constant.
- The original July raw predictions and exact execution provenance are unavailable.
- The Llama-3.2-3B arm remains unregenerated.
- No second human verifier has executed the package.

## Status

This repository began as a transparent reconstruction of a documented July 2026 protocol. It now contains a separate Qwen regeneration, explicit failed hypotheses, versioned historical/regenerated results, regression tests, CI, provenance records, and a hardened runner for future reruns.

— Robert “Bobby” Morong, independent researcher
