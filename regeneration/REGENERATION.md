# Independent regeneration of the MMLU option-order audit

**Run date:** 2026-08-12 · **Protocol:** unmodified `audit_full.py`, `--n 300 --seed 0`, four cyclic
rotations per question (1,200 predictions) · **Model:** `Qwen/Qwen2.5-0.5B-Instruct` · **Device:** CPU

This document reports an independent execution of this repository's own harness and compares the
result to the frozen reported table in `RESULTS.md`. It was run because the repository's status badge
reads *"transparent reconstruction"* and the profile README states it *"does not claim byte-for-byte
identity with the unavailable original raw run."* That caveat is correct, and this is the follow-up
that bounds it.

## 1. Headline comparison

| Metric | Frozen (reported) | Regenerated (bf16) | Δ |
|---|---:|---:|---:|
| Standard single-shot accuracy | 42.7% | **43.7%** | +1.0 |
| Accuracy over all four rotations | 41.2% | **43.6%** | +2.4 |
| Accuracy on flipping questions | 35.2% | **34.6%** | −0.6 |
| Answer flips under reordering | 64.3% | **78.7%** | **+14.4** |
| Stable across all four rotations | 35.7% | **21.3%** | **−14.4** |
| Accuracy on stable questions | 56.1% | **76.6%** | **+20.5** |
| Expected calibration error (10-bin) | 0.28 | **0.132** | **−14.8** |
| Mean stated confidence | 69.1% | **56.8%** | **−12.3** |

Per-rotation accuracy was flat (0.437 / 0.433 / 0.427 / 0.447), so the divergences are not a
rotation-handling bug.

### What reproduced
**Accuracy, and the paper's actual claim.** Single-shot accuracy landed within 1.0 pp. Accuracy on
flipping questions reproduced to 0.6 pp and sits barely above the 25% chance floor in both runs. The
central finding — *a model changes its underlying answer on a majority of questions under a
meaning-preserving reorder, and is near chance when it does* — held, and the flip rate came back
**higher**, not lower.

### What did not
**The entire confidence distribution.** ECE less than half the reported value; mean confidence down
12.3 pp. This is the claim most at risk, because the two-model headline in `RESULTS.md` is that the
larger model is *"more accurate and much better calibrated (ECE 0.09 vs 0.28), yet still flips."* If
0.28 is not stable, the quantitative contrast is much smaller than stated even if the qualitative
point survives.

## 2. Diagnosis: bfloat16 and argmax tie-breaking

`audit_full.py` loads with `torch_dtype="auto"`, which for this checkpoint resolves to
**`torch.bfloat16`**. Softmax over four quantised logits produces a measurable number of
**bit-identical** top-2 probabilities.

```
mean entropy over A/B/C/D      0.9887  (71.3% of the ln4 = 1.3863 maximum)
top1 − top2 gap < 0.05         20.6% of predictions
top1 − top2 gap < 0.01          8.3% of predictions
EXACT ties (top1 == top2)      100 / 1200 = 8.3% of predictions
```

### Tie-breaking hypothesis tested and refuted

Re-deciding every tied prediction by uniform random choice among tied labels instead of array position,
recomputed over 200 independent reseeds, gave a flip rate of **78.5% [77.7, 79.3]** versus **78.7%**
under positional `argmax`. The 0.2 pp difference shows tie-breaking policy does not drive the effect.

Tie multiplicity across 1,200 predictions: 1,100 unique maxima · 96 two-way · 3 three-way · 1 four-way.

| | stable | flips |
|---|---:|---:|
| **no exact tie** (219 questions) | 62 | 157 |
| **≥1 exact tie** (81 questions) | 2 | 79 |

```
P(flip | question contains an exact tie) = 97.5%
P(flip | no exact tie)                   = 71.7%
```

The 71.7% tie-excluded figure is a conservative lower bound, not a corrected estimate. Under positional,
random, or excluded tie policies, the flip rate exceeds the frozen 64.3%.

## 3. Secondary finding: positional bias

Predicted **display position** across all 1,200 bf16 predictions:

```
A: 290   B: 378   C: 377   D: 155
```

The fp32 control below replicates the pattern and shows underlying answers remain near-uniform.

## 4. Recommended repository changes

1. Pin and publish the prompt format and model revision.
2. Pin dtype for determinism, while noting fp32 shows dtype does not materially affect headline metrics.
3. Report tie rate, random tie-break control, and tie-excluded lower bound.
4. Treat the frozen calibration contrast as not regenerated.
5. Add the positional-bias table.
6. Upgrade status to regenerated with partial metric agreement, not full reproduction.

## 5. Float32 control — dtype hypothesis refuted

Identical settings, `torch.float32`, same seed and sample.

| Metric | Frozen | bf16 | **fp32** | fp32 − frozen |
|---|---:|---:|---:|---:|
| Standard single-shot accuracy | 42.7% | 43.7% | **44.0%** | +1.3 |
| Accuracy over all four rotations | 41.2% | 43.6% | **43.1%** | +1.9 |
| Stable across all four rotations | 35.7% | 21.3% | **21.7%** | −14.0 |
| Answer flips under reordering | 64.3% | 78.7% | **78.3%** | +14.0 |
| Accuracy on stable questions | 56.1% | 76.6% | **75.4%** | +19.3 |
| Accuracy on flipping questions | 35.2% | 34.6% | **34.1%** | −1.1 |
| Expected calibration error | 0.28 | 0.132 | **0.137** | −14.3 |
| Mean stated confidence | 69.1% | 56.8% | **56.8%** | −12.3 |

```
exact ties     bf16: 100/1200      fp32: 0/1200
mean entropy   bf16: 71.3% of max  fp32: 71.3% of max
```

Float32 eliminated every exact tie and changed essentially nothing. ECE moved 0.132 → 0.137,
mean confidence was unchanged to three significant figures, and entropy was unchanged. The bf16
compression hypothesis is therefore refuted.

Two precisions agree closely with each other while diverging from the frozen table on stability,
accuracy-on-stable, and calibration. Because the original raw run is unavailable, the cause cannot be
resolved from the surviving record. Prompt formatting, checkpoint revision, or sample provenance remain
plausible differences; none should be asserted as the cause without evidence.

## 5b. Positional bias — confirmed and controlled

| | A | B | C | D |
|---|---:|---:|---:|---:|
| **Predicted display position**, bf16 | 290 | 378 | 377 | **155** |
| **Predicted display position**, fp32 | 272 | 379 | 383 | **166** |
| Underlying answer chosen, bf16 | 311 | 294 | 306 | 289 |
| Underlying answer chosen, fp32 | 308 | 290 | 309 | 293 |

The model favors display positions B and C and avoids D by more than a factor of two across both
precisions, while underlying answers remain near-uniform. That contrast localizes the bias to display
position rather than content.

## 6. Environment used for the independent run

Python 3.11.15 · torch 2.13.0 · transformers 4.57.6 · datasets 3.6.0 · CPU.
The original 2026-08-12 bf16 run used the then-default model and dataset revisions; this PR pins the
current immutable revisions for all future reruns and writes an automatic provenance sidecar.

## 7. Bottom line

**Regenerated:** headline accuracy within 1.3 pp, accuracy on flipping questions within 1.1 pp, and the
central majority-flip/near-chance-on-flips result. Flip rate is higher than frozen under both precisions.

**Not regenerated:** stability, accuracy on stable questions, ECE, and mean four-label confidence.
The two-model calibration contrast must therefore be described as historical/unconfirmed until the
Llama arm and provenance gap are resolved.

**Not done here:** Llama-3.2-3B-Instruct was not rerun; no byte-level identity with the unavailable July
raw run is claimed; no second human verifier has executed this package.
