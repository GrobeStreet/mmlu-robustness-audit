# arXiv submission handoff

## Manuscript

**Title:** A Reproducible MMLU Option-Order Robustness Evaluation in Inspect AI

**Author:** Robert Morong

**Suggested primary category:** cs.CL (Computation and Language)

**Suggested cross-list:** cs.LG (Machine Learning), if arXiv permits and the author judges it appropriate.

## Abstract

Multiple-choice evaluations usually assume that a model's answer is invariant to semantically irrelevant changes in answer-option order. We implement a reproducible stress test of that assumption on a fixed sample of 300 MMLU test questions using four cyclic option rotations per question. For each prompt, a model is scored by normalized next-token logits over only A/B/C/D; displayed choices are then mapped back to the underlying answer identity before robustness is measured. On the pinned Qwen/Qwen2.5-0.5B-Instruct checkpoint in float32 on CPU, the model changes its underlying answer on 78.3% of sampled questions. Accuracy across all rotations is 43.1%, while accuracy on predictions from flipping questions is 34.1%. The displayed label distribution is strongly asymmetric even though underlying selected answers are close to uniform. A bfloat16 run, randomized tie-breaking, and a zero-tie float32 rerun do not remove the effect. We then re-implement the evaluation in Inspect AI using a separate task/provider/scorer path while preserving the raw-completion measurement boundary; the 1,200-sample Inspect run exactly matches all implemented headline float32 metrics at reported precision. The package pins model and dataset revisions, records provenance, includes regression tests and CI, and explicitly preserves historical values that did not regenerate. This paper documents the evaluation methodology and reproducibility evidence rather than claiming a benchmark-wide constant from a single small model and sample.

## Comments field

5 pages, 2 tables. Code, pinned provenance, tests, CI, and a validated Inspect AI implementation are available at https://github.com/GrobeStreet/mmlu-robustness-audit . The Inspect implementation reproduces the hardened fp32 headline metrics exactly at reported precision on the pinned 300-question protocol. AI systems assisted with software development, analysis, implementation review, and manuscript drafting; they are not authors.

## Source upload

Upload `manuscript.tex` as the primary source. It uses only standard TeX packages and compiles with pdfLaTeX.

Before pressing Submit, confirm:

- author name and email are correct;
- category choice is acceptable;
- title and abstract match the compiled PDF;
- generated PDF has all 5 pages and 2 tables;
- repository URL is public;
- AI-assistance disclosure is retained;
- no claim of independent human verification is added.

## After arXiv assigns an identifier

Record the versioned URL (for example `https://arxiv.org/abs/YYMM.NNNNNv1`) in this file and in `inspect_eval/INSPECT_EVALS_REGISTER_PREP.md`, then submit the Inspect Evals Register issue using the final source commit.
