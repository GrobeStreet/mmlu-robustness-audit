# Inspect AI port: MMLU option-order robustness

This directory translates the public MMLU option-order robustness audit into the UK AI Security Institute's [Inspect AI](https://inspect.aisi.org.uk/) evaluation framework while preserving the scientific boundary of the regenerated harness.

## What is preserved

- pinned MMLU dataset revision `c30699e8356da336a370243923dbaf21066bb9fe`;
- fixed-seed question selection (`n=300`, `seed=0` by default);
- four cyclic option rotations rather than all 24 permutations;
- the exact raw-completion prompt (`A. ...`, blank line, `Answer:`), with **no chat template**;
- inverse mapping from displayed answer position back to the underlying answer;
- normalized next-token scoring over only `A/B/C/D`;
- per-question flip-rate aggregation plus stable/flipping accuracy and mean four-label confidence;
- model-side metadata for dtype, device, revision, ties, margins, entropy, and four-label probabilities in Inspect logs.

The custom `mmlu-labels` model provider exists because Inspect's ordinary Hugging Face provider is chat-oriented. Reusing it directly would alter the prompt boundary and would not be a parity implementation of the published regeneration.

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-inspect.txt
```

Inspect AI is pinned to `0.3.249`, the current PyPI release used when this port was built.

## Run the parity configuration

From the repository root:

```bash
inspect eval inspect_eval/mmlu_option_order.py \
  --model mmlu-labels/Qwen/Qwen2.5-0.5B-Instruct \
  -M revision=7ae557604adf67be50417f59c2c2f167def9a775 \
  -M dtype=float32 \
  -M device=cpu \
  -T n=300 \
  -T seed=0
```

For a smoke test:

```bash
inspect eval inspect_eval/mmlu_option_order.py \
  --model mmlu-labels/Qwen/Qwen2.5-0.5B-Instruct \
  -M revision=7ae557604adf67be50417f59c2c2f167def9a775 \
  -M dtype=float32 \
  -M device=cpu \
  -T n=8 \
  -T seed=0
```

Inspect writes a structured evaluation log containing prompts, outputs, per-sample score metadata, and aggregate metrics. Use `inspect view` to inspect the run interactively.

## Metrics

The custom scorer reports:

- Inspect sample accuracy over all four rotations;
- rotation-0 (single-shot) accuracy;
- option-order flip rate, defined at the **question** level;
- accuracy on stable questions;
- accuracy on flipping questions;
- mean normalized four-label confidence.

Tie metadata is retained per sample so the existing tie-aware analysis can be reproduced from logs or compared against the parquet-based pipeline.

## Comparability boundary

This port is designed to be numerically comparable to the 2026-08-12 public regeneration **when run with the `mmlu-labels` provider and the same pinned model/dataset revisions, dtype, device, n, and seed**.

Using a normal Inspect provider (OpenAI, Anthropic, built-in Hugging Face chat mode, etc.) changes the elicitation/scoring protocol. Such runs may be useful as new experiments, but they are not reproductions of the frozen historical/regenerated table and must be labeled separately.

## Why this exists

The original audit established a benchmark-validity failure mode. This port makes that experiment legible inside a widely used evaluation framework without silently changing the measurement instrument. The design goal is portability **and** comparability, not portability at the expense of the original protocol.
