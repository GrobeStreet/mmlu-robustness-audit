# Inspect Evals Register preparation

Status: **implementation structurally ready; standard submission is blocked on the arXiv URL, then the post-PR two-model log requirement.**

Last audited against the live Inspect Evals Register guide: **2026-09-30**.

Inspect Evals uses a distributed register: new evaluations remain in the author's upstream repository and are registered through an issue that points to a pinned source commit.

## Current upstream requirements

The upstream eval repo must:

1. contain a `pyproject.toml` with a `[project]` table;
2. declare Inspect AI as a dependency;
3. define each task with Inspect AI's `@task` decorator;
4. host and pin external assets in stable version-controlled storage.

This repository satisfies those implementation-side requirements:

- package metadata: `pyproject.toml`
- Inspect dependency: `inspect-ai==0.3.249`
- task: `inspect_eval/mmlu_option_order.py::mmlu_option_order`
- dataset revision: `c30699e8356da336a370243923dbaf21066bb9fe`
- parity-model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- custom provider registered through the package's `inspect_ai` entry point
- validated full-run parity record: `inspect_eval/PARITY_2026-08-14.md`

## Standard registration issue

The live Register Eval Submission issue asks for:

- **arXiv URL** — preferably a versioned URL;
- **Source URL** — a GitHub blob URL pointing to the Python file containing the `@task`, pinned to a full 40-character commit SHA;
- **Maintainers** — optional additional GitHub usernames.

The bot validates the issue, derives metadata, and opens the register PR.

## Human blocker: arXiv

The source-code side is ready. The standard issue workflow requires an **arXiv paper whose methodology describes this evaluation**.

Prepared source:

- `preprint/manuscript.tex`
- `preprint/ARXIV_SUBMISSION.md`

No live arXiv identifier is recorded in this repository as of 2026-09-30.

After arXiv assigns an identifier, record the versioned URL here:

`ARXIV_URL = PENDING`

The original MMLU paper is not a substitute for this paper because it does not document this repository's cyclic option-order protocol, raw-completion scoring boundary, regeneration discrepancy record, or Inspect parity validation.

## Source pin after arXiv

After the arXiv URL is recorded and any final documentation change is committed, use the resulting **40-character commit SHA**.

Paste-ready source pattern:

`https://github.com/GrobeStreet/mmlu-robustness-audit/blob/[[PINNED_COMMIT_SHA]]/inspect_eval/mmlu_option_order.py#L1`

Do not submit `main`, a tag, or a shortened SHA.

## Post-PR requirement: two full model logs

Once the bot opens the Register PR, the current process requires **full eval logs from two different models**, uploaded to the Inspect Evals log store, followed by a PR comment confirming the upload.

These runs exist to demonstrate end-to-end execution and support Inspect's scanners. They can use small/inexpensive models.

Before upload:

- run all samples, not a smoke-test subset;
- record model/provider/version and exact command;
- inspect logs for local usernames, paths, secrets, or other PII;
- preserve the resulting `.eval` files and SHA-256 hashes locally or as GitHub artifacts.

The existing 1,200-sample Qwen Inspect parity run is useful validation evidence but does **not** by itself satisfy the current two-model Register requirement.

## Comparability warning

The frozen parity result uses the custom `mmlu-labels` provider to preserve raw-completion normalized next-token scoring over A/B/C/D.

Running ordinary OpenAI, Anthropic, or chat-oriented Hugging Face providers can be useful as Register validation runs, but those runs change the elicitation/scoring boundary and must be labeled as **new evaluation runs**, not reproductions of the frozen Qwen parity result.

## Suggested issue fields

**arXiv URL**

`[[VERSIONED_ARXIV_URL_FOR_MMLU_OPTION_ORDER_AUDIT]]`

**Source URL**

`https://github.com/GrobeStreet/mmlu-robustness-audit/blob/[[PINNED_COMMIT_SHA]]/inspect_eval/mmlu_option_order.py#L1`

**Maintainers**

Leave blank unless adding another maintainer; the submitting account is included automatically.

## Suggested register identity

- common title: `MMLU Option-Order Robustness`
- full title: `MMLU Option-Order Robustness Audit`
- task function: `mmlu_option_order`
- upstream repository: `https://github.com/GrobeStreet/mmlu-robustness-audit`
- implementation version: `1.0.0`
- framing: benchmark robustness / evaluation methodology

Suggested description:

> Measures whether a model's MMLU answer changes under four semantics-preserving cyclic option reorderings while preserving a raw-completion, normalized A/B/C/D next-token scoring protocol.

## Validation evidence for reviewers

1. `inspect_eval/PARITY_2026-08-14.md`
2. `inspect_eval/parity_summary_2026-08-14.json`
3. `inspect_eval/README.md`
4. `regeneration/REGENERATION.md`
5. GitHub Actions full parity execution referenced by the parity record

## Submission claim boundary

Safe claim:

> The Inspect implementation exactly reproduces the hardened fp32 regeneration at the reported precision for all headline metrics currently implemented in Inspect.

Do not claim:

- that the historical July calibration/stability numbers were recovered;
- that the Llama comparison has been regenerated;
- that a third-party human has independently verified the eval;
- that all 24 answer permutations were tested;
- that this is a contamination audit.

## Exact sequence from here

1. **Human:** submit `preprint/manuscript.tex` to arXiv and obtain the versioned v1 URL.
2. **GitHub:** record the arXiv URL in this file and `preprint/ARXIV_SUBMISSION.md`.
3. **GitHub:** freeze the final commit and copy its 40-character SHA.
4. **Human/GitHub UI:** open one Inspect Evals **Register Eval Submission** issue using the arXiv URL + pinned task source URL.
5. **Bot:** allow the Register bot to validate metadata and open its PR.
6. **Execution:** run two full model evaluations.
7. **Human/GitHub UI:** upload both logs to the Inspect Evals log store and confirm on the PR.
8. Address automated or maintainer feedback until merge.
