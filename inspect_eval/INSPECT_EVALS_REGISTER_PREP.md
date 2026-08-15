# Inspect Evals Register preparation

Status: **implementation ready for upstream registration workflow, pending the paper/metadata gate**

Inspect Evals changed its contribution model on 2026-05-08. New evaluations are no longer submitted as source-code PRs into `src/inspect_evals`. They remain in the author's upstream repository and are added to the Inspect Evals Register through a submission issue that points to a pinned source commit.

## Current upstream requirements

The upstream repository currently requires the eval repo to:

1. contain a `pyproject.toml` with a `[project]` table and be installable;
2. declare Inspect AI as a dependency;
3. define the task with Inspect AI's `@task` decorator;
4. pin external assets in stable version-controlled storage.

This repository now satisfies those implementation-side requirements:

- package metadata: `pyproject.toml`
- Inspect dependency: `inspect-ai==0.3.249`
- task: `inspect_eval/mmlu_option_order.py::mmlu_option_order`
- model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- dataset revision: `c30699e8356da336a370243923dbaf21066bb9fe`
- custom provider registered through the package's `inspect_ai` entry point
- validated full-run parity record: `inspect_eval/PARITY_2026-08-14.md`

## Standard registration form

The current Register Eval Submission issue asks for:

- **arXiv URL** — preferably versioned; the paper can be the original benchmark paper or a separate paper describing the replication;
- **Source URL** — GitHub blob URL to the `@task`-decorated Python source pinned to a full 40-character commit SHA;
- **Maintainers** — optional additional GitHub usernames.

The bot then validates the issue, derives metadata, and opens the register PR.

## Remaining blocker before standard submission

The source-code side is ready, but the standard issue workflow requires an **arXiv paper whose public methodology describes the evaluation**.

The original MMLU paper describes MMLU itself, but it does not document this repository's cyclic option-order robustness protocol, normalized four-label next-token scoring boundary, regeneration discrepancy record, or parity validation. For that reason, the strongest standard submission route is to first create a short versioned arXiv preprint describing this audit/replication rather than using the original MMLU paper as if it documented the new methodology.

If a paper-first route is not desired, Inspect Evals also documents a manual maintainer path for cases that need more control over register metadata. That would require contacting an Inspect Evals maintainer rather than assuming the standard bot submission will accept a non-matching paper.

## Paste-ready standard issue fields after the paper exists

**arXiv URL**

`[[VERSIONED_ARXIV_URL_FOR_MMLU_OPTION_ORDER_AUDIT]]`

**Source URL**

After this preparation branch is merged, replace `[[PINNED_COMMIT_SHA]]` with the final 40-character commit that contains the validated implementation:

`https://github.com/GrobeStreet/mmlu-robustness-audit/blob/[[PINNED_COMMIT_SHA]]/inspect_eval/mmlu_option_order.py#L1`

**Maintainers**

Leave blank unless an additional maintainer should be listed. The submitting GitHub account is included automatically by Inspect Evals.

## Suggested register identity

- common title: `MMLU Option-Order Robustness`
- full title: `MMLU Option-Order Robustness Audit`
- task function: `mmlu_option_order`
- upstream repository: `https://github.com/GrobeStreet/mmlu-robustness-audit`
- implementation version: `1.0.0`
- category framing: benchmark robustness / evaluation methodology

Suggested one-sentence description:

> Measures whether a model's MMLU answer changes under four semantics-preserving cyclic option reorderings while preserving a raw-completion, normalized A/B/C/D next-token scoring protocol.

## Validation evidence to point reviewers to

1. `inspect_eval/PARITY_2026-08-14.md` — formal full-run parity record.
2. `inspect_eval/parity_summary_2026-08-14.json` — machine-readable metrics and workflow provenance.
3. `inspect_eval/README.md` — implementation and usage documentation.
4. `REGENERATION.md` — historical-vs-regenerated evidence and remaining discrepancies.
5. GitHub Actions run `31845022558` — completed 1,200-sample fp32 Inspect execution.

## Submission claim boundary

Safe claim:

> The Inspect implementation exactly reproduces the hardened fp32 regeneration at the reported precision for all headline metrics currently implemented in Inspect.

Do not claim:

- that the historical July calibration/stability numbers were recovered;
- that the Llama comparison has been regenerated;
- that a third-party human has independently verified the eval;
- that all 24 answer permutations were tested;
- that this is a contamination audit.

## Final upstream sequence

1. merge this upstream-readiness branch after tests;
2. record the final pinned commit SHA;
3. produce/version the short methodology preprint or contact Inspect Evals maintainers for the manual metadata route;
4. open one Register Eval Submission issue with the versioned arXiv URL and pinned source URL;
5. respond to the bot/maintainer review without broadening the scientific claim beyond the parity record.
