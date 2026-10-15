# Paper discovery intake

## Source coverage

Use both sources on every normal scan:

- arXiv metadata/API for broad, date-ordered coverage. Focus on `cs.LG`,
  `stat.ML`, `cs.AI`, `eess.SP`, and relevant cross-lists.
- Hugging Face Papers for daily/trending discovery, paper metadata, project pages,
  and linked repositories.

Authoritative API references:

- <https://info.arxiv.org/help/api/index.html>
- <https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-papers>
- <https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.list_papers>

Search a small query lattice rather than claiming exhaustive coverage:

- `time series forecasting`
- `multivariate time series forecasting`
- `probabilistic time series forecasting`
- `spatiotemporal forecasting`
- `long-term time series forecasting`
- `time series foundation model` combined with forecasting

Add focused terms derived from current catalog gaps, but do not use architecture
families as repository directories or permanent model categories.

## Candidate brief

Every retained candidate contains:

```yaml
title: "paper title"
paper_url: "primary paper URL"
arxiv_id: "canonical id when available"
published: "YYYY-MM-DD"
source_hits: ["arxiv", "hugging-face"]
code_url: "authoritative repository or empty"
source_revision: "pinned revision or empty"
license: "verified license or unknown"
catalog_match: "none, exact, alias, or related"
relevance: 0
novelty: 0
implementability: 0
confidence: "high, medium, or low"
reason: "why this belongs in the intake queue"
open_questions: []
```

Scores are 0--100. Dispatch only if relevance is at least 75, novelty is at least
50, and there is no exact/alias catalog match. Missing code is allowed and leads to
paper-only implementation. Missing or unclear license must be recorded and prevents
copying, redistribution, or dependency on the external source; it does not prevent
a genuinely local implementation from public method facts.

## Deduplication

Deduplicate in this order:

1. canonical arXiv identifier with version suffix removed;
2. DOI when present;
3. normalized title after case, punctuation, and whitespace folding;
4. explicit aliases in current model cards;
5. semantic similarity, labeled as an inference rather than an exact match.

An updated version of an existing paper is an audit candidate, not a new model.

## Dispatch prompt contract

Each dispatched task is limited to one paper and asks for one of these outcomes:

- a rejection with evidence;
- an intake proposal with unresolved questions;
- an implementation only when the caller explicitly authorized implementation.

The task must check the paper, authoritative source, pinned revision, license,
input/output contract, reusable components, deviations, and verification plan. It
must not claim reproduction from a successful forward pass.

For recurring monitoring, use a durable task prompt that invokes
`$discover-papers`, states the lookback window and dispatch cap, and explicitly says
whether dispatch is authorized. A weekly 14-day lookback is a practical default;
overlap is intentional and removed by deduplication.

Suggested recurring-task prompt:

> Use $discover-papers to scan arXiv and Hugging Face Papers for time-series
> forecasting work published or updated in the last 14 days. Deduplicate against
> the current TSFLab catalog. Dispatch at most three separate review tasks for
> candidates that clear the harness thresholds; do not implement, merge, or publish
> them. If nothing qualifies, report a successful scan with no dispatch.
