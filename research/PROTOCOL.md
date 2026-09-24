# Experimental protocol — revision-aware learned context

**Status correction (2026-09-22):** this design was not validly executed. The
archived runner trained on evaluated questions and answers, used a defective
objective/scorer, and did not measure the declared complete cost or refresh
behavior. Git history does not verify pre-run freeze because the YAML and result
first appear in the same commit. This is historical intent; the later v2 and v3
protocols do not validate it.

## 1. Artifact manifest (every artifact records)

- Exact source identities and revisions; training-data provenance and permitted use
- Base-model revision, tokenizer, runtime, precision, positional convention, adapter/KV format
- Which documents supplied **training targets** vs which were present as **distractors**
- Training and serving budgets; composition regime; calibration scope; artifact checksum

## 2. Interfaces

```python
plan_refresh(old_snapshot, new_snapshot, dependency_graph) -> RefreshPlan
admit_artifacts(plan, development_checks) -> CorpusSnapshot | Hold
answer(question, snapshot_id) -> AnswerRecord          # must record snapshot + actual artifact IDs used
```

Every answer records the snapshot and the artifact IDs actually used. This is what makes the "old answer / update / new answer" demonstration inspectable.

## 3. Corpus

24 authored technical documents, **six families**, **three chronological revisions** each.

Must include: changed numbers, renamed entities, revoked facts, contradictory updates, cross-document references, and **unchanged controls**.

A hidden evaluator derives labels from the versioned authored truth table. **The model sees documents, not truth tables.** Human review checks the rendered documents actually support the labels.

Split development and test **by document family and revision pattern** — not by paraphrase. Separating query variability from document-family diversity is required.

> These are proposed **pilot dimensions, not a power calculation.**

## 4. Experimental arms

| # | Arm | Question it answers |
|---|---|---|
| 1 | Strong text retrieval, fresh indexing + ordinary prefix/KV caching | Is learned context worthwhile against ordinary engineering done well? |
| 2 | Full-context reading on a tractable subset | Exposes retrieval misses |
| 3 | Full learned-context refresh each revision | What does the expensive straightforward path achieve? |
| 4 | Selective refresh: changed artifact only | Is the document boundary a sufficient update boundary? |
| 5 | Selective refresh: conservative co-training impact set | Does a wider, selectively chosen refresh help? |
| 6 | Selective refresh + explicit text fallback during refresh | Can it stay useful while staying correct? |
| 7 | Stale artifact baseline | **Deliberately unsafe diagnostic. Not a deployment recommendation.** |

Hold matched: reader model, question set, available source facts, output contract, serving hardware.

## 5. Outcomes

**Primary:** current-version answer accuracy by change family, subject to a preregistered non-inferiority margin appropriate to a pilot. Publish uncertainty; do not announce equivalence from a non-significant difference.

**Secondary:** stale-answer frequency; cross-document accuracy; unsupported-answer rate; unchanged-task regression; refresh time; fallback coverage and duration; storage; peak memory; first-token and end-to-end latency; training and inference compute. **Track every failed build and inference attempt.**

**Lifecycle cost** = initial preparation + self-study generation + update/repair + query serving + retrieval/cache/storage + failed attempts + verification/review.

Report **physical units and measured wall time** before converting to prices. No stale provider prices. No extrapolating one GPU to another.

**Proposed benefit condition:** lower observed lifecycle cost while meeting the frozen quality margin, without increasing stale-answer frequency.

## 6. Leakage rules

- The scheduler is driven by **development** estimates of query frequency, source churn and measured rebuild cost.
- **Final test questions and labels must not drive the scheduler.**
- Enforced by `tests/test_no_test_label_access.py`.
