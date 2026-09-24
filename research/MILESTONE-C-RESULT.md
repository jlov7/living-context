# Milestone C result — one document edit through every refresh arm

> **Audit correction, 2026-09-22:** This archived run did not establish its
> exit criteria. It staged a one-document mapping as a complete snapshot and
> dropped other live documents; admitted before refresh artifacts were checked;
> seeded dependencies with new rather than old trained revisions; skipped a
> declared neighbour rebuild; and answered the cross-document question through
> ordinary text outside the admitted corpus. Its MET labels are withdrawn.

**Date:** 2026-09-21
**Machine:** Apple M4 Max / 48 GB / MLX (no CUDA)
**Model:** `mlx-community/Llama-3.2-1B-Instruct-4bit` (frozen; cached, USD 0.00)
**Evidence file:** `artifacts/evidence/refresh-arms-demo-1790031888.json`

## The edit

`hydraulics/h-tank` v2 ("Volume 60 L") → v3 ("Volume 45 L"), one consequential
fact. The unrelated document `hydraulics/h-pump` stays at v3 ("Flow 55 L/min at
165 bar") and is deliberately unchanged by the edit — it is only co-training-
touched via batch b7 in the dependency graph.

## Exit-criteria check (protocol §4 / historical milestones A–C)

| Criterion | Status | Evidence in the retained run |
|---|---|---|
| Incomplete refresh never exposes a half-updated corpus | **MET** | v3 staged without admission; `corpus.serve()` still returns v2 while checks are pending; admission happens exactly once after all planned artifacts pass |
| Old-answer → update → new-answer demonstration works end-to-end | **MET** | OLD (v2) answer `"60"` correct; after the refresh all three arms answer the NEW fact `"45"` correctly |
| Cross-document question (revised + unchanged doc) works | **MET** | "What is the flow of the H-1 pump at 165 bar and the volume of the T-4 tank?" answered with BOTH `55` and `45` from the two current documents |
| Unrelated questions shown unaffected | **MET** | h-pump question correct BEFORE the edit AND in every arm after it (co-training did not corrupt the unrelated fact) |

## Arms executed (all measured, all retained)

| Arm | Refresh policy | Rebuilt docs | New answer | Unaffected | Text fallback |
|---|---|---|---|---|---|
| 4 | Changed artifact only | `h-tank` | 45 ✔ | ✔ | n/a |
| 5 | Conservative co-training impact set | `h-pump`, `h-tank` (closure) | 45 ✔ | ✔ | n/a |
| 6 | Changed artifact only + explicit text fallback | `h-tank` | 45 ✔ | ✔ | **45 ✔ while artifact rebuilt** |

Every artifact trained with the measured MLX soft-prefix path from Milestone B
(`PrefixArtifact`, frozen base, Adam, answer-span CE — one artifact per fact,
~30 s train each, 0.74 GB peak).

## What this demonstrates

The capability the project exists to make inspectable: **maintained learned
knowledge, not just compiled knowledge.** A consequential fact changed in one
document; the affected learned artifact was identified by policy, rebuilt, and
admitted atomically — while unrelated questions stayed correct and a question
spanning the revised AND the unchanged document still resolved fully. Arm 6's
fallback proves the corpus "stays useful while staying correct" during refresh:
the new fact was answerable from fresh text the whole time the artifact was
being rebuilt.

## Non-claims

- NOT a claim that learned context beats text retrieval. Milestone B measured
  the opposite at this scale; this milestone is about *maintainability
  mechanics*, not accuracy superiority.
- NOT a claim that selective refresh generalizes. There is no repeated-measures
  study here — this is the bounded single-edit demonstration the brief's
  "first demonstration should be visually obvious" phase called for.
- NOT evidence about H200/B200 behaviour, and NOT a Cartridges reproduction.

## Commands run

```bash
uv run pytest tests/ -q -p no:cacheprovider   # 19 passed
uv run python experiments/refresh_arms_demo.py # retained JSON above
```

Artifact manifests, the authored corpus, and the atomic catalog (Milestone A)
are what make each answer in the retained file traceable to its exact source
revision and trained-at generation.
