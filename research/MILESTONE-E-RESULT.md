# Living Context — Milestone E result and reusable component

> **Audit correction, 2026-09-22:** `STATUS.md` is authoritative. The offline
> component has been repaired, but scientific completion, accuracy,
> selective-refresh, lifecycle-cost, and decision-boundary claims in this
> archived report are withdrawn. The learned objects were embedding soft
> prefixes, not persisted per-layer KV caches.

**Date:** 2026-09-21
**Historical status (withdrawn):** v1 complete / measured decision boundary.

## The result (measured, with the pilot protocol)

On this machine (Apple M4 Max, 48 GB, MLX, frozen `Llama-3.2-1B-Instruct-4bit`):

| Arm | Accuracy (3 questions × 3 repeats) | Serving cost | Refresh cost |
|---|---|---|---|
| 1. Strong cached text, fresh index | **9/9** | ~0.6 s total | — |
| 4. Learned selective refresh (soft-prefix/artifact) | **6/9** | ~0.1 s/q after refresh | **224.7 s** for 9 artifacts |

- The learned arm failed the pre-registered non-inferiority margin (protocol §5); the correct statement is **cannot claim non-inferiority at n=9**, not "learned loses".
- The failure is repeatable (3/3 repeats) and located exactly where the protocol predicts hardest: the supply family, whose changes are contradictory updates and renamed entities.
- The old-answer → edit → new-answer mechanics work: one edit through arms 4/5/6, atomically admitted, cross-document question preserved, unrelated question unaffected (Milestone C).
- **Decision boundary, first measured point:** at low query volume and low churn on this corpus, well-engineered cached text retrieval dominates on cost AND accuracy. The regime where learned composition might win (high query volume, high churn, learned cross-document dependencies) is unexplored locally and is the collaboration case for H200/B200-class hardware.

## The reusable component

`src/living_context/` — three small, dependency-light modules:

- `catalog.py` — append-only revision store + `VersionedCorpus` with **atomic snapshot admission**: a half-updated corpus is never observable; stale-base admissions raise.
- `refresh.py` — `plan_refresh(arm, old, new, present, graph)` decides which learned artifacts must rebuild for protocol arms 4/5/6.
- `dependencies.py` — co-training graph with transitive invalidation.

These are the maintenance machinery any learned-context runtime needs, independent of the model backend.

### External reproduction instructions

```bash
git clone <repo-url> LivingContext && cd LivingContext
uv venv --python 3.12 && uv pip install -e ".[dev]"
uv run pytest tests/                          # 22 tests, no model load
uv run python experiments/generate_revision_sequences.py   # corpus
uv run python experiments/baseline_cached_text.py          # arm 1
uv run python experiments/arm_learned_context.py           # learned arm
uv run python experiments/pilot_run.py                     # frozen pilot
uv run python scripts/check_reproduction.py                # must exit 0
```

Requires an Apple-Silicon Mac with enough RAM for a ~700 MB 4-bit model (MLX
stack; no CUDA). The model id is pinned in each experiment file.

## Non-claims / scoping (deletion and format)

- **Not a memory standard.** The artifact manifest is a profile over existing
  plain fields (source revisions, roles, checksum, trained-at generation) —
  it proposes no universal memory format.
- **Deletion scope, stated exactly.** The `VersionedCorpus` tests prove that
  a *deleted document id* is absent from the current snapshot while its
  history remains queryable by as-of. This is a statement about catalog
  entries, **not** model unlearning: no claim is made about erasing learned
  prefix representations from model weights, because none was tested.
- **Not a Cartridges reproduction.** This was a local soft-prefix mechanism
  probe. A historical hardware feasibility inspection informed its scope but
  is not part of the maintained replay.
- **Two same-author implementations caveat** (MCP repo) does not apply here;
  this repo's comparisons are one-implementation arms against a text baseline,
  which is the intended design.

## What the collaboration case now contains

1. The measured scaling curve (Milestone 0, retained).
2. A frozen pilot protocol with declared repeats (Milestone D).
3. A reproducible harness + corpus + retainable evidence (Milestones A–C).
4. A falsifiable prediction: artifact independence will be measurably
   non-diagonal at scale, and selective refresh's win regime is narrower than
   intuition suggests.
5. All USD spend to date: 0.00. No private data anywhere in the corpus.
