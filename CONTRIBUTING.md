# Contributing

Read [STATUS.md](STATUS.md), the [research document map](research/INDEX.md),
the [v2 result](research/study-v2/RESULT.md), and the separate
[v3 pilot result](research/study-v3/RESULT.md) and
[revision follow-up](research/study-v3-revision/RESULT.md) before changing code or claims.
The supported software is an in-memory reference. The v2 study ended with a
valid negative gate; the static v3 adapter pilot passed its own small quality
gate and tied cached text. A distinct post-pilot revision study passed a narrow
two-generation synthetic gate without measuring online update latency or
matched lifecycle cost. The old `experiments/pilot.yaml` freeze chronology
is unverified, so it cannot authorize new scientific claims.

## Core rules

1. Stage complete candidate snapshots and admit only after a trusted caller
   supplies an actual Boolean validation result. The optional checked path
   verifies artifact bytes and source revisions before issuing a process-local
   receipt; it does not authenticate the caller or the claimed training. One
   owner must serialize catalog mutation.
2. Preserve the 2026-09-21 evidence and frozen v2 protocol/results as history.
   Fixes to current replay or documentation must not rewrite observed output.
3. Keep development estimates separate from future test labels. The current
   YAML declaration test is not scheduler isolation proof.
4. Use synthetic, non-sensitive data. Deletion from active memory is not
   model unlearning.
5. Report failed checks, negative results, and unavailable prerequisites as
   such. Do not turn a source-only replay into a claim of model reproduction.

## Model-free development

```sh
uv sync --frozen --no-editable --extra dev --python 3.13
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pyright --pythonpath .venv/bin/python src tests scripts
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -p no:cacheprovider \
  --cov=src/living_context --cov-report=term-missing
.venv/bin/python -m scripts.verify_study_v2
.venv/bin/python scripts/replay_study_v3.py
.venv/bin/python -m scripts.replay_study_v3_revision
.venv/bin/python scripts/check_reproduction.py
.venv/bin/python -m living_context.example_lifecycle
.venv/bin/python -m living_context.text_reader_example
```

`scripts.verify_study_v2` is the current full-result text-evidence gate.
`replay_study_v3.py` checks the separate pilot's public synthetic raw outputs,
receipts and frozen source bytes without loading a model.
`replay_study_v3_revision.py` does the same for the separate two-generation
follow-up, including admission and stale-control evidence.
`check_reproduction.py` is the archived v1 custody and saved-output diagnostic.
It validates the public evidence subset against the unchanged original manifest,
then checks the retained synthetic files. The sole omitted original entry is an
internal panel close-out, not a pilot or canary input.
The `mlx` optional extra is for local Apple Silicon experiments only; no model
or tensor is shipped. Reproducing a model run requires the exact third-party
model under its own terms and the frozen protocol's cost accounting, as
explained in the [v2 inference capsule](research/replication/INFERENCE.md),
[v3 capsule](research/study-v3/REPLICATION.md),
[revision capsule](research/study-v3-revision/REPLICATION.md), and
[model terms](research/study-v2/THIRD_PARTY_MODEL.md). It is separate from
ordinary package verification.

A proposed change should include a focused regression test where behavior
changes, a `STATUS.md` update when claims change, and exact local check output.
AI-written contributions should be disclosed. Any new study requires a new
protocol with fresh held-out material and a declared falsifier before data
collection; the exposed Vela questions cannot be reused as fresh holdouts.

## AI assistance

OpenCode and Codex assisted code development, review, and documentation for this repository. Their cross-review is not independent human replication or external validation. Current claims are tied to retained tests, source and evidence replay, the bounded v2 result, static v3 pilot, and post-pilot revision follow-up; contributors should review those records before extending a claim. Disclose material AI assistance in future contributions and verify generated changes against the code and evidence.
