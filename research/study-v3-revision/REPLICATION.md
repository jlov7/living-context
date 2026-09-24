# Revision study replication capsule

`python -m scripts.replay_study_v3_revision` is a model-free replay of saved
synthetic responses, admission metadata, resource receipts, and exact scoring.
It checks the 15 frozen source/input hashes and the public evidence manifest.
The catalog source used by this study is retained at
`frozen-source/src/living_context/catalog.py`; the live package catalog may
receive later fixes without changing the study's pinned source hash. The
saved-output replay executes these exact verified bytes for the recorded
revision sequence. The hash-verified study runner still derives revision field
values, which the replay copies into historical catalog objects. Current
package behavior is checked by the separate core tests.

It does not rerun inference or rehash the private model/adapter files. An
outside person has not independently rerun the model study; that status is
**unverified**.

The runnable method was frozen at commit `59e82cfae6ba4b74a31a395c678db1d8516d07f5`
and amended before outcomes at commit `86965af479baee475d36b74fa235985e98d3c339`.
The current [`protocol.json`](protocol.json) pins the amended runner, method,
inputs, original freeze, and amendment bytes. The original version remains
[`protocol-v1.json`](protocol-v1.json). A source-only distribution must carry
these exact files; its Git history need not be available for preflight.

To perform a new local method run, obtain the pinned
`mlx-community/Llama-3.2-1B-Instruct-4bit` revision
`08231374eeacb049a0eade7922910865b8fce912` if you have the right to do
so under the Meta Llama 3.2 Community License. This repository redistributes
neither that model nor the trained adapters. Verify the six model file sizes
and SHA-256 values in `research/study-v2/protocol.json`. The recorded local
runtime was Apple M4 Max with Apple Metal, Python 3.13.15, MLX 0.32.2,
MLX-LM 0.31.3 and psutil 7.1.0. The installed MLX-LM LoRA trainer file hash
is also frozen in `protocol.json`. Hardware and numerical differences may
change outcomes even when sources match.

Fresh inference preflight still checks the original working-tree source paths,
including `src/living_context/catalog.py`, against the unchanged protocol.
Use the historical source at the amended method commit for a new run; the
frozen replay copy above does not relax that preflight. A new model run requires
separate authorization.

Use a new, absent output directory outside the public source tree. Preflight
checks source, training code and model bytes without training or downloading;
the run uses an operator-supplied **local** model path and offline flags:

```sh
python experiments/study_v3_revision.py --preflight --model /path/to/pinned-model --output /path/to/new-output
python experiments/study_v3_revision.py --run --model /path/to/pinned-model --output /path/to/new-output
python scripts/compare_inference.py --study v3-revision --candidate /path/to/new-output --report /path/to/new-output/discrepancies.json
```

The run performs seven 100-step training workers followed by one sequential
admission/inference worker. Its supervisor stops on worker failure, missing
memory evidence, a cumulative 900-second worker ceiling, sampled process-tree
RSS above 8 GiB, or reported MLX peak above 2 GB. Retain every worker log,
adapter tensor and config, all raw JSON and the complete comparison report,
including missing, extra, malformed, and divergent cells. The comparison
covers 48 primary arm outputs and four stale-adapter controls; it records raw
text and token IDs rather than hiding differences behind aggregate accuracy.
Use the public hashes and byte sizes as a readback target, but do not expect
byte-identical retraining. A new run is method replication, not replay of the
original private trained artifacts.

The frozen resource and quality gates apply once. A failed run or discrepancy
remains evidence of that attempt, not permission to choose easier facts,
additional seeds, changed prompts or new thresholds. The study still cannot
support a serving-cost crossover, online update latency, deletion from model
weights or generalization beyond these simple synthetic records.
