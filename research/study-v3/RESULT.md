# Static QLoRA pilot result

**The preregistered local quality gate passed, with a narrow claim.** At the
frozen amended source commit `2b94ce67830da4ff9ffd98095dc48b3a01869f96`,
the Cobalt held-in positive control answered exactly. Four per-document rank-4
LoRA adapters then answered both held-out question wordings for each of four
synthetic facts. The same frozen model with revision-keyed cached source text
also answered all eight. The no-context model answered none.

| Arm | Exact answers |
|---|---:|
| Per-document QLoRA adapter | 8/8 |
| Revision-keyed cached source text | 8/8 |
| No-context frozen model | 0/8 |

The four facts span two small record families, routing and permit. The facts
and answers were supplied as source-derived QA during adapter training; only
the evaluation **wordings** were unseen by the training worker. The worker
received generated training/validation JSONL containing one record's QA, not
the evaluation file. This was serialized input separation, not an OS denial
of evaluation-file access. One adapter was loaded with a fresh base model for
each document, so adapters did not accumulate across documents. Cached text
used a revision-keyed prompt cache and was checked against fresh text for
token-sequence parity. All arms used the same reader model, system instruction,
chat template, questions, greedy decoding, 32-token cap, and strict one-field
JSON scorer.

Five training workers, the control inference worker, and pilot inference worker
all completed. Their summed elapsed time was 53.197 seconds, including 44.472
seconds for training workers. Maximum sampled process-tree RSS was
2,415,116,288 bytes; maximum reported MLX peak was 1.646 GB. Each adapter
file is 5,647,955 bytes; the observed one-document cached KV entry is
8,388,608 bytes. Across the eight queries, adapter generation totalled 0.883
seconds and cached-text generation totalled 0.638 seconds. Cache build and
fresh/cached parity validation were recorded separately; these generation
figures exclude adapter/model load, training, parity validation and cache
construction. They are not lifecycle totals. Local units and one small sample
do not support a cost crossover or commercial saving.

The [protocol](PROTOCOL.md), [pre-outcome amendment](AMENDMENT-1.md), and
machine-readable [audit result](RESULTS.json) identify the tested source and
private evidence hashes. The [public raw JSON](results/) and
[manifest](PUBLIC-EVIDENCE-MANIFEST.json) allow a model-free
[`replay_study_v3.py`](../../scripts/replay_study_v3.py) to recompute all exact
scores and check seven successful worker receipts and resource bounds. Private
local output additionally retains logs, training worker inputs and adapter
tensors; the public source records their hashes and aggregate metrics. The
output was not externally notarized or independently rerun.
The model and tokenizer remain under the [Meta Llama 3.2 Community
License](../study-v2/THIRD_PARTY_MODEL.md), separate from this repository's
code license.

This pilot used a standard [LoRA](https://arxiv.org/abs/2106.09685) weight
adapter, a different representation from v2's virtual embedding soft prefix.
Its positive result does not alter the sealed v2 negative result: source facts,
training configurations and question sets differ. The pilot contains no
source revision, deletion, co-training dependency, or lifecycle policy test.
It shows that this small adapter can answer these four static synthetic facts
as accurately as cached text under the measured setup. A revision/cost study
would need a separate frozen protocol and outside replication before broader
claims. This AI-assisted account does not present the owner's personal
independent rerun as completed.

The [replication capsule](REPLICATION.md) gives rights-aware acquisition,
preflight, fresh output and full response-comparison steps. No outside runner
has completed them.
