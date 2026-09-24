# Soft-prefix pilot: a bounded negative result

The v2 study asked whether a 64-token learned embedding prefix could carry one
synthetic source fact into two held-out question wordings on a frozen Llama 3.2
1B 4-bit reader. The source was a short Vela code record. Six prefixes came
from two training objectives and three seeds. Evaluation required a single
JSON object with an exact one-string answer. The questions were unseen by the
training worker, but both asked about the same fact; they are not two
independent source cases.

Fresh text and a real revision-keyed cached-text entry each answered 2/2.
The bare reader answered 0/2. The six learned prefixes answered 0/12 exact
trials. A sealed diagnostic later recovered a source-derived training answer
and source record from two selected prefixes, so the narrow observation is a
failure to transfer into the held-out wording/output contract, not proof that
no learning occurred. The frozen development gate failed; the larger
two-document revision and composition phase was never run. The v2 gate stays
closed. No later tuning can turn this result into a positive v2 outcome.

The execution order matters. Nacre first checked strict JSON output and the
fresh/cached-text reader on a separate calibration fact. After an initial
numerical cache comparison failed, the frozen amendment changed only that
calibration threshold before development output. The amended calibration
passed. The Vela development run then trained all six prefixes under the
fixed 400-step, three-seed protocol and saved raw answers before scoring. A
passing gate would have allowed two new documents, revision, and composition;
the failed gate prevented those later phases.

```text
Vela source fact ──► 64 learned embedding tokens ──► frozen reader + unseen wording
       │                                             │
       └────────► fresh/revision-keyed cached text ──┘
                                 strict one-field JSON exact scorer
```

A malformed JSON response fails the output contract; a well-formed but wrong
answer fails exactness. A cache parity failure would invalidate the text
control, while missing or mismatched artifact evidence would invalidate the
run. Neither occurred in the amended development result. The observed prefix
failures alone do not isolate why transfer failed. The post-gate
training-question diagnostic checks two selected saved artifacts, not all six
or the held-out wording.

The six private prefix tensors totalled 3,146,860 bytes. The one-document
cached KV entry was 8,388,608 bytes and its lexical index 144 bytes. The
training worker took 335.683 seconds; the answer worker took 5.334 seconds.
All charged model calls totalled 363.511 seconds. These are local observations
with different roles: saved artifact bytes and cache bytes are storage
measurements; elapsed times are worker measurements. Since prefix quality
failed, smaller stored tensors do not show a useful cost tradeoff. The
experiment did not measure lifecycle cost across revisions, prices, a serving
deployment, or other document families.

The [protocol](../research/study-v2/PROTOCOL.md) and
[amendment](../research/study-v2/AMENDMENT-1.md) define the frozen test. The
[result](../research/study-v2/RESULT.md) maps counts and costs to the sealed
[raw evidence](../research/study-v2/results/). The offline
[`replay_study_v2.py`](../experiments/replay_study_v2.py) recomputes scoring
from saved responses. That replay does not run inference. Original private
prefix tensors and model weights are not in the public source tree. The
[inference capsule](../research/replication/INFERENCE.md) explains what a
future runner could and could not reproduce.

| Claim | Evidence | Limit |
|---|---|---|
| 2/2 cached-text and 0/12 prefix exact | Frozen development raw responses and offline replay | One fact, two wordings; seeds are related trials |
| Calibration valid | Nacre raw outputs and cache/logit checks | Local numeric tolerance, not exact-logit identity |
| Some source information retained | Post-gate diagnostic raw output | Two selected prefixes, training wording only |
| Storage and elapsed values | Private worker receipts and public summaries | No useful lifecycle or price comparison |

This AI-assisted technical note reports the tested method and its limit. It
does not claim that learned compression generally fails or that the owner
personally reran inference independently.
