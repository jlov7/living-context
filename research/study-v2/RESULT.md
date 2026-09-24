# Replacement study v2 result — 2026-09-22

**Decision: valid negative for this frozen local configuration.** The
development gate failed, so the larger two-document revision/composition study
was not run. This closes the tested Llama 3.2 1B 4-bit, 64-token soft-prefix,
400-step, two-objective configuration. It does not rule out other learned
context methods or budgets.

The source/data implementation was committed at `3d1791c286a30b05f529765ef2f20fe6e70ae27f`.
The first protocol freeze was `10ecc20885440d61f958039c6cefb79bdac890b4`.
The separate Nacre calibration exposed a numerical cache-logit threshold
problem. [AMENDMENT-1.md](AMENDMENT-1.md) records that failure and all
calibration-only diagnostics. Before any development output, the amended source
was committed at `a2a7ed3b6a7bb795bef4082639cf8319354e4526` and the amended
protocol at `3c9f399cfc1471e60a664383bbc1dc7b6311ee8b`.

The amended `protocol.json` **file bytes** hash to SHA-256
`ecdb4e7c497a903550401a31813167c6fd2b68c1d392e1bd1b40394a5d2377bd`.
The runner's `canonical_json(parsed_protocol)` hashes to
`becbfc4cade215795a3032489208307b93a4754b0b1c2dac1ca8a14c20a5578b`.
Raw outputs and gates carry the latter. The two hashes refer to the same
amended protocol; neither is the earlier frozen file hash.

The amended calibration passed: both Nacre questions were exact through fresh
and cached text; the complete generated token ID sequences matched; next-token
argmax matched; maximum absolute logit differences were 0.05859375 and
0.04296875 after float32 casting of float16 logits, below the pre-development
0.1 threshold. Revision invalidation removed one old cache entry, and the new
revision's cached and fresh outputs matched. This validates the reader/cache
contract for this local calibration. It is not exact-logit equivalence.

| Development arm | Exact answers | Denominator |
|---|---:|---:|
| Bare model | 0 | 2 |
| Fresh text | 2 | 2 |
| Real cached text | 2 | 2 |
| Document-reconstruction prefix, three seeds | 0 | 6 |
| Source-QA prefix, three seeds | 0 | 6 |

Every prefix response failed the strict single-JSON-object, one-string-answer
contract or the exact normalized answer. The text arms used the same model,
chat template, questions, greedy decoder, and 32-token cap. The trained prefix
arms used the source record only; held-out question and label files were
absent from the training worker payload. The source-derived training questions
used different wording from the two held-out Vela paraphrases. Those two
questions probe one fact, not independent fact coverage.

An additional sealed, post-gate diagnostic on existing artifacts confirmed
that source-QA seed 11 generated its exact source-derived training answer and
document-reconstruction seed 7 generated the exact full Vela record. For those
two checks, direct-forward first-token argmax, streamed first token and
teacher-forced first target agreed; teacher-forced accuracy was 1.0 with mean
cross-entropy below 0.0005. This supports interpreting the held-out failures
as bounded transfer/format failures rather than an obvious broken prefix
inference path. It does not validate every trained artifact or every training
example. No retraining, setting change, or final evaluation followed the
development result.

The six local prefix tensor files total 3,146,860 bytes (retained only in the
private local result directory). The one-document cached KV entry measured
8,388,608 bytes, and the lexical index 144 bytes. Training worker elapsed was
335.683 seconds and answer worker elapsed 5.334 seconds. Including successful
calibration, the failed calibration and the charged diagnostics, the cumulative
model time was 363.511 seconds under the 5,400-second ceiling. The sampled
training MLX active peak was about 701 MB and supervisor process RSS peak about
1.08 GB, below the declared 2 GiB and 8 GiB stops. These are local physical
measurements, not provider prices or a lifecycle cost result. Prefix accuracy
failed, so the storage difference cannot establish utility or dominance.

The source records are synthetic and non-sensitive. Development and proposed
final document IDs are disjoint, but they share a small code-record template.
No final cross-document answer, dependency-aware co-training advantage,
generalization to other document families, deletion/unlearning behavior, or
cost decision boundary was established. This study uses embedding soft
prefixes, not persisted per-layer KV artifacts. Training workers had
serialized input separation, not an OS filesystem denial.

`RESULTS.json` is a deterministic machine-readable derivation. The copied
`results/` directory contains raw text outputs, scored rows, gates, hashes,
journals and failure receipts, but no model, tokenizer or prefix tensor bytes.
The original local result root retains the six tensors. Reproduce strict
scoring and gate decisions without MLX or tensor files with:

```sh
python experiments/replay_study_v2.py --evidence research/study-v2/results
```

This replay verifies sealed text receipts, exact scoring and gate arithmetic.
It checks artifact metadata and hashes recorded in training receipts; without
the local tensors it cannot rehash tensor bytes or repeat model inference.
The evidence-only replay script was added after the result as an audit aid; the
runner, data, exact scorer and stopping gates used for the model run were
committed before development output.
`RESULTS.json` lists the evidence file hashes and uses distinct keys for the
literal protocol-file and canonical parsed-protocol digests.

Local verification after this result: 68 tests passed with 90.39% coverage
(90% gate); Ruff and the native `src tests scripts` Pyright gate passed. Both
new study scripts passed Pyright separately. The historical offline
`scripts/check_reproduction.py` succeeded. An offline wheel and source archive
built successfully; the wheel installed in a fresh local environment, where
the evidence-only replay reproduced `RESULTS.json` byte for byte. No hosted CI
or public export execution is claimed here.
