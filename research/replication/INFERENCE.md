# Inference replication capsule

The public v2 evidence replay verifies saved output bytes, seals, exact
scoring, and gate arithmetic. It does not repeat inference. An outside runner
has not rerun this study; independent replication remains **unverified**.

The frozen executable source and input hashes are in
[`study-v2/protocol.json`](../study-v2/protocol.json) and the copied
[`frozen-source/`](../study-v2/frozen-source/) tree. Use the exact pinned
model revision `mlx-community/Llama-3.2-1B-Instruct-4bit` at
`08231374eeacb049a0eade7922910865b8fce912` only if the runner has the
rights to access it under the Meta Llama 3.2 Community License. Verify every
file's byte size and SHA-256 from the protocol before load. The model and
tokenizer are not redistributed here. A machine with Apple Metal, Python 3.13,
MLX 0.32.2, and MLX-LM 0.31.3 matches the recorded runtime most closely;
hardware and numerical variation remain possible.

Run in a **new** output directory with the frozen source and inputs, retaining
both stdout/stderr and the model file readback. The frozen study runner has
`calibrate`, `train-development`, `answer-development`, and scoring commands;
inspect `--help` in the frozen source before execution. Never point it at the
archived `results/` directory. Preflight must compare the source and input
file hashes listed in the protocol and reject a mismatch. This is a costly
model rerun, not part of package tests or CI.

```sh
python scripts/preflight_v2_inference.py --model /path/to/pinned-model --output /path/to/new-output
```

To compare a rerun, retain every generated response, including malformed JSON,
and compare by phase, arm, seed, question ID, exact token sequence, and score
against `development-raw.json` and `development-scores.json`. Report all
discrepancies with model/runtime/hardware descriptors. The original six
private prefix tensors were not included in the public archive; a new runner
cannot rehash or directly reinfer from those exact tensors. Retraining from
the frozen inputs creates new artifacts and is a method replication, not
byte-identical replay of the original inference. Do not call a successful
evidence-only replay independent inference replication.

After a new development answer worker, retain a full comparison with:

```sh
python scripts/compare_inference.py --study v2 --candidate /path/to/new-output/development-raw.json --report /path/to/new-output/discrepancies.json
```

The comparison records text and generated token IDs for every arm/question
cell, including missing, extra and divergent cells. It does not hide
discrepancies behind an aggregate score.
