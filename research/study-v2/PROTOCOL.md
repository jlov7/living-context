# Frozen replacement study v2

The machine-readable protocol is `protocol.json`. This document states its
interpretation. Source, data, model and tokenizer hashes in that file are the
execution authority. The runner rejects a modified tracked tree or mismatched
hash. All runs use the cached Llama 3.2 1B Instruct 4-bit MLX snapshot, the
tokenizer's chat template, greedy decoding, and at most 32 output tokens.

The separate Nacre calibration must return the exact JSON schema and answers
through fresh text and a real stored KV cache. The cached and fresh paths must
match on complete greedy output and first-next-token argmax with maximum
absolute logit error at most 0.1 after float32 casting of float16 logits.
This is a calibration-derived numerical threshold, not universal exact-logit
equivalence. Full generated token ID sequences must also match. Every query
gets a clone of the stored cache.
The Nacre revision must invalidate the earlier cache key. Calibration failure
is a reader, cache, or schema failure, not evidence against learning.

The development source is one Vela document. The trainer receives that source
record and its source-derived training QA only. The two held-out questions ask
the same dispatch-code fact in different wording; neither questions nor labels
are in the training payload. Two objectives are compared prospectively:
document reconstruction cross-entropy and source-derived QA cross-entropy.
Both use 64 learned embedding tokens, a frozen base, 400 Adam steps, learning
rate 0.01, and seeds 7, 11, 19. The question prompts and exact JSON answer
contract match the text controls. The train and answer workers together have
20 minutes, 2 GB sampled MLX active memory and 8 GiB process RSS. Resource
censoring and infrastructure failure are reported separately from a valid
negative result. No outcome-dependent rerun or parameter tuning is allowed.

The development pass requires bare model 0/2, cached text 2/2, and both
held-out answers exact for at least two of three seeds of one objective. Among
passing objectives, select the one with the largest number of exact development
answers across all seeds, then lower measured training plus serving time, then
the fixed objective identifier. A valid fail closes this model, objective and
budget configuration; it does not rule out other learned-context methods.

Only a pass permits the final study. Orchid and Juniper have document IDs
disjoint from Vela, but all records share a small synthetic code template;
this is no claim of structural-family generalization. The final study trains
the selected objective on Orchid v1, Orchid v2, and Juniper v1, each at the
same three seeds. The initial snapshot serves Orchid v1 and Juniper v1. The
revised snapshot serves Orchid v2 and Juniper v1. Query-based lexical retrieval
selects at most two active sources, independent of evaluator `source_ids`.
Selected prefixes are concatenated in retrieval order. Cross-document answers
must use exactly one space on each side of `+`; the scorer only strips outer
whitespace and folds case. The final train and both answer workers share a
90-minute cumulative model-worker ceiling across calibration, development,
final work, and retained failed attempts. All attempt directories must share
one parent so the supervisor can count the prior receipts. Development also
has its own 20-minute ceiling. No final labels may affect selection.

Every raw response must be a single complete JSON object with one string
`answer` field. Duplicate keys, extra prose, extra fields, and substring matches
fail. The runner seals all raw outputs before the separate scorer loads labels.
It rejects missing, duplicate, or extra arm, seed, objective, and question rows.
Training artifact inventories and hashes must match their source revision and
the frozen protocol. Training uses serialized process-input separation only;
it is not an operating-system filesystem denial. This synthetic pilot cannot
establish a cost or compression advantage, unlearning, or dependency-aware
co-training advantage. Report measured physical time, storage, and memory;
do not infer provider pricing.

Run each phase once in the same new, unique local output directory. Keep
failed directories and supervisor logs under the same parent as the next
attempt. The commands are, in order:

```sh
MODEL=/absolute/path/to/the/pinned/snapshot
OUT=/absolute/path/to/a/new/local/results-directory
P=research/study-v2/protocol.json
python experiments/replacement_study_v2.py calibrate --protocol "$P" --model-path "$MODEL" --output-dir "$OUT"
python experiments/replacement_study_v2.py score --protocol "$P" --raw "$OUT/calibration-raw.json" --output "$OUT/calibration-scores.json"
python experiments/replacement_study_v2.py gate --protocol "$P" --score "$OUT/calibration-scores.json" --output "$OUT/calibration-gate.json"
python experiments/replacement_study_v2.py train-dev --protocol "$P" --model-path "$MODEL" --output-dir "$OUT"
python experiments/replacement_study_v2.py answer-dev --protocol "$P" --model-path "$MODEL" --output-dir "$OUT"
python experiments/replacement_study_v2.py score --protocol "$P" --raw "$OUT/development-raw.json" --output "$OUT/development-scores.json"
python experiments/replacement_study_v2.py gate --protocol "$P" --score "$OUT/development-scores.json" --output "$OUT/development-gate.json"
# Only when development-gate.json passes:
python experiments/replacement_study_v2.py train-final --protocol "$P" --model-path "$MODEL" --output-dir "$OUT"
python experiments/replacement_study_v2.py answer-final-initial --protocol "$P" --model-path "$MODEL" --output-dir "$OUT"
python experiments/replacement_study_v2.py answer-final-revised --protocol "$P" --model-path "$MODEL" --output-dir "$OUT"
python experiments/replacement_study_v2.py score --protocol "$P" --raw "$OUT/final-initial-raw.json" --output "$OUT/final-initial-scores.json"
python experiments/replacement_study_v2.py score --protocol "$P" --raw "$OUT/final-revised-raw.json" --output "$OUT/final-revised-scores.json"
```

The model path above is an operator-supplied absolute path, not part of the
public export. This local candidate contains no tensor bytes.
