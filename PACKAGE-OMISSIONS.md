# Distribution contents

The source distribution carries the maintained Python code, tests, synthetic
research records, frozen study source, and model-free replay tools. The wheel
carries the Python package and its metadata; it does not include the repository's
research files or replay scripts. The [research map](research/INDEX.md)
distinguishes current results from superseded v1 material.

The source distribution retains the scientific v1 evidence and v2 text
evidence, their manifests, the protocol-bound study-era source snapshot, and
the replay tools. The original v1 custody manifest is preserved unchanged.
The [derived public inventory](artifacts/evidence/PUBLIC-SHA256SUMS.json)
records one omitted internal panel close-out by path and original digest;
`check_reproduction.py` verifies that derivation before replaying the
remaining synthetic evidence. The source distribution also excludes
third-party model/tokenizer files and private learned-prefix tensors; those
were never tracked in this source tree. A package build and file-inventory
readback are required before claiming a particular archive contains these
files.

The source distribution also retains the v3 pilot's synthetic raw responses,
worker receipts, hash manifest, result, and model-free replay tool. Local
training logs and learned LoRA adapter tensors are not tracked or included;
their file hashes and sizes appear in the result inventory. A source-only
replay does not rerun model inference or rehash those private tensors.

The post-pilot v3 revision follow-up likewise includes its amended frozen
protocol, synthetic raw initial/final and stale-control responses, admission
hashes and byte counts for tensors **and** adapter configs, worker receipts,
result, and model-free replay. Its model files, trained tensor/config bytes,
worker logs, and local coordinator readback remain outside the source archive.
Public replay cannot independently rehash absent trained artifacts or repeat
inference.
