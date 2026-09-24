# Pre-outcome amendment 1: complete artifact and resource custody

This amendment was made after protocol/source commit `59e82cfae6ba4b74a31a395c678db1d8516d07f5`
and before any revision-study model worker or outcome. The original freeze is
retained as `protocol-v1.json`; the active `protocol.json` now pins the
amended executable hashes. Synthetic facts, question wordings, model,
rank, 100-step training, eight primary cells per phase, >=7/8 and >= cached
text threshold, stale-control count, and 900-second/8-GiB/2-GB bounds are
unchanged.

The initial runner bound only `adapters.safetensors`, while MLX also reads
`adapter_config.json`. The amended admission binds both files as separate
artifacts with the same exact source revision. Immediately before each model
load, both files are rehashed against the admitted receipt. Saved admission
records retain their hashes and byte counts. This is a trusted local caller
check; a digest is not authentication or a claim about hostile concurrent
filesystem mutation.

A successful worker with zero observed RSS or zero reported MLX peak now
fails the resource gate as a missing metric. The supervisor checks that after
each worker so subsequent training does not proceed on absent evidence. The
model-free scorer also requires exact seven training identities, one inference
identity, nonzero finite resource observations within the frozen totals, all
initial/final answer cells, both admission records, generation transition,
stale and Boolean candidate rejections, changed/deleted cache invalidation,
unchanged artifact reuse, deleted-revision absence, and exactly four separately
scored stale-adapter control cells. A model-free regression plants missing
lifecycle evidence and zero RSS to check these fail closed.

This fixes custody and fail-closed checks. It does not adjust the method or
acceptance threshold in response to quality results; no revision quality
results existed when this amendment was written.
