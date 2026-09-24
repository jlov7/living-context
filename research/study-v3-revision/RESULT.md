# Post-pilot revision follow-up: local result

The single frozen `lc-study-v3-revision` run passed its narrow quality gate.
Four initial per-document QLoRA adapters answered 8/8 strict JSON questions;
the revised live corpus also answered 8/8. The actual revision-keyed cached
text reader answered 8/8 initially and 7/8 after revision. The unadapted
model without source context answered 0/8 in each phase. The gate required
adapter >=7/8 and at least the cached-text count in **each** phase. It was
fixed in [PROTOCOL.md](PROTOCOL.md) and [protocol.json](protocol.json), with
one [pre-outcome amendment](AMENDMENT-1.md) to strengthen artifact and resource
custody. The original protocol version is retained as
[protocol-v1.json](protocol-v1.json). No revision outcome preceded that
amendment.

| Phase | Adapter | Cached text | Bare model | Text cache builds / hits |
|---|---:|---:|---:|---:|
| Initial, four live records | 8/8 | 8/8 | 0/8 | 4 / 4 |
| Final, two updates, one retained, one deletion, one addition | 8/8 | 7/8 | 0/8 | 3 / 5 |

The one final cached-text error was Cedar routing question 0: it returned
`CE-49616` when the revised source stated `CE-49615`. Fresh/cached token
parity passed. This was a wrong model answer with text present, not evidence
of a stale cache entry. Four separately retained old-adapter responses to
changed-record final questions answered 0/4 against current values (and 4/4
against their old values); they were excluded from the 8-cell final primary
denominator. This does not show that old adapters forgot the records or that
deletion unlearned anything.

The exact admission API bound both tensor and adapter-configuration files to
source revisions. Generation 1 held four r1 records; generation 2 held
Cedar r2, Drift r1, Larch r2, and Olive r1. Mesa was deleted. A candidate
with a stale Cedar artifact was rejected, and a verified candidate rejected
by its Boolean gate left generation 1 current. The next verified candidate
admitted generation 2. Drift reused its exact r1 artifact. The persistent
text-cache registry invalidated one entry each for changed Cedar, changed
Larch, and deleted Mesa; its unchanged Drift entry was hit twice in the final
phase. These are observed local checks of the in-memory reference, not a
distributed admission or hostile-filesystem security result.

Seven fixed 100-step training workers completed in 47.968 seconds total; the
single sequential inference/admission worker completed in 13.660 seconds.
Total worker elapsed was 61.628 seconds against 900 seconds. Maximum sampled
process-tree RSS was 2,663,972,864 bytes against 8 GiB; maximum reported MLX
peak was 1.682 GB against 2 GB. All eight workers exited successfully with
memory reports. Active final adapter tensor plus config bytes totaled
22,597,396; retaining all seven trained versions required 39,545,440 bytes.
Each cached-text KV entry reported 8,388,608 bytes. Initial adapter, cached
generation, and bare generation times summed to 0.941, 0.664, and 0.828
seconds across eight questions; final sums were 0.901, 0.666, and 0.788
seconds. Those generation totals exclude the seven training workers,
per-adapter loading (1.811 initial / 1.766 final seconds), and cache builds
(0.127 / 0.087 seconds). The total fresh/cached validation path took 1.904 /
1.830 seconds; it includes cache building and cached generation, so these
overlapping measurements must not be added. They are not matched lifecycle
costs or service prices. All seven
adapters were trained before the two snapshots were served, so the sequence
does not measure online update latency.

The [public raw evidence](results/) and
[manifest](PUBLIC-EVIDENCE-MANIFEST.json) contain synthetic answers, exact
source/admission hashes, tensor/config hashes and sizes, worker receipts, and
status. Run `python -m scripts.replay_study_v3_revision` from the source tree
to verify the source/evidence hashes, complete cells, exact scores, candidate
digests, cache hit/build pattern, lifecycle checks and resource bounds without
MLX. The separate local coordinator readback independently checked all 48
primary raw arm answers, four stale controls, and actual private tensor/config
files against these receipts; its receipt remains outside the public source
tree. Source-only replay cannot rehash private weights or repeat inference.
The [replication capsule](REPLICATION.md) gives a rights-aware new-run path.

This exploratory result covers four initial synthetic records, two changed
values, one retained record, one deletion and one addition across two simple
record templates. The source facts were seen in source-derived training QA;
only evaluation wordings were held out. It does not establish broader
generalization, dependency-aware co-training benefit, learned deletion,
unlearning, production serving behavior, or a cost crossover. The earlier v2
soft-prefix result remains its own sealed negative (cached text 2/2,
prefixes 0/12); this study cannot change that result. An outside person has
not rerun inference independently.

This AI-assisted report describes local saved outputs and their limits; it
does not attribute an independent rerun or review to the repository owner.
