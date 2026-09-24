# Post-pilot revision follow-up, fixed before revision outcomes

Identity: `lc-study-v3-revision`. This protocol was written after the separate
static `lc-study-v3-qlora` pilot passed its 8/8 development gate. It is a new,
exploratory follow-up, not a retrospective preregistration or a reopening of
the sealed v2 soft-prefix failure. The model, rank-4 last-eight-layer LoRA
configuration, 100 training iterations per document, greedy 32-token decoder,
chat template, strict one-string JSON answer, model snapshot, and resource
bounds remain fixed. No seed, step, prompt, target, or threshold search is
allowed after results.

Four new synthetic records cover two routing and two permit instances. Each
adapter trains on its document's source-derived QA only. Two evaluation
question wordings per document are absent from training serialization. Source
facts are seen during training; the evaluation wordings are held out. A
separate static pilot already established a functioning control for this
method on different records.

Train four initial adapters. After they all finish, train exactly three new
adapters: Cedar routing and Larch permit change values, Drift routing remains
unchanged, Mesa permit is deleted, and Olive permit is added. Training all
seven before serving either snapshot isolates admission/query behavior but
does not measure online update latency. One inference process first admits
and queries the initial four. It then stages the final four with exact artifact
receipts, deliberately rejects one stale-artifact candidate, rejects one
otherwise valid candidate by its Boolean gate, and checks the old generation
is still current. It admits a newly verified final candidate, explicitly
invalidates changed and deleted cached-text entries, retains the unchanged
entry, and queries all final live records. Adapter loading must hash the
actual file against the admitted receipt's artifact digest. Historical or
deleted revisions are never reported as current. Four answers from old
adapters on changed-document final questions are saved as separate stale
controls and excluded from primary accuracy.

Initial and final phases each have exactly eight primary question cells, each
with fresh-adapter, base-without-context, and revision-keyed cached-text arms.
The cached-text arm validates fresh/cached parity but reports cached-generation
time separately from parity and build time. The quality gate requires adapter
exact >=7/8 **in each phase** and no fewer correct than cached text in each
phase. Complete all fixed cells even if a quality threshold fails. Resource,
worker, source-custody, or parity failure stops the run and does not become a
quality result. No further training or modified retry is allowed after a
completed run.

The same model-free supervisor checks each direct MLX worker and descendants:
cumulative wall time <=900 seconds, sampled process-tree RSS <=8 GiB, reported
MLX peak <=2 GB, a present memory report, and successful worker exit. Report
all raw outputs, exact source/adaptor hashes and bytes, per-document train and
load time, cached generation/build/parity time, cache hit/build counts,
active versus retained adapter bytes, rejected candidates, and retired
visibility. These are local observations on five synthetic document identities
and two record families. They cannot establish deletion from model weights,
generalization, deployment cost, a price crossover, or a product advantage.

The separate `protocol.json` pins executable and input hashes. The preflight
must pass before a new output directory is created. Model weights and trained
adapters stay outside the public source tree; a later public replay may verify
saved synthetic outputs and receipt hashes without redistributing weights.
