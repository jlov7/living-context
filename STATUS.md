# STATUS

**Local validation baseline, 2026-09-24:** version 0.1.0 presents revision-aware
context research methods and an in-memory reference. The model-free lifecycle,
checked artifact admission, caller-owned text reader, and exact v2/v3 saved-output
replays are the current supported workflows. See [README.md](README.md) and
[research/INDEX.md](research/INDEX.md). The checks below describe local
observations, not a hosted CI result or independent model replication. Use the
current [workflow run](https://github.com/jlov7/living-context/actions) and
release records for those separate statuses.

**Engineering state:** local core, checked admission, text reader, revision
workload count model, and evidence replays verified on 2026-09-23.
**Scientific state:** **valid bounded negative** for the frozen local v2
Llama 3.2 1B 4-bit soft-prefix configuration. Cached text answered 2/2 held-out
Vela questions; prefixes answered 0/12. The final revision/composition phase
was closed by the predeclared development gate. A separate v3 static QLoRA
pilot passed its quality gate: per-document adapters and cached text each
answered 8/8 questions about four synthetic facts; the bare reader answered
0/8. A separate post-pilot revision follow-up passed its fixed gate on new
synthetic records: adapters 8/8 in each of two phases, cached text 8/8 then
7/8, bare model 0/8 in each. It exercised checked admission and a persistent
revision-keyed text cache; it did not establish an online update latency or
lifecycle-cost decision boundary.

## Verified now

- Immutable complete corpus snapshots; partial snapshots cannot silently drop
  live documents.
- The process-local API requires an actual Boolean `checks_passed` result from
  a trusted caller. Optional `verify_artifacts` checks complete manifest
  coverage, exact source revisions, and supplied bytes before issuing a
  candidate-bound receipt. Concurrent readers receive complete old or new
  snapshots; competing stale admission fails. A single trusted owner
  serializes catalog mutation. Digests do not authenticate the caller or prove
  training occurred; there is no durable or distributed admission.
- Admission revalidates manually constructed snapshots, rejects rollback and
  stale bases, and retains explicitly addressable historical generations.
- Staging an update and deletion for the same document now fails before a
  candidate is built, leaving the admitted generation unchanged. The v3
  revision saved-output replay reads the exact catalog source pinned by its
  protocol from `research/study-v3-revision/frozen-source/`. This keeps the
  historical replay stable while the current catalog changes. It does not
  rerun inference or change the study's protocol, results, or evidence.
- Dependency closure starts from the old trained revision and maps affected
  documents to their new rebuild revisions; deletions retire artifacts.
- Generated edit/delete/restore traces match a separate full-rebuild oracle;
  planted admission and planning faults are detected. The `TextReader` CLI
  caller returns only admitted text with generation/revision labels and rejects
  stale fallback plans. It does not generate language.
- A preregistered synthetic 128-document workload mapped 45 graph/edit/query
  cells to operation/storage units. Dependency closure reaches full rebuild
  in chain, star, and full-clique counterexamples. It makes no model-quality
  or price claim.
- Retained evidence and the archived protocol are SHA-256 pinned. Missing,
  changed, incomplete, duplicate, non-finite, or unlisted inputs fail the
  offline reproduction gate.
- CI dependencies for pyright and pytest-cov are declared. The former
  false-green optional model job was removed.
- The corrected study's separate reader/cache calibration passed after a
  pre-development numerical-tolerance amendment. All six development prefixes
  trained with a frozen base and true cross-entropy; all held-out outputs,
  exact scores, gate decisions and resource receipts are retained. The
  evidence-only replay works without model or tensor files. See
  `research/study-v2/RESULT.md` and `RESULTS.json`.
- The candidate v2 release check compares the complete deterministic replay
  with `RESULTS.json` and checks the distributed text inventory. The historical
  v1 reproduction check remains a custody and saved-output diagnostic.
- The separate v3 QLoRA pilot used a frozen model and pinned local trainer,
  an exact held-in positive control, four source facts in two families, and
  eight unseen question wordings. All seven workers completed within the
  preregistered resource bounds. Public synthetic raw outputs and receipts
  replay exactly without private model or adapter tensors. An outside
  inference rerun remains unverified.
- The post-pilot revision study used a separately frozen and pre-outcome
  amended protocol. Seven adapter training workers and one sequential
  admission/inference worker completed within resource bounds. Exact
  tensor/config receipts, two admitted generations, stale and failed-candidate
  rejection, unchanged reuse, deletion, cache invalidation, and all 52 saved
  response cells are retained. The source-only replay and separate local
  readback pass; outside inference replication remains unverified.

## Historical model evidence

The 2026-09-21 scientific observations remain preserved as historical evidence.
The [public subset inventory](artifacts/evidence/PUBLIC-SHA256SUMS.json) records
the omission of one internal panel close-out while retaining the original
manifest bytes. The old substring summaries reproduce, but the study claims
are withdrawn:

- the learned prefixes were trained on the evaluated question and answer;
- the training objective used raw target logits, not cross-entropy;
- the old substring scorer counted malformed repeated/glued answers;
- the arms used different decoding limits and incomplete cost accounting;
- the text path did not measure a persistent prompt/KV cache;
- three repeats of the same trained question are not independent samples;
- protocol-before-run chronology is unverified because protocol and result
  first appear in the same commit;
- the refresh demo did not exercise the claimed complete atomic admission or
  learned cross-document composition.

The saved-output boundary diagnostic is 9/9 for text and 0/9 for learned. It is
a lexical audit of truncated retained strings, not a corrected accuracy result.
The former 6/9 learned accuracy, non-inferiority, cached-text dominance, and
lifecycle decision-boundary statements are not supported.

## Claim ceiling

The software core is suitable for offline use. The v2 result is a negative
feasibility finding for its declared prefix budget, not a population result.
The v3 static pilot and its post-pilot revision follow-up show exact answers
for a few synthetic source facts under one standard adapter method. The
follow-up demonstrates local revision-specific adapter selection and checked
admission, but does not validate measured dependency training edges, deletion
from weights or unlearning, online update latency, a lifecycle-cost crossover,
a serving deployment, or broad generalization. The v2 larger study remains
stopped; neither v3 result alters that decision.
