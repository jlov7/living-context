# Revision workload protocol, frozen before execution

Study identity: `lc-revision-workload-v1`. This is a deterministic synthetic
operation-count exercise, not a measured model run or a price study.

The executable is `experiments/revision_workload.py`. Generate undirected
co-training graphs with 128 documents: disconnected, chain, star, four
32-document cliques, and one full clique. Seed the Python random generator
with 230923. For each graph choose exactly 1, 4, or 16 changed documents with
the same deterministic sampler. Count rebuilt live documents under changed-only,
dependency closure, and full rebuild. Simulate 1, 10, and 100 reads per edit;
these do not change rebuild counts. Report rebuild units per read, steady
storage units (128 for all policies), and temporary peak units (128 plus rebuild
count), assuming one equal-size retained artifact per document and a candidate
artifact copy while rebuilding. Historical source bytes, manifests, runtime
metadata, and old artifact generations are outside this unit; this assumes an
external artifact garbage collector retires old copies after admission. Report
closure/full ratio and exact seed set. Compare the count model on representative
graphs to `plan_refresh` in a test before interpreting the results.

This model assumes all declared edges are real, artifact size and rebuild cost
are equal, no deletion, and reads have one unit of retrieval work in every
policy. It does not measure answer quality, training time, cache behavior,
serving hardware, or money. The chain, star, and clique cases are deliberate
counterexamples to an assumed selective-refresh saving. Changed-only can be
unsafe with co-training edges; lower count is not proof of correctness.

Stop if the script does not reproduce the same JSON bytes on a second run.
Do not adjust graph shapes or counts after seeing output. The script may emit
one SVG chart of closure/full ratio and a complete JSON table.
