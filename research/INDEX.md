# Research document map

Use this map before interpreting older milestone pages. Dates and numerical
values in archived files remain historical evidence, including withdrawn
claims. The [current status](../STATUS.md) sets the claim ceiling.

| State | Documents | Meaning |
| --- | --- | --- |
| Current result | [v2 result](study-v2/RESULT.md), [machine result](study-v2/RESULTS.json), [v2 protocol](study-v2/PROTOCOL.md), [amendment](study-v2/AMENDMENT-1.md) | Valid bounded negative and predeclared stop; final study not admitted. |
| Separate pilot | [v3 result](study-v3/RESULT.md), [protocol](study-v3/PROTOCOL.md), [public raw evidence](study-v3/results/), [replay](../scripts/replay_study_v3.py) | Four static synthetic facts; QLoRA and cached text 8/8 each. No revision or lifecycle advantage claim. |
| Post-pilot follow-up | [revision result](study-v3-revision/RESULT.md), [frozen protocol and amendment](study-v3-revision/PROTOCOL.md), [public raw evidence](study-v3-revision/results/), [replay](../scripts/replay_study_v3_revision.py) | Two admitted synthetic generations; adapter 8/8 in each, cached text 8/8 then 7/8. No online latency, unlearning, or cost-crossover claim. |
| Count study | [revision workload](revision-workload/RESULT.md), [complete table](revision-workload/RESULTS.json) | Synthetic operation and storage units across graph/edit/query regimes; not model-quality evidence. |
| Current engineering | [README](../README.md), [architecture](../docs/ARCHITECTURE.md), [STATUS](../STATUS.md) | Supported in-memory catalog and planner, their trust boundary, and current claims. |
| Superseded v1 | [original protocol](PROTOCOL.md), [Milestone 0](MILESTONE-0-RESULT.md), [B](MILESTONE-B-RESULT.md), [C](MILESTONE-C-RESULT.md), [D](MILESTONE-D-RESULT.md), [E](MILESTONE-E-RESULT.md) | Historical design and results; old learned-accuracy, cost, completion, and freeze claims withdrawn or unverified. Read each correction banner. |
| Remaining questions | No further model run in this release | The bounded revision follow-up does not answer broad refresh quality or matched lifecycle cost. |

The public text evidence inventory is
[here](study-v2/PUBLIC-EVIDENCE-MANIFEST.json). It lists distributed v2
files and hashes; the private model and prefix tensors are deliberately absent.
The archived v1 diagnostic uses a [derived public inventory](../artifacts/evidence/PUBLIC-SHA256SUMS.json).
Its [original custody manifest](../artifacts/evidence/SHA256SUMS.json) remains
byte-for-byte intact; the public inventory records the single omitted internal
panel close-out and retains every scientific input needed for replay.
The [v3 public manifest](study-v3/PUBLIC-EVIDENCE-MANIFEST.json) identifies
distributed synthetic raw answers and worker receipts; adapter tensors and
third-party model bytes remain outside the source tree.
The [revision follow-up manifest](study-v3-revision/PUBLIC-EVIDENCE-MANIFEST.json)
identifies the corresponding synthetic two-generation raw answers, admission
metadata and worker receipts. Its tensor/config hashes can be checked against
private local files but those bytes are not distributed.
The [study-era source snapshot](study-v2/FROZEN-SOURCE.md) preserves the exact
protocol-bound code and lockfile separately from this candidate's current code.
[PACKAGE-OMISSIONS.md](../PACKAGE-OMISSIONS.md) describes what the source and
wheel distributions contain and why model weights are absent.

## Repository map

| Path | Role |
| --- | --- |
| Root `README.md`, `STATUS.md`, `CONTRIBUTING.md`, support and release files | Current entry, claim ceiling, developer recipe, and stewardship. |
| `.github/workflows/` and root configuration | Declared CI, package metadata, dependency lock, and repository settings; hosted status comes from actual workflow runs. |
| `src/living_context/` | Maintained in-memory catalog, optional artifact receipt, dependency graph, refresh planner, text reader, examples, and evidence-replay support. |
| `tests/` | Software and offline replay regression checks; optional MLX test requires its separate runtime. |
| `scripts/` | v2, static v3 and revision v3 evidence replay, v2 inference preflight, rerun comparison, and archived v1 diagnostic. |
| `experiments/` | Historical material, frozen v2 runner, deterministic workload count, and two frozen v3 QLoRA study runners; model execution is outside the ordinary quickstart. |
| `research/` | This map, methods and limits, v2 negative, v3 static and revision studies, workload count, and superseded v1 reports. `study-v2/frozen-source/` preserves study-era v2 source; public results contain text evidence, not model tensors. |
| `artifacts/evidence/` | Immutable v1 evidence and custody files; not a source of current learned-accuracy claims. |
| `docs/ARCHITECTURE.md` | Maintained design and API trust boundary. |
| `assets/readme/` | Self-contained reader-guide mark and revision-admission diagram; the same workflow and limits are stated in README text. |
