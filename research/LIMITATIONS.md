# Limitations

## 2026-09-22 audit additions

- Existing learned runs train on the evaluated Q/A and do not measure held-out generalization.
- The old objective and substring scorer are invalid for accuracy claims.
- Saved model outputs are truncated; rescoring is a lexical diagnostic.
- The text path did not measure a persistent prompt/KV cache, and cost totals omit phases.
- `mx.get_active_memory()` is an MLX active-allocation sample, not process RSS or a cap.
- The artifact is an embedding soft prefix, not a persisted per-layer KV cartridge.
- Cartridges-scale hardware does not prove a small corrected local test needs it.

## Known at design time

1. **Compute gate.** Cartridges at Scale uses H200/B200-class hardware. This machine is an M4 Max / 48 GB. Full reproduction is **not possible**. A small-scale port is a *separate feasibility experiment*, not a reproduction of the reported setup. Any result from it carries a correspondingly narrower claim.
2. **Pilot, not study.** 24 documents / 6 families / 3 revisions is a pilot design. No power analysis. No population claim.
3. **Authored corpus.** Synthetic documents with authored truth tables. This controls labels well but is not licensed real revision history. A later externally meaningful study needs real revision histories and separately reviewed labels.
4. **Deletion ≠ unlearning.** Removing a KV file proves nothing about comprehensive erasure. Any deletion statement must name the exact stored representations and observable behaviours tested.
5. **No independence.** Baseline and treatment share an author, a harness, and a reader model. Agreement between them is not external validation.
6. **Cost accounting is model-dependent.** Lifecycle-cost conclusions depend on the chosen price basis and hardware. Physical units and wall time must be published alongside any cost number.
7. **Optional components may not be needed.** The dependency controller may turn out to be unnecessary if artifact-level dependencies are tight. That is a legitimate finding.

## Structural risks

- **Baseline is strong.** Well-cached text retrieval with fresh indexing may simply win. Report it.
- **Impact set may be uninformative.** If measured dependencies are broad, selective refresh saves little.
- **Scheduler leakage.** Development/test separation must be enforced by test, not by convention.
- **Scope inflation.** The natural drift is toward "AI memory standard", "platform", or "unlearning system". All are out of scope.

## What a reviewer should attack

- The strongest overlooked predecessor (candidates: CAS follow-ups; update/invalidation work in cache-management literature).
- The simplest baseline that defeats the design (cached text retrieval done well).
- One fatal experimental flaw (candidate: contamination between authored revisions).
- The smallest additional contribution they would consider useful.

## Stop conditions (commit to these)

Stop expanding if:
- cached text dominates;
- the update mechanism merely reimplements existing libucks/CAS behaviour without a new measurement or improvement;
- the compute budget cannot support a meaningful comparison.

**Narrow to the demonstrated regime. Do not widen scope to compensate.**
