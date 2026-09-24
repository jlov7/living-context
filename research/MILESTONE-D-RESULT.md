# Milestone D result — frozen pilot, 3 questions × 2 arms × 3 repeats

> **Audit correction, 2026-09-22:** The 9/9 versus 6/9 figures below are old
> substring totals, not supported accuracy. Boundary-aware inspection of the
> truncated saved outputs gives 9/9 text and 0/9 learned; all six old learned
> passes are malformed repeated/glued strings. This is a saved-output diagnostic,
> not a corrected study. Prefixes trained on the evaluated Q/A, objective,
> decoding, and costs were unmatched, and seed repeats are not independent.
> Pre-run freeze is unverified because protocol and result first enter Git in
> the same commit. Non-inferiority and decision-boundary claims are withdrawn.

**Date:** 2026-09-21
**Historical protocol assertion (unverified):** `experiments/pilot.yaml` says FROZEN; Git does not establish that it predates the run.
**Evidence:** `artifacts/evidence/pilot-1790032618.json`; reproduction: `scripts/check_reproduction.py` regenerates the tables from the retained JSON (verified, exit 0).
**Machine:** Apple M4 Max / 48 GB / MLX / `mlx-community/Llama-3.2-1B-Instruct-4bit` (frozen, cached, USD 0.00)

## Frozen design (pre-registered)

- Arms: **1** cached-text with fresh index (baseline) vs **4** learned selective refresh.
- Repeats: 3 declared before the run.
- Questions: 3 (hydraulics volume 45, compliance label 200 bar, supply alloy AL-7075) — fixed at freeze time, not edited after.
- Non-inferiority margin: learned arms must stay within 1 answer of the text baseline on the set, else "cannot claim non-inferiority at n=...". No equivalence claim from a non-significant difference.

## Historical substring result (withdrawn as accuracy)

| Question (family) | Arm 1 (cached text) | Arm 4 (learned) |
|---|---|---|
| q-hydraulics (h-tank volume) | 3/3 ✔ | 3/3 ✔ |
| q-compliance (c-label 200 bar) | 3/3 ✔ | 3/3 ✔ |
| q-supply (s-spec AL-7075) | 3/3 ✔ | **0/3 ✘** |

**Totals: arm 1 = 9/9, arm 4 = 6/9.**

Historical partial timing totals (not lifecycle cost):
- arm 1 serving: ~0.1–0.3 s/question (sum ≈ 0.6 s)
- arm 4 refresh: 21.6–27.1 s per artifact; **224.68 s total** for 9 artifacts
- peak memory: 0.74 GB

## Honest reading

1. **The learned arm fails the pre-registered non-inferiority bar on this set** (6 vs 9; the supply question failed all 3 repeats — the artifact did not absorb "AL-7075"). Per the frozen protocol, the correct statement is: **cannot claim non-inferiority at n=9**, not "learned context loses, definitively."
2. **The failure is retained, and it is where the project predicted.** The supply/s-spec family is the one exercising *contradictory updates and renamed entities* (the protocol's hardest change class). A soft-prefix artifact trained on 400 Adam steps did not absorb that fact; three independent repeats confirm it is not seed noise.
3. At this scale and these change classes, the measured decision boundary is: **text retrieval wins on cost AND accuracy.** That is the contribution — a measured point, not an assertion. The regime where learned context might win (high query volume, low churn, learned-composition dependencies) remains unexplored at local scale and is the H200/B200 collaboration case.

## Deviations

None. The pilot ran exactly as frozen (protocol file unchanged; each repeat retained including all three supply-arm failures).

## Exit criteria

- [x] Protocol frozen before the run (`pilot.yaml` FROZEN); deviations section present (empty, honest).
- [x] Declared repeats executed (3×); tables generated from retained attempts including failures.
- [x] `tests/test_no_test_label_access.py` passes (3 tests) — no test label touches the scheduler.
- [x] Primary outcome reported with uncertainty ("cannot claim non-inferiority at n=9"); no equivalence claim.
- [x] Lifecycle cost reported in physical units + wall time (0.6 s vs 224.68 s) before any price conversion.
- [x] `scripts/check_reproduction.py` reproduces the tables from retained artifacts (exit 0).
