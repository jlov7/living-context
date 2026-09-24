"""Validate retained pilot evidence and independently rescore saved outputs.

This is an offline evidence-integrity gate. It does not rerun a model and does
not turn the historical trained-question demonstration into held-out evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from living_context.evaluation import answer_matches

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "evidence" / "PUBLIC-SHA256SUMS.json"
ORIGINAL_MANIFEST = "artifacts/evidence/SHA256SUMS.json"
ORIGINAL_MANIFEST_SHA256 = "5dd1abe1f206a306a93c3f18a05f9777f726a4caa0c96048e72c1c5f0b56c6c7"
OMITTED_CLOSEOUT = "artifacts/evidence/panel-ready-1790032900.closemd"
OMITTED_CLOSEOUT_SHA256 = "9a1f03ff351569baa309050f3169a6b7077ebf0dd2553d341d2b051f5bfd8cf2"


class ReproductionError(RuntimeError):
    """Retained evidence is missing, malformed, incomplete, or changed."""


@dataclass(frozen=True)
class VerifiedEvidence:
    manifest: dict[str, Any]
    files: MappingProxyType[str, bytes]


def _verify_public_subset(root: Path, manifest: dict[str, Any]) -> None:
    """Allow only the named internal close-out to be absent from frozen v1 custody."""
    expected_keys = {
        "subset_schema",
        "derived_from",
        "omitted",
        "schema_version",
        "purpose",
        "pilot_evidence",
        "pilot_protocol",
        "files",
    }
    if (
        set(manifest) != expected_keys
        or manifest.get("subset_schema") != "public-evidence-subset-v1"
    ):
        raise ReproductionError("public evidence subset schema mismatch")
    source = manifest.get("derived_from")
    omission = manifest.get("omitted")
    if source != {"path": ORIGINAL_MANIFEST, "sha256": ORIGINAL_MANIFEST_SHA256}:
        raise ReproductionError("public subset does not bind the original manifest")
    if not isinstance(omission, dict) or set(omission) != {"path", "sha256", "reason"}:
        raise ReproductionError("public subset needs one explained omission")
    if omission["path"] != OMITTED_CLOSEOUT or omission["sha256"] != OMITTED_CLOSEOUT_SHA256:
        raise ReproductionError("public subset omits an unapproved file")
    if not isinstance(omission["reason"], str) or not omission["reason"].strip():
        raise ReproductionError("public subset omission needs a reason")
    original_path = root / ORIGINAL_MANIFEST
    if not original_path.is_file():
        raise ReproductionError("original evidence manifest is missing")
    original_bytes = original_path.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != ORIGINAL_MANIFEST_SHA256:
        raise ReproductionError("original evidence manifest changed")
    try:
        original = json.loads(original_bytes)
    except json.JSONDecodeError as exc:
        raise ReproductionError("original evidence manifest is malformed") from exc
    original_files = original.get("files")
    if (
        not isinstance(original_files, dict)
        or original_files.get(OMITTED_CLOSEOUT) != OMITTED_CLOSEOUT_SHA256
    ):
        raise ReproductionError("original evidence manifest lacks the approved close-out")
    if (root / OMITTED_CLOSEOUT).exists():
        raise ReproductionError("internal close-out remains in the public evidence tree")
    expected_files = {
        path: digest for path, digest in original_files.items() if path != OMITTED_CLOSEOUT
    }
    if manifest.get("files") != expected_files:
        raise ReproductionError("public subset differs from the approved evidence inventory")
    if manifest.get("schema_version") != original.get("schema_version") or any(
        manifest.get(field) != original.get(field) for field in ("pilot_evidence", "pilot_protocol")
    ):
        raise ReproductionError("public subset changed pilot selection or schema")


def verify_manifest(root: Path = ROOT, manifest_path: Path = MANIFEST) -> VerifiedEvidence:
    if not manifest_path.is_file():
        raise ReproductionError(f"missing evidence manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except json.JSONDecodeError as exc:
        raise ReproductionError("evidence manifest is malformed") from exc
    if not isinstance(manifest, dict):
        raise ReproductionError("evidence manifest must be an object")
    if manifest_path.name == "PUBLIC-SHA256SUMS.json" or "subset_schema" in manifest:
        _verify_public_subset(root, manifest)
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ReproductionError("evidence manifest has no files")
    root_resolved = root.resolve()
    snapshots: dict[str, bytes] = {}
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ReproductionError("manifest paths and hashes must be strings")
        path = (root / relative).resolve()
        if not path.is_relative_to(root_resolved):
            raise ReproductionError(f"manifest path escapes repository root: {relative}")
        if not path.is_file():
            raise ReproductionError(f"missing retained file: {relative}")
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise ReproductionError(
                f"hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        snapshots[relative] = payload
    for selection in ("pilot_evidence", "pilot_protocol"):
        relative = manifest.get(selection)
        if not isinstance(relative, str) or relative not in snapshots:
            raise ReproductionError(f"{selection} must select a hashed manifest file")
    return VerifiedEvidence(manifest=manifest, files=MappingProxyType(snapshots))


def saved_canary_diagnostic(verified: VerifiedEvidence) -> dict[str, int]:
    """Rescore complete saved canary strings without treating them as a study."""
    points: list[dict[str, Any]] = []
    for relative, payload in verified.files.items():
        if not relative.startswith("artifacts/evidence/canary-"):
            continue
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ReproductionError(f"canary evidence is not an object: {relative}")
        if "points" in data:
            if not isinstance(data["points"], list):
                raise ReproductionError(f"canary points are malformed in {relative}")
            points.extend(data["points"])
        else:
            points.append(data)
    if not points:
        raise ReproductionError("no retained canary points")
    answer = "QX-7743-KB"
    legacy_passes = 0
    boundary_passes = 0
    baseline_boundary_passes = 0
    for index, point in enumerate(points):
        for field in ("baseline_answer", "with_artifact_answer", "canary_pass"):
            if field not in point:
                raise ReproductionError(f"canary point {index} missing {field}")
        if not isinstance(point["baseline_answer"], str) or not isinstance(
            point["with_artifact_answer"], str
        ):
            raise ReproductionError(f"canary point {index} answers must be strings")
        legacy = answer in point["with_artifact_answer"] and answer not in point["baseline_answer"]
        if point["canary_pass"] is not legacy:
            raise ReproductionError(f"canary point {index} disagrees with old scorer")
        legacy_passes += legacy
        boundary_passes += answer_matches(point["with_artifact_answer"], answer)
        baseline_boundary_passes += answer_matches(point["baseline_answer"], answer)
    return {
        "retained_attempts": len(points),
        "historical_substring_passes": legacy_passes,
        "saved_output_boundary_passes": boundary_passes,
        "baseline_boundary_passes": baseline_boundary_passes,
    }


def _legacy_substring_match(text: str, answer: str) -> bool:
    return answer.lower() in text.lower()


def validate_pilot(data: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    questions = protocol.get("questions")
    design = protocol.get("design")
    if not isinstance(questions, list) or not questions:
        raise ReproductionError("pilot protocol has no questions")
    if (
        not isinstance(design, dict)
        or isinstance(design.get("repeats"), bool)
        or not isinstance(design.get("repeats"), int)
        or design["repeats"] <= 0
    ):
        raise ReproductionError("pilot protocol needs a positive integer repeat count")
    repeats = design["repeats"]

    expected: set[tuple[str, str, str, int]] = set()
    declared_questions: set[tuple[str, str, str]] = set()
    for question in questions:
        if not isinstance(question, dict):
            raise ReproductionError("question entry is not an object")
        try:
            base = (question["question"], question["answer"], question["family"])
        except KeyError as exc:
            raise ReproductionError(f"question missing field: {exc.args[0]}") from exc
        if not all(isinstance(value, str) and value for value in base):
            raise ReproductionError("question, answer, and family must be non-empty strings")
        if base in declared_questions:
            raise ReproductionError(f"duplicate question declaration: {base}")
        declared_questions.add(base)
        expected.update((*base, repeat) for repeat in range(1, repeats + 1))

    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ReproductionError("pilot evidence rows must be a list")
    seen: set[tuple[str, str, str, int]] = set()
    strict_arm_1 = 0
    strict_arm_4 = 0
    legacy_arm_1 = 0
    legacy_arm_4 = 0
    by_family: dict[str, list[dict[str, Any]]] = {}
    required = {
        "question",
        "answer",
        "family",
        "repeat",
        "arm_1_answer",
        "arm_1_correct",
        "arm_1_serving_s",
        "arm_4_answer",
        "arm_4_correct",
        "arm_4_refresh_s",
        "arm_4_peak_mem_bytes",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ReproductionError(f"row {index} is not an object")
        missing = required - set(row)
        if missing:
            raise ReproductionError(f"row {index} missing fields: {sorted(missing)}")
        for field in ("question", "answer", "family", "arm_1_answer", "arm_4_answer"):
            if not isinstance(row[field], str):
                raise ReproductionError(f"row {index} field {field} must be a string")
        if isinstance(row["repeat"], bool) or not isinstance(row["repeat"], int):
            raise ReproductionError(f"row {index} repeat must be an integer")
        key = (row["question"], row["answer"], row["family"], row["repeat"])
        if key not in expected:
            raise ReproductionError(f"row {index} is not declared by the frozen file")
        if key in seen:
            raise ReproductionError(f"duplicate declared row: {key}")
        seen.add(key)
        for field in ("arm_1_serving_s", "arm_4_refresh_s", "arm_4_peak_mem_bytes"):
            if isinstance(row[field], bool) or not isinstance(row[field], (int, float)):
                raise ReproductionError(f"row {index} field {field} must be numeric")
            if row[field] < 0:
                raise ReproductionError(f"row {index} field {field} must be non-negative")
            if not math.isfinite(row[field]):
                raise ReproductionError(f"row {index} field {field} must be finite")

        arm_1_legacy = _legacy_substring_match(row["arm_1_answer"], row["answer"])
        arm_4_legacy = _legacy_substring_match(row["arm_4_answer"], row["answer"])
        if row["arm_1_correct"] is not arm_1_legacy:
            raise ReproductionError(f"row {index} arm_1_correct disagrees with old scorer")
        if row["arm_4_correct"] is not arm_4_legacy:
            raise ReproductionError(f"row {index} arm_4_correct disagrees with old scorer")
        legacy_arm_1 += arm_1_legacy
        legacy_arm_4 += arm_4_legacy
        strict_arm_1 += answer_matches(row["arm_1_answer"], row["answer"])
        strict_arm_4 += answer_matches(row["arm_4_answer"], row["answer"])
        by_family.setdefault(row["family"], []).append(row)

    missing_rows = expected - seen
    if missing_rows:
        raise ReproductionError(f"missing {len(missing_rows)} declared rows")

    recomputed = {
        "attempts": len(rows),
        "arm_1_correct": legacy_arm_1,
        "arm_4_correct": legacy_arm_4,
        "per_family": {
            family: {
                "arm_1": sum(bool(row["arm_1_correct"]) for row in family_rows),
                "arm_4": sum(bool(row["arm_4_correct"]) for row in family_rows),
                "n": len(family_rows),
            }
            for family, family_rows in by_family.items()
        },
        "lifecycle_cost_physical": {
            "arm_1_serving_total_s": round(sum(row["arm_1_serving_s"] for row in rows), 3),
            "arm_4_refresh_total_s": round(sum(row["arm_4_refresh_s"] for row in rows), 3),
            "peak_mem_bytes": max(row["arm_4_peak_mem_bytes"] for row in rows),
        },
    }
    for key, value in recomputed.items():
        if data.get(key) != value:
            raise ReproductionError(f"stored pilot summary mismatch: {key}")

    return {
        "historical_summary_reproduced": True,
        "declared_rows_complete": True,
        "historical_substring_score": {
            "arm_1": legacy_arm_1,
            "arm_4": legacy_arm_4,
            "n": len(rows),
        },
        "saved_output_boundary_diagnostic": {
            "arm_1": strict_arm_1,
            "arm_4": strict_arm_4,
            "n": len(rows),
        },
        "claim_ceiling": (
            "descriptive trained-question outputs only; no held-out generalization, "
            "matched lifecycle-cost, cached-retrieval, or non-inferiority claim"
        ),
    }


def main() -> int:
    try:
        verified = verify_manifest()
        pilot_relative = verified.manifest["pilot_evidence"]
        protocol_relative = verified.manifest["pilot_protocol"]
        data = json.loads(verified.files[pilot_relative])
        protocol = yaml.safe_load(verified.files[protocol_relative])
        if not isinstance(data, dict) or not isinstance(protocol, dict):
            raise ReproductionError("pilot evidence and protocol must be objects")
        result = validate_pilot(data, protocol)
        result["canary_saved_output_diagnostic"] = saved_canary_diagnostic(verified)
    except (KeyError, json.JSONDecodeError, yaml.YAMLError, ReproductionError) as exc:
        print(f"REPRODUCTION FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
