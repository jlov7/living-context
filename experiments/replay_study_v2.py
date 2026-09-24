"""Evidence-only replay for replacement study v2; never loads MLX or tensors.

This reproduces strict answers and gates from sealed text outputs and the
frozen synthetic labels. Tensor hashes are retained as custody metadata, but
tensor bytes are neither needed nor verified by this replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from living_context.replacement_study import (
    StudyValidationError,
    canonical_json,
    parse_json_bytes,
    parse_labels,
    parse_questions,
    parse_sources,
    read_sealed_json,
    score_exact_json,
    sha256_file,
    validate_data_separation,
    verify_input_snapshots,
)


def _sealed(folder: Path, name: str) -> dict[str, Any]:
    path = folder / name
    value = read_sealed_json(path, path.with_suffix(path.suffix + ".seal.json"))
    if not isinstance(value, dict):
        raise StudyValidationError(f"sealed {name} is not an object")
    return value


def _identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["question_id"], row["arm"], row.get("objective"), row.get("seed")


def _expand_calibration(raw: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in raw["rows"]:
        for arm, output_key, metrics_key in (
            ("bare", "bare_output", "bare_metrics"),
            ("text_fresh", "fresh_output", "fresh_metrics"),
            ("text_cached", "cached_output", "cached_metrics"),
        ):
            result.append(
                {
                    "question_id": row["question_id"],
                    "arm": arm,
                    "output": row[output_key],
                    "metrics": row[metrics_key],
                }
            )
    return result


def _check_training(
    folder: Path, protocol: dict[str, Any], sources: dict[str, dict[str, Any]], digest: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = folder / "development-training-results.json"
    training = _sealed(folder, path.name)
    if training.get("protocol_sha256") != digest or training.get("phase") != "development":
        raise StudyValidationError("training receipt protocol/phase mismatch")
    if (
        training.get("model_files_before") != protocol["model"]["files"]
        or training.get("model_files_after") != protocol["model"]["files"]
    ):
        raise StudyValidationError("training model-file receipt mismatch")
    source_id = protocol["phases"]["development"]["source_id"]
    expected = {
        (source_id, objective, seed)
        for objective in protocol["training"]["objectives"]
        for seed in protocol["training"]["seeds"]
    }
    observed = Counter(
        (row["source_id"], row["objective"], row["seed"]) for row in training["rows"]
    )
    if set(observed) != expected or any(count != 1 for count in observed.values()):
        raise StudyValidationError("training metadata inventory incomplete")
    for row in training["rows"]:
        source = sources[row["source_id"]]
        if (
            row["doc_id"] != source["doc_id"]
            or row["revision"] != source["revision"]
            or row["steps_completed"] != protocol["training"]["steps"]
            or row["prefix_shape"][0] != protocol["training"]["prefix_length"]
            or row["artifact_bytes"] < 1
            or len(row["artifact_sha256"]) != 64
        ):
            raise StudyValidationError("training metadata binding mismatch")
    return training, {"file": path.name, "sha256": sha256_file(path)}


def _check_phase(
    folder: Path,
    phase: str,
    protocol: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    labels: dict[str, str],
    sources: dict[str, dict[str, Any]],
    digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw_name = f"{phase}-raw.json"
    score_name = f"{phase}-scores.json"
    raw = _sealed(folder, raw_name)
    score = _sealed(folder, score_name)
    if raw.get("phase") != phase or raw.get("protocol_sha256") != digest:
        raise StudyValidationError("raw phase/protocol mismatch")
    if (
        raw.get("model_files_before") != protocol["model"]["files"]
        or raw.get("model_files_after") != protocol["model"]["files"]
    ):
        raise StudyValidationError("answer model-file receipt mismatch")
    if (
        score.get("phase") != phase
        or score.get("protocol_sha256") != digest
        or score.get("raw_sha256") != sha256_file(folder / raw_name)
        or score.get("raw_seal_sha256") != sha256_file(folder / f"{raw_name}.seal.json")
    ):
        raise StudyValidationError("score/raw binding mismatch")
    raw_rows = _expand_calibration(raw) if phase == "calibration" else raw["rows"]
    expected_questions = {key for key, row in questions.items() if row["split"] == phase}
    expected = {
        (qid, arm, None, None)
        for qid in expected_questions
        for arm in ("bare", "text_fresh", "text_cached")
    }
    if phase == "development":
        expected.update(
            (qid, "prefix", objective, seed)
            for qid in expected_questions
            for objective in protocol["training"]["objectives"]
            for seed in protocol["training"]["seeds"]
        )
    identities = Counter(_identity(row) for row in raw_rows)
    if set(identities) != expected or any(count != 1 for count in identities.values()):
        raise StudyValidationError("raw output inventory incomplete or duplicated")
    scored = score["rows"]
    if len(scored) != len(raw_rows):
        raise StudyValidationError("scored output inventory incomplete")
    for raw_row, scored_row in zip(raw_rows, scored, strict=True):
        qid = raw_row["question_id"]
        if _identity(raw_row) != _identity(scored_row) or raw_row["output"] != scored_row["output"]:
            raise StudyValidationError("scored row differs from sealed raw output")
        if any(scored_row.get(key) != value for key, value in raw_row.items()):
            raise StudyValidationError("scored row metadata differs from sealed raw output")
        exact = asdict(score_exact_json(raw_row["output"], labels[qid]))
        if scored_row["score"] != exact:
            raise StudyValidationError("stored exact score differs from replay")
        if phase == "development":
            retrieved = raw_row["retrieved_source_ids"]
            relevance = set(retrieved) >= set(questions[qid]["source_ids"])
            if scored_row["retrieval_contains_audit_sources"] != relevance:
                raise StudyValidationError("stored retrieval audit differs from replay")
            if raw_row["arm"] == "prefix":
                bound = [
                    {
                        "source_id": source_id,
                        "doc_id": sources[source_id]["doc_id"],
                        "revision": sources[source_id]["revision"],
                    }
                    for source_id in retrieved
                ]
                if raw_row["artifact_source_revisions"] != bound:
                    raise StudyValidationError("prefix/retrieval source binding mismatch")
    if phase == "calibration":
        for row in raw["rows"]:
            if (
                row["fresh_output"] != row["cached_output"]
                or row["fresh_metrics"]["generated_token_ids"]
                != row["cached_metrics"]["generated_token_ids"]
                or not row["next_token_argmax_equal"]
                or row["next_token_max_abs_logit_diff"] > protocol["retrieval"]["cache_logit_atol"]
            ):
                raise StudyValidationError("calibration cache parity failed")
        if not raw["invalidation_probe"]["old_revision_invalidated"]:
            raise StudyValidationError("old cache revision was not invalidated")
    else:
        for question_id in expected_questions:
            fresh = next(
                row
                for row in raw_rows
                if row["question_id"] == question_id and row["arm"] == "text_fresh"
            )
            cached = next(
                row
                for row in raw_rows
                if row["question_id"] == question_id and row["arm"] == "text_cached"
            )
            if (
                fresh["output"] != cached["output"]
                or fresh["metrics"]["generated_token_ids"]
                != cached["metrics"]["generated_token_ids"]
                or not cached["cache"]["next_token_argmax_equal"]
                or cached["cache"]["next_token_max_abs_logit_diff"]
                > protocol["retrieval"]["cache_logit_atol"]
            ):
                raise StudyValidationError("development cache parity failed")
    return scored, raw, score


def replay(root: Path, evidence: Path) -> dict[str, Any]:
    protocol_path = root / "research/study-v2/protocol.json"
    protocol = parse_json_bytes(protocol_path.read_bytes(), source=str(protocol_path))
    snapshots = verify_input_snapshots(root, protocol["inputs"])
    paths = protocol["data"]
    sources = parse_sources(parse_json_bytes(snapshots[paths["sources"]], source="sources"))
    questions = parse_questions(parse_json_bytes(snapshots[paths["questions"]], source="questions"))
    labels = parse_labels(parse_json_bytes(snapshots[paths["labels"]], source="labels"))
    validate_data_separation(sources, questions, labels)
    digest = hashlib.sha256(canonical_json(protocol).encode()).hexdigest()
    calibration, _cal_raw, cal_score = _check_phase(
        evidence, "calibration", protocol, questions, labels, sources, digest
    )
    development, dev_raw, dev_score = _check_phase(
        evidence, "development", protocol, questions, labels, sources, digest
    )
    training, training_file = _check_training(evidence, protocol, sources, digest)
    if dev_raw["training_receipt_sha256"] != training_file["sha256"]:
        raise StudyValidationError("raw/training receipt hash mismatch")
    cal_gate = _sealed(evidence, "calibration-gate.json")
    dev_gate = _sealed(evidence, "development-gate.json")
    for name, gate, score in (
        ("calibration", cal_gate, cal_score),
        ("development", dev_gate, dev_score),
    ):
        if (
            gate.get("phase") != name
            or gate.get("protocol_sha256") != digest
            or gate.get("score_sha256") != sha256_file(evidence / f"{name}-scores.json")
        ):
            raise StudyValidationError("gate/score binding mismatch")
    calibration_fresh_exact = sum(
        row["score"]["correct"] for row in calibration if row["arm"] == "text_fresh"
    )
    calibration_cached_exact = sum(
        row["score"]["correct"] for row in calibration if row["arm"] == "text_cached"
    )
    calibration_pass = len(calibration) == 6 and all(
        row["score"]["correct"]
        for row in calibration
        if row["arm"] in {"text_fresh", "text_cached"}
    )
    bare = [row for row in development if row["arm"] == "bare"]
    cached = [row for row in development if row["arm"] == "text_cached"]
    prefix = [row for row in development if row["arm"] == "prefix"]
    bare_zero = len(bare) == 2 and not any(row["score"]["correct"] for row in bare)
    cached_two = len(cached) == 2 and all(
        row["score"]["correct"] and row["retrieval_contains_audit_sources"] for row in cached
    )
    methods: list[dict[str, Any]] = []
    for objective in protocol["training"]["objectives"]:
        by_seed = {
            seed: [row for row in prefix if row["objective"] == objective and row["seed"] == seed]
            for seed in protocol["training"]["seeds"]
        }
        successful = sum(
            len(rows) == 2 and all(row["score"]["correct"] for row in rows)
            for rows in by_seed.values()
        )
        exact = sum(row["score"]["correct"] for rows in by_seed.values() for row in rows)
        serving = sum(row["metrics"]["elapsed_s"] for rows in by_seed.values() for row in rows)
        train = sum(row["training_s"] for row in training["rows"] if row["objective"] == objective)
        methods.append(
            {
                "objective": objective,
                "successful_seeds": successful,
                "exact_answers": exact,
                "serving_s": serving,
                "training_s": train,
                "selection_cost_s": train + serving,
                "passes_method_gate": successful >= 2,
            }
        )
    eligible = [row for row in methods if row["passes_method_gate"]]
    eligible.sort(
        key=lambda row: (-row["exact_answers"], row["selection_cost_s"], row["objective"])
    )
    dev_pass = bare_zero and cached_two and bool(eligible)
    selected = eligible[0]["objective"] if dev_pass else None
    if (
        cal_gate["passed"] != calibration_pass
        or dev_gate["passed"] != dev_pass
        or dev_gate["bare_zero_of_two"] != bare_zero
        or dev_gate["text_exact_two_of_two"] != cached_two
        or dev_gate["methods"] != methods
        or dev_gate["selected_objective"] != selected
    ):
        raise StudyValidationError("stored gate differs from independent replay")
    supervisor_names = (
        "_calibration-worker-supervisor.json",
        "_train-worker-supervisor.json",
        "_answer-worker-supervisor.json",
    )
    supervisor_elapsed = sum(
        parse_json_bytes((evidence / name).read_bytes(), source=name)["elapsed_s"]
        for name in supervisor_names
    )
    failed_name = "failed-calibration-a/_calibration-worker-supervisor.json"
    failed_elapsed = parse_json_bytes((evidence / failed_name).read_bytes(), source=failed_name)[
        "elapsed_s"
    ]
    extras = parse_json_bytes((evidence / "budget-extras.json").read_bytes(), source="extras")
    extra_elapsed = sum(row["elapsed_s"] for row in extras["model_calls"])
    return {
        "schema_version": 1,
        "source_commit": protocol["source_commit"],
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_canonical_json_sha256": digest,
        "calibration": {
            "passed": calibration_pass,
            "fresh_exact": calibration_fresh_exact,
            "cached_exact": calibration_cached_exact,
        },
        "development": {
            "passed": dev_pass,
            "bare_exact": sum(row["score"]["correct"] for row in bare),
            "fresh_text_exact": sum(
                row["score"]["correct"] for row in development if row["arm"] == "text_fresh"
            ),
            "cached_text_exact": sum(row["score"]["correct"] for row in cached),
            "prefix_exact": sum(row["score"]["correct"] for row in prefix),
            "prefix_denominator": len(prefix),
            "methods": methods,
            "selected_objective": selected,
            "larger_study_admitted": dev_pass,
        },
        "budget": {
            "successful_supervised_model_elapsed_s": supervisor_elapsed,
            "failed_calibration_supervised_elapsed_s": failed_elapsed,
            "diagnostic_model_elapsed_s": extra_elapsed,
            "total_charged_model_elapsed_s": supervisor_elapsed + failed_elapsed + extra_elapsed,
            "ceiling_s": protocol["limits"]["total_model_worker_elapsed_s"],
        },
        "storage": {
            "six_prefix_tensor_bytes_local_only": sum(
                row["artifact_bytes"] for row in training["rows"]
            ),
            "cached_kv_bytes": next(
                row["cache"]["cache_bytes"]
                for row in dev_raw["rows"]
                if row["arm"] == "text_cached"
            ),
            "lexical_index_bytes": dev_raw["index_storage_bytes"],
        },
        "evidence_files": {
            name: sha256_file(evidence / name)
            for name in (
                "calibration-raw.json",
                "calibration-scores.json",
                "calibration-gate.json",
                "development-training-results.json",
                "development-raw.json",
                "development-scores.json",
                "development-gate.json",
                "development-diagnostic.json",
                "development-evidence-inventory.json",
                "budget-extras.json",
            )
        },
        "replay_scope": "sealed text/scoring/gate and receipt metadata; tensor bytes deliberately absent",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(replay(args.root, args.evidence), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(data, end="")
    else:
        args.output.write_text(data)


if __name__ == "__main__":
    main()
