"""Model-free falsifiers for the fixed post-pilot revision gate."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.study_v3_revision import (
    adapter_config_file,
    adapter_file,
    admit_and_record,
    artifact_id,
    bindings,
    derive_status,
    records,
    revision,
)
from living_context.catalog import ArtifactAdmissionError, Catalog, VersionedCorpus


def fixture(tmp_path: Path):
    training, _ = records()
    receipts = [
        {
            "phase": "train",
            "doc_id": row["doc_id"],
            "revision": row["revision"],
            "status": "completed",
            "exit_code": 0,
            "elapsed_s": 1.0,
            "peak_rss_bytes": 1,
            "reported_mlx_peak_gb": 0.1,
            "memory_report_matches": 1,
        }
        for row in [*training["initial"], *training["final_new"]]
    ]
    receipts.append(
        {
            "phase": "revision-inference",
            "status": "completed",
            "exit_code": 0,
            "elapsed_s": 1.0,
            "peak_rss_bytes": 1,
            "reported_mlx_peak_gb": 0.1,
            "memory_report_matches": 1,
        }
    )
    for phase in ("initial", "final"):
        source_rows = (
            training["initial"]
            if phase == "initial"
            else [
                next(row for row in training["initial"] if row["doc_id"] == training["retained"]),
                *training["final_new"],
            ]
        )
        rows = [
            {
                "doc_id": row["doc_id"],
                "revision": row["revision"],
                "question_index": index,
                "arms": {
                    arm: {"raw": json.dumps({"answer": row["answer"]}), "score": {"exact": True}}
                    for arm in ("adapter", "bare", "text")
                },
            }
            for row in source_rows
            for index in range(2)
        ]
        (tmp_path / f"{phase}-raw.json").write_text(json.dumps({"rows": rows}))
    final_rows = [
        next(row for row in training["initial"] if row["doc_id"] == training["retained"]),
        *training["final_new"],
    ]
    for phase, source_rows, generation in (
        ("initial", training["initial"], 1),
        ("final", final_rows, 2),
    ):
        (tmp_path / f"admission-{phase}.json").write_text(
            json.dumps(
                {
                    "generation": generation,
                    "active_doc_revisions": {row["doc_id"]: row["revision"] for row in source_rows},
                    "deleted": [training["deleted"]] if phase == "final" else [],
                    "artifact_sha256": {
                        artifact_id(row, part): "a" * 64
                        for row in source_rows
                        for part in ("tensor", "config")
                    },
                }
            )
        )
    (tmp_path / "lifecycle.json").write_text(
        json.dumps(
            {
                "initial_generation": 1,
                "final_generation": 2,
                "stale_artifact_rejected": True,
                "failed_candidate_atomic": True,
                "deleted_current": False,
                "unchanged_artifact_reused": True,
                "invalidated_cache_entries": {
                    "routing-cedar": 1,
                    "permit-larch": 1,
                    "permit-mesa": 1,
                },
            }
        )
    )
    stale_rows = [
        {
            "doc_id": row["doc_id"],
            "old_revision": "r1",
            "current_revision": "r2",
            "question_index": index,
            "result": {"raw": json.dumps({"answer": row["answer"]}), "score": {"exact": True}},
        }
        for row in training["final_new"]
        if row["revision"] == "r2"
        for index in range(2)
    ]
    (tmp_path / "stale-controls-raw.json").write_text(json.dumps({"rows": stale_rows}))
    return training, receipts


def test_complete_fixed_cells_pass_and_duplicate_worker_fails(tmp_path: Path) -> None:
    training, receipts = fixture(tmp_path)
    assert derive_status(tmp_path, receipts, training) == "REVISION_GATE_PASS"
    duplicate = deepcopy(receipts)
    duplicate[1]["doc_id"] = duplicate[0]["doc_id"]
    duplicate[1]["revision"] = duplicate[0]["revision"]
    assert derive_status(tmp_path, duplicate, training) == "WORKER_INCOMPLETE"


def test_missing_cell_and_wrong_saved_score_fail(tmp_path: Path) -> None:
    training, receipts = fixture(tmp_path)
    path = tmp_path / "final-raw.json"
    data = json.loads(path.read_text())
    data["rows"][0]["question_index"] = 1
    path.write_text(json.dumps(data))
    assert derive_status(tmp_path, receipts, training) == "CELL_INCOMPLETE"
    data["rows"][0]["question_index"] = 0
    data["rows"][0]["arms"]["adapter"]["score"]["exact"] = False
    path.write_text(json.dumps(data))
    assert derive_status(tmp_path, receipts, training) == "SCORE_MISMATCH"


def test_missing_lifecycle_and_zero_memory_measurement_fail(tmp_path: Path) -> None:
    training, receipts = fixture(tmp_path)
    (tmp_path / "lifecycle.json").unlink()
    assert derive_status(tmp_path, receipts, training) == "LIFECYCLE_INCOMPLETE"
    receipts[0]["peak_rss_bytes"] = 0
    assert derive_status(tmp_path, receipts, training) == "WORKER_METRIC_UNAVAILABLE"


def test_exact_two_generation_artifact_admission_with_dummy_bytes(tmp_path: Path) -> None:
    training, _ = records()
    initial = training["initial"]
    new = training["final_new"]
    for row in [*initial, *new]:
        tensor = adapter_file(tmp_path, row)
        config = adapter_config_file(tmp_path, row)
        tensor.parent.mkdir(parents=True)
        tensor.write_bytes(f"tensor:{row['doc_id']}:{row['revision']}".encode())
        config.write_bytes(f"config:{row['doc_id']}:{row['revision']}".encode())
    catalog = Catalog()
    for row in [*initial, *new]:
        catalog.add_revision(revision(row))
    corpus = VersionedCorpus(catalog)
    first = corpus.stage({row["doc_id"]: revision(row) for row in initial})
    first_manifests, first_bytes = bindings(
        tmp_path, initial, {row["doc_id"]: 1 for row in initial}, "model"
    )
    admitted_first, first_record = admit_and_record(
        corpus, first, first_manifests, first_bytes, tmp_path, "initial"
    )
    assert admitted_first.generation == 1
    final_rows = [next(row for row in initial if row["doc_id"] == training["retained"]), *new]
    candidate = corpus.stage_changes(
        {row["doc_id"]: revision(row) for row in new},
        deleted=frozenset({training["deleted"]}),
    )
    final_manifests, final_bytes = bindings(
        tmp_path,
        final_rows,
        {row["doc_id"]: 1 if row["doc_id"] == training["retained"] else 2 for row in final_rows},
        "model",
    )
    stale_manifests, stale_bytes = dict(final_manifests), dict(final_bytes)
    changed_old, changed_new = initial[0], new[0]
    for part in ("tensor", "config"):
        stale_manifests.pop(artifact_id(changed_new, part))
        stale_bytes.pop(artifact_id(changed_new, part))
        stale_manifests[artifact_id(changed_old, part)] = first_manifests[
            artifact_id(changed_old, part)
        ]
        stale_bytes[artifact_id(changed_old, part)] = first_bytes[artifact_id(changed_old, part)]
    with pytest.raises(ArtifactAdmissionError):
        corpus.verify_artifacts(candidate, stale_manifests, stale_bytes)
    receipt = corpus.verify_artifacts(candidate, final_manifests, final_bytes)
    assert corpus.admit(candidate, checks_passed=False, receipt=receipt) is None
    assert corpus.serve() is admitted_first
    admitted_final, final_record = admit_and_record(
        corpus, candidate, final_manifests, final_bytes, tmp_path, "final"
    )
    assert admitted_final.generation == 2
    assert training["deleted"] not in corpus.serve().revisions
    assert not corpus.is_visible(changed_old["doc_id"], "r1")
    retained = final_rows[0]
    for part in ("tensor", "config"):
        key = artifact_id(retained, part)
        assert first_record["artifact_sha256"][key] == final_record["artifact_sha256"][key]
        assert final_record["artifact_bytes"][key] > 0
