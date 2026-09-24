"""Model-free replay of the public, synthetic revision-study evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

from experiments.study_v3_revision import artifact_id, derive_status, revision
from living_context.replacement_study import score_exact_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_frozen_catalog(path: Path, source_bytes: bytes) -> ModuleType:
    """Execute the hash-verified study catalog without replacing the live package."""
    name = "living_context_study_v3_revision_frozen_catalog"
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    # The caller checks the SHA-256 of this exact byte buffer against the frozen protocol.
    exec(compile(source_bytes, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def replay(root: Path) -> dict[str, Any]:
    study = root / "research/study-v3-revision"
    protocol = json.loads((study / "protocol.json").read_text())
    catalog_relative = "src/living_context/catalog.py"
    expected_catalog = protocol["source_sha256"].get(catalog_relative)
    require(isinstance(expected_catalog, str), "frozen catalog pin missing")
    frozen_catalog_path = study / "frozen-source/src/living_context/catalog.py"
    frozen_catalog_bytes = frozen_catalog_path.read_bytes()
    require(
        hashlib.sha256(frozen_catalog_bytes).hexdigest() == expected_catalog,
        f"frozen source mismatch: {catalog_relative}",
    )
    for relative, expected in protocol["source_sha256"].items():
        if relative != catalog_relative:
            require(digest(root / relative) == expected, f"frozen source mismatch: {relative}")
    frozen_catalog = load_frozen_catalog(frozen_catalog_path, frozen_catalog_bytes)

    def historical_revision(row: dict[str, str]) -> Any:
        value = revision(row)
        return frozen_catalog.DocumentRevision(
            value.doc_id, value.revision, value.content, value.edit_kind, value.as_of
        )

    manifest = json.loads((study / "PUBLIC-EVIDENCE-MANIFEST.json").read_text())
    require(
        digest(study / "RESULTS.json") == manifest["result_sha256"], "machine result hash mismatch"
    )
    evidence = study / "results"
    files = manifest["files"]
    require(
        {p.name for p in evidence.iterdir() if p.is_file()} == set(files),
        "evidence inventory mismatch",
    )
    for name, expected in files.items():
        require(digest(evidence / name) == expected, f"evidence hash mismatch: {name}")
    training = json.loads((study / "training.json").read_text())
    evaluation = json.loads((study / "evaluation.json").read_text())
    initial = training["initial"]
    final = [
        next(r for r in initial if r["doc_id"] == training["retained"]),
        *training["final_new"],
    ]
    preflight = json.loads((evidence / "preflight.json").read_text())
    require(
        preflight
        == {
            "status": "PREFLIGHT_PASS",
            "source_files": len(protocol["source_sha256"]),
            "model_files": 6,
            "training_workers": 7,
        },
        "preflight receipt mismatch",
    )
    receipts = json.loads((evidence / "worker-receipts.json").read_text())
    require(len(receipts) == 8, "worker inventory incomplete")
    expected_workers = {(r["doc_id"], r["revision"]) for r in [*initial, *training["final_new"]]}
    actual_workers = [
        (r.get("doc_id"), r.get("revision")) for r in receipts if r.get("phase") == "train"
    ]
    require(
        len(actual_workers) == 7 and set(actual_workers) == expected_workers,
        "training worker identities mismatch",
    )
    require(
        [r["phase"] for r in receipts].count("revision-inference") == 1,
        "inference worker identity mismatch",
    )
    require(
        all(
            r.get("status") == "completed"
            and r.get("exit_code") == 0
            and r.get("memory_report_matches", 0) >= 1
            for r in receipts
        ),
        "worker completion or memory report missing",
    )
    for row in receipts:
        for key in ("elapsed_s", "peak_rss_bytes", "reported_mlx_peak_gb"):
            value = row.get(key)
            require(
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
                f"missing worker metric: {key}",
            )
    require(sum(r["elapsed_s"] for r in receipts) <= 900, "worker time bound exceeded")
    require(max(r["peak_rss_bytes"] for r in receipts) <= 8 * 1024**3, "RSS bound exceeded")
    require(max(r["reported_mlx_peak_gb"] for r in receipts) <= 2, "MLX bound exceeded")
    counts: dict[str, dict[str, int]] = {}
    for phase, rows in (("initial", initial), ("final", final)):
        raw = json.loads((evidence / f"{phase}-raw.json").read_text())["rows"]
        expected_cells = {(r["doc_id"], r["revision"], i) for r in rows for i in range(2)}
        actual_cells = [(r.get("doc_id"), r.get("revision"), r.get("question_index")) for r in raw]
        require(
            len(raw) == 8 and len(set(actual_cells)) == 8 and set(actual_cells) == expected_cells,
            f"{phase} cell set mismatch",
        )
        by_id = {r["doc_id"]: r for r in rows}
        counts[phase] = {arm: 0 for arm in ("adapter", "bare", "text")}
        for row in raw:
            source = by_id[row["doc_id"]]
            require(
                row.get("question") == evaluation[phase][row["doc_id"]][row["question_index"]],
                "question differs from frozen input",
            )
            require(
                row.get("artifact_id") == artifact_id(source, "tensor"), "selected artifact differs"
            )
            require(set(row.get("arms", {})) == set(counts[phase]), "arm inventory mismatch")
            for arm, result in row["arms"].items():
                exact = score_exact_json(result["raw"], source["answer"]).correct
                require(result["score"]["exact"] == exact, "stored score differs from raw response")
                counts[phase][arm] += exact
            text = row["arms"]["text"]
            expected_hit = row["question_index"] == 1 or (
                phase == "final" and row["doc_id"] == training["retained"]
            )
            require(
                text["cache_hit"] is expected_hit and text["fresh_cached_parity"] is True,
                "cache lifecycle or parity mismatch",
            )
            require(
                text["cache_bytes"] > 0
                and text["cache_build_s"] >= 0
                and text["elapsed_s"] >= 0
                and text["validation_elapsed_s"] >= text["elapsed_s"],
                "cache metrics invalid",
            )
    catalog = frozen_catalog.Catalog()
    for row in [*initial, *training["final_new"]]:
        catalog.add_revision(historical_revision(row))
    corpus = frozen_catalog.VersionedCorpus(catalog)
    first = corpus.stage({row["doc_id"]: historical_revision(row) for row in initial})
    initial_admission = json.loads((evidence / "admission-initial.json").read_text())
    require(
        initial_admission["candidate_sha256"]
        == hashlib.sha256(corpus._candidate_bytes(first)).hexdigest(),
        "initial candidate digest mismatch",
    )
    corpus.admit(first, checks_passed=True)
    second = corpus.stage_changes(
        {row["doc_id"]: historical_revision(row) for row in training["final_new"]},
        deleted=frozenset({training["deleted"]}),
    )
    final_admission = json.loads((evidence / "admission-final.json").read_text())
    require(
        final_admission["candidate_sha256"]
        == hashlib.sha256(corpus._candidate_bytes(second)).hexdigest(),
        "final candidate digest mismatch",
    )
    for admission, rows, generation in (
        (initial_admission, initial, 1),
        (final_admission, final, 2),
    ):
        require(
            admission["generation"] == generation
            and admission["active_doc_revisions"] == {r["doc_id"]: r["revision"] for r in rows},
            "admitted generation or source revision mismatch",
        )
        require(
            admission["source_sha256"]
            == {r["doc_id"]: hashlib.sha256(r["content"].encode()).hexdigest() for r in rows},
            "admitted source digest mismatch",
        )
        keys = {artifact_id(r, part) for r in rows for part in ("tensor", "config")}
        require(
            set(admission["artifact_sha256"]) == keys and set(admission["artifact_bytes"]) == keys,
            "tensor/config inventory mismatch",
        )
        require(
            all(
                len(value) == 64 and int(value, 16) >= 0
                for value in admission["artifact_sha256"].values()
            ),
            "artifact digest form invalid",
        )
        require(
            all(
                isinstance(value, int) and value > 0
                for value in admission["artifact_bytes"].values()
            ),
            "artifact byte count invalid",
        )
    lifecycle = json.loads((evidence / "lifecycle.json").read_text())
    require(
        lifecycle["initial_generation"] == 1
        and lifecycle["final_generation"] == 2
        and lifecycle["stale_artifact_rejected"] is True
        and lifecycle["failed_candidate_atomic"] is True
        and lifecycle["deleted_current"] is False
        and lifecycle["unchanged_artifact_reused"] is True,
        "lifecycle admission flags mismatch",
    )
    require(
        lifecycle["active_artifact_bytes"] == sum(final_admission["artifact_bytes"].values()),
        "active artifact byte sum mismatch",
    )
    require(
        lifecycle["retained_artifact_bytes"]
        == sum(
            {**initial_admission["artifact_bytes"], **final_admission["artifact_bytes"]}.values()
        ),
        "retained artifact byte sum mismatch",
    )
    changed = {r["doc_id"] for r in training["final_new"] if r["revision"] == "r2"}
    require(
        lifecycle["invalidated_cache_entries"] == {r: 1 for r in changed | {training["deleted"]}},
        "cache invalidation record mismatch",
    )
    stale = json.loads((evidence / "stale-controls-raw.json").read_text())["rows"]
    require(
        len(stale) == 4
        and {(r["doc_id"], r["question_index"]) for r in stale}
        == {(doc, i) for doc in changed for i in range(2)},
        "stale controls incomplete",
    )
    current_answers = {r["doc_id"]: r["answer"] for r in training["final_new"]}
    stale_exact = 0
    for row in stale:
        require(
            row["old_revision"] == "r1" and row["current_revision"] == "r2",
            "stale revision label mismatch",
        )
        exact = score_exact_json(row["result"]["raw"], current_answers[row["doc_id"]]).correct
        require(row["result"]["score"]["exact"] == exact, "stale score mismatch")
        stale_exact += exact
    scores = json.loads((evidence / "scores.json").read_text())
    status = json.loads((evidence / "status.json").read_text())
    gate = (
        "REVISION_GATE_PASS"
        if all(
            counts[p]["adapter"] >= 7 and counts[p]["adapter"] >= counts[p]["text"] for p in counts
        )
        else "REVISION_GATE_FAIL"
    )
    require(
        scores == {"counts": counts, "denominator_per_phase": 8, "status": gate},
        "score summary mismatch",
    )
    require(status == {"status": gate, "worker_receipts": 8}, "status summary mismatch")
    summary = json.loads((study / "RESULTS.json").read_text())
    require(
        summary["study"] == protocol["study"] and summary["status"] == gate,
        "machine result identity mismatch",
    )
    require(
        summary["primary_exact"] == counts and summary["lifecycle"] == lifecycle,
        "machine result counts or lifecycle mismatch",
    )
    require(summary["resource"]["worker_count"] == len(receipts), "machine worker count mismatch")
    require(
        summary["resource"]["cumulative_worker_elapsed_s"]
        == round(sum(r["elapsed_s"] for r in receipts), 3),
        "machine worker elapsed mismatch",
    )
    require(
        summary["resource"]["peak_sampled_rss_bytes"] == max(r["peak_rss_bytes"] for r in receipts),
        "machine RSS mismatch",
    )
    require(
        summary["resource"]["max_reported_mlx_peak_gb"]
        == max(r["reported_mlx_peak_gb"] for r in receipts),
        "machine MLX peak mismatch",
    )
    for phase, admission in (("initial", initial_admission), ("final", final_admission)):
        require(
            summary["admitted_artifact_metadata"][phase]
            == {"sha256": admission["artifact_sha256"], "bytes": admission["artifact_bytes"]},
            "machine artifact metadata mismatch",
        )
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        for name in files:
            shutil.copyfile(evidence / name, folder / name)
        require(derive_status(folder, receipts, training) == gate, "frozen runner gate differs")
    return {
        "status": "REPLAY_PASS",
        "gate": gate,
        "counts": counts,
        "stale_control_exact": stale_exact,
        "scope": "saved synthetic outputs and receipt metadata only; no weight rehash or model inference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(replay(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
