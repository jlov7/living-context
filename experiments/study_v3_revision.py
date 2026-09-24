"""Frozen post-pilot revision follow-up. Model work requires explicit --run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from experiments import study_v3_qlora as pilot
from living_context.artifacts import ArtifactManifest, SourceRef
from living_context.catalog import (
    ArtifactAdmissionError,
    Catalog,
    DocumentRevision,
    VersionedCorpus,
)
from living_context.replacement_study import CacheKey, RevisionCacheRegistry

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/study-v3-revision"


def dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads((STUDY / "training.json").read_text()),
        json.loads((STUDY / "evaluation.json").read_text()),
    )


def preflight(model_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if output.exists():
        raise ValueError("new output directory must not exist")
    for relative, expected in protocol["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"frozen source differs: {relative}")
    old = pilot.preflight(model_path, output)
    training, evaluation = records()
    initial = training["initial"]
    final = training["final_new"]
    if len(initial) != 4 or len(final) != 3:
        raise ValueError("frozen training inventory differs")
    final_ids = {row["doc_id"] for row in final} | {training["retained"]}
    if final_ids != set(evaluation["final"]) or {r["doc_id"] for r in initial} != set(
        evaluation["initial"]
    ):
        raise ValueError("question inventory differs")
    for phase in ("initial", "final"):
        if any(len(questions) != 2 for questions in evaluation[phase].values()):
            raise ValueError("phase requires two evaluation wordings per live document")
    serialized_training = json.dumps(training)
    if any(
        question in serialized_training
        for phase in evaluation.values()
        for qs in phase.values()
        for question in qs
    ):
        raise ValueError("evaluation question appears in training serialization")
    return {
        "status": "PREFLIGHT_PASS",
        "source_files": len(protocol["source_sha256"]),
        "model_files": old["model_files"],
        "training_workers": 7,
    }


def revision(row: dict[str, str]) -> DocumentRevision:
    return DocumentRevision(
        row["doc_id"],
        row["revision"],
        row["content"],
        "initial" if row["revision"] == "r1" else "correction",
        "2026-09-21" if row["revision"] == "r1" else "2026-09-22",
    )


def adapter_file(output: Path, row: dict[str, str]) -> Path:
    return output / "adapters" / f"{row['doc_id']}@{row['revision']}" / "adapters.safetensors"


def adapter_config_file(output: Path, row: dict[str, str]) -> Path:
    return adapter_file(output, row).with_name("adapter_config.json")


def artifact_id(row: dict[str, str], part: str) -> str:
    return f"{row['doc_id']}@{row['revision']}#{part}"


def manifest(
    output: Path, row: dict[str, str], generation: int, model_revision: str, part: str
) -> ArtifactManifest:
    path = adapter_file(output, row) if part == "tensor" else adapter_config_file(output, row)
    return ArtifactManifest(
        artifact_id=artifact_id(row, part),
        sources=(SourceRef(row["doc_id"], row["revision"], "target"),),
        base_model_revision=model_revision,
        trained_at_generation=generation,
        checksum=sha256(path),
    )


def bindings(
    output: Path, rows: list[dict[str, str]], generations: dict[str, int], model_revision: str
) -> tuple[dict[str, ArtifactManifest], dict[str, bytes]]:
    manifests = {
        artifact_id(row, part): manifest(
            output, row, generations[row["doc_id"]], model_revision, part
        )
        for row in rows
        for part in ("tensor", "config")
    }
    payloads = {
        artifact_id(row, part): (
            adapter_file(output, row) if part == "tensor" else adapter_config_file(output, row)
        ).read_bytes()
        for row in rows
        for part in ("tensor", "config")
    }
    return manifests, payloads


def assert_bound_adapter(output: Path, row: dict[str, str], bound_hashes: dict[str, str]) -> Path:
    tensor = adapter_file(output, row)
    config = adapter_config_file(output, row)
    if (
        sha256(tensor) != bound_hashes[artifact_id(row, "tensor")]
        or sha256(config) != bound_hashes[artifact_id(row, "config")]
    ):
        raise ValueError("loaded adapter tensor or config differs from admitted receipt")
    return tensor.parent


def admit_and_record(
    corpus: VersionedCorpus,
    candidate: Any,
    manifests: dict[str, ArtifactManifest],
    payloads: dict[str, bytes],
    output: Path,
    label: str,
) -> tuple[Any, dict[str, Any]]:
    receipt = corpus.verify_artifacts(candidate, manifests, payloads)
    admitted = corpus.admit(candidate, checks_passed=True, receipt=receipt)
    if admitted is None:
        raise RuntimeError("checked candidate was not admitted")
    row = {
        "phase": label,
        "generation": admitted.generation,
        "candidate_sha256": receipt.candidate_sha256,
        "artifact_sha256": dict(receipt.artifact_sha256),
        "artifact_bytes": {key: len(value) for key, value in receipt.consumed_artifacts},
        "source_sha256": dict(receipt.source_sha256),
        "active_doc_revisions": {k: v.revision for k, v in sorted(admitted.revisions.items())},
        "deleted": sorted(admitted.deleted),
    }
    dump(output / f"admission-{label}.json", row)
    return admitted, row


def evaluate_phase(
    model_path: Path,
    output: Path,
    label: str,
    rows: list[dict[str, str]],
    questions: dict[str, list[str]],
    corpus: VersionedCorpus,
    registry: RevisionCacheRegistry[Any],
    base: Any,
    tokenizer: Any,
    model_hash: str,
    tokenizer_hash: str,
    bound_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    from mlx_lm import load

    from experiments.replacement_study_v2 import SYSTEM_PROMPT, _cache_check_and_generate

    answer_rows: list[dict[str, Any]] = []
    for record in rows:
        doc_id = record["doc_id"]
        rev = record["revision"]
        if not corpus.is_visible(doc_id, rev):
            raise ValueError("reader selected a retired source revision")
        selected_artifact = artifact_id(record, "tensor")
        adapter_dir = assert_bound_adapter(output, record, bound_hashes)
        started = time.monotonic()
        adapted, adapted_tokenizer = cast(
            tuple[Any, Any], load(str(model_path), adapter_path=str(adapter_dir))
        )
        adapted.eval()
        load_s = round(time.monotonic() - started, 3)
        key = CacheKey(
            model_hash,
            tokenizer_hash,
            hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            doc_id,
            rev,
            hashlib.sha256(record["content"].encode()).hexdigest(),
        )
        for index, question in enumerate(questions[doc_id]):
            cached, validation_elapsed = pilot.timed_cached_text(
                lambda record=record, question=question, key=key: _cache_check_and_generate(
                    base, tokenizer, [record["content"]], question, registry, key, 32, 0.1
                )
            )
            raw = cached["cached_output"]
            answer_rows.append(
                {
                    "phase": label,
                    "doc_id": doc_id,
                    "revision": rev,
                    "question_index": index,
                    "question": question,
                    "artifact_id": selected_artifact,
                    "adapter_load_s": load_s if index == 0 else 0.0,
                    "arms": {
                        "adapter": pilot.infer_one(adapted, adapted_tokenizer, record, question),
                        "bare": pilot.infer_one(base, tokenizer, record, question),
                        "text": {
                            "raw": raw,
                            "score": pilot.score(raw, record["answer"]),
                            "cache_hit": cached["cache_hit"],
                            "cache_bytes": cached["cache_bytes"],
                            "cache_build_s": cached["cache_build_s"],
                            "fresh_cached_parity": True,
                            "validation_elapsed_s": validation_elapsed,
                            "elapsed_s": cached["cached_metrics"]["elapsed_s"],
                            "metrics": cached["cached_metrics"],
                        },
                    },
                }
            )
        del adapted
    return answer_rows


def inference_worker(model_path: Path, output: Path) -> None:
    import mlx.core as mx
    from mlx_lm import load

    training, evaluation = records()
    protocol = json.loads((STUDY / "protocol.json").read_text())
    v2 = json.loads((ROOT / "research/study-v2/protocol.json").read_text())
    model_hash = v2["model"]["files"]["model.safetensors"]["sha256"]
    tokenizer_hash = v2["model"]["files"]["tokenizer.json"]["sha256"]
    started = time.monotonic()
    base, tokenizer = cast(tuple[Any, Any], load(str(model_path)))
    base.eval()
    base_load_s = round(time.monotonic() - started, 3)
    catalog = Catalog()
    initial_rows = training["initial"]
    new_rows = training["final_new"]
    for row in [*initial_rows, *new_rows]:
        catalog.add_revision(revision(row))
    corpus = VersionedCorpus(catalog)
    registry: RevisionCacheRegistry[Any] = RevisionCacheRegistry()
    candidate = corpus.stage({row["doc_id"]: revision(row) for row in initial_rows})
    initial_manifests, initial_bytes = bindings(
        output,
        initial_rows,
        {r["doc_id"]: 1 for r in initial_rows},
        protocol["model_snapshot_revision"],
    )
    first, first_receipt = admit_and_record(
        corpus, candidate, initial_manifests, initial_bytes, output, "initial"
    )
    initial_answers = evaluate_phase(
        model_path,
        output,
        "initial",
        initial_rows,
        evaluation["initial"],
        corpus,
        registry,
        base,
        tokenizer,
        model_hash,
        tokenizer_hash,
        first_receipt["artifact_sha256"],
    )
    dump(output / "initial-raw.json", {"rows": initial_answers})
    old_generation = corpus.current_generation
    final_rows = [
        next(row for row in initial_rows if row["doc_id"] == training["retained"]),
        *new_rows,
    ]
    final_candidate = corpus.stage_changes(
        {row["doc_id"]: revision(row) for row in new_rows}, deleted=frozenset({training["deleted"]})
    )
    final_manifests, final_bytes = bindings(
        output,
        final_rows,
        {r["doc_id"]: (1 if r["doc_id"] == training["retained"] else 2) for r in final_rows},
        protocol["model_snapshot_revision"],
    )
    stale = dict(final_manifests)
    changed = new_rows[0]["doc_id"]
    stale_bytes = dict(final_bytes)
    old_changed = next(row for row in initial_rows if row["doc_id"] == changed)
    for part in ("tensor", "config"):
        stale.pop(artifact_id(new_rows[0], part))
        stale_bytes.pop(artifact_id(new_rows[0], part))
        old_id = artifact_id(old_changed, part)
        stale[old_id] = initial_manifests[old_id]
        stale_bytes[old_id] = initial_bytes[old_id]
    try:
        corpus.verify_artifacts(final_candidate, stale, stale_bytes)
    except ArtifactAdmissionError:
        stale_rejected = corpus.current_generation == old_generation
    else:
        raise ValueError("stale artifact was not rejected")
    rejected_receipt = corpus.verify_artifacts(final_candidate, final_manifests, final_bytes)
    rejected = corpus.admit(final_candidate, checks_passed=False, receipt=rejected_receipt)
    failed_candidate_atomic = (
        rejected is None and corpus.current_generation == old_generation and corpus.serve() is first
    )
    if not (stale_rejected and failed_candidate_atomic):
        raise ValueError("failed candidate changed visible generation")
    second, final_receipt = admit_and_record(
        corpus, final_candidate, final_manifests, final_bytes, output, "final"
    )
    if (
        second.generation != 2
        or corpus.is_visible(training["deleted"], "r1")
        or corpus.is_visible(changed, "r1")
    ):
        raise ValueError("retired source remained current")
    invalidated = {
        doc_id: registry.invalidate_document(doc_id)
        for doc_id in (changed, new_rows[1]["doc_id"], training["deleted"])
    }
    final_answers = evaluate_phase(
        model_path,
        output,
        "final",
        final_rows,
        evaluation["final"],
        corpus,
        registry,
        base,
        tokenizer,
        model_hash,
        tokenizer_hash,
        final_receipt["artifact_sha256"],
    )
    dump(output / "final-raw.json", {"rows": final_answers})
    stale_controls = []
    for old in initial_rows:
        if old["doc_id"] not in {r["doc_id"] for r in new_rows if r["revision"] == "r2"}:
            continue
        adapter_dir = assert_bound_adapter(output, old, first_receipt["artifact_sha256"])
        adapted, token = cast(tuple[Any, Any], load(str(model_path), adapter_path=str(adapter_dir)))
        adapted.eval()
        current = next(r for r in new_rows if r["doc_id"] == old["doc_id"])
        for index, question in enumerate(evaluation["final"][old["doc_id"]]):
            stale_controls.append(
                {
                    "doc_id": old["doc_id"],
                    "old_revision": "r1",
                    "current_revision": "r2",
                    "question_index": index,
                    "result": pilot.infer_one(adapted, token, current, question),
                }
            )
        del adapted
    dump(output / "stale-controls-raw.json", {"rows": stale_controls})
    dump(
        output / "lifecycle.json",
        {
            "base_load_s": base_load_s,
            "initial_generation": first.generation,
            "final_generation": second.generation,
            "stale_artifact_rejected": stale_rejected,
            "failed_candidate_atomic": failed_candidate_atomic,
            "deleted_current": corpus.is_visible(training["deleted"], "r1"),
            "unchanged_artifact_reused": all(
                first_receipt["artifact_sha256"][artifact_id(final_rows[0], part)]
                == final_receipt["artifact_sha256"][artifact_id(final_rows[0], part)]
                for part in ("tensor", "config")
            ),
            "invalidated_cache_entries": invalidated,
            "active_artifact_bytes": sum(len(value) for value in final_bytes.values()),
            "retained_artifact_bytes": sum(
                len(value) for value in {**initial_bytes, **final_bytes}.values()
            ),
        },
    )
    print(f"MLX_PEAK_GB={mx.get_peak_memory() / 1e9:.3f}", flush=True)


def derive_status(output: Path, receipts: list[dict[str, Any]], training: dict[str, Any]) -> str:
    failure = next((r["status"] for r in receipts if r["status"] != "completed"), None)
    if failure is not None:
        return f"WORKER_{failure}"
    expected = {
        (r["doc_id"], r["revision"]) for r in [*training["initial"], *training["final_new"]]
    }
    actual = [(r.get("doc_id"), r.get("revision")) for r in receipts if r["phase"] == "train"]
    if (
        set(actual) != expected
        or len(actual) != 7
        or len(receipts) != 8
        or receipts[-1]["phase"] != "revision-inference"
    ):
        return "WORKER_INCOMPLETE"
    for row in receipts:
        if row.get("exit_code") != 0 or row.get("memory_report_matches", 0) < 1:
            return "WORKER_INCOMPLETE"
        if any(
            not isinstance(row.get(key), (int, float))
            or not math.isfinite(row[key])
            or row[key] <= 0
            for key in ("elapsed_s", "peak_rss_bytes", "reported_mlx_peak_gb")
        ):
            return "WORKER_METRIC_UNAVAILABLE"
    if (
        sum(row["elapsed_s"] for row in receipts) > pilot.LIMIT_SECONDS
        or max(row["peak_rss_bytes"] for row in receipts) > pilot.LIMIT_RSS
        or max(row["reported_mlx_peak_gb"] for row in receipts) > pilot.LIMIT_MLX_GB
    ):
        return "WORKER_RESOURCE_LIMIT"
    counts: dict[str, dict[str, int]] = {}
    for phase in ("initial", "final"):
        raw_path = output / f"{phase}-raw.json"
        if not raw_path.exists():
            return "CELL_INCOMPLETE"
        rows = json.loads(raw_path.read_text())["rows"]
        expected_rows = (
            training["initial"]
            if phase == "initial"
            else [
                next(r for r in training["initial"] if r["doc_id"] == training["retained"]),
                *training["final_new"],
            ]
        )
        expected_cells = {
            (r["doc_id"], r["revision"], index) for r in expected_rows for index in range(2)
        }
        actual_cells = [(r.get("doc_id"), r.get("revision"), r.get("question_index")) for r in rows]
        if len(rows) != 8 or len(set(actual_cells)) != 8 or set(actual_cells) != expected_cells:
            return "CELL_INCOMPLETE"
        answers = {r["doc_id"]: r["answer"] for r in expected_rows}
        counts[phase] = {arm: 0 for arm in ("adapter", "bare", "text")}
        for row in rows:
            if set(row.get("arms", {})) != set(counts[phase]):
                return "ARM_INCOMPLETE"
            for arm, result in row["arms"].items():
                exact = pilot.score(result["raw"], answers[row["doc_id"]])["exact"]
                if result["score"]["exact"] != exact:
                    return "SCORE_MISMATCH"
                counts[phase][arm] += bool(exact)
    required = [
        output / "lifecycle.json",
        output / "admission-initial.json",
        output / "admission-final.json",
        output / "stale-controls-raw.json",
    ]
    if any(not path.exists() for path in required):
        return "LIFECYCLE_INCOMPLETE"
    lifecycle = json.loads((output / "lifecycle.json").read_text())
    initial_admission = json.loads((output / "admission-initial.json").read_text())
    final_admission = json.loads((output / "admission-final.json").read_text())
    initial_rows = training["initial"]
    final_rows = [
        next(row for row in initial_rows if row["doc_id"] == training["retained"]),
        *training["final_new"],
    ]
    initial_map = {row["doc_id"]: row["revision"] for row in initial_rows}
    final_map = {row["doc_id"]: row["revision"] for row in final_rows}
    if (
        lifecycle.get("initial_generation") != 1
        or lifecycle.get("final_generation") != 2
        or lifecycle.get("stale_artifact_rejected") is not True
        or lifecycle.get("failed_candidate_atomic") is not True
        or lifecycle.get("deleted_current") is not False
        or lifecycle.get("unchanged_artifact_reused") is not True
        or initial_admission.get("generation") != 1
        or final_admission.get("generation") != 2
        or initial_admission.get("active_doc_revisions") != initial_map
        or final_admission.get("active_doc_revisions") != final_map
        or final_admission.get("deleted") != [training["deleted"]]
        or set(initial_admission.get("artifact_sha256", {}))
        != {artifact_id(row, part) for row in initial_rows for part in ("tensor", "config")}
        or set(final_admission.get("artifact_sha256", {}))
        != {artifact_id(row, part) for row in final_rows for part in ("tensor", "config")}
    ):
        return "LIFECYCLE_MISMATCH"
    retained = next(row for row in initial_rows if row["doc_id"] == training["retained"])
    if any(
        initial_admission["artifact_sha256"][artifact_id(retained, part)]
        != final_admission["artifact_sha256"][artifact_id(retained, part)]
        for part in ("tensor", "config")
    ):
        return "LIFECYCLE_MISMATCH"
    invalidated = lifecycle.get("invalidated_cache_entries", {})
    changed_ids = {row["doc_id"] for row in training["final_new"] if row["revision"] == "r2"}
    if set(invalidated) != changed_ids | {training["deleted"]} or any(
        invalidated[doc_id] != 1 for doc_id in invalidated
    ):
        return "CACHE_INVALIDATION_MISMATCH"
    stale_rows = json.loads((output / "stale-controls-raw.json").read_text())["rows"]
    expected_stale = {(doc_id, index) for doc_id in changed_ids for index in range(2)}
    actual_stale = [(row.get("doc_id"), row.get("question_index")) for row in stale_rows]
    if len(stale_rows) != 4 or len(set(actual_stale)) != 4 or set(actual_stale) != expected_stale:
        return "STALE_CONTROL_INCOMPLETE"
    final_answers = {row["doc_id"]: row["answer"] for row in training["final_new"]}
    for row in stale_rows:
        if row.get("old_revision") != "r1" or row.get("current_revision") != "r2":
            return "STALE_CONTROL_MISMATCH"
        exact = pilot.score(row["result"]["raw"], final_answers[row["doc_id"]])["exact"]
        if row["result"]["score"]["exact"] != exact:
            return "STALE_CONTROL_MISMATCH"
    passed = all(
        counts[p]["adapter"] >= 7 and counts[p]["adapter"] >= counts[p]["text"] for p in counts
    )
    status = "REVISION_GATE_PASS" if passed else "REVISION_GATE_FAIL"
    dump(output / "scores.json", {"counts": counts, "denominator_per_phase": 8, "status": status})
    return status


def supervise_checked(
    command: list[str], log_path: Path, deadline: float, env: dict[str, str]
) -> dict[str, Any]:
    receipt = pilot.supervise(command, log_path, deadline, env)
    if receipt["status"] == "completed" and (
        receipt["peak_rss_bytes"] <= 0 or receipt["reported_mlx_peak_gb"] <= 0
    ):
        receipt["status"] = "RESOURCE_METRIC_UNAVAILABLE"
    return receipt


def run(model_path: Path, output: Path) -> None:
    pre = preflight(model_path, output)
    output.mkdir(parents=True)
    dump(output / "preflight.json", pre)
    training, _ = records()
    deadline = time.monotonic() + pilot.LIMIT_SECONDS
    env = dict(os.environ)
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT),
        }
    )
    receipts: list[dict[str, Any]] = []
    for row in [*training["initial"], *training["final_new"]]:
        doc_id, rev = row["doc_id"], row["revision"]
        name = f"{doc_id}@{rev}"
        data_dir = output / "worker-data" / name
        pilot.training_data(row, data_dir)
        command = [
            sys.executable,
            "-m",
            "mlx_lm.lora",
            "--train",
            "--model",
            str(model_path),
            "--data",
            str(data_dir),
            "--adapter-path",
            str(output / "adapters" / name),
            "--iters",
            "100",
            "--config",
            str(ROOT / "research/study-v3/lora.yaml"),
        ]
        receipt = supervise_checked(command, output / f"train-{name}.log", deadline, env)
        receipts.append({"phase": "train", "doc_id": doc_id, "revision": rev, **receipt})
        dump(output / "worker-receipts.json", receipts)
        if receipt["status"] != "completed":
            break
    if len(receipts) == 7 and all(r["status"] == "completed" for r in receipts):
        receipt = supervise_checked(
            [
                sys.executable,
                __file__,
                "--infer-worker",
                "--model",
                str(model_path),
                "--output",
                str(output),
            ],
            output / "revision-infer.log",
            deadline,
            env,
        )
        receipts.append({"phase": "revision-inference", **receipt})
        dump(output / "worker-receipts.json", receipts)
    status = derive_status(output, receipts, training)
    dump(output / "status.json", {"status": status, "worker_receipts": len(receipts)})
    print(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--infer-worker", action="store_true")
    args = parser.parse_args()
    if args.infer_worker:
        inference_worker(args.model, args.output)
    elif args.preflight:
        print(json.dumps(preflight(args.model, args.output), sort_keys=True))
    else:
        run(args.model, args.output)


if __name__ == "__main__":
    main()
