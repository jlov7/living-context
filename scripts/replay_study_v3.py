"""Model-free source/evidence replay for the public v3 QLoRA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from living_context.replacement_study import score_exact_json


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_worker_receipts(receipts: list[dict[str, Any]], training: dict[str, Any]) -> None:
    expected = {
        ("train", row["doc_id"]) for row in [training["positive_control"], *training["pilot"]]
    } | {("control-inference", None), ("pilot-inference", None)}
    actual = [(row.get("phase"), row.get("doc_id")) for row in receipts]
    if len(receipts) != 7 or len(set(actual)) != 7 or set(actual) != expected:
        raise ValueError("worker receipt identities are incomplete or duplicated")
    if any(row.get("status") != "completed" or row.get("exit_code") != 0 for row in receipts):
        raise ValueError("worker did not complete successfully")
    for row in receipts:
        numeric = ("elapsed_s", "peak_rss_bytes", "reported_mlx_peak_gb", "memory_report_matches")
        if any(
            not isinstance(row.get(key), (int, float))
            or not math.isfinite(row[key])
            or row[key] < 0
            for key in numeric
        ):
            raise ValueError("worker metric is missing, negative, or non-finite")
        if row["memory_report_matches"] < 1:
            raise ValueError("worker memory report missing")
    if sum(row["elapsed_s"] for row in receipts) > 900:
        raise ValueError("cumulative worker elapsed exceeded frozen bound")
    if max(row["peak_rss_bytes"] for row in receipts) > 8 * 1024**3:
        raise ValueError("recorded RSS exceeded bound")
    if max(row["reported_mlx_peak_gb"] for row in receipts) > 2:
        raise ValueError("recorded MLX peak exceeded bound")


def replay(root: Path) -> dict[str, Any]:
    study = root / "research/study-v3"
    protocol = json.loads((study / "protocol.json").read_text())
    for relative, expected in protocol["source_sha256"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"frozen source mismatch: {relative}")
    manifest = json.loads((study / "PUBLIC-EVIDENCE-MANIFEST.json").read_text())
    evidence = study / "results"
    expected_files = manifest["files"]
    actual_files = {path.name for path in evidence.iterdir() if path.is_file()}
    if actual_files != set(expected_files):
        raise ValueError("public evidence inventory mismatch")
    for name, expected in expected_files.items():
        if sha256(evidence / name) != expected:
            raise ValueError(f"public evidence hash mismatch: {name}")
    training = json.loads((study / "training.json").read_text())
    questions = json.loads((study / "evaluation.json").read_text())
    control = json.loads((evidence / "control-raw.json").read_text())
    control_exact = score_exact_json(control["raw"], training["positive_control"]["answer"]).correct
    if not control_exact or control["score"]["exact"] is not True:
        raise ValueError("positive control did not pass")
    source_by_id = {row["doc_id"]: row for row in training["pilot"]}
    rows = json.loads((evidence / "pilot-raw.json").read_text())["rows"]
    expected_cells = {(doc_id, index) for doc_id in source_by_id for index in range(2)}
    actual_cells = [(row["doc_id"], row["question_index"]) for row in rows]
    if len(rows) != 8 or set(actual_cells) != expected_cells or len(set(actual_cells)) != 8:
        raise ValueError("pilot cell set is incomplete or duplicated")
    if any(len(questions[doc_id]) != 2 for doc_id in source_by_id):
        raise ValueError("question inventory mismatch")
    counts = {arm: 0 for arm in ("adapter", "bare", "text")}
    for row in rows:
        answer = source_by_id[row["doc_id"]]["answer"]
        if set(row["arms"]) != set(counts):
            raise ValueError("arm set mismatch")
        for arm, result in row["arms"].items():
            exact = score_exact_json(result["raw"], answer).correct
            if result["score"]["exact"] != exact:
                raise ValueError("stored score differs from raw response")
            counts[arm] += exact
        if row["arms"]["text"]["fresh_cached_parity"] is not True:
            raise ValueError("cached-text parity record missing")
    receipts = json.loads((evidence / "worker-receipts.json").read_text())
    validate_worker_receipts(receipts, training)
    scores = json.loads((evidence / "scores.json").read_text())
    status = json.loads((evidence / "status.json").read_text())
    derived_gate = (
        "PILOT_GATE_PASS"
        if counts["adapter"] >= 7 and counts["adapter"] >= counts["text"]
        else "PILOT_GATE_FAIL"
    )
    if scores != {"counts": counts, "denominator": 8, "status": derived_gate}:
        raise ValueError("score summary differs")
    if status != {"status": derived_gate, "worker_receipts": 7}:
        raise ValueError("status summary differs")
    return {
        "status": "REPLAY_PASS",
        "control_exact": True,
        "counts": counts,
        "denominator": 8,
        "gate": derived_gate,
        "scope": "saved-output replay only; no model inference or tensor rehash",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(replay(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
