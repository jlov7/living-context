"""Recompute the bounded v3 outcome from retained private raw worker output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from living_context.replacement_study import score_exact_json

ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def audit(output: Path) -> dict[str, Any]:
    training = json.loads((ROOT / "research/study-v3/training.json").read_text())
    questions = json.loads((ROOT / "research/study-v3/evaluation.json").read_text())
    records = {item["doc_id"]: item for item in training["pilot"]}
    control = json.loads((output / "control-raw.json").read_text())
    control_expected = training["positive_control"]["answer"]
    control_score = score_exact_json(control["raw"], control_expected).correct
    if not control_score or control_score != control["score"]["exact"]:
        raise ValueError("positive-control raw output does not pass exact scoring")
    raw = json.loads((output / "pilot-raw.json").read_text())
    rows = raw["rows"]
    expected_keys = {(doc, index) for doc in records for index in range(2)}
    actual_keys = {(row["doc_id"], row["question_index"]) for row in rows}
    if len(rows) != 8 or actual_keys != expected_keys:
        raise ValueError("pilot rows are missing, repeated, or unexpected")
    counts = {arm: 0 for arm in ("adapter", "bare", "text")}
    query_seconds = {arm: 0.0 for arm in counts}
    cache_bytes = {}
    for row in rows:
        doc, index = row["doc_id"], row["question_index"]
        if len(questions[doc]) != 2 or not isinstance(questions[doc][index], str):
            raise ValueError("frozen evaluation question missing")
        expected = records[doc]["answer"]
        for arm, result in row["arms"].items():
            exact = score_exact_json(result["raw"], expected).correct
            if exact != result["score"]["exact"]:
                raise ValueError(f"stored score differs for {doc}:{index}:{arm}")
            counts[arm] += exact
            query_seconds[arm] += float(result["metrics"]["elapsed_s"])
        text = row["arms"]["text"]
        if text["fresh_cached_parity"] is not True:
            raise ValueError("cached text parity not recorded")
        cache_bytes[doc] = int(text["cache_bytes"])
    scores = json.loads((output / "scores.json").read_text())
    if scores["counts"] != counts or scores["denominator"] != 8:
        raise ValueError("published counts differ from raw responses")
    receipts = json.loads((output / "worker-receipts.json").read_text())
    if len(receipts) != 7 or any(item["status"] != "completed" for item in receipts):
        raise ValueError("worker set incomplete or resource gate failed")
    if any(item["memory_report_matches"] < 1 for item in receipts):
        raise ValueError("missing MLX memory report")
    if max(item["peak_rss_bytes"] for item in receipts) > 8 * 1024**3:
        raise ValueError("RSS limit exceeded")
    if max(item["reported_mlx_peak_gb"] for item in receipts) > 2:
        raise ValueError("MLX peak limit exceeded")
    artifacts = {
        doc: file_record(output / "adapters" / doc / "adapters.safetensors")
        for doc in [training["positive_control"]["doc_id"], *records]
    }
    inventory = {}
    for name in (
        "preflight.json",
        "control-raw.json",
        "pilot-raw.json",
        "scores.json",
        "status.json",
        "worker-receipts.json",
    ):
        inventory[name] = file_record(output / name)
    for path in sorted(output.glob("*.log")):
        inventory[path.name] = file_record(path)
    return {
        "study": "lc-study-v3-qlora",
        "source_commit": "2b94ce67830da4ff9ffd98095dc48b3a01869f96",
        "control_exact": control_score,
        "pilot_counts": counts,
        "pilot_denominator": 8,
        "quality_gate": "PASS"
        if counts["adapter"] >= 7 and counts["adapter"] >= counts["text"]
        else "FAIL",
        "worker_elapsed_s": round(sum(item["elapsed_s"] for item in receipts), 3),
        "train_worker_elapsed_s": round(
            sum(item["elapsed_s"] for item in receipts if item["phase"] == "train"), 3
        ),
        "query_generation_seconds": {arm: round(value, 3) for arm, value in query_seconds.items()},
        "cache_bytes_per_document": cache_bytes,
        "artifact_files": artifacts,
        "max_sampled_rss_bytes": max(item["peak_rss_bytes"] for item in receipts),
        "max_reported_mlx_peak_gb": max(item["reported_mlx_peak_gb"] for item in receipts),
        "private_evidence_file_inventory": inventory,
        "scope": "synthetic four-fact static pilot; no revision or lifecycle outcome",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run_output)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"v3 audit: {result['quality_gate']} {result['pilot_counts']}")


if __name__ == "__main__":
    main()
