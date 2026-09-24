"""Write a complete saved-response discrepancy report for a separate rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def v2_rows(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    rows = raw["rows"]
    keys = [f"{row['arm']}|{row['question_id']}" for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate v2 inference cell")
    return {
        key: {"response": row["output"], "token_ids": row["metrics"]["generated_token_ids"]}
        for key, row in zip(keys, rows, strict=True)
    }


def v3_rows(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    result = {}
    for row in raw["rows"]:
        for arm, answer in row["arms"].items():
            key = f"{row['doc_id']}|{row['question_index']}|{arm}"
            if key in result:
                raise ValueError("duplicate v3 inference cell")
            result[key] = {
                "response": answer["raw"],
                "token_ids": answer["metrics"]["generated_token_ids"],
            }
    return result


def revision_rows(folder: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for phase in ("initial", "final"):
        raw = json.loads((folder / f"{phase}-raw.json").read_text())
        for row in raw["rows"]:
            for arm, answer in row["arms"].items():
                key = f"{phase}|{row['doc_id']}|{row['revision']}|{row['question_index']}|{arm}"
                if key in result:
                    raise ValueError("duplicate revision inference cell")
                result[key] = {
                    "response": answer["raw"],
                    "token_ids": answer["metrics"]["generated_token_ids"],
                }
    stale = json.loads((folder / "stale-controls-raw.json").read_text())
    for row in stale["rows"]:
        key = f"stale|{row['doc_id']}|{row['old_revision']}|{row['question_index']}"
        if key in result:
            raise ValueError("duplicate stale inference cell")
        answer = row["result"]
        result[key] = {
            "response": answer["raw"],
            "token_ids": answer["metrics"]["generated_token_ids"],
        }
    return result


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(reference) | set(candidate))
    cells = []
    for key in keys:
        before, after = reference.get(key), candidate.get(key)
        cells.append(
            {
                "cell": key,
                "state": "MISSING"
                if after is None
                else "EXTRA"
                if before is None
                else "MATCH"
                if after == before
                else "DIFFERENT",
                "reference": before,
                "candidate": after,
            }
        )
    return {
        "expected_cells": len(reference),
        "candidate_cells": len(candidate),
        "match_count": sum(item["state"] == "MATCH" for item in cells),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=("v2", "v3", "v3-revision"), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("discrepancy report already exists")
    if args.study == "v2":
        reference = v2_rows(ROOT / "research/study-v2/results/development-raw.json")
        candidate = v2_rows(args.candidate)
    elif args.study == "v3":
        reference = v3_rows(ROOT / "research/study-v3/results/pilot-raw.json")
        candidate = v3_rows(args.candidate)
    else:
        reference = revision_rows(ROOT / "research/study-v3-revision/results")
        candidate = revision_rows(args.candidate)
    args.report.write_text(
        json.dumps(compare(reference, candidate), indent=2, sort_keys=True) + "\n"
    )
    print(f"compared {len(reference)} reference cells with {len(candidate)} candidate cells")


if __name__ == "__main__":
    main()
