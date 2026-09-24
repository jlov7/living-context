"""Frozen synthetic revision-workload count study; no model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

N = 128
SEED = 230923


def edges(shape: str) -> set[tuple[int, int]]:
    if shape == "disconnected":
        return set()
    if shape == "chain":
        return {(i, i + 1) for i in range(N - 1)}
    if shape == "star":
        return {(0, i) for i in range(1, N)}
    if shape == "four-cliques":
        return {(i, j) for i in range(N) for j in range(i + 1, N) if i // 32 == j // 32}
    if shape == "full-clique":
        return {(i, j) for i in range(N) for j in range(i + 1, N)}
    raise ValueError(shape)


def closure(seeds: set[int], links: set[tuple[int, int]]) -> set[int]:
    seen = set(seeds)
    while True:
        expanded = seen | {b for a, b in links if a in seen} | {a for a, b in links if b in seen}
        if len(expanded) == len(seen):
            return seen
        seen = expanded


def run() -> dict[str, object]:
    rng = Random(SEED)
    rows: list[dict[str, object]] = []
    shapes = ("disconnected", "chain", "star", "four-cliques", "full-clique")
    for changed_count in (1, 4, 16):
        seeds = set(rng.sample(range(N), changed_count))
        for shape in shapes:
            links = edges(shape)
            impacted = len(closure(seeds, links))
            for reads_per_edit in (1, 10, 100):
                rows.append({
                    "shape": shape, "edge_count": len(links),
                    "changed_doc_ids": sorted(seeds), "reads_per_edit": reads_per_edit,
                    "changed_only_rebuild_units": changed_count,
                    "closure_rebuild_units": impacted,
                    "full_rebuild_units": N,
                    "steady_artifact_storage_units": N,
                    "changed_only_peak_storage_units": N + changed_count,
                    "closure_peak_storage_units": N + impacted,
                    "full_peak_storage_units": 2 * N,
                    "changed_only_rebuild_units_per_read": changed_count / reads_per_edit,
                    "closure_rebuild_units_per_read": impacted / reads_per_edit,
                    "full_rebuild_units_per_read": N / reads_per_edit,
                    "closure_to_full_ratio": impacted / N,
                })
    return {"study": "lc-revision-workload-v1", "n_documents": N, "seed": SEED, "rows": rows}


def chart(rows: list[dict[str, object]]) -> str:
    selected = [r for r in rows if r["reads_per_edit"] == 10 and len(r["changed_doc_ids"]) == 1]
    bars = []
    for index, row in enumerate(selected):
        y = 40 + index * 42
        ratio = float(row["closure_to_full_ratio"])
        bars.append(f'<text x="10" y="{y+17}" font-size="14">{row["shape"]}</text>')
        bars.append(f'<rect x="125" y="{y}" width="{ratio*400:.1f}" height="23" fill="#305b8a"/>')
        bars.append(f'<text x="{130+ratio*400:.1f}" y="{y+17}" font-size="14">{ratio:.3f}</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" width="620" height="280"><text x="10" y="22" font-size="16">Dependency closure / full rebuild, one edit</text>' + "".join(bars) + "</svg>\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = run()
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "closure.svg").write_text(chart(result["rows"]))


if __name__ == "__main__":
    main()
