"""Optional Matplotlib plot of the frozen operation-count output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    payload = json.loads(args.results.read_text())
    shapes = ("disconnected", "chain", "star", "four-cliques", "full-clique")
    changed = (1, 4, 16)
    values = [
        [
            next(
                row["closure_to_full_ratio"]
                for row in payload["rows"]
                if row["shape"] == shape
                and len(row["changed_doc_ids"]) == count
                and row["reads_per_edit"] == 10
            )
            for shape in shapes
        ]
        for count in changed
    ]
    fig, ax = plt.subplots(figsize=(9, 3.5), layout="constrained")
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(shapes)), labels=shapes, rotation=25, ha="right")
    ax.set_yticks(range(len(changed)), labels=[f"{count} changed" for count in changed])
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            ax.text(
                x,
                y,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.6 else "black",
            )
    ax.set_title("Dependency-closure rebuild units / full rebuild units (128 documents)")
    fig.colorbar(image, ax=ax, label="Ratio")
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
