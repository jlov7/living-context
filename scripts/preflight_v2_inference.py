"""Read-only custody preflight for a separate, rights-aware v2 model rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check_file(path: Path, expected: str, *, size: int | None = None) -> None:
    data = path.read_bytes()
    if size is not None and len(data) != size:
        raise ValueError(f"size mismatch: {path.name}")
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")


def preflight(root: Path, model: Path, output: Path) -> dict[str, object]:
    protocol = json.loads((root / "research/study-v2/protocol.json").read_text())
    if output.exists():
        raise ValueError("rerun output root already exists")
    frozen = root / "research/study-v2/frozen-source"
    for relative, digest in protocol["source_files"].items():
        check_file(frozen / relative, digest)
    for relative, digest in protocol["inputs"].items():
        check_file(root / relative, digest)
    for name, metadata in protocol["model"]["files"].items():
        check_file(model / name, metadata["sha256"], size=metadata["bytes"])
    return {
        "status": "PREFLIGHT_PASS",
        "study": protocol["study"],
        "source_commit": protocol["source_commit"],
        "source_files_checked": len(protocol["source_files"]),
        "input_files_checked": len(protocol["inputs"]),
        "model_files_checked": len(protocol["model"]["files"]),
        "output_root_state": "ABSENT",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            preflight(args.root.resolve(), args.model.resolve(), args.output.resolve()),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
