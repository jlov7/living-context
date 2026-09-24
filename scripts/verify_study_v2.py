"""Fail-closed public release check for the model-free v2 evidence replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.replay_study_v2 import replay
from living_context.replacement_study import StudyValidationError

ROOT = Path(__file__).resolve().parents[1]


class ReleaseReplayError(ValueError):
    """The distributed evidence or derived result differs from the release record."""


def verify_frozen_sources(root: Path, frozen: Path) -> int:
    """Check the study-era source snapshot against the sealed protocol hashes."""
    protocol = json.loads((root / "research/study-v2/protocol.json").read_bytes())
    files = protocol["source_files"]
    expected = set(files)
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in expected):
        raise ReleaseReplayError("unsafe frozen study source path")
    entries = list(frozen.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ReleaseReplayError("frozen study source contains a symbolic link")
    actual = {path.relative_to(frozen).as_posix() for path in entries if path.is_file()}
    if actual != expected:
        raise ReleaseReplayError("frozen study source inventory differs from protocol")
    for name, digest in files.items():
        if hashlib.sha256((frozen / name).read_bytes()).hexdigest() != digest:
            raise ReleaseReplayError(f"frozen study source hash mismatch: {name}")
    return len(files)


def verify_release_replay(
    root: Path, evidence: Path, manifest_path: Path, expected_path: Path
) -> dict[str, Any]:
    """Check exact distributed evidence files, then replay and compare all result bytes.

    The manifest is commit-bound metadata for the public source package. It
    describes text evidence only; it cannot authenticate absent tensor files.
    """
    try:
        verify_frozen_sources(root, root / "research/study-v2/frozen-source")
        manifest = json.loads(manifest_path.read_bytes())
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ReleaseReplayError("invalid public evidence manifest")
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ReleaseReplayError("public evidence manifest has no files")
        expected_names = set(files)
        if any(
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            for name in expected_names
        ):
            raise ReleaseReplayError("unsafe public evidence path")
        if any(not isinstance(digest, str) or len(digest) != 64 for digest in files.values()):
            raise ReleaseReplayError("invalid public evidence digest")
        actual_paths = list(evidence.rglob("*"))
        if any(path.is_symlink() for path in actual_paths):
            raise ReleaseReplayError("public evidence contains a symbolic link")
        actual_names = {
            path.relative_to(evidence).as_posix() for path in actual_paths if path.is_file()
        }
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ReleaseReplayError(
                f"public evidence inventory differs: missing={missing}, extra={extra}"
            )
        for name, digest in files.items():
            actual = hashlib.sha256((evidence / name).read_bytes()).hexdigest()
            if actual != digest:
                raise ReleaseReplayError(f"public evidence hash mismatch: {name}")
        result = replay(root, evidence)
        derived = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
        if derived != expected_path.read_bytes():
            raise ReleaseReplayError("full v2 replay differs from RESULTS.json")
        return result
    except (OSError, json.JSONDecodeError, StudyValidationError, KeyError, TypeError) as exc:
        raise ReleaseReplayError(f"public v2 replay failed: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    root = args.root
    evidence = args.evidence or root / "research/study-v2/results"
    manifest = args.manifest or root / "research/study-v2/PUBLIC-EVIDENCE-MANIFEST.json"
    expected = args.expected or root / "research/study-v2/RESULTS.json"
    result = verify_release_replay(root, evidence, manifest, expected)
    development = result["development"]
    print(
        "v2 exact replay: PASS; cached text "
        f"{development['cached_text_exact']}/2; prefixes "
        f"{development['prefix_exact']}/{development['prefix_denominator']}; "
        f"larger study admitted={development['larger_study_admitted']}"
    )


if __name__ == "__main__":
    main()
