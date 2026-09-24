"""Pure validation and scoring for the bounded replacement study.

This module intentionally has no MLX imports. Training consumes source records;
questions and sealed labels are loaded through separate functions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class StudyValidationError(ValueError):
    """Raised when a frozen study input or output is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def read_json(path: Path) -> Any:
    return parse_json_bytes(path.read_bytes(), source=str(path))


def parse_json_bytes(data: bytes, *, source: str) -> Any:
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudyValidationError(f"cannot read valid JSON: {source}") from exc


def verify_input_snapshots(root: Path, expected: dict[str, str]) -> dict[str, bytes]:
    """Read and hash small inputs once, returning the exact verified bytes."""
    snapshots: dict[str, bytes] = {}
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise StudyValidationError(f"input escapes repository: {relative}") from exc
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise StudyValidationError(f"frozen input is unreadable: {relative}") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise StudyValidationError(f"frozen input mismatch: {relative}")
        snapshots[relative] = data
    return snapshots


def load_sources(path: Path) -> dict[str, dict[str, Any]]:
    """Load the only data surface accepted by the trainer."""
    return parse_sources(read_json(path))


def parse_sources(payload: Any) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sources"), list):
        raise StudyValidationError("invalid source schema")
    sources: dict[str, dict[str, Any]] = {}
    for row in payload["sources"]:
        required = {"source_id", "split", "doc_id", "revision", "content", "training_qa"}
        if not isinstance(row, dict) or set(row) != required:
            raise StudyValidationError("source rows must have exactly the declared fields")
        source_id = row["source_id"]
        if not isinstance(source_id, str) or not source_id or source_id in sources:
            raise StudyValidationError("source_id must be a unique non-empty string")
        if row["split"] not in {"calibration", "development", "final"}:
            raise StudyValidationError(f"invalid split for {source_id}")
        if not all(
            isinstance(row[key], str) and row[key] for key in ("doc_id", "revision", "content")
        ):
            raise StudyValidationError(f"invalid source strings for {source_id}")
        if not isinstance(row["training_qa"], list) or not row["training_qa"]:
            raise StudyValidationError(f"training_qa must be non-empty for {source_id}")
        for qa in row["training_qa"]:
            if not isinstance(qa, dict) or set(qa) != {"question", "answer"}:
                raise StudyValidationError(f"invalid training QA for {source_id}")
            if not all(isinstance(qa[key], str) and qa[key] for key in ("question", "answer")):
                raise StudyValidationError(f"empty training QA for {source_id}")
        sources[source_id] = deepcopy(row)
    return sources


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    return parse_questions(read_json(path))


def parse_questions(payload: Any) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("questions"), list):
        raise StudyValidationError("invalid question schema")
    questions: dict[str, dict[str, Any]] = {}
    required = {"question_id", "split", "snapshot", "source_ids", "question"}
    for row in payload["questions"]:
        if not isinstance(row, dict) or set(row) != required:
            raise StudyValidationError("question rows must not contain labels")
        question_id = row["question_id"]
        if not isinstance(question_id, str) or not question_id or question_id in questions:
            raise StudyValidationError("question_id must be a unique non-empty string")
        if row["split"] not in {"calibration", "development", "final"}:
            raise StudyValidationError(f"invalid question split for {question_id}")
        if not isinstance(row["snapshot"], str) or not row["snapshot"]:
            raise StudyValidationError(f"invalid question snapshot for {question_id}")
        if not isinstance(row["source_ids"], list) or not row["source_ids"]:
            raise StudyValidationError(f"source_ids must be non-empty for {question_id}")
        if not all(isinstance(item, str) and item for item in row["source_ids"]):
            raise StudyValidationError(f"invalid source_ids for {question_id}")
        if not isinstance(row["question"], str) or not row["question"]:
            raise StudyValidationError(f"invalid question for {question_id}")
        questions[question_id] = deepcopy(row)
    return questions


def load_labels(path: Path) -> dict[str, str]:
    return parse_labels(read_json(path))


def parse_labels(payload: Any) -> dict[str, str]:
    if payload.get("schema_version") != 1 or not isinstance(payload.get("labels"), dict):
        raise StudyValidationError("invalid label schema")
    labels = payload["labels"]
    if not labels or not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in labels.items()
    ):
        raise StudyValidationError("labels must map non-empty strings to non-empty strings")
    return dict(labels)


def validate_data_separation(
    sources: dict[str, dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    labels: dict[str, str],
) -> None:
    if set(questions) != set(labels):
        raise StudyValidationError("question and label ids differ")
    split_docs: dict[str, set[str]] = {
        name: set() for name in ("calibration", "development", "final")
    }
    train_questions: set[str] = set()
    for source in sources.values():
        split_docs[source["split"]].add(source["doc_id"])
        train_questions.update(qa["question"] for qa in source["training_qa"])
    if any(
        split_docs[a] & split_docs[b]
        for a, b in (
            ("calibration", "development"),
            ("calibration", "final"),
            ("development", "final"),
        )
    ):
        raise StudyValidationError("document ids must be disjoint across splits")
    for question_id, row in questions.items():
        if row["question"] in train_questions:
            raise StudyValidationError(f"held-out wording appears in training QA: {question_id}")
        for source_id in row["source_ids"]:
            if source_id not in sources or sources[source_id]["split"] != row["split"]:
                raise StudyValidationError(f"question/source split mismatch: {question_id}")


class LexicalIndex:
    """Small deterministic overlap index used identically for every arm."""

    def __init__(self, sources: list[dict[str, Any]]) -> None:
        self._sources = {source["source_id"]: deepcopy(source) for source in sources}
        if len(self._sources) != len(sources):
            raise StudyValidationError("duplicate source id in lexical index")
        self._terms = {
            source_id: sorted(
                set(
                    re.findall(
                        r"[a-z0-9]+",
                        f"{source['doc_id']} {source['content']}".casefold(),
                    )
                )
            )
            for source_id, source in self._sources.items()
        }

    @property
    def storage_bytes(self) -> int:
        return len(canonical_json(self._terms).encode())

    def retrieve(self, question: str, *, limit: int) -> list[dict[str, Any]]:
        if limit < 1:
            raise StudyValidationError("retrieval limit must be positive")
        query_terms = set(re.findall(r"[a-z0-9]+", question.casefold()))
        ranked: list[tuple[int, str]] = []
        for source_id, terms in self._terms.items():
            score = len(query_terms & set(terms))
            if score:
                ranked.append((-score, source_id))
        ranked.sort()
        return [deepcopy(self._sources[source_id]) for _, source_id in ranked[:limit]]


def lexical_retrieve(
    question: str, sources: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    return LexicalIndex(sources).retrieve(question, limit=limit)


@dataclass(frozen=True)
class ExactScore:
    correct: bool
    parsed_answer: str | None
    error: str | None


def score_exact_json(output: str, expected: str) -> ExactScore:
    """Require one complete JSON object with exactly one string field."""
    duplicate = False

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(output, object_pairs_hook=pairs_hook)
    except json.JSONDecodeError:
        return ExactScore(False, None, "invalid_json")
    if duplicate or not isinstance(value, dict) or set(value) != {"answer"}:
        return ExactScore(False, None, "invalid_schema")
    answer = value["answer"]
    if not isinstance(answer, str):
        return ExactScore(False, None, "invalid_schema")
    normalized = answer.strip().casefold()
    expected_normalized = expected.strip().casefold()
    if normalized != expected_normalized:
        return ExactScore(False, answer, "wrong_answer")
    return ExactScore(True, answer, None)


@dataclass(frozen=True)
class CacheKey:
    model_sha256: str
    tokenizer_sha256: str
    prompt_sha256: str
    doc_id: str
    revision: str
    content_sha256: str


class RevisionCacheRegistry(Generic[T]):
    """Revision-keyed cache with explicit clone-on-read isolation."""

    def __init__(self, clone: Callable[[T], T] = deepcopy) -> None:
        self._items: dict[CacheKey, T] = {}
        self._clone = clone

    def put(self, key: CacheKey, value: T) -> None:
        self.invalidate_document(key.doc_id)
        self._items[key] = self._clone(value)

    def get(self, key: CacheKey) -> T:
        if key not in self._items:
            raise KeyError(key)
        return self._clone(self._items[key])

    def invalidate_document(self, doc_id: str) -> int:
        doomed = [key for key in self._items if key.doc_id == doc_id]
        for key in doomed:
            del self._items[key]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._items)


def cross_entropy_from_logits(logits: list[float], target: int) -> float:
    """Reference CE used to test the MLX target/logit formulation."""
    if not logits or target < 0 or target >= len(logits):
        raise StudyValidationError("invalid logits or target")
    if not all(math.isfinite(value) for value in logits):
        raise StudyValidationError("logits must be finite")
    largest = max(logits)
    logsumexp = largest + math.log(sum(math.exp(value - largest) for value in logits))
    return logsumexp - logits[target]


def target_prediction_slice(prefix_length: int, sequence_length: int, target_start: int) -> slice:
    """Return causal-logit positions that predict targets[target_start:]."""
    if prefix_length < 1 or not 0 < target_start < sequence_length:
        raise StudyValidationError("invalid causal alignment dimensions")
    return slice(prefix_length + target_start - 1, prefix_length + sequence_length - 1)


def assert_prefix_only_training(base_parameter_names: list[str], gradient_names: list[str]) -> None:
    if base_parameter_names:
        raise StudyValidationError("base model still has trainable parameters")
    if gradient_names != ["weights"]:
        raise StudyValidationError(f"unexpected trainable prefix parameters: {gradient_names}")


def write_sealed_json(path: Path, payload: Any) -> Path:
    """Durably write raw output and a hash sidecar before labels can be read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seal_path = path.with_suffix(path.suffix + ".seal.json")
    if path.exists() or seal_path.exists():
        raise StudyValidationError(f"sealed output already exists: {path.name}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    seal = {
        "schema_version": 1,
        "file": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    seal_data = (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode()
    with seal_path.open("wb") as handle:
        handle.write(seal_data)
        handle.flush()
        os.fsync(handle.fileno())
    return seal_path


def read_sealed_json(path: Path, seal_path: Path) -> Any:
    seal = parse_json_bytes(seal_path.read_bytes(), source=str(seal_path))
    data = path.read_bytes()
    if (
        not isinstance(seal, dict)
        or seal.get("file") != path.name
        or seal.get("bytes") != len(data)
        or seal.get("sha256") != hashlib.sha256(data).hexdigest()
    ):
        raise StudyValidationError("raw output seal mismatch")
    return parse_json_bytes(data, source=str(path))


def validate_frozen_inputs(root: Path, protocol: dict[str, Any]) -> dict[str, bytes]:
    inputs = protocol.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise StudyValidationError("protocol inputs are missing")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in inputs.items()):
        raise StudyValidationError("protocol inputs must map paths to hashes")
    return verify_input_snapshots(root, inputs)
