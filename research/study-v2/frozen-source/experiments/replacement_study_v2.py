"""Bounded, fail-closed local replacement study.

Public commands supervise a single MLX worker. Training workers receive only a
serialized source record and training settings; evaluation questions and
sealed labels are absent from their payload. Scoring is a separate command and
will only read raw outputs whose durable SHA-256 seal verifies.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from living_context.replacement_study import (
    CacheKey,
    LexicalIndex,
    RevisionCacheRegistry,
    StudyValidationError,
    assert_prefix_only_training,
    canonical_json,
    parse_json_bytes,
    parse_labels,
    parse_questions,
    parse_sources,
    read_sealed_json,
    score_exact_json,
    sha256_file,
    target_prediction_slice,
    validate_data_separation,
    validate_frozen_inputs,
    write_sealed_json,
)

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "Answer from the available synthetic reference. Return exactly one JSON object with "
    'exactly one key named "answer" and a string value. Do not add prose.'
)
RECONSTRUCTION_QUESTION = "What is the complete synthetic reference record?"
OBJECTIVES = ("document_reconstruction", "source_qa")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _protocol(path: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    data = path.read_bytes()
    protocol = parse_json_bytes(data, source=str(path))
    if protocol.get("state") != "FROZEN" or protocol.get("schema_version") != 1:
        raise StudyValidationError("protocol is not frozen schema version 1")
    relative = path.resolve().relative_to(ROOT)
    tracked = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if tracked.stdout:
        raise StudyValidationError("tracked repository files changed after protocol freeze")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    if committed != data:
        raise StudyValidationError("protocol bytes differ from HEAD")
    snapshots = validate_frozen_inputs(ROOT, protocol)
    for relative, digest in protocol["source_files"].items():
        if sha256_file(ROOT / relative) != digest:
            raise StudyValidationError(f"source changed after protocol freeze: {relative}")
    return protocol, snapshots


def _small_inputs(protocol: dict[str, Any], snapshots: dict[str, bytes]) -> tuple[dict, dict, dict]:
    paths = protocol["data"]
    sources = parse_sources(
        parse_json_bytes(snapshots[paths["sources"]], source="sources snapshot")
    )
    questions = parse_questions(
        parse_json_bytes(snapshots[paths["questions"]], source="questions snapshot")
    )
    labels = parse_labels(parse_json_bytes(snapshots[paths["labels"]], source="labels snapshot"))
    validate_data_separation(sources, questions, labels)
    return sources, questions, labels


def _redact(text: str) -> str:
    return text.replace(str(Path.home()), "<HOME>")


def _supervise(
    worker_args: list[str],
    *,
    payload: dict[str, Any],
    output_dir: Path,
    timeout_s: float,
    rss_limit_bytes: int,
    total_budget_s: float,
    receipt_name: str | None = None,
) -> dict[str, Any]:
    """Run one model worker with external elapsed/RSS enforcement."""
    import psutil

    output_dir.mkdir(parents=True, exist_ok=True)
    budget_root = output_dir.parent
    prior_elapsed = 0.0
    for receipt_path in budget_root.glob("*/**/*-supervisor.json"):
        receipt = parse_json_bytes(receipt_path.read_bytes(), source=str(receipt_path))
        prior_elapsed += float(receipt["elapsed_s"])
    extras_path = budget_root / "budget-extras.json"
    if extras_path.exists():
        extras = parse_json_bytes(extras_path.read_bytes(), source=str(extras_path))
        prior_elapsed += sum(float(item["elapsed_s"]) for item in extras["model_calls"])
    remaining_budget = total_budget_s - prior_elapsed
    if remaining_budget <= 0:
        raise RuntimeError("cumulative model-worker budget exhausted")
    effective_timeout_s = min(timeout_s, remaining_budget)
    receipt_stem = receipt_name or worker_args[0]
    for suffix in ("-input-access.json", "-stdout.log", "-stderr.log", "-supervisor.json"):
        if (output_dir / f"{receipt_stem}{suffix}").exists():
            raise StudyValidationError("attempt already exists; use a new result directory")
    with tempfile.TemporaryDirectory(prefix="living-context-study-") as temporary:
        temp = Path(temporary)
        payload_path = temp / "worker-input.json"
        payload_data = _json_bytes(payload)
        payload_path.write_bytes(payload_data)
        input_manifest = {
            "schema_version": 1,
            "worker": worker_args[0],
            "payload_sha256": hashlib.sha256(payload_data).hexdigest(),
            "allowed_payload_fields": sorted(payload),
            "contains_evaluation_questions": "questions" in payload,
            "contains_heldout_labels": "labels" in payload,
            "boundary": (
                "serialized process-input separation; this is not an OS filesystem sandbox"
            ),
        }
        _write_json(output_dir / f"{receipt_stem}-input-access.json", input_manifest)
        env = {
            "HOME": str(temp / "empty-home"),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TMPDIR": str(temp),
            "LIVING_CONTEXT_MODEL_PATH": os.environ["LIVING_CONTEXT_MODEL_PATH"],
        }
        Path(env["HOME"]).mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), *worker_args, str(payload_path)]
        start = time.monotonic()
        stdout_path = output_dir / f"{receipt_stem}-stdout.log"
        stderr_path = output_dir / f"{receipt_stem}-stderr.log"
        stdout_handle = stdout_path.open("x")
        stderr_handle = stderr_path.open("x")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        observed_peak_rss = 0
        stop_reason: str | None = None
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            elapsed = time.monotonic() - start
            try:
                rss = monitored.memory_info().rss
            except psutil.Error:
                rss = 0
            observed_peak_rss = max(observed_peak_rss, rss)
            if elapsed > effective_timeout_s:
                stop_reason = "elapsed_limit"
            elif rss > rss_limit_bytes:
                stop_reason = "rss_limit"
            if stop_reason:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(0.25)
        process.wait()
        stdout_handle.close()
        stderr_handle.close()
        stdout = _redact(stdout_path.read_text())
        stderr = _redact(stderr_path.read_text())
        stdout_path.write_text(stdout)
        stderr_path.write_text(stderr)
        elapsed = time.monotonic() - start
        receipt = {
            "schema_version": 1,
            "worker": worker_args[0],
            "returncode": process.returncode,
            "stop_reason": stop_reason,
            "elapsed_s": elapsed,
            "prior_cumulative_model_worker_elapsed_s": prior_elapsed,
            "total_model_worker_budget_s": total_budget_s,
            "effective_timeout_s": effective_timeout_s,
            "sampled_process_peak_rss_bytes": observed_peak_rss,
            "rss_sample_interval_s": 0.25,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
        _write_json(output_dir / f"{receipt_stem}-supervisor.json", receipt)
        if process.returncode != 0 or stop_reason:
            raise RuntimeError(
                f"worker {worker_args[0]} failed: {stop_reason or process.returncode}"
            )
        return receipt


def _model_file_hashes(model_path: Path, expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    actual_names = {
        item.name for item in model_path.iterdir() if item.is_file() or item.is_symlink()
    }
    if actual_names != set(expected):
        raise StudyValidationError("model snapshot file set differs from protocol")
    result: dict[str, Any] = {}
    for name, spec in expected.items():
        path = model_path / name
        size = path.resolve().stat().st_size
        digest = sha256_file(path)
        if size != spec["bytes"] or digest != spec["sha256"]:
            raise StudyValidationError(f"model file mismatch: {name}")
        result[name] = {"bytes": size, "sha256": digest}
    return result


def _load_mlx(model_path: Path, expected: dict[str, dict[str, Any]]):
    import mlx.core as mx
    from mlx_lm import load

    before = _model_file_hashes(model_path, expected)
    started = time.perf_counter()
    model, tokenizer = cast(tuple[Any, Any], load(str(model_path), return_config=False))
    mx.eval(model.parameters())
    load_s = time.perf_counter() - started
    after = _model_file_hashes(model_path, expected)
    if before != after:
        raise StudyValidationError("model snapshot changed during load")
    if tokenizer.chat_template is None:
        raise StudyValidationError("pinned tokenizer has no chat template")
    model.eval()
    model.freeze()
    return model, tokenizer, load_s, before


def _chat_tokens(tokenizer, question: str, documents: list[str] | None = None) -> list[int]:
    content = question
    if documents:
        content = "Reference:\n" + "\n\n".join(documents) + "\n\nQuestion:\n" + question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return list(tokens)


def _training_examples(
    tokenizer, source: dict[str, Any], objective: str
) -> list[tuple[list[int], int]]:
    if objective == "document_reconstruction":
        examples = [(RECONSTRUCTION_QUESTION, source["content"])]
    elif objective == "source_qa":
        examples = [(row["question"], row["answer"]) for row in source["training_qa"]]
    else:
        raise StudyValidationError(f"unknown objective: {objective}")
    sequences: list[tuple[list[int], int]] = []
    for question, answer in examples:
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        full_messages = [
            *prompt_messages,
            {"role": "assistant", "content": canonical_json({"answer": answer})},
        ]
        prompt_ids = list(
            tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True
            )
        )
        full_ids = list(
            tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False)
        )
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
            raise StudyValidationError("chat template does not preserve the training prompt prefix")
        sequences.append((full_ids, len(prompt_ids)))
    return sequences


def _rss_bytes() -> int:
    import psutil

    return psutil.Process().memory_info().rss


def _flat_names(value: Any) -> list[str]:
    from mlx.utils import tree_flatten

    return sorted(name for name, _ in tree_flatten(value))


def _train_one(
    model,
    tokenizer,
    source: dict[str, Any],
    settings: dict[str, Any],
    objective: str,
    seed: int,
    journal_path: Path,
):
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
    from mlx.utils import tree_flatten

    class PrefixArtifact(nn.Module):
        def __init__(self, length: int, dimension: int) -> None:
            super().__init__()
            self.weights = (mx.random.normal((length, dimension)) * 0.02).astype(mx.float32)

    mx.random.seed(seed)
    prefix_length = settings["prefix_length"]
    dimension = int(model.model.embed_tokens(mx.arange(2)).shape[-1])
    artifact = PrefixArtifact(prefix_length, dimension)
    assert_prefix_only_training(_flat_names(model.trainable_parameters()), ["weights"])
    examples = _training_examples(tokenizer, source, objective)
    optimizer = optim.Adam(learning_rate=settings["learning_rate"])
    started = time.monotonic()
    peak_mlx = int(mx.get_active_memory())
    peak_rss = _rss_bytes()
    final_loss = None

    def loss_fn(ids: list[int], target_start: int):
        input_ids = mx.array(ids)[None, :]
        token_embeddings = model.model.embed_tokens(input_ids)
        embeddings = mx.concatenate([artifact.weights[None, :, :], token_embeddings], axis=1)
        dummy = mx.zeros((1, prefix_length), dtype=input_ids.dtype)
        combined_ids = mx.concatenate([dummy, input_ids], axis=1)
        logits = model(inputs=combined_ids, input_embeddings=embeddings)
        prediction_slice = target_prediction_slice(prefix_length, len(ids), target_start)
        predictions = logits[:, prediction_slice, :].astype(mx.float32)
        targets = input_ids[:, target_start:]
        if predictions.shape[1] != targets.shape[1]:
            raise StudyValidationError("causal target/logit alignment failed")
        return nn.losses.cross_entropy(predictions, targets, reduction="mean")  # pyright: ignore[reportPrivateImportUsage]

    loss_and_grad = nn.value_and_grad(artifact, loss_fn)  # pyright: ignore[reportPrivateImportUsage]
    for step in range(settings["steps"]):
        ids, target_start = examples[step % len(examples)]
        loss, gradients = loss_and_grad(ids, target_start)
        assert_prefix_only_training([], _flat_names(gradients))
        mx.eval(loss, gradients)
        final_loss = float(loss.item())
        finite_gradients = all(
            bool(mx.all(mx.isfinite(cast(Any, value))).item())
            for _, value in tree_flatten(gradients)
        )
        if not bool(mx.isfinite(loss).item()) or not finite_gradients:
            raise RuntimeError("non_finite_loss_or_gradient")
        optimizer.update(artifact, gradients)
        mx.eval(artifact.weights, optimizer.state)
        peak_mlx = max(peak_mlx, int(mx.get_active_memory()))
        peak_rss = max(peak_rss, _rss_bytes())
        if peak_mlx > settings["mlx_active_limit_bytes"]:
            raise RuntimeError("mlx_active_limit")
        if peak_rss > settings["rss_limit_bytes"]:
            raise RuntimeError("rss_limit")
        if time.monotonic() - started > settings["worker_elapsed_limit_s"]:
            raise RuntimeError("elapsed_limit")
        if step % 10 == 0 or step + 1 == settings["steps"]:
            with journal_path.open("a") as handle:
                handle.write(
                    canonical_json(
                        {
                            "objective": objective,
                            "seed": seed,
                            "step_completed": step + 1,
                            "loss": final_loss,
                            "elapsed_s": time.monotonic() - started,
                            "sampled_mlx_active_bytes": peak_mlx,
                            "sampled_process_rss_bytes": peak_rss,
                        }
                    )
                    + "\n"
                )
                handle.flush()
    return artifact, {
        "objective": objective,
        "seed": seed,
        "steps_completed": settings["steps"],
        "final_loss": final_loss,
        "training_s": time.monotonic() - started,
        "sampled_mlx_active_peak_bytes": peak_mlx,
        "sampled_process_rss_peak_bytes": peak_rss,
        "prefix_shape": list(artifact.weights.shape),
    }


def _generate(
    model,
    tokenizer,
    tokens: list[int],
    max_tokens: int,
    *,
    prompt_cache=None,
    input_embeddings=None,
):
    from mlx_lm.generate import stream_generate

    started = time.perf_counter()
    responses = list(
        stream_generate(
            model,
            tokenizer,
            tokens,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            input_embeddings=input_embeddings,
        )
    )
    elapsed = time.perf_counter() - started
    output = "".join(response.text for response in responses)
    final = responses[-1]
    prompt_s = final.prompt_tokens / final.prompt_tps if final.prompt_tps else None
    return output, {
        "generated_token_ids": [int(response.token) for response in responses[:-1]],
        "elapsed_s": elapsed,
        "prompt_tokens": final.prompt_tokens,
        "prompt_s": prompt_s,
        "decode_s": elapsed - prompt_s if prompt_s is not None else None,
        "generation_tokens": final.generation_tokens,
        "finish_reason": final.finish_reason,
        "reported_peak_memory_gb": final.peak_memory,
    }


def _generate_prefix(model, tokenizer, prompt_tokens: list[int], weights, max_tokens: int):
    import mlx.core as mx

    ids = mx.array(prompt_tokens)[None, :]
    token_embeddings = model.model.embed_tokens(ids)
    embeddings = mx.concatenate([weights[None, :, :], token_embeddings], axis=1)[0]
    dummy_tokens = [0] * int(weights.shape[0]) + prompt_tokens
    return _generate(
        model,
        tokenizer,
        dummy_tokens,
        max_tokens,
        input_embeddings=embeddings,
    )


def _common_text_prefix(tokenizer, documents: list[str]) -> list[int]:
    first = _chat_tokens(tokenizer, "AAA-QUESTION-SENTINEL", documents)
    second = _chat_tokens(tokenizer, "ZZZ-QUESTION-SENTINEL", documents)
    length = 0
    for left, right in zip(first, second):
        if left != right:
            break
        length += 1
    if length == 0 or length == min(len(first), len(second)):
        raise StudyValidationError("could not isolate a stable cached text prefix")
    return first[:length]


def _build_cache(model, prefix_tokens: list[int]):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    started = time.perf_counter()
    model(mx.array(prefix_tokens)[None, :], cache=cache)
    mx.eval([entry.state for entry in cache])
    elapsed = time.perf_counter() - started
    return cache, elapsed, sum(entry.nbytes for entry in cache)


def _cache_check_and_generate(
    model, tokenizer, documents, question, registry, key, max_tokens, logit_atol
):
    import mlx.core as mx

    full = _chat_tokens(tokenizer, question, documents)
    prefix = _common_text_prefix(tokenizer, documents)
    if full[: len(prefix)] != prefix:
        raise StudyValidationError("question prompt does not share the frozen cache prefix")
    tail = full[len(prefix) :]
    if not tail:
        raise StudyValidationError("cached prompt tail is empty")
    cache_build_s = 0.0
    try:
        base_cache = registry.get(key)
        cache_hit = True
    except KeyError:
        base_cache, cache_build_s, cache_bytes = _build_cache(model, prefix)
        registry.put(key, base_cache)
        base_cache = registry.get(key)
        cache_hit = False
        registry.cache_bytes = cache_bytes
    parity_started = time.perf_counter()
    fresh_logits_raw = model(mx.array(full)[None, :])[:, -1, :]
    cached_for_check = copy.deepcopy(base_cache)
    cached_logits_raw = model(mx.array(tail)[None, :], cache=cached_for_check)[:, -1, :]
    fresh_logits = fresh_logits_raw.astype(mx.float32)
    cached_logits = cached_logits_raw.astype(mx.float32)
    mx.eval(fresh_logits, cached_logits)
    max_abs_diff = float(cast(float, mx.max(mx.abs(fresh_logits - cached_logits)).item()))
    max_abs_fresh_logit = float(cast(float, mx.max(mx.abs(fresh_logits)).item()))
    max_abs_cached_logit = float(cast(float, mx.max(mx.abs(cached_logits)).item()))
    fresh_argmax = int(cast(int, mx.argmax(fresh_logits, axis=-1).item()))
    cached_argmax = int(cast(int, mx.argmax(cached_logits, axis=-1).item()))
    parity_check_s = time.perf_counter() - parity_started
    if cached_argmax != fresh_argmax or max_abs_diff > logit_atol:
        raise StudyValidationError("cached and fresh next-token logits differ")
    fresh_output, fresh_metrics = _generate(model, tokenizer, full, max_tokens)
    cached_output, cached_metrics = _generate(
        model, tokenizer, tail, max_tokens, prompt_cache=registry.get(key)
    )
    if (
        cached_output != fresh_output
        or cached_metrics["generated_token_ids"] != fresh_metrics["generated_token_ids"]
    ):
        raise StudyValidationError("cached and fresh greedy token sequences differ")
    return {
        "fresh_output": fresh_output,
        "cached_output": cached_output,
        "fresh_metrics": fresh_metrics,
        "cached_metrics": cached_metrics,
        "cache_hit": cache_hit,
        "cache_build_s": cache_build_s,
        "cache_bytes": registry.cache_bytes,
        "next_token_max_abs_logit_diff": max_abs_diff,
        "next_token_max_abs_fresh_logit": max_abs_fresh_logit,
        "next_token_max_abs_cached_logit": max_abs_cached_logit,
        "next_token_fresh_logit_dtype": str(fresh_logits_raw.dtype),
        "next_token_cached_logit_dtype": str(cached_logits_raw.dtype),
        "next_token_comparison_dtype": "mlx.core.float32",
        "next_token_argmax_equal": True,
        "parity_check_s": parity_check_s,
    }


def _worker_setup(payload_path: Path):
    payload_data = payload_path.read_bytes()
    payload = parse_json_bytes(payload_data, source="worker payload")
    model_path = Path(os.environ["LIVING_CONTEXT_MODEL_PATH"])
    model, tokenizer, load_s, hashes = _load_mlx(model_path, payload["model_files"])
    return payload, model, tokenizer, load_s, hashes


def _calibration_worker(payload_path: Path) -> None:
    payload, model, tokenizer, load_s, hashes = _worker_setup(payload_path)
    registry = RevisionCacheRegistry(clone=copy.deepcopy)
    rows = []
    source = payload["source"]
    documents = [source["content"]]
    prompt_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    key = CacheKey(
        payload["model_digest"],
        payload["tokenizer_digest"],
        prompt_hash,
        source["doc_id"],
        source["revision"],
        hashlib.sha256(source["content"].encode()).hexdigest(),
    )
    for question in payload["questions"]:
        text = _cache_check_and_generate(
            model,
            tokenizer,
            documents,
            question["question"],
            registry,
            key,
            payload["max_output_tokens"],
            payload["cache_logit_atol"],
        )
        bare_output, bare_metrics = _generate(
            model,
            tokenizer,
            _chat_tokens(tokenizer, question["question"]),
            payload["max_output_tokens"],
        )
        rows.append(
            {
                "question_id": question["question_id"],
                "bare_output": bare_output,
                "bare_metrics": bare_metrics,
                **text,
            }
        )
    invalidation_source = payload["invalidation_source"]
    invalidation_key = CacheKey(
        payload["model_digest"],
        payload["tokenizer_digest"],
        prompt_hash,
        invalidation_source["doc_id"],
        invalidation_source["revision"],
        hashlib.sha256(invalidation_source["content"].encode()).hexdigest(),
    )
    invalidation_started = time.perf_counter()
    invalidated_entries = registry.invalidate_document(source["doc_id"])
    invalidation_s = time.perf_counter() - invalidation_started
    if invalidated_entries != 1:
        raise StudyValidationError("expected exactly one old revision cache entry")
    invalidation_probe = _cache_check_and_generate(
        model,
        tokenizer,
        [invalidation_source["content"]],
        payload["questions"][0]["question"],
        registry,
        invalidation_key,
        payload["max_output_tokens"],
        payload["cache_logit_atol"],
    )
    old_key_invalidated = False
    try:
        registry.get(key)
    except KeyError:
        old_key_invalidated = True
    if not old_key_invalidated:
        raise StudyValidationError("old revision cache survived invalidation")
    post_hashes = _model_file_hashes(
        Path(os.environ["LIVING_CONTEXT_MODEL_PATH"]), payload["model_files"]
    )
    write_sealed_json(
        Path(payload["output_path"]),
        {
            "schema_version": 1,
            "phase": "calibration",
            "protocol_sha256": payload["protocol_sha256"],
            "model_load_s": load_s,
            "model_files_before": hashes,
            "model_files_after": post_hashes,
            "rows": rows,
            "invalidation_probe": {
                "old_revision_invalidated": old_key_invalidated,
                "new_revision_cache_matches_fresh": invalidation_probe["cached_output"]
                == invalidation_probe["fresh_output"],
                "new_revision_next_token_max_abs_logit_diff": invalidation_probe[
                    "next_token_max_abs_logit_diff"
                ],
                "invalidation_s": invalidation_s,
                "invalidated_entries": invalidated_entries,
            },
        },
    )


def _train_worker(payload_path: Path) -> None:
    payload, model, tokenizer, load_s, hashes = _worker_setup(payload_path)
    import mlx.core as mx

    output_dir = Path(payload["output_dir"])
    rows = []
    for source in payload["sources"]:
        for objective in payload["objectives"]:
            for seed in payload["seeds"]:
                artifact, metrics = _train_one(
                    model,
                    tokenizer,
                    source,
                    payload["settings"],
                    objective,
                    seed,
                    output_dir / f"{payload['phase']}-training-journal.jsonl",
                )
                artifact_name = f"prefix-{source['source_id']}-{objective}-seed-{seed}.safetensors"
                artifact_path = output_dir / artifact_name
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                mx.save_safetensors(
                    str(artifact_path),
                    {"weights": artifact.weights},
                    metadata={
                        "source_id": source["source_id"],
                        "revision": source["revision"],
                        "objective": objective,
                        "seed": str(seed),
                    },
                )
                metrics.update(
                    {
                        "source_id": source["source_id"],
                        "doc_id": source["doc_id"],
                        "revision": source["revision"],
                        "artifact": artifact_path.name,
                        "artifact_bytes": artifact_path.stat().st_size,
                        "artifact_sha256": sha256_file(artifact_path),
                    }
                )
                rows.append(metrics)
                _write_json(
                    output_dir / f"{payload['phase']}-training-progress.json", {"rows": rows}
                )
    post_hashes = _model_file_hashes(
        Path(os.environ["LIVING_CONTEXT_MODEL_PATH"]), payload["model_files"]
    )
    write_sealed_json(
        output_dir / payload["result_name"],
        {
            "schema_version": 1,
            "phase": payload["phase"],
            "protocol_sha256": payload["protocol_sha256"],
            "snapshot": payload.get("snapshot"),
            "model_load_s": load_s,
            "model_files_before": hashes,
            "model_files_after": post_hashes,
            "rows": rows,
        },
    )


def _answer_worker(payload_path: Path) -> None:
    payload, model, tokenizer, load_s, hashes = _worker_setup(payload_path)
    import mlx.core as mx

    registry = RevisionCacheRegistry(clone=copy.deepcopy)
    rows = []
    questions = payload["questions"]
    sources = payload["sources"]
    index_started = time.perf_counter()
    index = LexicalIndex(sources)
    index_build_s = time.perf_counter() - index_started
    artifacts = payload.get("artifacts", [])
    for question in questions:
        retrieval_started = time.perf_counter()
        retrieved = index.retrieve(question["question"], limit=payload["retrieval_limit"])
        retrieval_s = time.perf_counter() - retrieval_started
        if not retrieved:
            raise StudyValidationError("lexical retrieval returned no source")
        documents = [source["content"] for source in retrieved]
        bare_output, bare_metrics = _generate(
            model,
            tokenizer,
            _chat_tokens(tokenizer, question["question"]),
            payload["max_output_tokens"],
        )
        # Single-document development/calibration cache keys are revision-specific.
        key = CacheKey(
            payload["model_digest"],
            payload["tokenizer_digest"],
            hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "+".join(item["doc_id"] for item in retrieved),
            "+".join(item["revision"] for item in retrieved),
            hashlib.sha256("\n\n".join(documents).encode()).hexdigest(),
        )
        text = _cache_check_and_generate(
            model,
            tokenizer,
            documents,
            question["question"],
            registry,
            key,
            payload["max_output_tokens"],
            payload["cache_logit_atol"],
        )
        rows.append(
            {
                "question_id": question["question_id"],
                "arm": "bare",
                "output": bare_output,
                "metrics": bare_metrics,
                "retrieved_source_ids": [source["source_id"] for source in retrieved],
                "retrieval_s": retrieval_s,
            }
        )
        rows.append(
            {
                "question_id": question["question_id"],
                "arm": "text_fresh",
                "output": text["fresh_output"],
                "metrics": text["fresh_metrics"],
                "retrieved_source_ids": [source["source_id"] for source in retrieved],
                "retrieval_s": retrieval_s,
            }
        )
        rows.append(
            {
                "question_id": question["question_id"],
                "arm": "text_cached",
                "output": text["cached_output"],
                "metrics": text["cached_metrics"],
                "retrieved_source_ids": [source["source_id"] for source in retrieved],
                "retrieval_s": retrieval_s,
                "cache": {
                    key: value
                    for key, value in text.items()
                    if key
                    not in {"fresh_output", "cached_output", "fresh_metrics", "cached_metrics"}
                },
            }
        )
        artifact_index = {
            (row["source_id"], row["objective"], row["seed"]): row for row in artifacts
        }
        methods = sorted({(row["objective"], row["seed"]) for row in artifacts})
        for objective, seed in methods:
            selected_specs = []
            for source in retrieved:
                spec = artifact_index.get((source["source_id"], objective, seed))
                if spec is None:
                    raise StudyValidationError("no active-revision prefix for retrieved source")
                selected_specs.append(spec)
            weights = []
            for artifact_spec in selected_specs:
                artifact_path = Path(payload["artifact_dir"]) / artifact_spec["artifact"]
                if sha256_file(artifact_path) != artifact_spec["artifact_sha256"]:
                    raise StudyValidationError("prefix artifact hash mismatch")
                arrays = cast(dict[str, Any], mx.load(str(artifact_path), format="safetensors"))
                weights.append(arrays["weights"])
            composed_weights = weights[0] if len(weights) == 1 else mx.concatenate(weights, axis=0)
            output, metrics = _generate_prefix(
                model,
                tokenizer,
                _chat_tokens(tokenizer, question["question"]),
                composed_weights,
                payload["max_output_tokens"],
            )
            rows.append(
                {
                    "question_id": question["question_id"],
                    "arm": "prefix",
                    "objective": objective,
                    "seed": seed,
                    "artifact_source_revisions": [
                        {
                            "source_id": spec["source_id"],
                            "doc_id": spec["doc_id"],
                            "revision": spec["revision"],
                        }
                        for spec in selected_specs
                    ],
                    "output": output,
                    "metrics": metrics,
                    "retrieved_source_ids": [source["source_id"] for source in retrieved],
                    "retrieval_s": retrieval_s,
                }
            )
    post_hashes = _model_file_hashes(
        Path(os.environ["LIVING_CONTEXT_MODEL_PATH"]), payload["model_files"]
    )
    write_sealed_json(
        Path(payload["output_path"]),
        {
            "schema_version": 1,
            "phase": payload["phase"],
            "protocol_sha256": payload["protocol_sha256"],
            "training_receipt_sha256": payload.get("training_receipt_sha256"),
            "model_load_s": load_s,
            "model_files_before": hashes,
            "model_files_after": post_hashes,
            "index_build_s": index_build_s,
            "index_storage_bytes": index.storage_bytes,
            "rows": rows,
        },
    )


def _worker_environment(model_path: Path) -> dict[str, str]:
    os.environ["LIVING_CONTEXT_MODEL_PATH"] = str(model_path)
    return {}


def _model_digests(protocol: dict[str, Any]) -> tuple[str, str]:
    files = protocol["model"]["files"]
    return files["model.safetensors"]["sha256"], files["tokenizer.json"]["sha256"]


def _base_payload(protocol: dict[str, Any]) -> dict[str, Any]:
    model_digest, tokenizer_digest = _model_digests(protocol)
    return {
        "protocol_sha256": hashlib.sha256(canonical_json(protocol).encode()).hexdigest(),
        "model_files": protocol["model"]["files"],
        "model_digest": model_digest,
        "tokenizer_digest": tokenizer_digest,
        "max_output_tokens": protocol["generation"]["max_output_tokens"],
        "retrieval_limit": protocol["retrieval"]["limit"],
        "cache_logit_atol": protocol["retrieval"]["cache_logit_atol"],
    }


def _protocol_digest(protocol: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(protocol).encode()).hexdigest()


def _read_gate(path: Path, phase: str, protocol: dict[str, Any]) -> dict[str, Any]:
    gate = read_sealed_json(path, path.with_suffix(path.suffix + ".seal.json"))
    if (
        gate.get("phase") != phase
        or gate.get("protocol_sha256") != _protocol_digest(protocol)
        or not gate.get("passed")
    ):
        raise StudyValidationError(f"valid {phase} gate is required")
    score_path = path.with_name(f"{phase}-scores.json")
    if sha256_file(score_path) != gate.get("score_sha256"):
        raise StudyValidationError(f"{phase} gate score binding mismatch")
    return gate


def _validated_training(
    output_dir: Path,
    phase: str,
    protocol: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    objectives: list[str],
) -> dict[str, Any]:
    path = output_dir / f"{phase}-training-results.json"
    training = read_sealed_json(path, path.with_suffix(path.suffix + ".seal.json"))
    if training.get("phase") != phase or training.get("protocol_sha256") != _protocol_digest(
        protocol
    ):
        raise StudyValidationError("training receipt is not bound to this protocol")
    source_ids = (
        [protocol["phases"]["development"]["source_id"]]
        if phase == "development"
        else protocol["phases"]["final"]["training_source_ids"]
    )
    expected = {
        (source_id, objective, seed)
        for source_id in source_ids
        for objective in objectives
        for seed in protocol["training"]["seeds"]
    }
    observed: set[tuple[str, str, int]] = set()
    for row in training.get("rows", []):
        identity = (row["source_id"], row["objective"], row["seed"])
        if identity in observed or identity not in expected:
            raise StudyValidationError("duplicate or undeclared training artifact")
        observed.add(identity)
        source = sources[row["source_id"]]
        if row["doc_id"] != source["doc_id"] or row["revision"] != source["revision"]:
            raise StudyValidationError("training artifact source binding mismatch")
        if row["steps_completed"] != protocol["training"]["steps"]:
            raise StudyValidationError("training artifact has incomplete steps")
        artifact_path = output_dir / row["artifact"]
        if artifact_path.resolve().parent != output_dir.resolve():
            raise StudyValidationError("training artifact escapes result directory")
        if (
            artifact_path.stat().st_size != row["artifact_bytes"]
            or sha256_file(artifact_path) != row["artifact_sha256"]
        ):
            raise StudyValidationError("training artifact hash or size mismatch")
    if observed != expected:
        raise StudyValidationError("training artifact inventory incomplete")
    return training


def _supervised_command(args: argparse.Namespace) -> None:
    protocol, snapshots = _protocol(args.protocol)
    paths = protocol["data"]
    sources = parse_sources(
        parse_json_bytes(snapshots[paths["sources"]], source="sources snapshot")
    )
    questions: dict[str, dict[str, Any]] = {}
    if args.command not in {"train-dev", "train-final"}:
        questions = parse_questions(
            parse_json_bytes(snapshots[paths["questions"]], source="questions snapshot")
        )
    model_path = args.model_path.resolve()
    _model_file_hashes(model_path, protocol["model"]["files"])
    _worker_environment(model_path)
    output_dir = args.output_dir.resolve()
    limits = protocol["limits"]
    base = _base_payload(protocol)
    if args.command == "calibrate":
        source = sources[protocol["phases"]["calibration"]["source_id"]]
        invalidation_source = sources[protocol["phases"]["calibration"]["invalidation_source_id"]]
        selected_questions = [row for row in questions.values() if row["split"] == "calibration"]
        output_path = output_dir / "calibration-raw.json"
        payload = {
            **base,
            "source": source,
            "invalidation_source": invalidation_source,
            "questions": selected_questions,
            "output_path": str(output_path),
        }
        _supervise(
            ["_calibration-worker"],
            payload=payload,
            output_dir=output_dir,
            timeout_s=limits["calibration_elapsed_s"],
            rss_limit_bytes=limits["process_rss_bytes"],
            total_budget_s=limits["total_model_worker_elapsed_s"],
        )
    elif args.command == "train-dev":
        _read_gate(output_dir / "calibration-gate.json", "calibration", protocol)
        source = sources[protocol["phases"]["development"]["source_id"]]
        payload = {
            **base,
            "phase": "development",
            "sources": [source],
            "objectives": protocol["training"]["objectives"],
            "seeds": protocol["training"]["seeds"],
            "settings": {
                **protocol["training"],
                "mlx_active_limit_bytes": limits["mlx_active_bytes"],
                "rss_limit_bytes": limits["process_rss_bytes"],
                "worker_elapsed_limit_s": limits["development_elapsed_s"],
            },
            "output_dir": str(output_dir),
            "result_name": "development-training-results.json",
        }
        _supervise(
            ["_train-worker"],
            payload=payload,
            output_dir=output_dir,
            timeout_s=limits["development_elapsed_s"],
            rss_limit_bytes=limits["process_rss_bytes"],
            total_budget_s=limits["total_model_worker_elapsed_s"],
        )
    elif args.command == "answer-dev":
        _read_gate(output_dir / "calibration-gate.json", "calibration", protocol)
        training = _validated_training(
            output_dir, "development", protocol, sources, protocol["training"]["objectives"]
        )
        selected_questions = [row for row in questions.values() if row["split"] == "development"]
        source_id = protocol["phases"]["development"]["source_id"]
        payload = {
            **base,
            "phase": "development",
            "snapshot": "development",
            "training_receipt_sha256": sha256_file(
                output_dir / "development-training-results.json"
            ),
            "sources": [sources[source_id]],
            "questions": selected_questions,
            "artifacts": training["rows"],
            "artifact_dir": str(output_dir),
            "output_path": str(output_dir / "development-raw.json"),
        }
        elapsed = parse_json_bytes(
            (output_dir / "_train-worker-supervisor.json").read_bytes(),
            source="training supervisor",
        )["elapsed_s"]
        remaining = limits["development_elapsed_s"] - elapsed
        if remaining <= 0:
            raise RuntimeError("development elapsed budget exhausted before answering")
        _supervise(
            ["_answer-worker"],
            payload=payload,
            output_dir=output_dir,
            timeout_s=remaining,
            rss_limit_bytes=limits["process_rss_bytes"],
            total_budget_s=limits["total_model_worker_elapsed_s"],
        )
    elif args.command == "train-final":
        gate = _read_gate(output_dir / "development-gate.json", "development", protocol)
        if not gate.get("selected_objective"):
            raise StudyValidationError("development gate did not permit the final phase")
        selected_sources = [
            sources[source_id] for source_id in protocol["phases"]["final"]["training_source_ids"]
        ]
        payload = {
            **base,
            "phase": "final",
            "sources": selected_sources,
            "objectives": [gate["selected_objective"]],
            "seeds": protocol["training"]["seeds"],
            "settings": {
                **protocol["training"],
                "mlx_active_limit_bytes": limits["mlx_active_bytes"],
                "rss_limit_bytes": limits["process_rss_bytes"],
                "worker_elapsed_limit_s": limits["final_elapsed_s"],
            },
            "output_dir": str(output_dir),
            "result_name": "final-training-results.json",
        }
        _supervise(
            ["_train-worker"],
            payload=payload,
            output_dir=output_dir,
            timeout_s=limits["final_elapsed_s"],
            rss_limit_bytes=limits["process_rss_bytes"],
            total_budget_s=limits["total_model_worker_elapsed_s"],
            receipt_name="final-train",
        )
    elif args.command in {"answer-final-initial", "answer-final-revised"}:
        snapshot = args.command.removeprefix("answer-final-")
        gate = _read_gate(output_dir / "development-gate.json", "development", protocol)
        training = _validated_training(
            output_dir, "final", protocol, sources, [gate["selected_objective"]]
        )
        source_ids = protocol["phases"]["final"][f"{snapshot}_source_ids"]
        selected_questions = [
            row
            for row in questions.values()
            if row["split"] == "final" and row["snapshot"] == snapshot
        ]
        active_artifacts = [row for row in training["rows"] if row["source_id"] in source_ids]
        payload = {
            **base,
            "phase": "final",
            "snapshot": snapshot,
            "training_receipt_sha256": sha256_file(output_dir / "final-training-results.json"),
            "sources": [sources[source_id] for source_id in source_ids],
            "questions": selected_questions,
            "artifacts": active_artifacts,
            "artifact_dir": str(output_dir),
            "output_path": str(output_dir / f"final-{snapshot}-raw.json"),
        }
        prior_receipts = [output_dir / "final-train-supervisor.json"]
        if snapshot == "revised":
            prior_receipts.append(output_dir / "final-initial-supervisor.json")
        used = sum(
            parse_json_bytes(path.read_bytes(), source=path.name)["elapsed_s"]
            for path in prior_receipts
        )
        remaining = limits["final_elapsed_s"] - used
        if remaining <= 0:
            raise RuntimeError("final elapsed budget exhausted before answering")
        _supervise(
            ["_answer-worker"],
            payload=payload,
            output_dir=output_dir,
            timeout_s=remaining,
            rss_limit_bytes=limits["process_rss_bytes"],
            total_budget_s=limits["total_model_worker_elapsed_s"],
            receipt_name=f"final-{snapshot}",
        )
    else:
        raise AssertionError(args.command)


def _score_command(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    protocol, snapshots = _protocol(args.protocol)
    sources, questions, labels = _small_inputs(protocol, snapshots)
    raw = read_sealed_json(args.raw, args.raw.with_suffix(args.raw.suffix + ".seal.json"))
    if raw.get("protocol_sha256") != _protocol_digest(protocol):
        raise StudyValidationError("raw output is not bound to this protocol")
    split = raw["phase"]
    raw_rows = raw["rows"]
    if split == "calibration":
        expanded = []
        for row in raw_rows:
            common = {"question_id": row["question_id"]}
            expanded.extend(
                [
                    {
                        **common,
                        "arm": "bare",
                        "output": row["bare_output"],
                        "metrics": row["bare_metrics"],
                    },
                    {
                        **common,
                        "arm": "text_fresh",
                        "output": row["fresh_output"],
                        "metrics": row["fresh_metrics"],
                    },
                    {
                        **common,
                        "arm": "text_cached",
                        "output": row["cached_output"],
                        "metrics": row["cached_metrics"],
                    },
                ]
            )
        raw_rows = expanded
    rows = []
    seen: set[tuple[Any, ...]] = set()
    expected_questions = {
        key
        for key, row in questions.items()
        if row["split"] == split
        and (raw.get("snapshot") is None or row["snapshot"] == raw["snapshot"])
    }
    if split == "calibration":
        methods: list[tuple[str | None, int | None]] = []
    elif split == "development":
        _validated_training(
            args.raw.parent, "development", protocol, sources, protocol["training"]["objectives"]
        )
        methods = [
            (objective, seed)
            for objective in protocol["training"]["objectives"]
            for seed in protocol["training"]["seeds"]
        ]
        if raw.get("training_receipt_sha256") != sha256_file(
            args.raw.parent / "development-training-results.json"
        ):
            raise StudyValidationError("development raw/training binding mismatch")
    elif split == "final":
        gate = _read_gate(args.raw.parent / "development-gate.json", "development", protocol)
        _validated_training(
            args.raw.parent, "final", protocol, sources, [gate["selected_objective"]]
        )
        methods = [(gate["selected_objective"], seed) for seed in protocol["training"]["seeds"]]
        if raw.get("training_receipt_sha256") != sha256_file(
            args.raw.parent / "final-training-results.json"
        ):
            raise StudyValidationError("final raw/training binding mismatch")
    else:
        raise StudyValidationError("unknown raw phase")
    expected_identities: set[tuple[Any, ...]] = {
        (question_id, arm, None, None)
        for question_id in expected_questions
        for arm in ("bare", "text_fresh", "text_cached")
    }
    expected_identities.update(
        (question_id, "prefix", objective, seed)
        for question_id in expected_questions
        for objective, seed in methods
    )
    for row in raw_rows:
        question_id = row["question_id"]
        if question_id not in expected_questions:
            raise StudyValidationError("raw output split mismatch")
        identity = (question_id, row["arm"], row.get("objective"), row.get("seed"))
        if identity in seen:
            raise StudyValidationError("duplicate raw output row")
        seen.add(identity)
        if identity not in expected_identities:
            raise StudyValidationError("undeclared raw output row")
        if row["arm"] == "prefix":
            retrieved_ids = row.get("retrieved_source_ids")
            bound = row.get("artifact_source_revisions")
            if not isinstance(retrieved_ids, list) or not isinstance(bound, list):
                raise StudyValidationError("prefix retrieval/artifact binding missing")
            expected_bound = [
                {
                    "source_id": source_id,
                    "doc_id": sources[source_id]["doc_id"],
                    "revision": sources[source_id]["revision"],
                }
                for source_id in retrieved_ids
            ]
            if bound != expected_bound:
                raise StudyValidationError("prefix artifact binding differs from retrieved sources")
        score = score_exact_json(row["output"], labels[question_id])
        relevance = set(
            row.get("retrieved_source_ids", questions[question_id]["source_ids"])
        ) >= set(questions[question_id]["source_ids"])
        rows.append({**row, "score": asdict(score), "retrieval_contains_audit_sources": relevance})
    if seen != expected_identities:
        raise StudyValidationError("raw output inventory is incomplete")
    result = {
        "schema_version": 1,
        "phase": split,
        "snapshot": raw.get("snapshot"),
        "protocol_sha256": _protocol_digest(protocol),
        "raw_file": args.raw.name,
        "raw_sha256": sha256_file(args.raw),
        "raw_seal_sha256": sha256_file(args.raw.with_suffix(args.raw.suffix + ".seal.json")),
        "scoring_s": time.perf_counter() - started,
        "rows": rows,
    }
    write_sealed_json(args.output, result)


def _gate_command(args: argparse.Namespace) -> None:
    protocol, snapshots = _protocol(args.protocol)
    sources, _questions, _labels = _small_inputs(protocol, snapshots)
    score = read_sealed_json(args.score, args.score.with_suffix(args.score.suffix + ".seal.json"))
    if score.get("protocol_sha256") != _protocol_digest(protocol):
        raise StudyValidationError("score is not bound to this protocol")
    raw_path = args.score.parent / score["raw_file"]
    if (
        sha256_file(raw_path) != score["raw_sha256"]
        or sha256_file(raw_path.with_suffix(raw_path.suffix + ".seal.json"))
        != score["raw_seal_sha256"]
    ):
        raise StudyValidationError("score raw-output binding mismatch")
    if score["phase"] == "calibration":
        text_rows = [row for row in score["rows"] if row["arm"] in {"text_fresh", "text_cached"}]
        passed = len(text_rows) == 4 and all(row["score"]["correct"] for row in text_rows)
        result = {
            "schema_version": 1,
            "phase": "calibration",
            "passed": passed,
            "interpretation_if_failed": "reader_or_schema_gate_failure_not_prefix_evidence",
        }
    elif score["phase"] == "development":
        training = _validated_training(
            args.score.parent,
            "development",
            protocol,
            sources,
            protocol["training"]["objectives"],
        )
        bare = [row for row in score["rows"] if row["arm"] == "bare"]
        text = [row for row in score["rows"] if row["arm"] == "text_cached"]
        prefix = [row for row in score["rows"] if row["arm"] == "prefix"]
        bare_ok = len(bare) == 2 and not any(row["score"]["correct"] for row in bare)
        text_ok = len(text) == 2 and all(
            row["score"]["correct"] and row["retrieval_contains_audit_sources"] for row in text
        )
        by_method: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for row in prefix:
            by_method.setdefault(row["objective"], {}).setdefault(row["seed"], []).append(row)
        methods = []
        for objective in OBJECTIVES:
            seeds = by_method.get(objective, {})
            successful_seeds = sum(
                len(rows) == 2 and all(row["score"]["correct"] for row in rows)
                for rows in seeds.values()
            )
            exact_answers = sum(row["score"]["correct"] for rows in seeds.values() for row in rows)
            serving_s = sum(row["metrics"]["elapsed_s"] for rows in seeds.values() for row in rows)
            training_s = sum(
                row["training_s"] for row in training["rows"] if row["objective"] == objective
            )
            methods.append(
                {
                    "objective": objective,
                    "successful_seeds": successful_seeds,
                    "exact_answers": exact_answers,
                    "serving_s": serving_s,
                    "training_s": training_s,
                    "selection_cost_s": training_s + serving_s,
                    "passes_method_gate": successful_seeds >= 2,
                }
            )
        eligible = [row for row in methods if row["passes_method_gate"]]
        eligible.sort(
            key=lambda row: (-row["exact_answers"], row["selection_cost_s"], row["objective"])
        )
        result = {
            "schema_version": 1,
            "phase": "development",
            "passed": bare_ok and text_ok and bool(eligible),
            "bare_zero_of_two": bare_ok,
            "text_exact_two_of_two": text_ok,
            "methods": methods,
            "selected_objective": eligible[0]["objective"]
            if eligible and bare_ok and text_ok
            else None,
            "stop_larger_study": not (bare_ok and text_ok and bool(eligible)),
        }
    else:
        raise StudyValidationError("gate supports calibration or development only")
    result["protocol_sha256"] = _protocol_digest(protocol)
    result["score_sha256"] = sha256_file(args.score)
    write_sealed_json(args.output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in (
        "calibrate",
        "train-dev",
        "answer-dev",
        "train-final",
        "answer-final-initial",
        "answer-final-revised",
    ):
        command = sub.add_parser(name)
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--model-path", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--protocol", type=Path, required=True)
    score.add_argument("--raw", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    gate = sub.add_parser("gate")
    gate.add_argument("--protocol", type=Path, required=True)
    gate.add_argument("--score", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    for name in ("_calibration-worker", "_train-worker", "_answer-worker"):
        command = sub.add_parser(name)
        command.add_argument("payload", type=Path)
    args = parser.parse_args()
    if args.command in {
        "calibrate",
        "train-dev",
        "answer-dev",
        "train-final",
        "answer-final-initial",
        "answer-final-revised",
    }:
        _supervised_command(args)
    elif args.command == "score":
        _score_command(args)
    elif args.command == "gate":
        _gate_command(args)
    elif args.command == "_calibration-worker":
        _calibration_worker(args.payload)
    elif args.command == "_train-worker":
        _train_worker(args.payload)
    elif args.command == "_answer-worker":
        _answer_worker(args.payload)


if __name__ == "__main__":
    main()
