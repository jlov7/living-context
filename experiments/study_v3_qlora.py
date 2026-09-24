"""Frozen local QLoRA pilot supervisor. No model access occurs without --run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "research/study-v3"
LIMIT_SECONDS = 900
LIMIT_RSS = 8 * 1024**3
LIMIT_MLX_GB = 2.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(model_path: Path, output: Path) -> dict[str, Any]:
    protocol = json.loads((STUDY / "protocol.json").read_text())
    if output.exists():
        raise ValueError("new output directory must not exist")
    for relative, expected in protocol["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"source input mismatch: {relative}")
    from mlx_lm import lora  # model import is deliberately confined to explicit preflight

    if digest(Path(lora.__file__)) != protocol["installed_lora_sha256"]:
        raise ValueError("installed MLX-LM LoRA trainer differs")
    v2 = json.loads((ROOT / "research/study-v2/protocol.json").read_text())
    if v2["model"]["snapshot_revision"] != protocol["model_snapshot_revision"]:
        raise ValueError("model revision mismatch")
    for name, metadata in v2["model"]["files"].items():
        path = model_path / name
        if path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]:
            raise ValueError(f"model file mismatch: {name}")
    return {"status": "PREFLIGHT_PASS", "source_files": len(protocol["source_sha256"]), "model_files": len(v2["model"]["files"]), "model_revision": protocol["model_snapshot_revision"]}


def supervise(command: list[str], log_path: Path, deadline: float, env: dict[str, str]) -> dict[str, Any]:
    import psutil

    started = time.monotonic()
    peak_rss = 0
    peak_mlx_gb = 0.0
    report_count = 0
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        reason = "completed"
        while process.poll() is None:
            if time.monotonic() > deadline:
                reason = "TIME_LIMIT"
            try:
                root = psutil.Process(process.pid)
                children = root.children(recursive=True)
                rss = sum(p.memory_info().rss for p in [root, *children])
                peak_rss = max(peak_rss, rss)
                if rss > LIMIT_RSS:
                    reason = "RSS_LIMIT"
            except psutil.NoSuchProcess:
                pass
            except (psutil.AccessDenied, PermissionError):
                reason = "RSS_METRIC_UNAVAILABLE"
            report_count, peak_mlx_gb = memory_reports(log_path.read_text(errors="replace"))
            if peak_mlx_gb > LIMIT_MLX_GB:
                reason = "MLX_PEAK_LIMIT"
            if reason != "completed":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.25)
        code = process.wait()
    if code != 0 and reason == "completed":
        reason = "WORKER_FAILED"
    report_count, peak_mlx_gb = memory_reports(log_path.read_text(errors="replace"))
    if peak_mlx_gb > LIMIT_MLX_GB:
        reason = "MLX_PEAK_LIMIT"
    if report_count == 0 and reason == "completed":
        reason = "MLX_MEMORY_REPORT_MISSING"
    return {"status": reason, "exit_code": code, "elapsed_s": round(time.monotonic() - started, 3), "peak_rss_bytes": peak_rss, "reported_mlx_peak_gb": peak_mlx_gb, "memory_report_matches": report_count}


def memory_reports(output: str) -> tuple[int, float]:
    """Parse whole log snapshots, including a report split across prior polls."""
    matches = re.findall(r"Peak mem ([0-9.]+) GB|MLX_PEAK_GB=([0-9.]+)\r?\n", output)
    values = [float(training or inference) for training, inference in matches]
    return len(values), max(values, default=0.0)


def training_data(record: dict[str, str], directory: Path) -> None:
    from experiments.replacement_study_v2 import SYSTEM_PROMPT

    directory.mkdir(parents=True)
    row = {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["question"]},
        {"role": "assistant", "content": json.dumps({"answer": record["answer"]}, separators=(",", ":"))},
    ]}
    data = json.dumps(row, separators=(",", ":")) + "\n"
    (directory / "train.jsonl").write_text(data)
    (directory / "valid.jsonl").write_text(data)


def score(text: str, expected: str) -> dict[str, Any]:
    from living_context.replacement_study import score_exact_json

    result = score_exact_json(text, expected)
    return {"exact": result.correct, "answer": result.parsed_answer, "error": result.error}


def timed_cached_text(call: Any) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    result = call()
    return result, round(time.monotonic() - started, 3)


def infer_one(model: Any, tokenizer: Any, record: dict[str, str], question: str) -> dict[str, Any]:
    from experiments.replacement_study_v2 import _chat_tokens, _generate

    started = time.monotonic()
    output, metrics = _generate(model, tokenizer, _chat_tokens(tokenizer, question), 32)
    return {"raw": output, "score": score(output, record["answer"]), "metrics": metrics, "elapsed_s": round(time.monotonic() - started, 3)}


def inference_worker(model_path: Path, output: Path, phase: str) -> None:
    from mlx_lm import load

    training = json.loads((STUDY / "training.json").read_text())
    if phase == "control":
        record = training["positive_control"]
        model, tokenizer = cast(tuple[Any, Any], load(str(model_path), adapter_path=str(output / "adapters" / record["doc_id"])))
        model.eval()
        result = infer_one(model, tokenizer, record, record["question"])
        (output / "control-raw.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        import mlx.core as mx
        print(f"MLX_PEAK_GB={mx.get_peak_memory() / 1e9:.3f}", flush=True)
        return
    from experiments.replacement_study_v2 import _cache_check_and_generate
    from living_context.replacement_study import CacheKey, RevisionCacheRegistry

    questions = json.loads((STUDY / "evaluation.json").read_text())
    v2 = json.loads((ROOT / "research/study-v2/protocol.json").read_text())
    base, tokenizer = cast(tuple[Any, Any], load(str(model_path)))
    base.eval()
    rows = []
    for record in training["pilot"]:
        adapted, adapted_tokenizer = cast(tuple[Any, Any], load(str(model_path), adapter_path=str(output / "adapters" / record["doc_id"])))
        adapted.eval()
        registry = RevisionCacheRegistry()
        key = CacheKey(
            model_sha256=v2["model"]["files"]["model.safetensors"]["sha256"],
            tokenizer_sha256=v2["model"]["files"]["tokenizer.json"]["sha256"],
            prompt_sha256=hashlib.sha256(b"study-v3-system-prompt").hexdigest(),
            doc_id=record["doc_id"], revision="r1",
            content_sha256=hashlib.sha256(record["content"].encode()).hexdigest(),
        )
        for index, question in enumerate(questions[record["doc_id"]]):
            cached, validation_elapsed = timed_cached_text(
                partial(
                    _cache_check_and_generate,
                    base, tokenizer, [record["content"]], question, registry, key, 32, 0.1,
                )
            )
            text_answer = cached["cached_output"]
            arms = {
                "adapter": infer_one(adapted, adapted_tokenizer, record, question),
                "bare": infer_one(base, tokenizer, record, question),
                "text": {
                    "raw": text_answer, "score": score(text_answer, record["answer"]),
                    "cache_hit": cached["cache_hit"], "cache_bytes": cached["cache_bytes"],
                    "fresh_cached_parity": True, "metrics": cached["cached_metrics"],
                    "validation_elapsed_s": validation_elapsed,
                    "elapsed_s": cached["cached_metrics"]["elapsed_s"],
                    "cache_build_s": cached["cache_build_s"],
                },
            }
            rows.append({"doc_id": record["doc_id"], "question_index": index, "arms": arms})
        del adapted
    (output / "pilot-raw.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    import mlx.core as mx
    print(f"MLX_PEAK_GB={mx.get_peak_memory() / 1e9:.3f}", flush=True)


def decide_status(output: Path, receipts: list[dict[str, Any]]) -> str:
    worker_failure = next((r["status"] for r in receipts if r["status"] != "completed"), None)
    if worker_failure is not None:
        return f"WORKER_{worker_failure}"
    if not (output / "control-raw.json").exists():
        return "CONTROL_UNAVAILABLE"
    if not json.loads((output / "control-raw.json").read_text())["score"]["exact"]:
        return "CONTROL_FAILED"
    if not (output / "pilot-raw.json").exists() or len(receipts) != 7:
        return "PILOT_UNAVAILABLE"
    rows = json.loads((output / "pilot-raw.json").read_text())["rows"]
    counts = {
        arm: sum(row["arms"][arm]["score"]["exact"] for row in rows)
        for arm in ("adapter", "bare", "text")
    }
    status = (
        "PILOT_GATE_PASS"
        if len(rows) == 8 and counts["adapter"] >= 7 and counts["adapter"] >= counts["text"]
        else "PILOT_GATE_FAIL"
    )
    (output / "scores.json").write_text(
        json.dumps({"status": status, "counts": counts, "denominator": len(rows)}, indent=2, sort_keys=True) + "\n"
    )
    return status


def run(model_path: Path, output: Path) -> None:
    pre = preflight(model_path, output)
    output.mkdir(parents=True)
    (output / "preflight.json").write_text(json.dumps(pre, indent=2, sort_keys=True) + "\n")
    training = json.loads((STUDY / "training.json").read_text())
    settings = json.loads((STUDY / "protocol.json").read_text())
    deadline = time.monotonic() + LIMIT_SECONDS
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)})
    receipts = []
    records = [training["positive_control"], *training["pilot"]]
    for index, record in enumerate(records):
        if index and not json.loads((output / "control-raw.json").read_text())["score"]["exact"]:
            break
        doc_id = record["doc_id"]
        training_data(record, output / "worker-data" / doc_id)
        command = [sys.executable, "-m", "mlx_lm.lora", "--train", "--model", str(model_path), "--data", str(output / "worker-data" / doc_id), "--adapter-path", str(output / "adapters" / doc_id), "--iters", str(settings["control_iters"] if index == 0 else settings["pilot_iters"]), "--config", str(STUDY / "lora.yaml")]
        receipt = supervise(command, output / f"train-{doc_id}.log", deadline, env)
        receipts.append({"phase": "train", "doc_id": doc_id, **receipt})
        (output / "worker-receipts.json").write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
        if receipt["status"] != "completed":
            break
        if index == 0:
            answer = supervise([sys.executable, __file__, "--infer-worker", "control", "--model", str(model_path), "--output", str(output)], output / "control-infer.log", deadline, env)
            receipts.append({"phase": "control-inference", **answer})
            (output / "worker-receipts.json").write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
            if answer["status"] != "completed":
                break
    if len([r for r in receipts if r["phase"] == "train" and r["status"] == "completed"]) == 5:
        answer = supervise([sys.executable, __file__, "--infer-worker", "pilot", "--model", str(model_path), "--output", str(output)], output / "pilot-infer.log", deadline, env)
        receipts.append({"phase": "pilot-inference", **answer})
        (output / "worker-receipts.json").write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
    status = decide_status(output, receipts)
    (output / "status.json").write_text(json.dumps({"status": status, "worker_receipts": len(receipts)}, indent=2, sort_keys=True) + "\n")
    print(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--infer-worker", choices=("control", "pilot"))
    args = parser.parse_args()
    if args.model is None or args.output is None:
        parser.error("--model and --output are required")
    if args.infer_worker:
        inference_worker(args.model, args.output, args.infer_worker)
    elif args.preflight:
        print(json.dumps(preflight(args.model, args.output), sort_keys=True))
    elif args.run:
        run(args.model, args.output)
    else:
        parser.error("choose --preflight or --run")


if __name__ == "__main__":
    main()
