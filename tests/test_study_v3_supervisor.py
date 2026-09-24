"""Model-free load-bearing checks for the frozen v3 supervisor."""

import json
import sys
import time
from types import SimpleNamespace

import psutil

from experiments.replacement_study_v2 import SYSTEM_PROMPT
from experiments.study_v3_qlora import (
    decide_status,
    memory_reports,
    supervise,
    timed_cached_text,
    training_data,
)


def test_memory_report_whole_snapshots_and_final_report(tmp_path, monkeypatch) -> None:
    class FakeObservedProcess:
        def __init__(self, _pid):
            pass

        def children(self, recursive=False):
            return []

        def memory_info(self):
            return SimpleNamespace(rss=1024)

    monkeypatch.setattr(psutil, "Process", FakeObservedProcess)
    assert memory_reports("Peak mem 0.7 G") == (0, 0.0)
    assert memory_reports("Peak mem 0.7 GB\n") == (1, 0.7)
    assert memory_reports("Peak mem 0.7 GB\nMLX_PEAK_GB=1.3\n") == (2, 1.3)
    missing = supervise(
        [sys.executable, "-c", "print('done')"], tmp_path / "missing.log", time.monotonic() + 10, {}
    )
    assert missing["status"] == "MLX_MEMORY_REPORT_MISSING"
    final = supervise(
        [sys.executable, "-c", "print('MLX_PEAK_GB=0.4')"], tmp_path / "final.log", time.monotonic() + 10, {}
    )
    assert final["status"] == "completed" and final["memory_report_matches"] == 1


def test_inaccessible_rss_fails_closed(tmp_path, monkeypatch) -> None:
    class InaccessibleProcess:
        def __init__(self, _pid):
            pass

        def children(self, recursive=False):
            raise PermissionError("process table denied")

    monkeypatch.setattr(psutil, "Process", InaccessibleProcess)
    result = supervise(
        [sys.executable, "-c", "print('MLX_PEAK_GB=0.4')"],
        tmp_path / "denied.log", time.monotonic() + 10, {},
    )
    assert result["status"] == "RSS_METRIC_UNAVAILABLE"


def test_worker_failure_overrides_valid_raw_output(tmp_path) -> None:
    (tmp_path / "control-raw.json").write_text(json.dumps({"score": {"exact": True}}))
    (tmp_path / "pilot-raw.json").write_text(json.dumps({"rows": [
        {"arms": {arm: {"score": {"exact": True}} for arm in ("adapter", "bare", "text")}}
        for _ in range(8)
    ]}))
    receipts = [{"status": "completed"} for _ in range(6)] + [{"status": "MLX_PEAK_LIMIT"}]
    assert decide_status(tmp_path, receipts) == "WORKER_MLX_PEAK_LIMIT"
    assert not (tmp_path / "scores.json").exists()


def test_cached_text_timer_is_isolated_and_train_chat_context_matches_reader(tmp_path) -> None:
    result, elapsed = timed_cached_text(lambda: {"cached_output": "x"})
    assert result["cached_output"] == "x"
    time.sleep(0.02)
    assert elapsed < 0.02
    training_data({"question": "Q", "answer": "A"}, tmp_path / "data")
    row = json.loads((tmp_path / "data/train.jsonl").read_text())
    assert row["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert row["messages"][1] == {"role": "user", "content": "Q"}
    assert row["messages"][2] == {"role": "assistant", "content": '{"answer":"A"}'}
