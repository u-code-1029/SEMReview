import io
import json
import os
import subprocess
import sys

import pytest

from semreview.jobs import JobController


def test_worker_command_error_is_json_protocol(tmp_path):
    payload = tmp_path / "request.json"
    payload.write_text("{}", encoding="utf-8")
    env = dict(os.environ, PYTHONPATH="src")
    result = subprocess.run([sys.executable, "-m", "semreview", "--worker", "unknown",
                             "--request", str(payload)], env=env, capture_output=True, text=True)
    assert result.returncode == 1
    event = json.loads(result.stdout.strip().splitlines()[-1])
    assert event["type"] == "error"
    assert "unknown" in event["message"]


def test_qprocess_scan_roundtrip_and_unicode(qtbot, tmp_path):
    base = tmp_path / "이미지"
    leaf = base / "배치" / "시편"
    leaf.mkdir(parents=True)
    (leaf / "10.tif").write_bytes(b"10")
    (leaf / "2.tif").write_bytes(b"2")
    controller = JobController()
    with qtbot.waitSignal(controller.finished, timeout=20000) as blocker:
        controller.start("scan", {"base": str(base), "profiles": [{"levels": [
            {"include": ["*"], "exclude": []}, {"include": ["*"], "exclude": []}]}]})
    assert controller.busy is False
    result = blocker.args[0]
    assert len(result) == 1
    assert [x["number"] for x in result[0]["images"]] == [2, 10]


def test_qprocess_reports_actionable_error(qtbot):
    controller = JobController()
    with qtbot.waitSignal(controller.failed, timeout=20000) as blocker:
        controller.start("unknown", {})
    assert "unknown" in blocker.args[0]
    assert not controller.busy


def test_qprocess_cancellation(qtbot, tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    for index in range(30):
        leaf = base / str(index)
        leaf.mkdir()
        (leaf / "1.tif").write_bytes(b"x" * 1_000_000)
    controller = JobController()
    with qtbot.waitSignal(controller.cancelled, timeout=20000):
        controller.start("scan", {"base": str(base), "profiles": [{"levels": [{"include": ["*"]}]}]})
        controller.cancel()
    assert not controller.busy


def test_worker_file_channel_without_console(tmp_path, monkeypatch):
    from semreview.worker import run_worker

    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert run_worker("unknown", str(request), None) == 1
    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event["type"] == "error"


class BrokenConsole:
    def write(self, text):
        raise BrokenPipeError("The inherited console pipe has closed")

    def flush(self):
        raise BrokenPipeError("The inherited console pipe has closed")


@pytest.mark.parametrize("console", ["cp1252", "broken_pipe"])
def test_worker_scan_survives_unusable_stdout(tmp_path, monkeypatch, console):
    from semreview.worker import run_worker

    base = tmp_path / "한글 이미지"
    leaf = base / "촬영" / "패턴"
    leaf.mkdir(parents=True)
    (leaf / "10.tif").write_bytes(b"10")
    (leaf / "2.tif").write_bytes(b"2")
    request_dir = tmp_path / "한글 요청"
    request_dir.mkdir()
    request = request_dir / "request.json"
    request.write_text(json.dumps({"base": str(base), "profiles": [{"levels": [
        {"include": ["*"]}, {"include": ["*"]}]}]}, ensure_ascii=False), encoding="utf-8")
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict") if console == "cp1252" else BrokenConsole()
    monkeypatch.setattr(sys, "stdout", stdout)

    assert run_worker("scan", str(request), None) == 0

    events = [json.loads(line) for line in (request_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["progress", "result"]
    result = json.loads((request_dir / "result.json").read_text(encoding="utf-8"))
    assert result[0]["relative_path"] == "촬영/패턴"
    assert [image["number"] for image in result[0]["images"]] == [2, 10]


@pytest.mark.parametrize("console", ["cp1252", "broken_pipe"])
def test_worker_error_is_persisted_when_traceback_console_fails(tmp_path, monkeypatch, console):
    from semreview.worker import run_worker

    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict") if console == "cp1252" else BrokenConsole()
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "stdout", BrokenConsole())

    assert run_worker("지원하지 않는 명령", str(request), None) == 1

    event = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert event == {"type": "error", "message": "지원하지 않는 작업: 지원하지 않는 명령"}
    assert not (tmp_path / "result.json").exists()
