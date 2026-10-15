import json
import os
import subprocess
import sys

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
