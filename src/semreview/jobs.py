"""Responsive QProcess bridge with cooperative cancellation and isolated requests."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal


class JobController(QObject):
    progress = Signal(dict)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    busyChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: QProcess | None = None
        self._temporary = None
        self._buffer = bytearray()
        self._stderr = ""
        self._terminal_event = None
        self._busy = False
        self.command = ""
        self._event_offset = 0
        self._event_timer = QTimer(self)
        self._event_timer.setInterval(100)
        self._event_timer.timeout.connect(self._read_events)

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def process(self):
        return self._process

    def start(self, command: str, payload: dict) -> None:
        if self.busy:
            raise RuntimeError("현재 작업이 끝난 뒤 다시 실행하세요.")
        self._temporary = tempfile.TemporaryDirectory(prefix="semreview-job-")
        root = Path(self._temporary.name)
        request = root / "request.json"
        try:
            request.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        except Exception:
            self._temporary.cleanup()
            self._temporary = None
            raise
        self._cancel_file = root / "cancel"
        self._event_file = root / "events.jsonl"
        self._event_offset = 0
        self._buffer.clear()
        self._stderr = ""
        self._terminal_event = None
        self.command = command
        process = QProcess(self)
        self._process = process
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONUTF8", "1")
        source_root = str(Path(__file__).resolve().parent.parent)
        env.insert("PYTHONPATH", source_root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        process.setProcessEnvironment(env)
        process.setProgram(sys.executable)
        arguments = [] if getattr(sys, "frozen", False) else ["-m", "semreview"]
        arguments += ["--worker", command, "--request", str(request), "--cancel-file", str(self._cancel_file)]
        process.setArguments(arguments)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.finished.connect(self._on_finished)
        process.errorOccurred.connect(self._on_process_error)
        self._busy = True
        self.busyChanged.emit(True)
        process.start()
        self._event_timer.start()

    def cancel(self) -> None:
        if self.busy:
            self._cancel_file.touch(exist_ok=True)
            self.progress.emit({"stage": "cancelling", "message": "진행 중인 안전한 단계를 마친 후 취소합니다."})

    def _read_stdout(self):
        if self._process is None:
            return
        self._process.readAllStandardOutput()
        self._read_events()

    def _read_events(self):
        if not self._busy or not self._event_file.exists():
            return
        with self._event_file.open("rb") as stream:
            stream.seek(self._event_offset)
            incoming = stream.read()
            self._event_offset = stream.tell()
        self._buffer.extend(incoming)
        while b"\n" in self._buffer:
            raw, _, rest = self._buffer.partition(b"\n")
            self._buffer = bytearray(rest)
            try:
                event = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "progress":
                self.progress.emit(event["data"])
            elif event.get("type") in {"result", "error", "cancelled"}:
                self._terminal_event = event

    def _read_stderr(self):
        if self._process is not None:
            self._stderr = (self._stderr + bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace"))[-12000:]

    def _on_process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            message = self._process.errorString() if self._process else "실행할 수 없습니다."
            self._cleanup()
            self.failed.emit(f"백그라운드 작업을 시작할 수 없습니다: {message}")

    def _on_finished(self, exit_code, exit_status):
        self._read_stdout()
        self._read_stderr()
        event = self._terminal_event
        result = None
        error = None
        was_cancelled = False
        if event and event["type"] == "result" and exit_code == 0:
            try:
                expected = Path(self._temporary.name) / "result.json"
                if Path(event["path"]).resolve() != expected.resolve():
                    raise ValueError("잘못된 결과 파일 경로입니다.")
                result = json.loads(expected.read_text(encoding="utf-8"))
                was_cancelled = event.get("cancelled", False)
            except Exception as exc:
                error = f"작업 결과를 읽을 수 없습니다: {exc}"
        elif event and event["type"] == "cancelled":
            was_cancelled = True
        elif event and event["type"] == "error":
            error = event["message"]
        else:
            error = f"작업 프로세스가 종료되었습니다 (코드 {exit_code}).\n{self._stderr[-3000:]}"
        self._cleanup()
        if error:
            self.failed.emit(error)
        elif result is not None:
            self.finished.emit(result)
        elif was_cancelled:
            self.cancelled.emit()

    def _cleanup(self):
        self._event_timer.stop()
        if self._process is not None:
            self._process.deleteLater()
            self._process = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._busy = False
        self.busyChanged.emit(False)
