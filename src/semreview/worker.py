"""Small JSON-lines worker protocol; only explicit apply/restore jobs mutate source paths."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Callable


def dispatch(command: str, payload: dict, progress: Callable, cancelled: Callable):
    from . import core

    if command == "scan":
        return core.scan(payload["base"], payload["profiles"], progress=progress, cancelled=cancelled)
    if command == "rules":
        from .imaging import rule_analysis
        return rule_analysis(payload["sequence"], progress=progress, cancelled=cancelled)
    if command == "train":
        from .ml import train
        from .project import Project, sequence_signature

        # Reject stale annotations even when called without the desktop UI.
        with Project(payload["project_dir"]) as project:
            saved = project.annotations()
        for sequence in payload["sequences"]:
            annotation = saved.get(sequence["relative_path"])
            if annotation and annotation["sequence_signature"] != sequence_signature(sequence):
                raise ValueError(f"이미지가 변경되어 라벨 재확인이 필요합니다: {sequence['relative_path']}")
        return train(payload["project_dir"], payload["sequences"], payload["annotations"],
                     payload["classes"], payload["model_name"], config=payload.get("config"),
                     progress=progress, cancelled=cancelled)
    if command == "infer":
        from .ml import infer
        return infer(payload["model_path"], payload["sequences"], payload["project_dir"],
                     progress=progress, cancelled=cancelled)
    if command == "apply":
        return core.apply_plans(payload["plans"], payload["project_dir"], progress=progress, cancelled=cancelled)
    if command == "resume":
        return core.resume_run(payload["project_dir"], payload["run_id"], progress=progress, cancelled=cancelled)
    if command == "restore":
        return core.restore_run(payload["project_dir"], payload["run_id"], progress=progress, cancelled=cancelled)
    if command == "inspect_model":
        from .ml import inspect_model
        return inspect_model(payload["model_path"])
    raise ValueError(f"지원하지 않는 작업: {command}")


def run_worker(command: str, request_path: str, cancel_file: str | None) -> int:
    def emit(event: dict):
        line = json.dumps(event, ensure_ascii=False, allow_nan=False)
        # A Windows windowed executable has no sys.stdout; the file channel works
        # identically there and for console/frozen/macOS launches.
        with Path(request_path).with_name("events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
        if sys.stdout is not None:
            print(line, flush=True)

    def cancelled() -> bool:
        return bool(cancel_file and Path(cancel_file).exists())

    def progress(data: dict):
        emit({"type": "progress", "data": data})

    try:
        payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("작업 요청은 JSON 객체여야 합니다.")
        result = dispatch(command, payload, progress, cancelled)
        # Keep large manifests out of the event pipe.
        result_path = Path(request_path).with_name("result.json")
        result_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        emit({"type": "result", "path": str(result_path), "cancelled": cancelled()})
        return 0
    except Exception as error:
        if sys.stderr is not None:
            traceback.print_exc(file=sys.stderr)
        emit({"type": "cancelled" if cancelled() else "error", "message": str(error)})
        return 2 if cancelled() else 1
