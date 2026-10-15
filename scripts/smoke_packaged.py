"""Exercise the frozen app through its file-based worker protocol.

Run on the platform that built the executable. Results, diagnostics, synthetic
images and a genuinely trained model are retained under --output. This checks
packaging and execution; synthetic data cannot establish SEM accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np
import tifffile


WEIGHT_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
CLASS_DEFINITIONS = [
    {"id": f"boundary-{i + 1}", "name": f"사용자 단계 {i + 1}",
     "description": f"합성 패턴의 {i + 1}번째 전환 시작점", "color": "#62d6bb"}
    for i in range(12)
]
CLASSES = [definition["id"] for definition in CLASS_DEFINITIONS]
FRAMES = 16
PROFILES = [{"name": "Synthetic smoke test", "levels": [
    {"include": ["촬영*"], "exclude": []},
    {"include": ["패턴*"], "exclude": []},
]}]


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def weights_file(output: Path, supplied: Path | None) -> Path:
    path = supplied.resolve() if supplied else output / "weights" / "resnet18-f37072fd.pth"
    if not path.is_file():
        if supplied:
            raise FileNotFoundError(f"Pretrained weights not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(WEIGHT_URL, timeout=60) as response, temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if digest(temporary) != WEIGHT_SHA256:
                raise ValueError("Downloaded ResNet18 weights failed SHA256 verification")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(path) != WEIGHT_SHA256:
        raise ValueError(f"ResNet18 weights failed SHA256 verification: {path}")
    return path


def generate_images(output: Path) -> tuple[Path, Path]:
    base = output / "images"
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:128, :192]
    for group in range(3):
        folder = base / f"촬영{group + 1}" / "패턴01"
        folder.mkdir(parents=True)
        for index in range(FRAMES):
            phase = min(index, len(CLASSES))
            shift = index % 3 + group * 2
            waves = np.sin((xx + shift) / 4.0) + np.cos((yy - shift) / 5.0)
            image = waves * (12 + phase * 2) + 40 + phase * 10 + rng.normal(0, 3, xx.shape)
            if phase >= 2:
                image += 12 * np.cos((xx + yy + group * 2) / 8.0)
            tifffile.imwrite(folder / f"{index + 5}.tif", np.clip(image * 257, 0, 65535).astype(np.uint16),
                             photometric="minisblack", compression="deflate")
    # Also check decoding at the actual SEM resolution without expanding the
    # small training test into an accuracy/performance benchmark.
    full_base = output / "full-resolution"
    full_folder = full_base / "촬영1" / "패턴16비트"
    full_folder.mkdir(parents=True)
    yy, xx = np.ogrid[:2048, :3072]
    image = np.clip(30000 + 10000 * np.sin(xx / 31) + 8000 * np.cos(yy / 43), 0, 65535)
    tifffile.imwrite(full_folder / "10.tif", image.astype(np.uint16),
                     photometric="minisblack", compression="deflate")
    return base, full_base


def worker(executable: Path, output: Path, env: dict, name: str, command: str, payload: dict,
           timeout: int = 600):
    directory = output / name
    directory.mkdir()
    request = directory / "request.json"
    save_json(request, payload)
    started = time.monotonic()
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        result = subprocess.run([str(executable), "--worker", command, "--request", str(request)],
                                cwd=directory, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
    events_path = directory / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    errors = [event for event in events if event.get("type") in {"error", "cancelled"}]
    if result.returncode != 0 or errors:
        raise RuntimeError(f"Frozen {command} failed (exit {result.returncode}): {errors}; logs: {directory}")
    if not events or events[-1].get("type") != "result" or events[-1].get("cancelled"):
        raise RuntimeError(f"Frozen {command} did not send a successful result event: {events[-1:]}")
    # Read the expected file ourselves: stdout is absent for Windows GUI builds.
    value = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    seconds = round(time.monotonic() - started, 3)
    print(f"PASS {name}: {seconds}s", flush=True)
    return value, seconds


def gui_startup(executable: Path, output: Path, env: dict) -> dict:
    directory = output / "gui"
    directory.mkdir()
    project = directory / "project"
    # The native executable still loads Qt and constructs MainWindow. Offscreen
    # permits CI sessions without an interactive desktop and avoids user popups.
    gui_env = dict(env, QT_QPA_PLATFORM="offscreen")
    started = time.monotonic()
    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
        process = subprocess.Popen([str(executable), "--project", str(project)], cwd=directory,
                                   env=gui_env, stdout=stdout, stderr=stderr)
        try:
            deadline = started + 45
            while not (project / "project.sqlite").is_file():
                if process.poll() is not None:
                    raise RuntimeError(f"GUI exited early ({process.returncode}); logs: {directory}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"GUI did not initialize its project; logs: {directory}")
                time.sleep(0.2)
            # Give deferred Qt initialization time to fail before accepting it.
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                pass
            else:
                raise RuntimeError(f"GUI exited after initialization ({process.returncode}); logs: {directory}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    seconds = round(time.monotonic() - started, 3)
    print(f"PASS gui-startup: {seconds}s", flush=True)
    return {"seconds": seconds, "qt_platform": "offscreen", "project_created": True,
            "scope": "Process stayed alive after project initialization; visual interaction was not tested."}


def run(args) -> dict:
    executable = args.executable.expanduser().resolve(strict=True)
    if not executable.is_file():
        raise ValueError("--executable must point to the native binary, not the .app directory")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Use an empty --output directory to exclude stale test results: {output}")
    summary = {"success": False, "platform": platform.platform(), "executable": str(executable),
               "notice": "Synthetic execution check only; no real SEM accuracy or CUDA/VRAM validation.",
               "seconds": {}}
    try:
        weights = weights_file(output, args.weights)
        base, full_base = generate_images(output)
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                   CUDA_VISIBLE_DEVICES="", PYTORCH_ENABLE_MPS_FALLBACK="1")
        timings = summary["seconds"]
        sequences, timings["scan"] = worker(executable, output, env, "scan", "scan",
                                             {"base": str(base), "profiles": PROFILES})
        assert len(sequences) == 3, "Expected exactly three discovered folders"
        for sequence in sequences:
            assert not sequence["issues"], sequence["issues"]
            assert [image["number"] for image in sequence["images"]] == list(range(5, 5 + FRAMES))
        rules, timings["rules"] = worker(executable, output, env, "rules", "rules", {"sequence": sequences[0]})
        assert len(rules["scores"]) == FRAMES and len(rules["shifts"]) == FRAMES
        assert rules["candidates"] and all(0 < index < FRAMES for index in rules["candidates"])
        project = output / "training-project"
        annotations = {seq["relative_path"]: dict(zip(CLASSES, range(1, 13))) for seq in sequences}
        model, timings["train"] = worker(executable, output, env, "train", "train", {
            "project_dir": str(project), "sequences": sequences, "annotations": annotations,
            "classes": CLASSES, "class_definitions": CLASS_DEFINITIONS,
            "model_name": "Synthetic packaged app smoke test · 12 custom classes",
            "config": {"pretrained_weights_path": str(weights), "device": "cpu", "epochs": 2, "offset": 0},
        })
        assert Path(model["path"]).is_file() and model["classes"] == CLASSES
        assert model["class_definitions"] == CLASS_DEFINITIONS
        assert model["metrics"]["validated"] and model["metrics"]["folds"] == 3
        inspected, timings["inspect"] = worker(executable, output, env, "inspect", "inspect_model",
                                               {"model_path": model["path"]})
        assert inspected["id"] == model["id"] and inspected["classes"] == CLASSES
        assert inspected["class_definitions"] == CLASS_DEFINITIONS
        # A separate project forces inference to decode images and extract fresh
        # features instead of merely reading the training feature cache.
        predictions, timings["infer"] = worker(executable, output, env, "infer", "infer", {
            "model_path": model["path"], "sequences": sequences, "project_dir": str(output / "inference-project"),
        })
        assert len(predictions) == 3
        for prediction in predictions:
            assert set(prediction["landmarks"]) == set(CLASSES), prediction
            indices = [prediction["landmarks"][label] for label in CLASSES]
            assert 0 <= indices[0] and indices[-1] < FRAMES, indices
            assert all(left < right for left, right in zip(indices, indices[1:])), indices
        full_sequences, timings["full_scan"] = worker(executable, output, env, "full-scan", "scan",
                                                      {"base": str(full_base), "profiles": PROFILES})
        assert len(full_sequences) == 1 and len(full_sequences[0]["images"]) == 1
        full_rules, timings["full_rules"] = worker(executable, output, env, "full-rules", "rules",
                                                   {"sequence": full_sequences[0]})
        assert len(full_rules["scores"]) == 1
        summary["gui"] = gui_startup(executable, output, env)
        summary.update(success=True, model_path=model["path"], folders=3, frames_per_folder=FRAMES,
                       class_count=len(CLASSES), class_definitions_roundtrip=True,
                       dtype="uint16", full_resolution_shape=[2048, 3072], weights_sha256=WEIGHT_SHA256,
                       training_device="cpu", inference_device="auto", cold_inference=True)
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        save_json(output / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path,
                        help="SEMReview.exe or SEMReview.app/Contents/MacOS/SEMReview")
    parser.add_argument("--output", required=True, type=Path, help="Empty directory for test evidence")
    parser.add_argument("--weights", type=Path, help="Optional official resnet18-f37072fd.pth")
    args = parser.parse_args()
    try:
        summary = run(args)
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
