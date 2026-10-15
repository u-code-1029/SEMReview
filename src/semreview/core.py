"""Filesystem-only SEM sequence discovery and lossless review operations.

Landmarks and plan boundaries are zero-based indices in ``sequence['images']``.
Every destructive-looking action is a journalled, no-overwrite same-volume move.
This module deliberately has no Qt, TIFF, or machine-learning dependencies.
"""

from __future__ import annotations

import copy
import ctypes
import errno
import fnmatch
import hashlib
import json
import os
import re
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class CancelledError(RuntimeError):
    """Discovery was cancelled; a partial folder inventory must not be used."""


Progress = Callable[[dict], None] | None
Cancelled = Callable[[], bool] | None
_NUMERIC_STEM = re.compile(r"^[0-9]+$")
_TIFF_SUFFIXES = {".tif", ".tiff"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(progress: Progress, **event) -> None:
    if progress is not None:
        if "current" in event:
            event.setdefault("completed", event["current"])
        if "path" in event:
            event.setdefault("message", event["path"])
        progress(event)


def _cancelled(cancelled: Cancelled) -> bool:
    return bool(cancelled and cancelled())


def _hash_file(path: Path, cancelled: Cancelled = None) -> str:
    if _cancelled(cancelled):
        raise CancelledError("작업이 취소되었습니다.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if _cancelled(cancelled):
                raise CancelledError("작업이 취소되었습니다.")
            digest.update(chunk)
    if _cancelled(cancelled):
        raise CancelledError("작업이 취소되었습니다.")
    return digest.hexdigest()


def _is_link(path: Path) -> bool:
    # A Windows junction is a directory reparse point but is_symlink() is false.
    return path.is_symlink() or path.is_junction()


def _under(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_path(path: Path, base: Path) -> None:
    """Reject changed/symlink ancestors, even for a not-yet-created target."""
    if (not path.is_absolute() or path != Path(os.path.abspath(path))
            or not _under(path, base)):
        raise ValueError(f"기준 디렉터리 밖의 경로입니다: {path}")
    if base.resolve() != base or _is_link(base):
        raise ValueError(f"기준 디렉터리가 변경되었습니다: {base}")
    cursor = path
    while cursor != base:
        if _is_link(cursor):
            raise ValueError(f"심볼릭 링크는 처리하지 않습니다: {cursor}")
        cursor = cursor.parent


def _inspect_sequence(folder: Path, base: Path, cancelled: Cancelled = None) -> dict:
    result = {"path": str(folder), "relative_path": folder.relative_to(base).as_posix(),
              "images": [], "issues": []}
    try:
        _safe_path(folder, base)
        candidates = sorted(folder.iterdir(), key=lambda p: (p.name.casefold(), p.name))
    except (OSError, ValueError) as exc:
        result["issues"].append(f"폴더를 읽을 수 없습니다: {exc}")
        return result
    for path in candidates:
        if _cancelled(cancelled):
            raise CancelledError("폴더 탐색이 취소되었습니다.")
        if _is_link(path) or path.suffix.lower() not in _TIFF_SUFFIXES:
            continue
        if not path.is_file():
            continue
        if not _NUMERIC_STEM.fullmatch(path.stem):
            result["issues"].append(f"숫자가 아닌 TIFF 파일 이름: {path.name}")
            continue
        try:
            before = path.stat()
            digest = _hash_file(path, cancelled)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ino):
                raise OSError("해시 계산 중 파일이 변경되었습니다")
            result["images"].append({"path": str(path), "name": path.name,
                                     "number": int(path.stem), "size": after.st_size,
                                     "mtime_ns": after.st_mtime_ns, "sha256": digest})
        except OSError as exc:
            result["issues"].append(f"이미지 파일 읽기 실패 ({path.name}): {exc}")
    if _cancelled(cancelled):
        raise CancelledError("폴더 탐색이 취소되었습니다.")
    result["images"].sort(key=lambda item: (item["number"], item["name"]))
    numbers = [item["number"] for item in result["images"]]
    duplicates = sorted({number for i, number in enumerate(numbers[1:], 1)
                         if number == numbers[i - 1]})
    if duplicates:
        result["issues"].append(f"중복 이미지 번호: {', '.join(map(str, duplicates))}")
    if not result["images"]:
        result["issues"].append("숫자 이름의 TIFF 이미지가 없습니다.")
    return result


def scan(base: str, profiles: list[dict], progress: Progress = None,
         cancelled: Cancelled = None) -> list[dict]:
    """Discover the union of explicit depth profiles, without following links."""
    if _cancelled(cancelled):
        raise CancelledError("폴더 탐색이 취소되었습니다.")
    root = Path(base).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("베이스 경로는 디렉터리여야 합니다.")
    folders: set[Path] = set()
    for profile in profiles:
        levels = profile.get("levels", [])
        current = [root]
        for level in levels:
            include = level.get("include", ["*"])
            exclude = level.get("exclude", [])
            if isinstance(include, str) or isinstance(exclude, str):
                raise ValueError("폴더 패턴은 문자열 목록이어야 합니다.")
            following = []
            for parent in current:
                if _cancelled(cancelled):
                    raise CancelledError("폴더 탐색이 취소되었습니다.")
                _safe_path(parent, root)
                for child in sorted(parent.iterdir(), key=lambda p: p.name):
                    if _cancelled(cancelled):
                        raise CancelledError("폴더 탐색이 취소되었습니다.")
                    if child.name == ".sem-review" or _is_link(child) or not child.is_dir():
                        continue
                    if (any(fnmatch.fnmatchcase(child.name, pattern) for pattern in include)
                            and not any(fnmatch.fnmatchcase(child.name, pattern) for pattern in exclude)):
                        following.append(child)
            current = following
        folders.update(current)
    results = []
    for index, folder in enumerate(sorted(folders), 1):
        if _cancelled(cancelled):
            raise CancelledError("폴더 탐색이 취소되었습니다.")
        results.append(_inspect_sequence(folder, root, cancelled))
        _emit(progress, stage="scan", current=index, total=len(folders), path=str(folder))
    if _cancelled(cancelled):
        raise CancelledError("폴더 탐색이 취소되었습니다.")
    return results


def make_review_plan(base: str, sequence: dict, landmarks: dict[str, int],
                     classes: list[str], end_class: str, offset: int = 30,
                     model_id: str = "manual") -> dict:
    """Compute an inclusive contiguous keep interval; never classify its members."""
    root = Path(base).expanduser().resolve()
    result = {"id": uuid.uuid4().hex, "base": str(root), "sequence": copy.deepcopy(sequence),
              "landmarks": dict(landmarks), "classes": list(classes), "end_class": end_class,
              "offset": offset, "model_id": model_id, "start": None, "end": None,
              "start_index": None, "end_index": None, "keep": [], "move": [],
              "warnings": [], "errors": list(sequence.get("issues", []))}
    images = sequence.get("images", [])
    if not images:
        result["errors"].append("이미지가 없습니다.")
    if not classes or len(classes) != len(set(classes)):
        result["errors"].append("클래스는 비어 있지 않은 고유한 이름 목록이어야 합니다.")
    if end_class not in classes:
        result["errors"].append("끝점 클래스가 모델에 없습니다.")
    if isinstance(offset, bool) or not isinstance(offset, int):
        result["errors"].append("오프셋은 정수여야 합니다.")
    positions = []
    for label in classes:
        index = landmarks.get(label)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(images):
            result["errors"].append(f"클래스 기준점이 없거나 범위를 벗어났습니다: {label}")
        else:
            positions.append(index)
    if len(positions) == len(classes) and any(a >= b for a, b in zip(positions, positions[1:])):
        result["errors"].append("기준점은 바닥부터 천장까지 엄격한 오름차순이어야 합니다.")
    try:
        folder = Path(sequence["path"])
        _safe_path(folder, root)
        if ".sem-review" in folder.relative_to(root).parts:
            raise ValueError("격리 폴더는 검수할 수 없습니다.")
        previous_number = None
        for record in images:
            path = Path(record["path"])
            _safe_path(path, root)
            if path.parent != folder or path.name != record["name"]:
                raise ValueError("이미지 경로가 시퀀스 폴더와 일치하지 않습니다.")
            if (path.suffix.lower() not in _TIFF_SUFFIXES
                    or not _NUMERIC_STEM.fullmatch(path.stem)
                    or type(record["number"]) is not int or record["number"] != int(path.stem)):
                raise ValueError("이미지 번호가 파일명과 일치하지 않습니다.")
            if previous_number is not None and record["number"] <= previous_number:
                raise ValueError("전체 이미지 목록은 중복 없이 번호 순서로 정렬되어야 합니다.")
            previous_number = record["number"]
            if (type(record.get("size")) is not int or record["size"] < 0
                    or type(record.get("mtime_ns")) is not int):
                raise ValueError("파일 크기 또는 수정 시각이 없습니다. 폴더를 다시 탐색해 주세요.")
            if not re.fullmatch(r"[a-f0-9]{64}", record.get("sha256", "")):
                raise ValueError("파일 내용 지문이 없습니다. 폴더를 다시 탐색해 주세요.")
    except (KeyError, TypeError, ValueError) as exc:
        result["errors"].append(str(exc))
    if result["errors"]:
        return result
    start = positions[0]
    raw_end = landmarks[end_class] + offset
    end = min(raw_end, len(images) - 1)
    result.update(start=start, end=end, start_index=start, end_index=end)
    if end < start:
        result["errors"].append("계산된 끝점이 바닥보다 앞에 있습니다.")
        return result
    if raw_end >= len(images):
        result["warnings"].append(f"뒤쪽 이미지가 {raw_end - len(images) + 1}장 부족하여 마지막 이미지까지 보존합니다.")
    for left, right in zip(images[start:end + 1], images[start + 1:end + 1]):
        if right["number"] != left["number"] + 1:
            result["errors"].append(f"보존 구간의 이미지 번호가 연속적이지 않습니다: {left['number']} → {right['number']}")
    result["keep"] = [record["path"] for record in images[start:end + 1]]
    result["move"] = [record["path"] for record in images[:start] + images[end + 1:]]
    return result


def _sync_dir(path: Path) -> None:
    if os.name == "posix":
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass  # Some network/filesystem implementations cannot fsync directories.


def _save(path: Path, document: dict) -> None:
    document["updated_at"] = _now()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _runs_dir(project_dir: str) -> Path:
    path = Path(project_dir).expanduser().resolve() / "review-runs"
    if _is_link(path):
        raise ValueError("검수 기록 디렉터리는 심볼릭 링크일 수 없습니다.")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _run_lock(directory: Path):
    with (directory / ".operation.lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            try:
                # Windows byte-range locks also deny reads. Inspect the length
                # without reading the byte another process may already hold.
                if os.fstat(stream.fileno()).st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("다른 검수 파일 작업이 진행 중입니다.") from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("다른 검수 파일 작업이 진행 중입니다.") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _move_noreplace(source: Path, destination: Path) -> None:
    """Use an atomic no-clobber rename when available, with a safe link fallback."""
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise OSError(errno.EXDEV, "다른 볼륨으로 이동할 수 없습니다", str(destination))
    if os.name == "nt":
        os.rename(source, destination)  # Windows rename fails when destination exists.
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        function = None
        arguments = None
        if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
            function = libc.renamex_np
            function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
            arguments = (os.fsencode(source), os.fsencode(destination), 0x00000004)
        elif hasattr(libc, "renameat2"):
            function = libc.renameat2
            function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            arguments = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if function is not None:
            function.restype = ctypes.c_int
            if function(*arguments) == 0:
                _sync_dir(source.parent)
                _sync_dir(destination.parent)
                return
            error = ctypes.get_errno()
            if error not in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
                raise OSError(error, os.strerror(error), str(destination))
        # link() never overwrites. The durable 'moving' journal handles a crash
        # between these operations; recovery only unlinks two paths to one inode.
        os.link(source, destination, follow_symlinks=False)
        _sync_dir(destination.parent)
        if not os.path.samefile(source, destination):
            raise OSError("이동 중 원본이 교체되어 두 파일을 모두 보존했습니다.")
        source.unlink()
    _sync_dir(source.parent)
    _sync_dir(destination.parent)


def _safe_mkdir(path: Path, base: Path) -> None:
    _safe_path(path, base)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        _safe_path(directory, base)
        directory.mkdir(exist_ok=True)
        _sync_dir(directory.parent)
    _safe_path(path, base)


def _matches(path: Path, record: dict, base: Path, cancelled: Cancelled = None) -> bool:
    try:
        _safe_path(path, base)
        if not path.is_file() or _is_link(path):
            return False
        before = path.stat()
        digest = _hash_file(path, cancelled)
        after = path.stat()
        return (before.st_ino == after.st_ino and before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns
                and after.st_size == record["size"] and after.st_mtime_ns == record["mtime_ns"]
                and digest == record["sha256"])
    except (OSError, ValueError):
        return False


def _verify_snapshot(item: dict, cancelled: Cancelled = None) -> None:
    """Validate kept files too, while allowing only this run's recorded moves."""
    plan = item["plan"]
    base = Path(plan["base"])
    folder = Path(plan["sequence"]["path"])
    _safe_path(folder, base)
    expected = {record["name"]: record for record in plan["sequence"]["images"]}
    moves = {entry["source"]: entry for entry in item["entries"]}
    visible = {path.name for path in folder.iterdir()
               if path.suffix.lower() in _TIFF_SUFFIXES and path.is_file() and not _is_link(path)}
    if visible - set(expected):
        raise ValueError("분석 후 TIFF 파일 목록이 변경되었습니다. 다시 분석해 주세요.")
    for record in expected.values():
        if _cancelled(cancelled):
            raise CancelledError("파일 검증이 취소되었습니다.")
        source = Path(record["path"])
        if source.exists() or source.is_symlink():
            if not _matches(source, record, base, cancelled):
                raise ValueError(f"분석 후 파일이 변경되었습니다: {source.name}")
        else:
            entry = moves.get(str(source))
            if not entry or entry["status"] not in {"moving", "moved", "restoring", "conflict"}:
                raise ValueError(f"분석 후 파일이 없어졌습니다: {source.name}")
            if not _matches(Path(entry["destination"]), record, base, cancelled):
                raise ValueError(f"원본과 격리 파일을 확인할 수 없습니다: {source.name}")


def _summary(document: dict) -> dict:
    result = copy.deepcopy(document)
    entries = [entry for item in document["plans"] for entry in item["entries"]]
    result.update(total=len(entries), moved=sum(e["status"] == "moved" for e in entries),
                  restored=sum(e["status"] == "restored" for e in entries),
                  pending=sum(e["status"] in {"pending", "moving", "restoring"} for e in entries),
                  conflicts=sum(e["status"] == "conflict" for e in entries))
    result["errors"] = [error for item in document["plans"] for error in item.get("errors", [])]
    return result


def _load(directory: Path, run_id: str) -> tuple[Path, dict]:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("올바르지 않은 실행 ID입니다.")
    path = directory / f"{run_id}.json"
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema_version") != 1 or document.get("run_id") != run_id:
        raise ValueError("지원하지 않거나 손상된 검수 기록입니다.")
    # A partially written or edited journal must never widen the original move
    # set or redirect a restoration to an arbitrary path.
    try:
        for item in document["plans"]:
            plan = item["plan"]
            if item.get("invalid"):
                if item["entries"]:
                    raise ValueError("무효한 계획에 이동 기록이 있습니다.")
                continue
            calculated = make_review_plan(plan["base"], plan["sequence"], plan["landmarks"],
                                          plan["classes"], plan["end_class"], plan["offset"],
                                          plan["model_id"])
            if (calculated["errors"] or calculated["keep"] != plan["keep"]
                    or calculated["move"] != plan["move"]):
                raise ValueError("검수 기록의 보존 구간이 올바르지 않습니다.")
            records = {record["path"]: record for record in plan["sequence"]["images"]}
            if sorted(entry["source"] for entry in item["entries"]) != sorted(plan["move"]):
                raise ValueError("검수 기록의 이동 목록이 올바르지 않습니다.")
            base = Path(plan["base"])
            for entry in item["entries"]:
                target = base / ".sem-review" / "quarantine" / run_id / Path(entry["source"]).relative_to(base)
                record = records[entry["source"]]
                if (entry["destination"] != str(target)
                        or any(entry[key] != record[key] for key in ("size", "mtime_ns", "sha256"))):
                    raise ValueError("검수 기록의 파일 경로 또는 지문이 올바르지 않습니다.")
    except (KeyError, TypeError) as exc:
        raise ValueError("검수 기록에 필수 정보가 없습니다.") from exc
    return path, document


def _process_move(entry: dict, base: Path) -> None:
    source, destination = Path(entry["source"]), Path(entry["destination"])
    _safe_path(source, base)
    _safe_path(destination, base)
    src_exists = source.exists() or source.is_symlink()
    dst_exists = destination.exists() or destination.is_symlink()
    if dst_exists:
        if not _matches(destination, entry, base):
            raise FileExistsError(f"격리 대상 위치에 다른 파일이 있습니다: {destination}")
        if not src_exists:
            return  # rename completed, journal completion was interrupted.
        if entry.get("recovering_link") and os.path.samefile(source, destination):
            if not _matches(source, entry, base):
                raise ValueError(f"이동 중 원본이 변경되었습니다: {source}")
            source.unlink()
            _sync_dir(source.parent)
            return
        raise FileExistsError(f"격리 대상 파일이 이미 있습니다: {destination}")
    if not src_exists or not _matches(source, entry, base):
        raise ValueError(f"원본 파일이 없거나 변경되었습니다: {source}")
    _safe_mkdir(destination.parent, base)
    _checked_move(source, destination, entry, base)


def _checked_move(source: Path, destination: Path, record: dict, base: Path) -> None:
    _move_noreplace(source, destination)
    if not _matches(destination, record, base):
        # A producer may replace/update a TIFF between verification and rename.
        # Return the changed file when that can be done without overwriting.
        if not source.exists() and not source.is_symlink():
            _safe_path(destination, base)
            _move_noreplace(destination, source)
            raise ValueError(f"이동 도중 파일이 변경되어 원래 위치로 되돌렸습니다: {source}")
        raise ValueError(f"이동 도중 파일이 변경되었습니다. 두 경로를 보존했습니다: {source}, {destination}")


def _execute(path: Path, document: dict, progress: Progress, cancelled: Cancelled) -> dict:
    document["operation"] = "apply"
    document["status"] = "running"
    _save(path, document)
    processed = 0
    total = sum(len(item["entries"]) for item in document["plans"])
    for item in document["plans"]:
        if item.get("invalid"):
            continue
        item["errors"] = []
        if _cancelled(cancelled):
            document["status"] = "cancelled"
            _save(path, document)
            return _summary(document)
        try:
            _verify_snapshot(item, cancelled)
        except CancelledError:
            document["status"] = "cancelled"
            _save(path, document)
            return _summary(document)
        except (OSError, ValueError) as exc:
            item["errors"].append(str(exc))
            item["status"] = "blocked"
            processed += len(item["entries"])
            _save(path, document)
            _emit(progress, stage="move", current=processed, total=total,
                  path=item["plan"]["sequence"]["path"], status="blocked", run_id=document["run_id"])
            continue
        item["status"] = "running"
        for entry in item["entries"]:
            if _cancelled(cancelled):
                document["status"] = "cancelled"
                _save(path, document)
                return _summary(document)
            if entry["status"] == "moved":
                # Snapshot verification checked originals; verify the destination
                # as well so missing/tampered quarantined files are not ignored.
                if (not Path(entry["source"]).exists()
                        and _matches(Path(entry["destination"]), entry, Path(item["plan"]["base"]))):
                    processed += 1
                    _emit(progress, stage="move", current=processed, total=total, path=entry["source"],
                          status=entry["status"], run_id=document["run_id"])
                    continue
            entry["recovering_link"] = entry["status"] == "moving"
            entry["status"] = "moving"
            entry.pop("error", None)
            _save(path, document)
            stop_folder = False
            try:
                _process_move(entry, Path(item["plan"]["base"]))
                entry["status"] = "moved"
            except (OSError, ValueError) as exc:
                entry["status"] = "conflict"
                entry["error"] = str(exc)
                item["errors"].append(str(exc))
                stop_folder = isinstance(exc, ValueError)
            entry.pop("recovering_link", None)
            processed += 1
            _save(path, document)
            _emit(progress, stage="move", current=processed, total=total, path=entry["source"],
                  status=entry["status"], run_id=document["run_id"])
            if stop_folder:
                break  # A changed source invalidates this folder's remaining decisions.
        item["status"] = "blocked" if item["errors"] else "complete"
    document["status"] = "partial" if any(item["errors"] for item in document["plans"]) else "complete"
    _save(path, document)
    return _summary(document)


def apply_plans(plans: list[dict], project_dir: str, progress: Progress = None,
                cancelled: Cancelled = None) -> dict:
    """Persist the exact approved plan, recheck files, then quarantine exclusions."""
    directory = _runs_dir(project_dir)
    with _run_lock(directory):
        run_id = uuid.uuid4().hex
        document = {"schema_version": 1, "run_id": run_id, "created_at": _now(),
                    "updated_at": _now(), "operation": "apply", "status": "prepared", "plans": []}
        seen = set()
        for supplied in plans:
            plan = make_review_plan(supplied["base"], supplied["sequence"], supplied["landmarks"],
                                    supplied["classes"], supplied["end_class"], supplied["offset"],
                                    supplied.get("model_id", "manual"))
            plan["id"] = supplied["id"]
            errors = list(plan["errors"])
            for error in supplied.get("errors", []):
                if error not in errors:
                    errors.append(error)
            if plan["keep"] != supplied.get("keep") or plan["move"] != supplied.get("move"):
                errors.append("승인된 검수 계획과 계산 결과가 일치하지 않습니다.")
            folder = plan["sequence"]["path"]
            if folder in seen:
                errors.append("같은 폴더를 한 번의 실행에 여러 번 적용할 수 없습니다.")
            seen.add(folder)
            item = {"plan": plan, "status": "blocked" if errors else "pending",
                    "invalid": bool(errors), "errors": errors, "entries": []}
            if not errors:
                records = {record["path"]: record for record in plan["sequence"]["images"]}
                base = Path(plan["base"])
                for source in plan["move"]:
                    destination = base / ".sem-review" / "quarantine" / run_id / Path(source).relative_to(base)
                    record = records[source]
                    item["entries"].append({"source": source, "destination": str(destination),
                                            "size": record["size"], "mtime_ns": record["mtime_ns"],
                                            "sha256": record["sha256"], "status": "pending"})
            document["plans"].append(item)
        path = directory / f"{run_id}.json"
        _save(path, document)
        return _execute(path, document, progress, cancelled)


def list_runs(project_dir: str) -> list[dict]:
    """List readable run journals, newest first; corrupt journals stay visible."""
    directory = Path(project_dir).expanduser().resolve() / "review-runs"
    if not directory.exists():
        return []
    results = []
    for path in directory.glob("*.json"):
        try:
            _, document = _load(directory, path.stem)
            results.append(_summary(document))
        except (OSError, ValueError, KeyError) as exc:
            results.append({"run_id": path.stem, "status": "unreadable", "errors": [str(exc)],
                            "created_at": "", "total": 0, "moved": 0, "restored": 0,
                            "pending": 0, "conflicts": 0, "plans": []})
    return sorted(results, key=lambda item: item.get("created_at", ""), reverse=True)


def resume_run(project_dir: str, run_id: str, progress: Progress = None,
               cancelled: Cancelled = None) -> dict:
    directory = _runs_dir(project_dir)
    with _run_lock(directory):
        path, document = _load(directory, run_id)
        if document["operation"] == "restore":
            return _restore(path, document, progress, cancelled)
        return _execute(path, document, progress, cancelled)


def _restore(path: Path, document: dict, progress: Progress, cancelled: Cancelled) -> dict:
    document.update(operation="restore", status="running")
    _save(path, document)
    total = sum(len(item["entries"]) for item in document["plans"])
    processed = 0
    for item in reversed(document["plans"]):
        if item.get("invalid"):
            continue
        item["errors"] = []
        base = Path(item["plan"]["base"])
        for entry in reversed(item["entries"]):
            if _cancelled(cancelled):
                document["status"] = "cancelled"
                _save(path, document)
                return _summary(document)
            original, quarantined = Path(entry["source"]), Path(entry["destination"])
            previous_status = entry["status"]
            try:
                _safe_path(original, base)
                _safe_path(quarantined, base)
                source_exists = original.exists() or original.is_symlink()
                target_exists = quarantined.exists() or quarantined.is_symlink()
                if not target_exists:
                    if not _matches(original, entry, base):
                        raise ValueError(f"복원할 파일을 찾거나 확인할 수 없습니다: {original}")
                    if previous_status != "pending":
                        entry["status"] = "restored"
                    processed += 1
                    _save(path, document)
                    _emit(progress, stage="restore", current=processed, total=total, path=str(original),
                          status=entry["status"], run_id=document["run_id"])
                    continue
                if not _matches(quarantined, entry, base):
                    raise ValueError(f"격리된 파일이 변경되었습니다: {quarantined}")
                if source_exists:
                    if (previous_status in {"restoring", "moving"}
                            and _matches(original, entry, base)
                            and os.path.samefile(original, quarantined)):
                        quarantined.unlink()
                        _sync_dir(quarantined.parent)
                    else:
                        raise FileExistsError(f"원래 위치에 파일이 있어 복원하지 않았습니다: {original}")
                else:
                    entry["status"] = "restoring"
                    _save(path, document)
                    _safe_mkdir(original.parent, base)
                    _checked_move(quarantined, original, entry, base)
                entry["status"] = "restored"
                entry.pop("error", None)
            except (OSError, ValueError) as exc:
                entry["status"] = "conflict"
                entry["error"] = str(exc)
                item["errors"].append(str(exc))
            processed += 1
            _save(path, document)
            _emit(progress, stage="restore", current=processed, total=total, path=str(original),
                  status=entry["status"], run_id=document["run_id"])
        item["status"] = "blocked" if item["errors"] else "restored"
    document["status"] = "partial" if any(item["errors"] for item in document["plans"]) else "restored"
    _save(path, document)
    return _summary(document)


def restore_run(project_dir: str, run_id: str, progress: Progress = None,
                cancelled: Cancelled = None) -> dict:
    directory = _runs_dir(project_dir)
    with _run_lock(directory):
        path, document = _load(directory, run_id)
        return _restore(path, document, progress, cancelled)
