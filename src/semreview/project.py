"""Versioned, atomic project state. Source images are never stored in the database."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_CLASSES = ["바닥", "1층 천장", "2층 천장"]
DEFAULT_PROFILES = [{"name": "기본 · 2단계", "levels": [
    {"include": ["*"], "exclude": []}, {"include": ["*"], "exclude": []},
]}]


def sequence_signature(sequence: dict) -> str:
    """Bind labels to exact ordered file contents, independently of base location."""
    items = [(im["name"], im["number"], im["size"], im["sha256"]) for im in sequence["images"]]
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()


class Project:
    def __init__(self, project_dir: str | Path):
        self.root = Path(project_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.root / "project.sqlite", timeout=10)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            self._db.close()
            raise ValueError("더 새로운 앱에서 만든 프로젝트입니다. 앱을 업데이트하세요.")
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._db.execute("""CREATE TABLE IF NOT EXISTS annotations (
                relative_path TEXT PRIMARY KEY, landmarks TEXT NOT NULL,
                sequence_signature TEXT NOT NULL, status TEXT NOT NULL,
                cursor INTEGER NOT NULL, updated_at TEXT NOT NULL)""")
            self._db.execute("CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value: Any) -> None:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        with self._db:
            self._db.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, encoded))

    def annotations(self) -> dict:
        rows = self._db.execute("SELECT relative_path,landmarks,sequence_signature,status,cursor,updated_at FROM annotations")
        return {r[0]: {"landmarks": json.loads(r[1]), "sequence_signature": r[2],
                       "status": r[3], "cursor": r[4], "updated_at": r[5]} for r in rows}

    def save_annotation(self, relative_path: str, landmarks: dict, sequence_signature: str,
                        status: str = "draft", cursor: int = 0) -> None:
        if status not in {"draft", "complete", "completed", "skipped", "invalid"}:
            raise ValueError(f"지원하지 않는 라벨 상태: {status}")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in landmarks.values()):
            raise ValueError("라벨 위치는 0 이상의 정수여야 합니다.")
        with self._db:
            self._db.execute("""INSERT INTO annotations VALUES (?,?,?,?,?,?)
              ON CONFLICT(relative_path) DO UPDATE SET landmarks=excluded.landmarks,
              sequence_signature=excluded.sequence_signature,status=excluded.status,
              cursor=excluded.cursor,updated_at=excluded.updated_at""",
              (relative_path, json.dumps(landmarks, ensure_ascii=False), sequence_signature,
               status, max(0, int(cursor)), datetime.now(timezone.utc).isoformat()))

    def models(self) -> list[dict]:
        records = [json.loads(r[0]) for r in self._db.execute("SELECT metadata FROM models ORDER BY rowid DESC")]
        for item in records:
            if item.get("project_relative_path"):
                item["path"] = str(self.root / item["project_relative_path"])
        return records

    def add_model(self, metadata: dict) -> None:
        data = dict(metadata)
        if not data.get("id") or not data.get("path"):
            raise ValueError("모델 ID와 경로가 필요합니다.")
        path = Path(data["path"]).resolve()
        try:
            data["project_relative_path"] = str(path.relative_to(self.root))
        except ValueError:
            pass
        with self._db:
            self._db.execute("INSERT INTO models VALUES (?,?) ON CONFLICT(id) DO UPDATE SET metadata=excluded.metadata",
                             (data["id"], json.dumps(data, ensure_ascii=False, allow_nan=False)))

    def import_model(self, model_path: str) -> dict:
        from .ml import inspect_model

        data = inspect_model(model_path)
        destination = self.root / "models" / f"{data['id']}.sem-model"
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = Path(model_path).resolve()
        if source != destination.resolve():
            if destination.exists():
                if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
                    raise ValueError("같은 ID의 다른 모델이 이미 있습니다.")
            else:
                from .core import _move_noreplace

                temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with source.open("rb") as reader, temporary.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
                        writer.flush()
                        os.fsync(writer.fileno())
                    # Validate the completed copy before atomically publishing it.
                    inspect_model(temporary)
                    _move_noreplace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
        data["path"] = str(destination)
        self.add_model(data)
        return data

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
