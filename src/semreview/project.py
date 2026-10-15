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

from .classdefs import (DEFAULT_CLASSES, DEFAULT_CLASS_DEFINITIONS, LEGACY_DEFAULT_NAMES,
                        class_ids, legacy_definitions, normalize_model_metadata, remap_keys,
                        validate_definitions)

SCHEMA_VERSION = 2
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
        existed = bool(self._db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='settings'").fetchone())
        if version > SCHEMA_VERSION:
            self._db.close()
            raise ValueError("더 새로운 앱에서 만든 프로젝트입니다. 앱을 업데이트하세요.")
        if version < 2 and (existed or version == 1):
            backup = self.root / "project.v1-backup.sqlite"
            while True:
                try:
                    with backup.open("xb"):
                        pass
                    break
                except FileExistsError:
                    backup = self.root / f"project.v1-backup-{uuid.uuid4().hex}.sqlite"
            try:
                destination = sqlite3.connect(backup)
                try:
                    self._db.backup(destination)
                finally:
                    destination.close()
            except Exception:
                backup.unlink(missing_ok=True)
                self._db.close()
                raise
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self._db.execute("""CREATE TABLE IF NOT EXISTS annotations (
                relative_path TEXT PRIMARY KEY, landmarks TEXT NOT NULL,
                sequence_signature TEXT NOT NULL, status TEXT NOT NULL,
                cursor INTEGER NOT NULL, updated_at TEXT NOT NULL)""")
            self._db.execute("CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
            if version < 2:
                self._migrate_classes(legacy=existed or version == 1)
            self._db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _write_setting(self, key: str, value: Any) -> None:
        """Caller owns the transaction, including multi-table schema changes."""
        self._db.execute("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                         (key, json.dumps(value, ensure_ascii=False, allow_nan=False)))

    def _migrate_classes(self, *, legacy: bool) -> None:
        old_names = self.get("classes", LEGACY_DEFAULT_NAMES)
        defs = legacy_definitions(old_names) if legacy else validate_definitions(DEFAULT_CLASS_DEFINITIONS)
        mapping = dict(zip(old_names, class_ids(defs))) if legacy else {}
        self._write_setting("class_definitions", defs)
        self._write_setting("classes", class_ids(defs))
        for relative, labels in self._db.execute("SELECT relative_path,landmarks FROM annotations").fetchall():
            self._db.execute("UPDATE annotations SET landmarks=? WHERE relative_path=?",
                             (json.dumps(remap_keys(json.loads(labels), mapping), ensure_ascii=False), relative))
        model_maps = {}
        for model_id, encoded in self._db.execute("SELECT id,metadata FROM models").fetchall():
            metadata = json.loads(encoded)
            normalized = normalize_model_metadata(metadata)
            model_maps[model_id] = dict(zip(metadata["classes"], normalized["classes"]))
            self._db.execute("UPDATE models SET metadata=? WHERE id=?",
                             (json.dumps(normalized, ensure_ascii=False, allow_nan=False), model_id))
        predictions = self.get("review_predictions", [])
        for prediction in predictions:
            model_mapping = model_maps.get(prediction.get("model_id"), mapping)
            for key in ("landmarks", "candidates"):
                if key in prediction:
                    prediction[key] = remap_keys(prediction[key], model_mapping)
        if predictions:
            self._write_setting("review_predictions", predictions)
        end = self.get("review_end_class")
        if end is not None:
            model_mapping = model_maps.get(self.get("review_model_id"), mapping)
            self._write_setting("review_end_class", model_mapping.get(end, end))

    def get(self, key: str, default: Any = None) -> Any:
        row = self._db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key: str, value: Any) -> None:
        if key == "class_definitions":
            self.update_class_definitions(value)
            return
        if key == "classes":
            # Compatibility for legacy scripts. New UI always edits definitions.
            current = self.class_definitions()
            if value == class_ids(current):
                return
            if value == DEFAULT_CLASSES:
                self.update_class_definitions(DEFAULT_CLASS_DEFINITIONS)
            else:
                by_id = {item["id"]: item for item in current}
                defs = [by_id[item] for item in value] if isinstance(value, list) and all(isinstance(item, str) and item in by_id for item in value) else legacy_definitions(value)
                self.update_class_definitions(defs)
            return
        with self._db:
            self._write_setting(key, value)

    def class_definitions(self) -> list[dict]:
        return validate_definitions(self.get("class_definitions", DEFAULT_CLASS_DEFINITIONS))

    def update_class_definitions(self, defs: list[dict]) -> None:
        definitions = validate_definitions(defs)
        old = self.class_definitions()
        old_ids, new_ids = class_ids(old), class_ids(definitions)
        criteria_changed = {item["id"]: item["description"] for item in old} != {
            item["id"]: item["description"] for item in definitions}
        reconfirm = old_ids != new_ids or criteria_changed
        with self._db:
            self._write_setting("class_definitions", definitions)
            self._write_setting("classes", new_ids)
            if reconfirm:
                now = datetime.now(timezone.utc).isoformat()
                for relative, labels, status in self._db.execute("SELECT relative_path,landmarks,status FROM annotations").fetchall():
                    surviving = {key: value for key, value in json.loads(labels).items() if key in new_ids}
                    if status in {"complete", "completed"}:
                        status = "draft"
                    self._db.execute("UPDATE annotations SET landmarks=?,status=?,updated_at=? WHERE relative_path=?",
                                     (json.dumps(surviving, ensure_ascii=False), status, now, relative))

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
        definitions = self.class_definitions()
        ids = set(class_ids(definitions))
        # Legacy callers can still submit current display names. An exact ID
        # always takes precedence if a display name happens to equal another ID.
        aliases = {item["name"]: item["id"] for item in definitions if item["name"] not in ids}
        landmarks = remap_keys(landmarks, aliases)
        with self._db:
            self._db.execute("""INSERT INTO annotations VALUES (?,?,?,?,?,?)
              ON CONFLICT(relative_path) DO UPDATE SET landmarks=excluded.landmarks,
              sequence_signature=excluded.sequence_signature,status=excluded.status,
              cursor=excluded.cursor,updated_at=excluded.updated_at""",
              (relative_path, json.dumps(landmarks, ensure_ascii=False), sequence_signature,
               status, max(0, int(cursor)), datetime.now(timezone.utc).isoformat()))

    def models(self) -> list[dict]:
        records = [normalize_model_metadata(json.loads(r[0])) for r in self._db.execute("SELECT metadata FROM models ORDER BY rowid DESC")]
        for item in records:
            if item.get("project_relative_path"):
                item["path"] = str(self.root / item["project_relative_path"])
        return records

    def add_model(self, metadata: dict) -> None:
        data = normalize_model_metadata(metadata)
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

        data = normalize_model_metadata(inspect_model(model_path))
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
