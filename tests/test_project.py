import json
import sqlite3

import pytest

from semreview.project import Project, sequence_signature
from semreview.classdefs import (DEFAULT_CLASSES, LEGACY_DEFAULT_NAMES, class_ids,
                                legacy_definitions, new_definition)


def example_sequence():
    return {"path": "/some/base/a", "relative_path": "a", "images": [
        {"name": "5.tif", "number": 5, "size": 100, "sha256": "abc"},
        {"name": "6.tif", "number": 6, "size": 100, "sha256": "def"},
    ]}


def test_roundtrip_annotations_and_settings(tmp_path):
    seq = example_sequence()
    with Project(tmp_path) as project:
        project.set("base", "D:/한글 이미지")
        project.save_annotation("a", {"바닥": 0, "1층 천장": 1}, sequence_signature(seq), "complete", 1)
    with Project(tmp_path) as project:
        assert project.get("base") == "D:/한글 이미지"
        saved = project.annotations()["a"]
        assert saved["landmarks"] == {"바닥": 0, "1층 천장": 1}
        assert saved["cursor"] == 1
        assert saved["status"] == "complete"


def test_signature_tracks_order_contents_but_not_base(tmp_path):
    seq = example_sequence()
    original = sequence_signature(seq)
    seq["path"] = "E:/new-location/a"
    assert sequence_signature(seq) == original
    seq["images"][0]["sha256"] = "changed"
    assert sequence_signature(seq) != original
    seq["images"][0]["sha256"] = "abc"
    seq["images"].reverse()
    assert sequence_signature(seq) != original


def test_latest_annotation_wins_atomically(tmp_path):
    with Project(tmp_path) as project:
        project.save_annotation("x", {"floor": 1}, "a")
        project.save_annotation("x", {"floor": 2}, "b", "skipped", 3)
        assert len(project.annotations()) == 1
        assert project.annotations()["x"]["sequence_signature"] == "b"


@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_invalid_landmark_is_not_persisted(tmp_path, bad):
    with Project(tmp_path) as project:
        with pytest.raises(ValueError):
            project.save_annotation("x", {"floor": bad}, "a")
        assert not project.annotations()


def test_newer_schema_rejected(tmp_path):
    db = sqlite3.connect(tmp_path / "project.sqlite")
    db.execute("PRAGMA user_version=99")
    db.close()
    with pytest.raises(ValueError, match="새로운"):
        Project(tmp_path)


def test_model_paths_rebase_after_project_move(tmp_path):
    import shutil
    original = tmp_path / "old"
    with Project(original) as project:
        model = original / "models" / "m.sem-model"
        model.parent.mkdir()
        model.write_bytes(b"fixture")
        project.add_model({"id": "m", "path": str(model), "classes": ["바닥", "천장"]})
    shutil.move(str(original), str(tmp_path / "new"))
    with Project(tmp_path / "new") as project:
        assert project.models()[0]["path"] == str(tmp_path / "new/models/m.sem-model")


def test_settings_refuse_non_finite_values(tmp_path):
    with Project(tmp_path) as project:
        with pytest.raises(ValueError):
            project.set("invalid", float("nan"))
        assert project.get("invalid") is None


def test_interrupted_import_does_not_publish_partial_model(tmp_path, monkeypatch):
    import shutil
    from semreview import ml

    source = tmp_path / "source.sem-model"
    source.write_bytes(b"model contents")
    monkeypatch.setattr(ml, "inspect_model", lambda p: {"id": "model-id", "path": str(p), "classes": ["floor", "roof"]})
    original_copy = shutil.copyfileobj

    def interrupted(reader, writer):
        writer.write(b"partial")
        raise OSError("simulated interrupted copy")

    with Project(tmp_path / "project") as project:
        monkeypatch.setattr(shutil, "copyfileobj", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            project.import_model(str(source))
        assert not list((project.root / "models").iterdir())
        assert not project.models()
        monkeypatch.setattr(shutil, "copyfileobj", original_copy)
        imported = project.import_model(str(source))
        assert imported["id"] == "model-id"
        assert (project.root / "models/model-id.sem-model").read_bytes() == source.read_bytes()


def _legacy_project(path, *, explicit_classes=True):
    with sqlite3.connect(path / "project.sqlite") as db:
        db.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE annotations (relative_path TEXT PRIMARY KEY, landmarks TEXT NOT NULL, sequence_signature TEXT NOT NULL, status TEXT NOT NULL, cursor INTEGER NOT NULL, updated_at TEXT NOT NULL)")
        db.execute("CREATE TABLE models (id TEXT PRIMARY KEY, metadata TEXT NOT NULL)")
        db.execute("PRAGMA user_version=1")
        names = ["기판", "막 표면"] if explicit_classes else LEGACY_DEFAULT_NAMES
        if explicit_classes:
            db.execute("INSERT INTO settings VALUES ('classes',?)", (json.dumps(names),))
        db.execute("INSERT INTO annotations VALUES ('sample',?,'signature','complete',5,'old')",
                   (json.dumps(dict(zip(names, range(len(names))))),))
        # The saved model can have a different taxonomy than the project.
        model = {"id": "old-model", "format_version": 1, "classes": ["A", "B"], "path": "/old.sem-model"}
        db.execute("INSERT INTO models VALUES ('old-model',?)", (json.dumps(model),))
        for key, value in {"review_model_id": "old-model", "review_end_class": "B",
                           "review_predictions": [{"model_id": "old-model", "landmarks": {"A": 4, "B": 8},
                                                   "candidates": {"A": [3, 4], "B": [8]}}]}.items():
            db.execute("INSERT INTO settings VALUES (?,?)", (key, json.dumps(value)))
    return names


@pytest.mark.parametrize("explicit", [True, False])
def test_schema1_migration_preserves_annotations_models_reviews_and_backup(tmp_path, explicit):
    names = _legacy_project(tmp_path, explicit_classes=explicit)
    (tmp_path / "project.v1-backup.sqlite").write_bytes(b"previous backup")
    with Project(tmp_path) as project:
        ids = class_ids(legacy_definitions(names))
        assert project.get("classes") == ids
        assert [item["name"] for item in project.class_definitions()] == names
        assert project.annotations()["sample"]["landmarks"] == dict(zip(ids, range(len(ids))))
        assert project.annotations()["sample"]["status"] == "complete"
        model_ids = class_ids(legacy_definitions(["A", "B"]))
        assert project.models()[0]["classes"] == model_ids
        assert project.get("review_end_class") == model_ids[1]
        prediction = project.get("review_predictions")[0]
        assert prediction["landmarks"] == dict(zip(model_ids, [4, 8]))
        assert prediction["candidates"] == dict(zip(model_ids, [[3, 4], [8]]))
    backups = list(tmp_path.glob("project.v1-backup-*.sqlite"))
    assert len(backups) == 1
    assert (tmp_path / "project.v1-backup.sqlite").read_bytes() == b"previous backup"
    with sqlite3.connect(backups[0]) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
        assert json.loads(db.execute("SELECT landmarks FROM annotations").fetchone()[0]) == dict(zip(names, range(len(names))))
    with Project(tmp_path) as project:
        assert project.get("classes") == ids
    assert len(list(tmp_path.glob("project.v1-backup-*.sqlite"))) == 1


def test_rename_color_preserve_annotations_but_criteria_order_reconfirm(tmp_path):
    with Project(tmp_path) as project:
        assert project.get("classes") == DEFAULT_CLASSES
        assert not list(tmp_path.glob("*backup*"))
        for relative, status in [("done", "complete"), ("invalid", "invalid"), ("skip", "skipped")]:
            project.save_annotation(relative, dict(zip(DEFAULT_CLASSES, [1, 3, 5])), "sig", status)
        defs = project.class_definitions()
        defs[0].update(name="새 이름", color="#112233")
        project.update_class_definitions(defs)
        assert project.annotations()["done"]["status"] == "complete"
        assert project.annotations()["done"]["landmarks"]["start"] == 1
        defs[0]["description"] = "처음 나타나는 선명한 경계"
        project.update_class_definitions(defs)
        assert project.annotations()["done"]["status"] == "draft"
        assert project.annotations()["invalid"]["status"] == "invalid"
        assert project.annotations()["skip"]["status"] == "skipped"
        project.save_annotation("done", dict(zip(DEFAULT_CLASSES, [1, 3, 5])), "sig", "complete")
        project.update_class_definitions(list(reversed(defs)))
        assert project.annotations()["done"]["landmarks"] == dict(zip(DEFAULT_CLASSES, [1, 3, 5]))
        assert project.annotations()["done"]["status"] == "draft"


def test_class_add_remove_preserves_surviving_values_and_model_snapshot(tmp_path):
    with Project(tmp_path) as project:
        defs = project.class_definitions()
        project.save_annotation("sample", dict(zip(DEFAULT_CLASSES, [1, 3, 5])), "sig", "complete")
        project.add_model({"id": "fixed", "path": str(tmp_path / "model.sem-model"),
                           "classes": DEFAULT_CLASSES, "class_definitions": defs})
        added = new_definition("새 경계", 3)
        project.update_class_definitions([defs[0], added, defs[2]])
        assert project.annotations()["sample"]["landmarks"] == {"start": 1, "end": 5}
        assert project.annotations()["sample"]["status"] == "draft"
        assert project.models()[0]["class_definitions"] == defs


def test_definition_update_rolls_back_all_writes_on_failure(tmp_path):
    with Project(tmp_path) as project:
        project.save_annotation("sample", {"start": 0}, "sig", "complete")
        original = project.class_definitions()
        project._db.execute("CREATE TRIGGER prevent_annotation_update BEFORE UPDATE ON annotations BEGIN SELECT RAISE(ABORT, 'test failure'); END")
        with pytest.raises(sqlite3.IntegrityError):
            project.update_class_definitions(list(reversed(original)))
        assert project.class_definitions() == original
        assert project.get("classes") == DEFAULT_CLASSES
        assert project.annotations()["sample"]["status"] == "complete"


def test_legacy_script_name_labels_are_saved_as_stable_ids(tmp_path):
    with Project(tmp_path) as project:
        project.set("classes", ["진입", "완료"])
        project.save_annotation("sample", {"진입": 2, "완료": 7}, "sig", "complete")
        ids = class_ids(project.class_definitions())
        assert project.annotations()["sample"]["landmarks"] == dict(zip(ids, [2, 7]))


def test_schema1_long_class_names_are_preserved_without_truncation(tmp_path):
    _legacy_project(tmp_path)
    names = ["경계 설명이 포함된 긴 기존 클래스 " * 30, "종료 " * 50]
    names = [name.rstrip() for name in names]
    with sqlite3.connect(tmp_path / "project.sqlite") as db:
        db.execute("UPDATE settings SET value=? WHERE key='classes'", (json.dumps(names),))
        db.execute("UPDATE annotations SET landmarks=?", (json.dumps(dict(zip(names, [2, 6]))),))
    with Project(tmp_path) as project:
        assert [item["name"] for item in project.class_definitions()] == names
        assert project.annotations()["sample"]["landmarks"] == dict(zip(project.get("classes"), [2, 6]))
