import sqlite3

import pytest

from semreview.project import Project, sequence_signature


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
