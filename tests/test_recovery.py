import os
from pathlib import Path

import pytest

from semreview import core


def setup_run(tmp_path):
    base = tmp_path / "data"
    folder = base / "one"
    folder.mkdir(parents=True)
    for number in range(1, 8):
        (folder / f"{number}.tif").write_bytes(f"unique-{number}".encode())
    sequence = core.scan(str(base), [{"levels": [{"include": ["*"]}]}])[0]
    plan = core.make_review_plan(str(base), sequence, {"floor": 1, "ceiling1": 3, "ceiling2": 4},
                                 ["floor", "ceiling1", "ceiling2"], "ceiling2", 0)
    return base, plan, str(tmp_path / "project")


@pytest.mark.parametrize("failure", ["before_move", "after_move", "after_link"])
def test_interrupted_move_can_resume(tmp_path, monkeypatch, failure):
    base, plan, project = setup_run(tmp_path)
    original_move = core._move_noreplace
    def interrupted(source, destination):
        if failure == "after_move":
            original_move(source, destination)
        elif failure == "after_link":
            os.link(source, destination)
        raise KeyboardInterrupt("simulated process termination")
    monkeypatch.setattr(core, "_move_noreplace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        core.apply_plans([plan], project)
    journal = core.list_runs(project)[0]
    assert journal["plans"][0]["entries"][0]["status"] == "moving"
    monkeypatch.setattr(core, "_move_noreplace", original_move)
    run = core.resume_run(project, journal["run_id"])
    assert run["status"] == "complete" and run["moved"] == 3
    assert all(Path(path).exists() for path in plan["keep"])
    assert all(not Path(path).exists() for path in plan["move"])
    assert core.restore_run(project, run["run_id"])["status"] == "restored"


@pytest.mark.parametrize("failure", ["before_move", "after_move", "after_link"])
def test_interrupted_restore_can_resume(tmp_path, monkeypatch, failure):
    base, plan, project = setup_run(tmp_path)
    run = core.apply_plans([plan], project)
    original_move = core._move_noreplace
    def interrupted(source, destination):
        if failure == "after_move":
            original_move(source, destination)
        elif failure == "after_link":
            os.link(source, destination)
        raise KeyboardInterrupt("simulated process termination")
    monkeypatch.setattr(core, "_move_noreplace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        core.restore_run(project, run["run_id"])
    monkeypatch.setattr(core, "_move_noreplace", original_move)
    restored = core.resume_run(project, run["run_id"])
    assert restored["status"] == "restored" and restored["restored"] == 3
    assert all(Path(path).exists() for path in plan["keep"] + plan["move"])


def test_new_same_content_destination_is_still_a_collision(tmp_path):
    base, plan, project = setup_run(tmp_path)
    run = core.apply_plans([plan], project, cancelled=lambda: True)
    entry = run["plans"][0]["entries"][0]
    destination = Path(entry["destination"])
    destination.parent.mkdir(parents=True)
    # Even a hardlink supplied externally before this run starts moving is
    # a collision, not permission to unlink the original.
    os.link(entry["source"], destination)
    resumed = core.resume_run(project, run["run_id"])
    assert resumed["status"] == "partial"
    assert Path(entry["source"]).exists() and destination.exists()


def test_kept_image_change_blocks_partial_run_resume(tmp_path):
    base, plan, project = setup_run(tmp_path)
    events = []
    run = core.apply_plans([plan], project, progress=events.append,
                           cancelled=lambda: len(events) == 1)
    Path(plan["keep"][0]).write_bytes(b"changed kept image")
    resumed = core.resume_run(project, run["run_id"])
    assert resumed["status"] == "partial" and resumed["moved"] == 1
    assert Path(plan["move"][1]).exists()


def test_restore_after_cancel_only_restores_completed_moves(tmp_path):
    base, plan, project = setup_run(tmp_path)
    events = []
    run = core.apply_plans([plan], project, progress=events.append,
                           cancelled=lambda: len(events) == 1)
    restored = core.restore_run(project, run["run_id"])
    assert restored["status"] == "restored" and restored["restored"] == 1
    assert all(Path(path).exists() for path in plan["keep"] + plan["move"])


def test_completed_move_does_not_ignore_recreated_original(tmp_path):
    base, plan, project = setup_run(tmp_path)
    run = core.apply_plans([plan], project)
    entry = run["plans"][0]["entries"][0]
    os.link(entry["destination"], entry["source"])
    resumed = core.resume_run(project, run["run_id"])
    assert resumed["status"] == "partial"
    assert Path(entry["source"]).exists()


def test_tampered_journal_destination_is_rejected(tmp_path):
    import json
    base, plan, project = setup_run(tmp_path)
    run = core.apply_plans([plan], project)
    journal_path = Path(project) / "review-runs" / f"{run['run_id']}.json"
    document = json.loads(journal_path.read_text())
    document["plans"][0]["entries"][0]["destination"] = str(tmp_path / "other.tif")
    journal_path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        core.restore_run(project, run["run_id"])
    assert core.list_runs(project)[0]["status"] == "unreadable"


def test_source_changed_between_validation_and_rename_is_returned(tmp_path, monkeypatch):
    base, plan, project = setup_run(tmp_path)
    original_move = core._move_noreplace
    changed = []
    def producer_replaced_source(source, destination):
        if not changed:
            source.write_bytes(b"new producer contents")
            changed.append(str(source))
        original_move(source, destination)
    monkeypatch.setattr(core, "_move_noreplace", producer_replaced_source)
    run = core.apply_plans([plan], project)
    assert run["status"] == "partial"
    assert Path(changed[0]).read_bytes() == b"new producer contents"
    entry = next(entry for entry in run["plans"][0]["entries"] if entry["source"] == changed[0])
    assert not Path(entry["destination"]).exists()
    assert all(Path(path).exists() for path in plan["keep"])
    assert all(Path(path).exists() for path in plan["move"])


@pytest.mark.skipif(os.name == "nt", reason="POSIX fallback; Windows uses no-clobber rename")
def test_link_fallback_never_unlinks_replaced_source(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"original")
    original_link = os.link
    def producer_replaced_source(src, dst, **kwargs):
        original_link(src, dst, **kwargs)
        src.unlink()
        src.write_bytes(b"new file")
    monkeypatch.setattr(core.ctypes, "CDLL", lambda *args, **kwargs: object())
    monkeypatch.setattr(os, "link", producer_replaced_source)
    with pytest.raises(OSError):
        core._move_noreplace(source, destination)
    assert source.read_bytes() == b"new file"
    assert destination.read_bytes() == b"original"
