from pathlib import Path

import pytest

from semreview import core


CLASSES = ["바닥", "1층 천장", "2층 천장"]
PROFILES = [{"name": "default", "levels": [{"include": ["n*"], "exclude": []},
                                           {"include": ["m*"], "exclude": ["m_skip"]}]}]


def files(folder: Path, numbers=range(1, 11)):
    folder.mkdir(parents=True, exist_ok=True)
    for number in numbers:
        (folder / f"{number}.tif").write_bytes(f"image-{number}".encode())


@pytest.fixture
def dataset(tmp_path):
    base = tmp_path / "data"
    files(base / "n1" / "m1")
    return base


def plan_for(base, *, offset=0, floor=2, first=4, last=6):
    sequence = core.scan(str(base), PROFILES)[0]
    return core.make_review_plan(str(base), sequence,
                                 dict(zip(CLASSES, [floor, first, last])),
                                 CLASSES, CLASSES[-1], offset=offset)


def test_scan_numerical_order_and_hashes(dataset):
    sequence = core.scan(str(dataset), PROFILES)[0]
    assert [image["number"] for image in sequence["images"]] == list(range(1, 11))
    assert sequence["relative_path"] == "n1/m1"
    assert sequence["issues"] == []
    assert all(len(image["sha256"]) == 64 for image in sequence["images"])


def test_multiple_depth_profiles_union_exclusion_and_symlinks(dataset):
    files(dataset / "n1" / "m_skip")
    files(dataset / "n2" / "m2" / "deep" / "leaf")
    files(dataset / ".sem-review" / "m1")
    files(dataset / "outside")
    (dataset / "n_link").symlink_to(dataset / "n1", target_is_directory=True)
    profiles = PROFILES + PROFILES + [{"name": "four", "levels": [
        {"include": ["n2"]}, {"include": ["m2"]}, {"include": ["deep"]}, {"include": ["leaf"]}]}]
    assert [seq["relative_path"] for seq in core.scan(str(dataset), profiles)] == [
        "n1/m1", "n2/m2", "n2/m2/deep/leaf"]


def test_scan_cancel_does_not_return_partial(dataset):
    with pytest.raises(core.CancelledError):
        core.scan(str(dataset), PROFILES, cancelled=lambda: True)


def test_scan_reports_duplicate_and_bad_names(dataset):
    folder = dataset / "n1" / "m1"
    (folder / "01.tiff").write_bytes(b"duplicate")
    (folder / "notes.tif").write_bytes(b"unknown")
    sequence = core.scan(str(dataset), PROFILES)[0]
    assert any("중복" in issue for issue in sequence["issues"])
    assert any("숫자가 아닌" in issue for issue in sequence["issues"])
    assert plan_for(dataset)["errors"]


@pytest.mark.parametrize(("offset", "expected", "warning"), [
    (0, [3, 4, 5, 6, 7], False),
    (2, [3, 4, 5, 6, 7, 8, 9], False),
    (-2, [3, 4, 5], False),
    (30, [3, 4, 5, 6, 7, 8, 9, 10], True),
])
def test_inclusive_interval_offsets(dataset, offset, expected, warning):
    plan = plan_for(dataset, offset=offset)
    assert plan["errors"] == []
    assert [int(Path(name).stem) for name in plan["keep"]] == expected
    assert bool(plan["warnings"]) is warning
    assert set(plan["keep"]).isdisjoint(plan["move"])
    assert len(plan["keep"]) + len(plan["move"]) == 10


def test_end_class_choice_and_negative_invalid(dataset):
    sequence = core.scan(str(dataset), PROFILES)[0]
    plan = core.make_review_plan(str(dataset), sequence, dict(zip(CLASSES, [2, 4, 6])),
                                 CLASSES, CLASSES[1], 0)
    assert [int(Path(name).stem) for name in plan["keep"]] == [3, 4, 5]
    assert plan_for(dataset, offset=-10)["errors"]
    assert plan_for(dataset, floor=4, first=2)["errors"]


def test_non_one_start_and_only_internal_gaps_block(tmp_path):
    base = tmp_path / "data"
    files(base / "n1" / "m1", [5, 6, 7, 8, 9, 10, 20])
    plan = plan_for(base, floor=0, first=2, last=5)
    assert not plan["errors"]
    assert len(plan["keep"]) == 6
    (base / "n1" / "m1" / "7.tif").unlink()
    assert plan_for(base, floor=0, first=2, last=5)["errors"]


def test_apply_and_restore_preserve_exact_bytes_and_paths(dataset, tmp_path):
    before = {str(path): path.read_bytes() for path in dataset.rglob("*.tif")}
    plan = plan_for(dataset)
    project = str(tmp_path / "project")
    run = core.apply_plans([plan], project)
    assert run["status"] == "complete"
    assert run["moved"] == 5
    assert all(Path(path).exists() for path in plan["keep"])
    assert all(not Path(path).exists() for path in plan["move"])
    for entry in run["plans"][0]["entries"]:
        assert Path(entry["destination"]).read_bytes() == before[entry["source"]]
    assert len(core.list_runs(project)) == 1
    restored = core.restore_run(project, run["run_id"])
    assert restored["status"] == "restored"
    assert restored["restored"] == 5
    assert {str(path): path.read_bytes() for path in dataset.rglob("*.tif")} == before
    assert core.restore_run(project, run["run_id"])["status"] == "restored"


@pytest.mark.parametrize("change", ["contents", "added", "deleted"])
def test_changed_folder_blocked_but_unrelated_plan_applies(dataset, tmp_path, change):
    files(dataset / "n1" / "m2")
    sequences = core.scan(str(dataset), PROFILES)
    plans = [core.make_review_plan(str(dataset), seq, dict(zip(CLASSES, [2, 4, 6])),
                                   CLASSES, CLASSES[-1], 0) for seq in sequences]
    folder = dataset / "n1" / "m1"
    if change == "contents":
        (folder / "5.tif").write_bytes(b"changed keep image")
    elif change == "added":
        (folder / "11.tif").write_bytes(b"new image")
    else:
        (folder / "5.tif").unlink()
    run = core.apply_plans(plans, str(tmp_path / "project"))
    assert run["status"] == "partial"
    assert run["plans"][0]["status"] == "blocked"
    assert run["plans"][1]["status"] == "complete"
    assert (folder / "1.tif").exists()


def test_cancel_then_resume_and_restore(dataset, tmp_path):
    project = str(tmp_path / "project")
    events = []
    run = core.apply_plans([plan_for(dataset)], project, progress=events.append,
                           cancelled=lambda: len(events) >= 2)
    assert run["status"] == "cancelled" and run["moved"] == 2
    resumed = core.resume_run(project, run["run_id"])
    assert resumed["status"] == "complete" and resumed["moved"] == 5
    assert core.restore_run(project, run["run_id"])["status"] == "restored"


def test_restore_does_not_overwrite_new_original(dataset, tmp_path):
    project = str(tmp_path / "project")
    run = core.apply_plans([plan_for(dataset)], project)
    entry = run["plans"][0]["entries"][0]
    Path(entry["source"]).write_bytes(b"new important file")
    restored = core.restore_run(project, run["run_id"])
    assert restored["status"] == "partial"
    assert restored["conflicts"] == 1
    assert Path(entry["source"]).read_bytes() == b"new important file"
    assert Path(entry["destination"]).exists()
    Path(entry["source"]).unlink()
    assert core.restore_run(project, run["run_id"])["status"] == "restored"


def test_no_overwrite_move_primitive(tmp_path):
    source, destination = tmp_path / "a", tmp_path / "b"
    source.write_bytes(b"source")
    destination.write_bytes(b"destination")
    with pytest.raises(FileExistsError):
        core._move_noreplace(source, destination)
    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"destination"


def test_quarantine_symlink_blocked(dataset, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (dataset / ".sem-review").symlink_to(outside, target_is_directory=True)
    plan = plan_for(dataset)
    run = core.apply_plans([plan], str(tmp_path / "project"))
    assert run["status"] == "partial"
    assert all(Path(path).exists() for path in plan["move"])
    assert list(outside.iterdir()) == []


def test_tampered_keep_move_lists_rejected(dataset, tmp_path):
    plan = plan_for(dataset)
    plan["move"].append(plan["keep"].pop())
    run = core.apply_plans([plan], str(tmp_path / "project"))
    assert run["status"] == "partial" and run["moved"] == 0


def test_caller_validation_errors_are_respected(dataset, tmp_path):
    plan = plan_for(dataset)
    plan["errors"].append("선택한 모델과 분석 모델이 다릅니다.")
    run = core.apply_plans([plan], str(tmp_path / "project"))
    assert run["status"] == "partial" and run["moved"] == 0
    assert all(Path(path).exists() for path in plan["move"])


def test_no_cross_volume_copy_fallback(dataset, tmp_path, monkeypatch):
    def cross_volume(source, destination):
        import errno
        raise OSError(errno.EXDEV, "different volume")
    monkeypatch.setattr(core, "_move_noreplace", cross_volume)
    plan = plan_for(dataset)
    run = core.apply_plans([plan], str(tmp_path / "project"))
    assert run["status"] == "partial"
    assert all(Path(path).exists() for path in plan["move"])


def test_list_runs_is_readonly_for_missing_project(tmp_path):
    project = tmp_path / "missing"
    assert core.list_runs(str(project)) == []
    assert not project.exists()


def test_tiff_symlink_excluded_from_scan_and_review(dataset, tmp_path):
    target = tmp_path / "external.tif"
    target.write_bytes(b"do not touch")
    (dataset / "n1" / "m1" / "200.tif").symlink_to(target)
    plan = plan_for(dataset)
    assert not plan["errors"] and len(plan["sequence"]["images"]) == 10
    run = core.apply_plans([plan], str(tmp_path / "project"))
    assert run["status"] == "complete"
    assert target.read_bytes() == b"do not touch"


def test_plan_cannot_escape_base_with_dotdot(dataset):
    plan = plan_for(dataset)
    sequence = plan["sequence"]
    sequence["path"] = str(dataset / ".." / "elsewhere")
    invalid = core.make_review_plan(str(dataset), sequence, plan["landmarks"], CLASSES, CLASSES[-1], 0)
    assert invalid["errors"]


@pytest.mark.parametrize("change", ["duplicate", "reorder"])
def test_invalid_order_outside_keep_interval_is_rejected(dataset, change):
    plan = plan_for(dataset)
    sequence = plan["sequence"]
    if change == "duplicate":
        sequence["images"][0] = sequence["images"][1].copy()
    else:
        sequence["images"][0], sequence["images"][1] = sequence["images"][1], sequence["images"][0]
    invalid = core.make_review_plan(str(dataset), sequence, plan["landmarks"], CLASSES, CLASSES[-1], 0)
    assert invalid["errors"]


def test_scan_cancel_after_final_progress_never_returns_inventory(dataset):
    events = []
    with pytest.raises(core.CancelledError):
        core.scan(str(dataset), PROFILES, progress=events.append, cancelled=lambda: bool(events))


def test_full_folder_preflight_can_be_cancelled(dataset, tmp_path, monkeypatch):
    plan = plan_for(dataset)
    project = str(tmp_path / "project")
    cancelled = [False]
    original_hash = core._hash_file
    def cancel_after_first_hash(path, cancel_check=None):
        result = original_hash(path, cancel_check)
        cancelled[0] = True
        return result
    monkeypatch.setattr(core, "_hash_file", cancel_after_first_hash)
    run = core.apply_plans([plan], project, cancelled=lambda: cancelled[0])
    assert run["status"] == "cancelled" and run["moved"] == 0
    assert all(Path(path).exists() for path in plan["keep"] + plan["move"])
    assert core.list_runs(project)[0]["status"] == "cancelled"


def test_progress_aliases_and_completed_run_replay(dataset, tmp_path):
    events = []
    project = str(tmp_path / "project")
    core.scan(str(dataset), PROFILES, progress=events.append)
    run = core.apply_plans([plan_for(dataset)], project, progress=events.append)
    core.resume_run(project, run["run_id"], progress=events.append)
    core.restore_run(project, run["run_id"], progress=events.append)
    core.restore_run(project, run["run_id"], progress=events.append)
    assert all(event["current"] == event["completed"] for event in events)
    assert events[-1]["completed"] == events[-1]["total"] == 5


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows junction integration")
def test_windows_directory_junctions_are_excluded(dataset):
    import subprocess
    junction = dataset / "n_junction"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(dataset / "n1")],
                   check=True, capture_output=True)
    assert junction.is_junction()
    assert [seq["relative_path"] for seq in core.scan(str(dataset), PROFILES)] == ["n1/m1"]
    with pytest.raises(ValueError):
        core._safe_path(junction / "m1" / "1.tif", dataset.resolve())


def test_operation_mutex_blocks_other_process_then_releases(tmp_path):
    import os
    import subprocess
    import sys
    directory = tmp_path / "runs"
    directory.mkdir()
    code = ("import sys; from pathlib import Path; from semreview.core import _run_lock; "
            "lock = _run_lock(Path(sys.argv[1])); lock.__enter__(); "
            "print('locked', flush=True); input(); lock.__exit__(None, None, None)")
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    child = subprocess.Popen([sys.executable, "-c", code, str(directory)], env=environment,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError):
            with core._run_lock(directory):
                pytest.fail("Concurrent mutation acquired the same project lock")
        child.communicate("release\n", timeout=10)
        assert child.returncode == 0
        with core._run_lock(directory):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=10)
