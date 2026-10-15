from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from semreview.core import scan
from semreview.project import DEFAULT_CLASSES, DEFAULT_PROFILES, Project, sequence_signature
from semreview.ui import MainWindow, SettingsDialog


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    base = tmp_path / "source"
    for batch in ["batch-a", "batch-b"]:
        leaf = base / batch / "sample"
        leaf.mkdir(parents=True)
        for i in range(12):
            yy, xx = np.mgrid[:32, :48]
            image = ((xx * 3 + yy * 5 + i * 17) % 256).astype(np.uint8)
            tifffile.imwrite(leaf / f"{10 + i}.tif", image)
    sequences = scan(str(base), DEFAULT_PROFILES)
    project_dir = tmp_path / "project"
    with Project(project_dir) as project:
        project.set("base", str(base))
        project.set("sequences", sequences)
        project.set("classes", DEFAULT_CLASSES)
        for seq in sequences:
            project.save_annotation(seq["relative_path"], dict(zip(DEFAULT_CLASSES, [2, 5, 8])),
                                    sequence_signature(seq), "complete", 2)
        # UI model-selection fixture, never submitted to the ML loader.
        project.add_model({"id": "fixture-model", "name": "검증용 모델", "classes": DEFAULT_CLASSES,
                           "path": str(project_dir / "fixture.sem-model"), "metrics": {"validated": False}})
    widget = MainWindow(str(project_dir))
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitUntil(lambda: widget.current_sequence is not None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **kw: QMessageBox.StandardButton.Yes)
    yield widget
    if widget.jobs.busy:
        widget.jobs.cancel()
        qtbot.waitUntil(lambda: not widget.jobs.busy, timeout=20000)
    if widget.rule_jobs.busy:
        widget.rule_jobs.cancel()
        qtbot.waitUntil(lambda: not widget.rule_jobs.busy, timeout=20000)
    widget.editor.cache.pool.waitForDone(5000)
    widget.review_editor.cache.pool.waitForDone(5000)
    widget.close()


def populate_review(window):
    window.tabs.setCurrentIndex(1)
    window.offset.setValue(2)
    window.predictions = [{"relative_path": s["relative_path"], "landmarks": dict(zip(DEFAULT_CLASSES, [2, 5, 8])),
                           "model_id": "fixture-model", "priority": .4, "warnings": [], "candidates": {}}
                          for s in window.sequences]
    window._rebuild_plans()
    window.review_table.setCurrentCell(0, 0)


def test_open_restores_labels_and_cursor(window):
    assert window.editor.cursor == 2
    assert window.editor.landmarks == dict(zip(DEFAULT_CLASSES, [2, 5, 8]))
    assert window.folder_list.count() == 2
    assert window.review_model.currentData()["id"] == "fixture-model"


def test_annotation_edit_autosaves_and_undoes(window):
    relative = window.current_sequence["relative_path"]
    window.editor.assign(DEFAULT_CLASSES[0], 1)
    assert window.project.annotations()[relative]["landmarks"][DEFAULT_CLASSES[0]] == 1
    assert window.project.annotations()[relative]["status"] == "draft"
    window.editor.undo()
    assert window.project.annotations()[relative]["landmarks"][DEFAULT_CLASSES[0]] == 2


def test_training_payload_contains_only_checked_completed_folders(window, monkeypatch):
    collected = []
    monkeypatch.setattr(window, "_start", lambda command, payload: collected.append((command, payload)))
    window.project.set("group_depth", 2)
    window._select_completed()
    window.train()
    command, payload = collected[0]
    assert command == "train"
    assert len(payload["sequences"]) == 2
    assert payload["config"]["group_level"] == 2
    assert payload["config"]["allow_download"] is False


def test_review_edit_is_independent_and_preserves_other_approvals(window):
    populate_review(window)
    for row in range(2):
        window.review_table.item(row, 1).setCheckState(Qt.CheckState.Checked)
    before = window.project.annotations()
    window._review_labels_changed({DEFAULT_CLASSES[0]: 1, DEFAULT_CLASSES[1]: 5, DEFAULT_CLASSES[2]: 8})
    assert window.project.annotations() == before
    assert window.review_table.item(0, 1).checkState() == Qt.CheckState.Unchecked
    assert window.review_table.item(1, 1).checkState() == Qt.CheckState.Checked


def test_changing_offset_invalidates_approvals(window):
    populate_review(window)
    window.review_table.item(0, 1).setCheckState(Qt.CheckState.Checked)
    window.offset.setValue(0)
    assert window.review_table.item(0, 1).checkState() == Qt.CheckState.Unchecked
    assert window.plans[0]["end"] == 8
    assert len(window.plans[0]["keep"]) == 7


def test_rescan_invalidates_old_predictions_and_changed_labels(window):
    populate_review(window)
    seq = window.sequences[0]
    tifffile.imwrite(seq["images"][0]["path"], np.zeros((32, 48), np.uint8))
    rescanned = scan(window.base_path.text(), DEFAULT_PROFILES)
    window.operation = "scan"
    window._finished(rescanned)
    assert window.predictions == []
    assert window.project.annotations()[seq["relative_path"]]["status"] == "invalid"
    assert window.editor.landmarks == {}


def test_ui_real_apply_and_restore_worker_roundtrip(window, qtbot):
    populate_review(window)
    originals = {im["path"]: Path(im["path"]).read_bytes() for seq in window.sequences for im in seq["images"]}
    window.review_table.item(0, 1).setCheckState(Qt.CheckState.Checked)
    expected_keep = set(window.plans[0]["keep"])
    folder = Path(window.sequences[0]["path"])
    with qtbot.waitSignal(window.jobs.finished, timeout=20000):
        window.apply_reviews()
    assert {str(p) for p in folder.glob("*.tif")} == expected_keep
    assert window.runs_table.rowCount() == 1
    window.runs_table.setCurrentCell(0, 0)
    with qtbot.waitSignal(window.jobs.finished, timeout=20000):
        window.run_action("restore")
    assert all(Path(path).read_bytes() == data for path, data in originals.items())


def test_settings_support_arbitrary_depth_and_classes(window, qtbot):
    dialog = SettingsDialog(window.project)
    qtbot.addWidget(dialog)
    dialog._add_level()
    dialog._add_level()
    dialog.class_names.setText("바닥, 1층, 2층, 3층, 4층")
    dialog.accept()
    assert len(dialog.profiles[0]["levels"]) == 4
    assert dialog.classes == ["바닥", "1층", "2층", "3층", "4층"]


def test_input_fields_do_not_trigger_label_shortcuts(window, qtbot):
    window.activateWindow()
    window.model_name.setFocus()
    before = dict(window.editor.landmarks)
    qtbot.keyClicks(window.model_name, "123")
    assert window.editor.landmarks == before
