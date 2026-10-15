from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from semreview.core import scan
from semreview.project import DEFAULT_CLASSES, DEFAULT_PROFILES, Project, sequence_signature
from semreview.ui import MainWindow, SettingsDialog
from semreview.class_editor import ClassDefinitionsDialog


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
                           "class_definitions": project.class_definitions(),
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
    assert payload["class_definitions"] == window.project.class_definitions()


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


def test_settings_support_arbitrary_depth(window, qtbot):
    dialog = SettingsDialog(window.project)
    qtbot.addWidget(dialog)
    dialog._add_level()
    dialog._add_level()
    dialog.accept()
    assert len(dialog.profiles[0]["levels"]) == 4
    assert not hasattr(dialog, "class_names")


def test_editing_class_name_keeps_labels_and_saved_model_snapshot(window, monkeypatch):
    original = window.project.class_definitions()
    changed = [dict(item) for item in original]
    changed[0].update(name="기판 노출")
    before = window.project.annotations()

    def save_dialog(dialog):
        dialog.definitions = changed
        return ClassDefinitionsDialog.DialogCode.Accepted

    monkeypatch.setattr(ClassDefinitionsDialog, "exec", save_dialog)
    window.edit_class_definitions()
    assert window.project.class_definitions() == changed
    assert window.project.annotations() == before
    assert "기판 노출" in window.classes_summary.text()
    assert window.editor.landmarks == before[window.current_sequence["relative_path"]]["landmarks"]
    assert window.review_model.currentData()["class_definitions"] == original
    assert window.end_class.itemText(0) == original[0]["name"]
    assert window.end_class.itemData(0) == DEFAULT_CLASSES[0]


def test_review_end_class_uses_id_not_display_name(window):
    populate_review(window)
    window.end_class.setCurrentIndex(1)
    assert window.end_class.currentText() != window.end_class.currentData()
    assert window.project.get("review_end_class") == DEFAULT_CLASSES[1]
    assert window.plans[0]["end"] == 7
    window._review_model_changed()
    assert window.end_class.currentData() == DEFAULT_CLASSES[1]


def test_missing_boundary_error_displays_model_class_name(window, monkeypatch):
    populate_review(window)
    missing_id = DEFAULT_CLASSES[1]
    missing_name = window.review_model.currentData()["class_definitions"][1]["name"]
    del window.predictions[0]["landmarks"][missing_id]
    window._rebuild_plans()
    expected = f"클래스 기준점이 없거나 범위를 벗어났습니다: {missing_name}"
    assert expected in window.plans[0]["errors"]
    assert expected in window.review_table.item(0, 5).text()
    captured = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, message: captured.append(message))
    window.review_table.setCurrentCell(0, 0)
    window.approve_current_review()
    assert expected in captured[0]
    assert f": {missing_id}" not in captured[0]


def test_class_addition_refreshes_labeling_and_invalidates_completion(window, monkeypatch):
    from semreview.classdefs import new_definition
    changed = window.project.class_definitions() + [new_definition("상부 연결")]

    def save_dialog(dialog):
        dialog.definitions = changed
        return ClassDefinitionsDialog.DialogCode.Accepted

    monkeypatch.setattr(ClassDefinitionsDialog, "exec", save_dialog)
    window.edit_class_definitions()
    assert len(window.editor.classes) == 4
    assert all(item["status"] == "draft" for item in window.project.annotations().values())
    assert "4개" in window.classes_summary.text()


def test_input_fields_do_not_trigger_label_shortcuts(window, qtbot):
    window.activateWindow()
    window.model_name.setFocus()
    before = dict(window.editor.landmarks)
    qtbot.keyClicks(window.model_name, "123")
    assert window.editor.landmarks == before


def test_class_shortcuts_preserve_folder_and_review_table_navigation(window, qtbot):
    window.activateWindow()
    window.folder_list.setCurrentRow(0)
    window.folder_list.setFocus()
    qtbot.keyClick(window.folder_list, Qt.Key.Key_Down)
    assert window.folder_list.currentRow() == 1
    populate_review(window)
    window.review_table.setCurrentCell(0, 1)
    window.review_table.setFocus()
    before = dict(window.review_editor.landmarks)
    qtbot.keyClick(window.review_table, Qt.Key.Key_Space)
    assert window.review_table.item(0, 1).checkState() == Qt.CheckState.Checked
    assert window.review_editor.landmarks == before
    qtbot.keyClick(window.review_table, Qt.Key.Key_Down)
    assert window.review_table.currentRow() == 1


def test_keyboard_can_select_and_assign_twelfth_class(window, qtbot):
    from semreview.classdefs import new_definition
    definitions = [new_definition(f"패턴 {i+1}", index=i) for i in range(12)]
    window.project.update_class_definitions(definitions)
    window.annotations = window.project.annotations()
    window._load_training_sequence(window.current_sequence)
    window.activateWindow()
    window.editor.label_table.setCurrentCell(10, 0)
    window.editor.label_table.setFocus()
    window.editor.set_cursor(11)
    qtbot.keyClick(window.editor.label_table, Qt.Key.Key_Down)
    assert window.editor.label_table.currentRow() == 11
    qtbot.keyClick(window.editor.label_table, Qt.Key.Key_Space)
    assert window.editor.landmarks[definitions[11]["id"]] == 11
    saved = window.project.annotations()[window.current_sequence["relative_path"]]
    assert saved["landmarks"][definitions[11]["id"]] == 11
