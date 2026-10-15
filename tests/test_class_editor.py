import json

from PySide6.QtWidgets import QDialog, QFileDialog

from semreview.class_editor import ClassDefinitionsDialog
from semreview.classdefs import DEFAULT_CLASS_DEFINITIONS, new_definition


def make_dialog(qtbot, definitions=None):
    dialog = ClassDefinitionsDialog(definitions or DEFAULT_CLASS_DEFINITIONS)
    qtbot.addWidget(dialog)
    return dialog


def test_many_arbitrary_classes_bulk_add_and_reorder_keeps_identity(qtbot):
    dialog = make_dialog(qtbot)
    dialog.bulk_names.setPlainText("\n".join(f"패턴 상태 {i}" for i in range(4, 13)))
    dialog.add_bulk()
    assert dialog.table.rowCount() == 12
    definitions = dialog.current_definitions()
    dialog.table.selectRow(11)
    dialog.move_selected(-1)
    dialog.table.item(10, 1).setText("상부 연결 시작")
    dialog.table.item(10, 2).setText("분리된 패턴이 처음 연결되는 장")
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.definitions[10]["id"] == definitions[11]["id"]
    assert dialog.definitions[10]["description"] == "분리된 패턴이 처음 연결되는 장"


def test_duplicate_or_empty_names_and_invalid_color_are_rejected(qtbot):
    dialog = make_dialog(qtbot)
    dialog.table.item(1, 1).setText(dialog.table.item(0, 1).text())
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.error.text()
    dialog.table.item(1, 1).setText("")
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted
    dialog.table.item(1, 1).setText("테스트")
    dialog.table.item(1, 3).setText("red")
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_bulk_limit_and_minimum_count(qtbot):
    dialog = make_dialog(qtbot)
    dialog.bulk_names.setPlainText("\n".join(f"상태 {i}" for i in range(62)))
    dialog.add_bulk()
    assert dialog.table.rowCount() == 3
    assert dialog.error.text()
    dialog.table.selectRow(0)
    dialog.remove_selected()
    dialog.table.selectRow(0)
    dialog.remove_selected()
    dialog.accept()
    assert dialog.result() != QDialog.DialogCode.Accepted


def test_template_is_optional_and_supports_sixty_four_classes(qtbot):
    dialog = make_dialog(qtbot)
    assert dialog.current_definitions()[0]["name"] == "시작 기준점"
    dialog.ceiling_count.setValue(63)
    dialog.apply_floor_template()
    assert dialog.table.rowCount() == 64
    assert dialog.current_definitions()[-1]["name"] == "63층 천장"
    assert dialog.current_definitions()[0]["id"] != DEFAULT_CLASS_DEFINITIONS[0]["id"]
    dialog.accept()
    assert dialog.result() == QDialog.DialogCode.Accepted


def test_definition_json_roundtrip_and_cancel_leave_original_untouched(qtbot, tmp_path, monkeypatch):
    definitions = [new_definition("초기 상태"), new_definition("완성 상태")]
    definitions[1]["description"] = "패턴 전체가 연결된 첫 이미지"
    dialog = make_dialog(qtbot, definitions)
    filename = tmp_path / "classes.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(filename), ""))
    dialog.export_definitions()
    assert json.loads(filename.read_text())["class_definitions"] == definitions
    another = make_dialog(qtbot)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(filename), ""))
    another.import_definitions()
    assert another.current_definitions() == definitions
    another.table.item(0, 1).setText("이름 수정")
    another.reject()
    assert definitions[0]["name"] == "초기 상태"
