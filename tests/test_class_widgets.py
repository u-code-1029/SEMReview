"""Class presentation must never change landmark identity or filtered-row assignment."""
from copy import deepcopy

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage

from semreview.classdefs import DEFAULT_CLASS_DEFINITIONS, DEFAULT_CLASSES
from semreview.widgets import SequenceEditor


@pytest.fixture
def many_editor(qtbot, monkeypatch):
    editor = SequenceEditor()
    qtbot.addWidget(editor)
    image = QImage(48, 32, QImage.Format.Format_Grayscale8)
    image.fill(100)
    monkeypatch.setattr(editor.cache, 'request', lambda path, size=0: (str(path), image))
    definitions = [{'id': f'stable_{i}', 'name': f'맞춤 상태 {i+1:02d}',
                    'description': f'특징 {i+1:02d}의 전환이 처음 나타나는 이미지',
                    'color': f'#{i+20:02x}aabb'} for i in range(12)]
    sequence = {'relative_path': 'batch/sample', 'path': '/sample',
                'images': [{'path': f'/sample/{i+1}.tif', 'name': f'{i+1}.tif', 'number': i+1}
                           for i in range(36)]}
    editor.set_sequence(sequence, [d['id'] for d in definitions], class_definitions=definitions)
    editor.resize(1480, 960)
    editor.show()
    yield editor, definitions, sequence
    editor.cache.pool.waitForDone(5000)


def test_class_twelve_can_be_selected_and_assigned_by_navigation(many_editor):
    editor, definitions, _ = many_editor
    for _ in range(11):
        editor.select_next_class(1)
    assert editor.label_table.currentRow() == 11
    editor.set_cursor(25)
    editor.assign_selected()
    assert editor.landmarks == {definitions[11]['id']: 25}
    assert editor.class_progress.text() == '1 / 12개 지정'
    assert definitions[11]['description'] in editor.class_criterion.text()
    editor.select_next_class(1)
    assert editor.label_table.currentRow() == 0
    editor.select_next_class(-1)
    assert editor.label_table.currentRow() == 11


def test_filter_matches_criteria_and_preserves_high_index_identity(many_editor):
    editor, definitions, _ = many_editor
    editor.class_filter.setText('특징 12')
    assert editor.label_table.currentRow() == 11
    assert sum(not editor.label_table.isRowHidden(i) for i in range(12)) == 1
    editor.set_cursor(30)
    editor.assign_selected()
    assert editor.landmarks == {definitions[11]['id']: 30}
    editor.set_cursor(1)
    editor._jump_landmark(11, 0)
    assert editor.cursor == 30
    editor.remove_selected()
    assert editor.landmarks == {}
    editor.undo()
    assert editor.landmarks == {definitions[11]['id']: 30}
    editor.class_filter.setText('검색 결과 없음')
    editor.assign_selected()
    assert editor.landmarks == {definitions[11]['id']: 30}
    editor.assign_number(1)
    assert editor.class_filter.text() == ''
    assert editor.landmarks[definitions[1]['id']] == 30


def test_renaming_class_keeps_landmarks_review_range_and_analysis(many_editor, qtbot):
    editor, definitions, sequence = many_editor
    ids = [d['id'] for d in definitions]
    landmarks = {key: i+2 for i, key in enumerate(ids)}
    editor.set_sequence(sequence, ids, landmarks, class_definitions=definitions)
    editor.set_cursor(20)
    editor.label_table.selectRow(11)
    editor.assign_selected()
    editor.set_keep_range(2, 32)
    editor.set_analysis({'scores': [0.1]*36, 'candidates': {ids[11]: [20]}})
    renamed = deepcopy(definitions)
    renamed[11].update(name='마지막 전환점', description='이름과 설명을 수정한 기준', color='#ff00aa')
    editor.set_sequence(sequence, ids, editor.landmarks, class_definitions=renamed)
    assert editor.landmarks[ids[11]] == 20
    assert editor.undo_stack
    assert editor.cursor == 20
    assert editor.timeline.keep_range == (2, 32)
    assert editor.timeline.scores == [0.1]*36
    assert editor.label_table.item(11, 0).text() == '12. 마지막 전환점'
    assert editor.label_table.item(11, 0).foreground().color().name() == '#ff00aa'
    assert editor.timeline.class_name(ids[11]) == '마지막 전환점'
    assert editor.timeline.class_color(ids[11]) == '#ff00aa'
    assert '마지막 전환점' in editor.candidates.itemText(0)
    assert '이름과 설명을 수정한 기준' in editor.class_criterion.text()
    qtbot.mouseMove(editor.timeline, QPoint(round(editor.timeline._x(20)), 100))
    assert '마지막 전환점' in editor.timeline.toolTip()
    assert ids[11] not in editor.timeline.toolTip()


def test_default_id_callers_display_names_and_many_errors_stay_compact(many_editor):
    editor, _, sequence = many_editor
    assert len(editor.validation_errors()) == 12
    assert len(editor.warning.text().splitlines()) == 4
    assert 'stable_' not in editor.warning.text()
    editor.set_sequence(sequence, DEFAULT_CLASSES)
    for row, definition in enumerate(DEFAULT_CLASS_DEFINITIONS):
        assert definition['name'] in editor.label_table.item(row, 0).text()
        assert editor.label_table.item(row, 0).data(Qt.ItemDataRole.UserRole) == definition['id']


def test_legacy_name_keys_still_work_without_definitions(many_editor):
    editor, _, sequence = many_editor
    names = ['기존 시작', '기존 종료']
    editor.set_sequence(sequence, names, {names[0]: 2, names[1]: 25})
    assert editor.label_table.item(1, 0).text() == '2. 기존 종료'
    assert editor.validation_errors() == []
