"""Explicit, ordered class definitions with persistent identities."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QColorDialog, QDialog, QDialogButtonBox, QFileDialog,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .classdefs import new_definition, validate_definitions


class ClassDefinitionsDialog(QDialog):
    """Edit names separately from the IDs used by labels and saved models."""

    def __init__(self, definitions, parent=None):
        super().__init__(parent)
        self.setWindowTitle('클래스 정의 · 이름, 순서, 라벨 기준')
        self.resize(970, 750)
        self.definitions = [dict(item) for item in definitions]
        self._original_ids = [item['id'] for item in definitions]
        self._original_descriptions = {item['id']: item.get('description', '') for item in definitions}
        layout = QVBoxLayout(self)
        title = QLabel('분류할 상태를 직접 정의하세요')
        title.setObjectName('sectionTitle')
        layout.addWidget(title)
        intro = QLabel('각 클래스는 이미지 흐름에서 한 번씩 나타나는 상태의 시작점입니다. 이름에는 구조나 의미 제한이 없습니다.\n'
                       '위에서 아래로 등장 순서를 정하세요. 검수 시 첫 클래스부터 선택한 끝점 클래스 + 추가 장수까지 연속 보존합니다.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['순서', '클래스 이름', '정의 · 무엇을 보고 시작점을 지정하나요?', '색상'])
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(1, 200)
        self.table.itemChanged.connect(self._update_summary)
        self.table.cellDoubleClicked.connect(self._choose_color)
        layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        for text, callback in [('+ 클래스 추가', self.add_class), ('선택 삭제', self.remove_selected),
                               ('↑ 위로', lambda: self.move_selected(-1)), ('↓ 아래로', lambda: self.move_selected(1))]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            actions.addWidget(button)
        actions.addStretch()
        self.summary = QLabel()
        actions.addWidget(self.summary)
        layout.addLayout(actions)
        self.bulk_names = QPlainTextEdit()
        self.bulk_names.setPlaceholderText('여러 클래스를 한 번에 추가 · 한 줄에 하나의 이름\n예: 하부 표면\n패턴 출현\n상부 연결')
        self.bulk_names.setMaximumHeight(85)
        bulk_row = QHBoxLayout()
        bulk_row.addWidget(self.bulk_names, 1)
        bulk_button = QPushButton('여러 이름 추가')
        bulk_button.clicked.connect(self.add_bulk)
        bulk_row.addWidget(bulk_button)
        layout.addLayout(bulk_row)
        templates = QHBoxLayout()
        templates.addWidget(QLabel('선택용 예시 · 바닥 + 천장'))
        self.ceiling_count = QSpinBox()
        self.ceiling_count.setRange(1, 63)
        self.ceiling_count.setValue(2)
        self.ceiling_count.setSuffix('개 층')
        templates.addWidget(self.ceiling_count)
        template_button = QPushButton('예시로 새 구성')
        template_button.clicked.connect(self.apply_floor_template)
        templates.addWidget(template_button)
        templates.addStretch()
        for text, callback in [('정의 가져오기', self.import_definitions), ('정의 내보내기', self.export_definitions)]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            templates.addWidget(button)
        layout.addLayout(templates)
        self.change_note = QLabel()
        self.change_note.setObjectName('warning')
        self.change_note.setWordWrap(True)
        layout.addWidget(self.change_note)
        note = QLabel('이름·색상을 바꾸면 기존 라벨과 완료 상태를 유지합니다. 정의 설명·추가·삭제·순서 변경 시 완료 상태를 재확인합니다.\n'
                      '저장된 모델은 학습 당시의 클래스 정의를 유지합니다. 색상 칸을 더블 클릭하면 색을 선택할 수 있습니다.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        layout.addWidget(note)
        self.error = QLabel()
        self.error.setObjectName('warning')
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText('정의 저장')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._set_rows(definitions)

    def _set_rows(self, definitions):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for definition in definitions:
            self._append_row(definition)
        self.table.blockSignals(False)
        if definitions:
            self.table.selectRow(0)
        self._update_summary()

    def _append_row(self, definition):
        row = self.table.rowCount()
        self.table.insertRow(row)
        order = QTableWidgetItem(str(row + 1))
        order.setFlags(order.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 0, order)
        name = QTableWidgetItem(definition['name'])
        name.setData(Qt.ItemDataRole.UserRole, definition['id'])
        self.table.setItem(row, 1, name)
        self.table.setItem(row, 2, QTableWidgetItem(definition.get('description', '')))
        color = QTableWidgetItem(definition['color'])
        color.setForeground(QColor(definition['color']))
        self.table.setItem(row, 3, color)

    def current_definitions(self):
        return [{'id': self.table.item(row, 1).data(Qt.ItemDataRole.UserRole),
                 'name': self.table.item(row, 1).text().strip(),
                 'description': self.table.item(row, 2).text().strip(),
                 'color': self.table.item(row, 3).text().strip()}
                for row in range(self.table.rowCount())]

    def _update_summary(self, *_):
        # A row is populated one cell at a time; only inspect finished rows.
        if not hasattr(self, 'summary') or any(self.table.item(row, 3) is None for row in range(self.table.rowCount())):
            return
        count = self.table.rowCount()
        self.summary.setText(f'{count} / 64개 클래스 · 최소 2개')
        ids = [definition['id'] for definition in self.current_definitions()]
        deleted = len(set(self._original_ids) - set(ids))
        if deleted:
            self.change_note.setText(f'기존 클래스 {deleted}개 삭제 예정 · 해당 클래스의 학습 라벨도 삭제됩니다. 저장 전 구성을 확인하세요.')
        elif ids != self._original_ids or {item['id']: item['description'] for item in self.current_definitions()} != self._original_descriptions:
            self.change_note.setText('클래스 구성이 변경되었습니다. 저장 후 학습 폴더의 시작점과 등장 순서를 다시 확인하세요.')
        else:
            self.change_note.setText('클래스의 고유 ID를 유지하여 기존 학습 라벨과 연결합니다.')

    def add_class(self):
        if self.table.rowCount() >= 64:
            self.error.setText('클래스는 최대 64개까지 정의할 수 있습니다.')
            return
        row = self.table.rowCount()
        self._append_row(new_definition(f'클래스 {row + 1}', index=row))
        self.table.selectRow(row)
        self.table.editItem(self.table.item(row, 1))
        self._update_summary()

    def remove_selected(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            for index in range(self.table.rowCount()):
                self.table.item(index, 0).setText(str(index + 1))
            self._update_summary()

    def move_selected(self, direction):
        row = self.table.currentRow()
        destination = row + direction
        if row < 0 or not 0 <= destination < self.table.rowCount():
            return
        definitions = self.current_definitions()
        definitions[row], definitions[destination] = definitions[destination], definitions[row]
        self._set_rows(definitions)
        self.table.selectRow(destination)

    def add_bulk(self):
        names = [name.strip() for name in self.bulk_names.toPlainText().splitlines() if name.strip()]
        if not names:
            self.error.setText('한 줄에 하나씩 클래스 이름을 입력하세요.')
            return
        definitions = self.current_definitions()
        start = len(definitions)
        definitions.extend(new_definition(name, index=start + i) for i, name in enumerate(names))
        try:
            normalized = validate_definitions(definitions)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self._set_rows(normalized)
        self.bulk_names.clear()
        self.error.clear()

    def apply_floor_template(self):
        names = ['바닥'] + [f'{index}층 천장' for index in range(1, self.ceiling_count.value() + 1)]
        definitions = [new_definition(name, index=index) for index, name in enumerate(names)]
        for definition in definitions:
            definition['description'] = f'{definition["name"]} 상태가 처음 나타나는 이미지'
        self._set_rows(definitions)
        self.error.clear()

    def _choose_color(self, row, column):
        if column != 3:
            return
        item = self.table.item(row, column)
        color = QColorDialog.getColor(QColor(item.text()), self, '클래스 표시 색상')
        if color.isValid():
            item.setText(color.name())
            item.setForeground(color)

    def import_definitions(self):
        filename, _ = QFileDialog.getOpenFileName(self, '클래스 정의 가져오기', '', '클래스 정의 (*.json)')
        if not filename:
            return
        try:
            path = Path(filename)
            if path.stat().st_size > 1_000_000:
                raise ValueError('클래스 정의 파일은 1 MB 이하만 지원합니다.')
            payload = json.loads(path.read_text(encoding='utf-8'))
            definitions = validate_definitions(payload.get('class_definitions', []) if isinstance(payload, dict) else payload)
            self._set_rows(definitions)
            self.error.clear()
        except (OSError, ValueError, TypeError) as exc:
            self.error.setText(f'정의를 가져오지 못했습니다: {exc}')

    def export_definitions(self):
        try:
            definitions = validate_definitions(self.current_definitions())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        filename, _ = QFileDialog.getSaveFileName(self, '클래스 정의 내보내기', 'SEM-클래스-정의.json', '클래스 정의 (*.json)')
        if filename:
            try:
                Path(filename).write_text(json.dumps({'format_version': 1, 'class_definitions': definitions}, ensure_ascii=False, indent=2), encoding='utf-8')
                self.error.setText('클래스 정의를 내보냈습니다.')
            except OSError as exc:
                QMessageBox.warning(self, '내보내기 실패', str(exc))

    def accept(self):
        try:
            self.definitions = validate_definitions(self.current_definitions())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        super().accept()
