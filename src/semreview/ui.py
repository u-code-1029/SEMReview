"""Korean desktop application for ordered SEM landmark training and review."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QEvent
from PySide6.QtGui import QKeySequence, QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QDialog, QDialogButtonBox, QTabWidget,
    QLineEdit, QComboBox, QSpinBox, QCheckBox, QListWidget, QListWidgetItem, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QProgressBar,
    QFormLayout, QPlainTextEdit, QInputDialog, QStackedWidget, QAbstractButton)

from .project import Project, DEFAULT_CLASSES, DEFAULT_PROFILES, sequence_signature
from .classdefs import definitions_for_metadata
from .class_editor import ClassDefinitionsDialog
from .jobs import JobController
from .widgets import SequenceEditor

STYLE = '''
QMainWindow,QDialog,QWidget { background:#101722; color:#e5ecf6; font-size:13px; }
QWidget#card,QGroupBox { background:#172130; border:1px solid #2b3a50; border-radius:8px; }
QGroupBox { margin-top:14px; padding:14px 10px 8px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#b9c8dc; }
QLabel { background:transparent; }
QLabel#heading { font-size:25px; font-weight:700; }
QLabel#sectionTitle { font-size:15px; font-weight:600; }
QLabel#muted { color:#93a4bc; font-size:12px; }
QLabel#warning { color:#f4c180; font-size:12px; }
QPushButton { background:#25354c; border:1px solid #344962; border-radius:6px; padding:7px 11px; }
QPushButton:hover { background:#304762; border-color:#6385aa; }
QPushButton:pressed { background:#1b2b3e; }
QPushButton:disabled { color:#5a6e87; background:#1b2635; border-color:#28384b; }
QPushButton#primary { background:#247d73; border-color:#3ba294; color:white; font-weight:600; }
QPushButton#primary:hover { background:#2b9387; }
QPushButton#danger { background:#6e3945; border-color:#92505d; }
QLineEdit,QSpinBox,QComboBox,QPlainTextEdit { background:#0e1621; border:1px solid #34475f; border-radius:5px; padding:6px; selection-background-color:#2a776f; }
QComboBox::drop-down { width:23px; border:0; }
QComboBox QAbstractItemView { background:#1c293c; selection-background-color:#31516e; }
QTabWidget::pane { border:1px solid #273950; border-radius:6px; }
QTabBar::tab { background:#172335; color:#a7b9d0; padding:11px 26px; border-bottom:3px solid transparent; }
QTabBar::tab:selected { background:#1c2d41; color:#7de0d1; border-bottom-color:#4ed5c1; }
QListWidget,QTableWidget { background:#141e2d; alternate-background-color:#192536; border:1px solid #293c54; border-radius:5px; gridline-color:#28394c; }
QListWidget::item { padding:10px 6px; border-bottom:1px solid #223246; }
QListWidget::item:selected,QTableWidget::item:selected { background:#284a61; color:white; }
QHeaderView::section { background:#203046; color:#afc0d6; border:0; padding:7px; }
QProgressBar { border:1px solid #31465e; border-radius:5px; text-align:center; background:#152030; min-height:17px; }
QProgressBar::chunk { background:#339c8d; border-radius:4px; }
QSlider::groove:horizontal { height:5px; background:#2a3e57; border-radius:2px; }
QSlider::handle:horizontal { background:#63d3c1; width:13px; margin:-5px 0; border-radius:6px; }
QSplitter::handle { background:#223247; width:4px; height:4px; }
QToolTip { background:#263a52; color:#e4edf8; border:1px solid #56718d; padding:5px; }
QScrollBar:vertical { background:#111b29; width:11px; }
QScrollBar::handle:vertical { background:#3c526e; border-radius:4px; min-height:25px; }
QScrollBar:horizontal { background:#111b29; height:11px; }
QScrollBar::handle:horizontal { background:#3c526e; border-radius:4px; min-width:25px; }
'''


def _button(text, callback, primary=False):
    button=QPushButton(text)
    if primary:
        button.setObjectName('primary')
    button.clicked.connect(callback)
    return button


def _table(headers):
    table=QTableWidget(0,len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().hide()
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
    return table


class SettingsDialog(QDialog):
    """Editable traversal profiles: one exact depth per profile; multiple depths supported."""
    def __init__(self,project,parent=None):
        super().__init__(parent)
        self.setWindowTitle('프로젝트 설정 · 폴더 탐색과 학습 장치')
        self.resize(760,600)
        self.profiles=json.loads(json.dumps(project.get('profiles',DEFAULT_PROFILES)))
        self._current=-1
        layout=QVBoxLayout(self)
        intro=QLabel('단계마다 허용할 폴더 이름 또는 패턴을 지정하세요.\n쉼표로 여러 이름을 구분하며 *와 ?를 사용할 수 있습니다. 서로 다른 깊이는 프로필을 추가하세요.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row=QHBoxLayout()
        self.profile_combo=QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._select_profile)
        row.addWidget(self.profile_combo,1)
        row.addWidget(_button('프로필 추가',self._add_profile))
        row.addWidget(_button('삭제',self._remove_profile))
        layout.addLayout(row)
        self.profile_name=QLineEdit()
        self.profile_name.setPlaceholderText('프로필 이름')
        layout.addWidget(self.profile_name)
        self.levels=_table(['단계','허용 이름 / 패턴','제외 이름 / 패턴'])
        self.levels.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        self.levels.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.ResizeToContents)
        self.levels.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.Stretch)
        self.levels.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.levels,1)
        row=QHBoxLayout()
        row.addWidget(_button('+ 하위 단계',self._add_level))
        row.addWidget(_button('선택 단계 삭제',self._remove_level))
        row.addStretch()
        layout.addLayout(row)
        form=QFormLayout()
        self.group_depth=QSpinBox()
        self.group_depth.setRange(1,30)
        self.group_depth.setValue(project.get('group_depth',1))
        form.addRow('촬영 그룹 경로 단계',self.group_depth)
        config=project.get('training_config',{})
        self.device=QComboBox()
        for label,value in [('자동 선택','auto'),('CPU','cpu'),('NVIDIA CUDA','cuda'),('Apple MPS','mps')]:
            self.device.addItem(label,value)
        self.device.setCurrentIndex(max(0,self.device.findData(config.get('device','auto'))))
        form.addRow('학습 장치',self.device)
        weights_row=QHBoxLayout()
        self.weights_path=QLineEdit(config.get('pretrained_weights_path',''))
        self.weights_path.setPlaceholderText('선택 사항 · 공식 .pth 또는 기존 .sem-model')
        weights_row.addWidget(self.weights_path,1)
        weights_row.addWidget(_button('찾아보기',self._browse_weights))
        form.addRow('로컬 사전학습 가중치',weights_row)
        layout.addLayout(form)
        note=QLabel('학습할 상태의 이름·순서·정의는 메인 화면의 클래스 정의에서 관리합니다.')
        note.setObjectName('muted')
        layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh_profiles()

    def _browse_weights(self):
        path,_=QFileDialog.getOpenFileName(self,'로컬 사전학습 가중치 선택',self.weights_path.text(),'사전학습 가중치 (*.pth *.sem-model)')
        if path:
            self.weights_path.setText(path)

    def _store_profile(self):
        if 0<=self._current<len(self.profiles):
            levels=[]
            for row in range(self.levels.rowCount()):
                values=[]
                for column in [1,2]:
                    item=self.levels.item(row,column)
                    values.append([v.strip() for v in (item.text() if item else '').split(',') if v.strip()])
                levels.append({'include':values[0] or ['*'],'exclude':values[1]})
            self.profiles[self._current]={'name':self.profile_name.text().strip() or f'프로필 {self._current+1}','levels':levels}

    def _refresh_profiles(self,selected=0):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems([p.get('name',f'프로필 {i+1}') for i,p in enumerate(self.profiles)])
        self.profile_combo.setCurrentIndex(selected)
        self.profile_combo.blockSignals(False)
        self._current=-1
        self._select_profile(selected)

    def _select_profile(self,index):
        self._store_profile()
        self._current=index
        if not 0<=index<len(self.profiles):
            return
        profile=self.profiles[index]
        self.profile_name.setText(profile.get('name',''))
        self.levels.setRowCount(0)
        for level in profile.get('levels',[]):
            self._add_level(level)

    def _add_level(self,level=None):
        if not isinstance(level,dict):
            level={'include':['*'],'exclude':[]}
        row=self.levels.rowCount()
        self.levels.insertRow(row)
        for column,value in enumerate([str(row+1),', '.join(level.get('include',['*'])),', '.join(level.get('exclude',[]))]):
            item=QTableWidgetItem(value)
            if column==0:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.levels.setItem(row,column,item)

    def _remove_level(self):
        row=self.levels.currentRow()
        if row>=0:
            self.levels.removeRow(row)
            for idx in range(self.levels.rowCount()):
                self.levels.item(idx,0).setText(str(idx+1))

    def _add_profile(self):
        self._store_profile()
        self.profiles.append({'name':f'프로필 {len(self.profiles)+1}','levels':[{'include':['*'],'exclude':[]}]})
        self._refresh_profiles(len(self.profiles)-1)

    def _remove_profile(self):
        if len(self.profiles)>1:
            self.profiles.pop(max(0,self._current))
            self._refresh_profiles()

    def accept(self):
        self._store_profile()
        if any(not p['levels'] for p in self.profiles):
            QMessageBox.warning(self,'탐색 단계 확인','각 프로필에 최소 한 개의 하위 단계를 추가하세요.')
            return
        weights=self.weights_path.text().strip()
        if weights and (not Path(weights).is_file() or Path(weights).suffix not in {'.pth','.sem-model'}):
            QMessageBox.warning(self,'가중치 파일 확인','존재하는 .pth 또는 .sem-model 파일을 선택하세요.')
            return
        self.training_config={'device':self.device.currentData(),'pretrained_weights_path':weights}
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self,project_dir=None):
        super().__init__()
        self.setWindowTitle('SEM Review · 학습과 검수')
        self.resize(1480,960)
        self.setMinimumSize(1080,720)
        self.setStyleSheet(STYLE)
        self.project=None
        self.sequences=[]
        self.annotations={}
        self.predictions=[]
        self.plans=[]
        self.current_sequence=None
        self.current_review=-1
        self.busy=False
        self.rule_busy=False
        self.operation=''
        self.rules_relative=None
        self._loading=False
        self._closing=False
        self.jobs=JobController(self)
        self.jobs.progress.connect(self._progress)
        self.jobs.finished.connect(self._finished)
        self.jobs.failed.connect(self._failed)
        self.jobs.cancelled.connect(self._cancelled)
        self.jobs.busyChanged.connect(self._busy_changed)
        self.rule_jobs=JobController(self)
        self.rule_jobs.finished.connect(self._rules_finished)
        self.rule_jobs.failed.connect(self._rules_failed)
        self.rule_jobs.progress.connect(self._rule_progress)
        self.rule_jobs.cancelled.connect(self._rules_cancelled)
        self.rule_jobs.busyChanged.connect(self._rule_busy_changed)
        self._build()
        QApplication.instance().installEventFilter(self)
        if project_dir:
            self._open_project(project_dir)

    def _build(self):
        container=QWidget()
        self.setCentralWidget(container)
        layout=QVBoxLayout(container)
        layout.setContentsMargins(20,16,20,14)
        top=QHBoxLayout()
        heading=QLabel('SEM Review')
        heading.setObjectName('heading')
        top.addWidget(heading)
        subtitle=QLabel('연속 이미지의 시작과 끝을 정확하게')
        subtitle.setObjectName('muted')
        top.addWidget(subtitle)
        top.addStretch()
        top.addWidget(_button('프로젝트 열기',self.open_project_dialog))
        top.addWidget(_button('새 프로젝트',self.create_project_dialog))
        self.classes_button=_button('클래스 정의',self.edit_class_definitions,True)
        top.addWidget(self.classes_button)
        self.settings_button=_button('설정',self.settings)
        top.addWidget(self.settings_button)
        layout.addLayout(top)
        self.project_label=QLabel('프로젝트를 만들거나 열어 시작하세요.')
        self.project_label.setObjectName('muted')
        layout.addWidget(self.project_label)
        self.stack=QStackedWidget()
        welcome=QWidget()
        wl=QVBoxLayout(welcome)
        wl.addStretch()
        title=QLabel('SEM 이미지, 흐름을 보며 분류하세요')
        title.setObjectName('heading')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl.addWidget(title)
        desc=QLabel('폴더 규칙을 한 번 정하고, 이미지의 기준점만 지정하세요.\n학습한 모델로 여러 폴더를 분석하고 보존 구간 밖의 이미지를 복원 가능한 위치로 이동합니다.')
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setObjectName('muted')
        wl.addWidget(desc)
        actions=QHBoxLayout()
        actions.addStretch()
        actions.addWidget(_button('새 프로젝트 만들기',self.create_project_dialog,True))
        actions.addWidget(_button('기존 프로젝트 열기',self.open_project_dialog))
        actions.addStretch()
        wl.addLayout(actions)
        wl.addStretch()
        self.stack.addWidget(welcome)
        workspace=QWidget()
        body=QVBoxLayout(workspace)
        body.setContentsMargins(0,0,0,0)
        source=QHBoxLayout()
        source.addWidget(QLabel('베이스 폴더'))
        self.base_path=QLineEdit()
        self.base_path.setReadOnly(True)
        self.base_path.setPlaceholderText('학습 또는 검수할 이미지의 베이스 디렉터리')
        source.addWidget(self.base_path,1)
        self.choose_base_button=_button('폴더 선택',self.choose_base)
        source.addWidget(self.choose_base_button)
        self.scan_button=_button('탐색 / 새로고침',self.scan,True)
        source.addWidget(self.scan_button)
        body.addLayout(source)
        self.tabs=QTabWidget()
        self.tabs.addTab(self._training_page(),'01  학습 · 라벨링')
        self.tabs.addTab(self._review_page(),'02  검수')
        self.tabs.addTab(self._models_page(),'모델 · 평가')
        self.tabs.addTab(self._history_page(),'이동 기록 · 복원')
        body.addWidget(self.tabs,1)
        self.stack.addWidget(workspace)
        layout.addWidget(self.stack,1)
        footer=QHBoxLayout()
        self.status=QLabel('준비됨 · 모든 이미지 처리는 로컬에서 수행됩니다')
        self.status.setObjectName('muted')
        footer.addWidget(self.status,1)
        self.progress=QProgressBar()
        self.progress.setMaximumWidth(280)
        self.progress.setValue(0)
        footer.addWidget(self.progress)
        self.cancel_button=_button('작업 취소',self.cancel_jobs)
        self.cancel_button.setEnabled(False)
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        self.settings_button.setEnabled(False)
        self.classes_button.setEnabled(False)

    def _training_page(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        definition_row=QHBoxLayout()
        self.classes_summary=QLabel('학습할 클래스를 정의하세요')
        self.classes_summary.setWordWrap(True)
        definition_row.addWidget(self.classes_summary,1)
        self.training_classes_button=_button('클래스 편집',self.edit_class_definitions)
        definition_row.addWidget(self.training_classes_button)
        layout.addLayout(definition_row)
        split=QSplitter(Qt.Orientation.Horizontal)
        queue=QWidget()
        ql=QVBoxLayout(queue)
        ql.setContentsMargins(0,0,10,0)
        self.queue_summary=QLabel('폴더 0개')
        self.queue_summary.setObjectName('sectionTitle')
        ql.addWidget(self.queue_summary)
        self.folder_filter=QComboBox()
        self.folder_filter.addItems(['모든 폴더','미완료','완료','건너뜀','확인 필요'])
        self.folder_filter.currentIndexChanged.connect(self._refresh_queue)
        ql.addWidget(self.folder_filter)
        self.folder_search=QLineEdit()
        self.folder_search.setPlaceholderText('폴더 이름 검색')
        self.folder_search.textChanged.connect(self._refresh_queue)
        ql.addWidget(self.folder_search)
        self.folder_list=QListWidget()
        self.folder_list.currentItemChanged.connect(self._folder_selected)
        ql.addWidget(self.folder_list,1)
        folder_navigation=QHBoxLayout()
        folder_navigation.addWidget(_button('이전 폴더',lambda:self._navigate_folder(-1)))
        folder_navigation.addWidget(_button('다음 폴더',lambda:self._navigate_folder(1)))
        ql.addLayout(folder_navigation)
        ql.addWidget(_button('완료 폴더 전체 선택',self._select_completed))
        note=QLabel('체크한 완료 폴더를 학습에 사용합니다.\n원본 변경 시 기존 라벨을 다시 확인하세요.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        ql.addWidget(note)
        queue.setMinimumWidth(190)
        queue.setMaximumWidth(300)
        split.addWidget(queue)
        self.editor=SequenceEditor()
        self.editor.changed.connect(self._labels_changed)
        self.editor.cursorChanged.connect(self._cursor_changed)
        self.editor.nextRequested.connect(self.complete_current)
        self.editor.skipRequested.connect(self.skip_current)
        split.addWidget(self.editor)
        split.setSizes([220,1180])
        split.setStretchFactor(1,1)
        layout.addWidget(split,1)
        controls=QHBoxLayout()
        self.rules_button=_button('변화량 분석 · 라벨 추천',self.analyze_rules)
        controls.addWidget(self.rules_button)
        controls.addStretch()
        controls.addWidget(QLabel('모델 이름'))
        self.model_name=QLineEdit('SEM 경계 모델')
        self.model_name.setMaximumWidth(220)
        controls.addWidget(self.model_name)
        self.allow_download=QCheckBox('사전학습 가중치 다운로드 허용')
        self.allow_download.setChecked(False)
        self.allow_download.setToolTip('ResNet-18 가중치가 로컬에 없을 때만 공식 PyTorch 서버에서 다운로드합니다. 이미지 데이터는 전송하지 않습니다.')
        controls.addWidget(self.allow_download)
        self.train_button=_button('선택 폴더로 학습',self.train,True)
        controls.addWidget(self.train_button)
        layout.addLayout(controls)
        return page

    def _review_page(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        row=QHBoxLayout()
        row.addWidget(QLabel('모델'))
        self.review_model=QComboBox()
        self.review_model.setMinimumWidth(190)
        self.review_model.currentIndexChanged.connect(self._review_model_changed)
        row.addWidget(self.review_model,1)
        row.addWidget(QLabel('끝점 클래스'))
        self.end_class=QComboBox()
        self.end_class.currentIndexChanged.connect(self._review_options_changed)
        row.addWidget(self.end_class)
        row.addWidget(QLabel('뒤에 보존할 장수'))
        self.offset=QSpinBox()
        self.offset.setRange(-100000,100000)
        self.offset.setValue(30)
        self.offset.setToolTip('정렬된 실제 이미지 목록 기준입니다. 음수는 해당 기준점 이전까지 보존합니다.')
        self.offset.valueChanged.connect(self._review_options_changed)
        row.addWidget(self.offset)
        self.infer_button=_button('전체 폴더 분석',self.infer,True)
        row.addWidget(self.infer_button)
        layout.addLayout(row)
        note=QLabel('모델의 첫 클래스부터 선택한 끝점까지 모든 이미지를 보존합니다. 결과를 확인하고 체크한 폴더만 격리 이동합니다.')
        note.setObjectName('muted')
        layout.addWidget(note)
        split=QSplitter(Qt.Orientation.Vertical)
        table_area=QWidget()
        table_layout=QVBoxLayout(table_area)
        table_layout.setContentsMargins(0,0,0,0)
        self.review_table=_table(['폴더','확인','보존 구간','보존 / 이동','확인 우선순위','안내'])
        self.review_table.currentCellChanged.connect(self._review_selected)
        self.review_table.itemChanged.connect(self._review_check_changed)
        table_layout.addWidget(self.review_table)
        actions=QHBoxLayout()
        actions.addWidget(_button('현재 폴더 확인',self.approve_current_review))
        actions.addWidget(_button('오류 없는 결과 모두 선택',self._approve_valid_reviews))
        actions.addWidget(_button('확인 체크 모두 해제',self._uncheck_reviews))
        actions.addStretch()
        self.review_summary=QLabel('분석을 실행하면 폴더별 결과가 표시됩니다.')
        actions.addWidget(self.review_summary)
        self.apply_button=_button('확인한 결과 격리 이동',self.apply_reviews,True)
        actions.addWidget(self.apply_button)
        table_layout.addLayout(actions)
        split.addWidget(table_area)
        self.review_editor=SequenceEditor()
        self.review_editor.changed.connect(self._review_labels_changed)
        self.review_editor.next_button.setText('확인하고 다음 결과  ↵')
        self.review_editor.nextRequested.connect(self.approve_and_next_review)
        self.review_editor.skip_button.setVisible(False)
        split.addWidget(self.review_editor)
        split.setSizes([230,520])
        layout.addWidget(split,1)
        return page

    def _models_page(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        row=QHBoxLayout()
        label=QLabel('학습 결과와 저장된 모델')
        label.setObjectName('sectionTitle')
        row.addWidget(label,1)
        row.addWidget(_button('모델 가져오기',self.import_model))
        row.addWidget(_button('선택 모델 내보내기',self.export_model))
        layout.addLayout(row)
        self.model_table=_table(['모델 이름','클래스','생성 시각','모델 ID'])
        self.model_table.currentCellChanged.connect(self._model_selected)
        layout.addWidget(self.model_table,1)
        self.metrics_summary=QLabel('모델을 선택하면 검증 결과가 표시됩니다.')
        self.metrics_summary.setWordWrap(True)
        layout.addWidget(self.metrics_summary)
        self.metrics_table=_table(['클래스','평균 경계 오차','정확 일치','±1장','±3장'])
        self.metrics_table.setMaximumHeight(240)
        layout.addWidget(self.metrics_table)
        self.show_model_details=QCheckBox('상세 모델 정보 보기')
        layout.addWidget(self.show_model_details)
        self.model_details=QPlainTextEdit()
        self.model_details.setReadOnly(True)
        self.model_details.setPlaceholderText('모델을 선택하면 학습 조건과 검증 결과가 표시됩니다.')
        layout.addWidget(self.model_details,1)
        self.model_details.setVisible(False)
        self.show_model_details.toggled.connect(self.model_details.setVisible)
        return page

    def _history_page(self):
        page=QWidget()
        layout=QVBoxLayout(page)
        text=QLabel('격리 이동은 되돌릴 수 있습니다. 중단된 실행은 재개하거나 원래 위치로 복원하세요.\n복원할 위치에 다른 파일이 있으면 덮어쓰지 않고 충돌을 표시합니다.')
        text.setObjectName('muted')
        layout.addWidget(text)
        self.runs_table=_table(['실행 ID','상태','생성 시각','파일 수'])
        layout.addWidget(self.runs_table,1)
        row=QHBoxLayout()
        row.addWidget(_button('기록 새로고침',self._refresh_runs))
        row.addStretch()
        self.resume_button=_button('선택 실행 재개',lambda:self.run_action('resume'))
        row.addWidget(self.resume_button)
        self.restore_button=_button('선택 실행 복원',lambda:self.run_action('restore'),True)
        row.addWidget(self.restore_button)
        layout.addLayout(row)
        self.run_details=QPlainTextEdit()
        self.run_details.setReadOnly(True)
        self.run_details.setMaximumHeight(180)
        layout.addWidget(self.run_details)
        return page

    def create_project_dialog(self):
        if self.busy:
            return
        parent=QFileDialog.getExistingDirectory(self,'프로젝트를 저장할 상위 폴더 선택')
        if not parent:
            return
        name,ok=QInputDialog.getText(self,'새 프로젝트','프로젝트 폴더 이름',text='SEM 프로젝트')
        if not ok or not name.strip():
            return
        if Path(name).name!=name or name in {'.','..'}:
            QMessageBox.warning(self,'이름 확인','경로 구분자가 없는 폴더 이름을 입력하세요.')
            return
        self._open_project(str(Path(parent)/name.strip()))

    def open_project_dialog(self):
        if self.busy:
            return
        directory=QFileDialog.getExistingDirectory(self,'project.sqlite가 있는 프로젝트 폴더 선택')
        if directory:
            if not (Path(directory)/'project.sqlite').is_file():
                QMessageBox.warning(self,'프로젝트를 찾을 수 없음','선택한 폴더에 project.sqlite 파일이 없습니다. 새 프로젝트를 만들려면 새 프로젝트 버튼을 사용하세요.')
                return
            self._open_project(directory)

    def _open_project(self,directory):
        try:
            project=Project(directory)
            if self.project:
                self.project.close()
            self.project=project
            self.current_sequence=None
            self.current_review=-1
            self.rule_jobs.cancel()
            self.rules_relative=None
            self._loading=True
            self.editor.set_sequence(None,project.get('classes',DEFAULT_CLASSES),class_definitions=project.class_definitions())
            self.review_editor.set_sequence(None,[])
            self.base_path.setText(project.get('base',''))
            self.sequences=project.get('sequences',[])
            self.annotations=project.annotations()
            self.predictions=project.get('review_predictions',[])
            self.offset.setValue(project.get('review_offset',30))
            self.project_label.setText(str(project.root))
            self.stack.setCurrentIndex(1)
            self.settings_button.setEnabled(True)
            self.classes_button.setEnabled(True)
            self._refresh_class_summary()
            self._refresh_queue()
            self._refresh_models()
            self._refresh_runs()
            self._loading=False
            self._rebuild_plans()
            last=project.get('current_folder')
            for row in range(self.folder_list.count()):
                if self.folder_list.item(row).data(Qt.ItemDataRole.UserRole)==last:
                    self.folder_list.setCurrentRow(row)
                    break
            else:
                if self.folder_list.count():
                    self.folder_list.setCurrentRow(0)
            self.status.setText('프로젝트를 열었습니다. 라벨은 자동 저장됩니다.')
        except Exception as exc:
            self._loading=False
            self._show_error('프로젝트 열기 실패',exc)

    def settings(self):
        if not self.project or self.busy:
            return
        dialog=SettingsDialog(self.project,self)
        if dialog.exec()!=QDialog.DialogCode.Accepted:
            return
        self.project.set('profiles',dialog.profiles)
        self.project.set('group_depth',dialog.group_depth.value())
        self.project.set('training_config',dialog.training_config)
        self.status.setText('설정을 저장했습니다. 탐색 / 새로고침을 누르면 폴더 규칙이 적용됩니다.')

    def _refresh_class_summary(self):
        definitions=self.project.class_definitions()
        names=[item['name'] for item in definitions]
        preview=' → '.join(names[:5])
        if len(names)>5:
            preview+=f' → … +{len(names)-5}개'
        self.classes_summary.setText(f'학습 클래스 {len(names)}개 · {preview}')
        self.classes_summary.setToolTip('\n'.join(f'{i+1}. {item["name"]} — {item["description"]}' for i,item in enumerate(definitions)))

    def edit_class_definitions(self):
        if not self.project or self.busy:
            return
        dialog=ClassDefinitionsDialog(self.project.class_definitions(),self)
        if dialog.exec()!=QDialog.DialogCode.Accepted:
            return
        try:
            self.project.update_class_definitions(dialog.definitions)
            self.annotations=self.project.annotations()
            self._refresh_class_summary()
            self._refresh_queue()
            if self.current_sequence:
                self._load_training_sequence(self.current_sequence)
            else:
                self.editor.set_sequence(None,self.project.get('classes'),class_definitions=self.project.class_definitions())
            self.status.setText('클래스 정의를 저장했습니다. 저장된 모델은 학습 당시의 정의로 검수합니다.')
        except Exception as exc:
            self._show_error('클래스 정의 저장 실패',exc)

    def choose_base(self):
        if not self.project or self.busy:
            return
        directory=QFileDialog.getExistingDirectory(self,'이미지 베이스 폴더 선택',self.base_path.text())
        if directory:
            old=self.project.get('base','')
            if old and str(Path(directory).resolve())!=str(Path(old).resolve()):
                answer=QMessageBox.question(self,'베이스 폴더 변경','새 베이스 폴더의 이미지 목록과 검수 결과로 바꿉니다. 기존 라벨은 파일 지문이 같은 경우에만 다시 사용할 수 있습니다. 계속하시겠습니까?')
                if answer!=QMessageBox.StandardButton.Yes:
                    return
                self.predictions=[]
                self.project.set('review_predictions',[])
                self._rebuild_plans()
            self.base_path.setText(directory)
            self.project.set('base',directory)
            self.scan()

    def scan(self):
        if not self.project or self.busy:
            return
        if not self.base_path.text():
            self.choose_base()
            return
        self._start('scan',{'base':self.base_path.text(),'profiles':self.project.get('profiles',DEFAULT_PROFILES)})

    def _refresh_queue(self,*args):
        if not self.project:
            return
        current=self.current_sequence.get('relative_path') if self.current_sequence else None
        checked={self.folder_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.folder_list.count()) if self.folder_list.item(i).checkState()==Qt.CheckState.Checked}
        self.folder_list.blockSignals(True)
        self.folder_list.clear()
        mode=self.folder_filter.currentIndex()
        search=self.folder_search.text().casefold()
        complete=0
        for sequence in self.sequences:
            relative=sequence['relative_path']
            ann=self.annotations.get(relative,{})
            status=ann.get('status','draft')
            done=status in {'complete','completed'}
            complete+=int(done)
            if search not in relative.casefold():
                continue
            if mode==1 and (done or status=='skipped') or mode==2 and not done or mode==3 and status!='skipped' or mode==4 and status!='invalid' and not sequence.get('issues'):
                continue
            symbol='✓' if done else '—' if status=='skipped' else '!' if status=='invalid' or sequence.get('issues') else '○'
            state={'draft':'미완료','complete':'완료','completed':'완료','skipped':'건너뜀','invalid':'원본 변경 · 확인 필요'}.get(status,status)
            item=QListWidgetItem(f'{symbol}  {relative}\n{len(sequence["images"])}장 · {state}')
            item.setData(Qt.ItemDataRole.UserRole,relative)
            item.setToolTip(sequence['path']+'\n'+'\n'.join(map(str,sequence.get('issues',[]))))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if relative in checked else Qt.CheckState.Unchecked)
            self.folder_list.addItem(item)
            if relative==current:
                self.folder_list.setCurrentItem(item)
        self.folder_list.blockSignals(False)
        self.queue_summary.setText(f'{complete} / {len(self.sequences)} 폴더 완료')

    def _folder_selected(self,item,previous=None):
        if item is None or self._loading:
            return
        relative=item.data(Qt.ItemDataRole.UserRole)
        sequence=next((s for s in self.sequences if s['relative_path']==relative),None)
        if sequence:
            self._load_training_sequence(sequence)

    def _navigate_folder(self,direction):
        row=self.folder_list.currentRow()+direction
        if 0<=row<self.folder_list.count():
            self.folder_list.setCurrentRow(row)

    def _load_training_sequence(self,sequence):
        self._loading=True
        self.current_sequence=sequence
        relative=sequence['relative_path']
        ann=self.annotations.get(relative,{})
        stale=ann and ann.get('sequence_signature')!=sequence_signature(sequence)
        landmarks={} if stale else ann.get('landmarks',{})
        self.editor.set_sequence(sequence,self.project.get('classes',DEFAULT_CLASSES),landmarks,ann.get('cursor',0),class_definitions=self.project.class_definitions())
        self.project.set('current_folder',relative)
        if stale:
            self.editor.warning.setText('원본 파일이 변경되었습니다. 기준점을 다시 지정하고 확인하세요.')
        self._loading=False

    def _labels_changed(self,landmarks):
        self._save_current('draft')

    def _cursor_changed(self,index):
        if not self._loading and self.current_sequence:
            ann=self.annotations.get(self.current_sequence['relative_path'],{})
            self._save_current(ann.get('status','draft'),refresh=False)

    def _save_current(self,status='draft',refresh=True):
        if self._loading or not self.project or not self.current_sequence:
            return
        relative=self.current_sequence['relative_path']
        self.project.save_annotation(relative,self.editor.landmarks,sequence_signature(self.current_sequence),status,self.editor.cursor)
        self.annotations=self.project.annotations()
        if refresh:
            self._refresh_queue()

    def complete_current(self):
        if not self.current_sequence:
            return
        errors=self.editor.validation_errors()
        if errors:
            QMessageBox.warning(self,'기준점 확인','\n'.join(errors))
            return
        current=self.current_sequence['relative_path']
        self._save_current('complete')
        self._go_next(current)

    def skip_current(self):
        if self.current_sequence:
            relative=self.current_sequence['relative_path']
            self._save_current('skipped')
            self._go_next(relative)

    def _go_next(self,relative):
        all_rel=[s['relative_path'] for s in self.sequences]
        start=all_rel.index(relative) if relative in all_rel else -1
        visible={self.folder_list.item(i).data(Qt.ItemDataRole.UserRole):i for i in range(self.folder_list.count())}
        for rel in all_rel[start+1:]+all_rel[:start]:
            if rel in visible and self.annotations.get(rel,{}).get('status') not in {'complete','completed','skipped'}:
                self.folder_list.setCurrentRow(visible[rel])
                return
        self.status.setText('현재 목록의 미완료 폴더를 모두 확인했습니다.')

    def _select_completed(self):
        for idx in range(self.folder_list.count()):
            item=self.folder_list.item(idx)
            completed=self.annotations.get(item.data(Qt.ItemDataRole.UserRole),{}).get('status') in {'complete','completed'}
            item.setCheckState(Qt.CheckState.Checked if completed else Qt.CheckState.Unchecked)

    def analyze_rules(self):
        editor=self.review_editor if self.tabs.currentIndex()==1 else self.editor
        if not editor.sequence or self.rule_busy:
            return
        self.rules_relative=(self.tabs.currentIndex(),editor.sequence['relative_path'])
        self.status.setText('이동 보정과 변화량을 계산하고 있습니다…')
        self.rule_jobs.start('rules',{'sequence':editor.sequence})

    def _rules_finished(self,result):
        if not self.rules_relative:
            return
        tab,relative=self.rules_relative
        editor=self.review_editor if tab==1 else self.editor
        if editor.sequence and editor.sequence['relative_path']==relative:
            editor.set_analysis(result)
        if not self.busy:
            self.progress.setRange(0,100)
            self.progress.setValue(100)
            self.status.setText('변화량 분석 완료 · 추천은 후보이며 직접 기준점을 확정해야 합니다.')

    def _rules_failed(self,message):
        if not self.busy:
            self.progress.setRange(0,100)
            self.progress.setValue(0)
            self.status.setText(f'변화량 분석 실패: {message}')

    def _rule_progress(self,event):
        if not self.busy:
            self._progress(event)

    def _rules_cancelled(self):
        if not self.busy:
            self.progress.setRange(0,100)
            self.progress.setValue(0)
            self.status.setText('변화량 분석을 취소했습니다.')

    def cancel_jobs(self):
        self.jobs.cancel()
        self.rule_jobs.cancel()

    def train(self):
        if not self.project or self.busy:
            return
        checked={self.folder_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.folder_list.count()) if self.folder_list.item(i).checkState()==Qt.CheckState.Checked}
        sequences=[s for s in self.sequences if s['relative_path'] in checked]
        if not sequences:
            QMessageBox.information(self,'학습 데이터 선택','왼쪽 목록에서 완료된 폴더를 체크하거나 완료 폴더 전체 선택을 누르세요.')
            return
        invalid=[s['relative_path'] for s in sequences if self.annotations.get(s['relative_path'],{}).get('status') not in {'complete','completed'} or self.annotations.get(s['relative_path'],{}).get('sequence_signature')!=sequence_signature(s)]
        if invalid:
            QMessageBox.warning(self,'라벨 확인 필요','완료되지 않았거나 원본이 변경된 폴더가 포함되어 있습니다:\n'+'\n'.join(invalid[:10]))
            return
        self._start('train',{'project_dir':str(self.project.root),'sequences':sequences,
            'annotations':{s['relative_path']:self.annotations[s['relative_path']]['landmarks'] for s in sequences},
            'classes':self.project.get('classes',DEFAULT_CLASSES),'model_name':self.model_name.text().strip() or 'SEM 경계 모델',
            'class_definitions':self.project.class_definitions(),
            'config':{**self.project.get('training_config',{}),'allow_download':self.allow_download.isChecked(),'group_level':self.project.get('group_depth',1)}})

    def _refresh_models(self):
        if not self.project:
            return
        selected=self.review_model.currentData()
        selected_id=selected.get('id') if isinstance(selected,dict) else self.project.get('review_model_id')
        self.models=self.project.models()
        self.review_model.blockSignals(True)
        self.review_model.clear()
        self.model_table.setRowCount(len(self.models))
        for row,model in enumerate(self.models):
            name=model.get('name',model.get('model_name',model.get('id','모델')))
            self.review_model.addItem(name,model)
            for col,value in enumerate([name,' → '.join(item['name'] for item in definitions_for_metadata(model)),model.get('created_at',''),model.get('id','')]):
                self.model_table.setItem(row,col,QTableWidgetItem(str(value)))
            if model.get('id')==selected_id:
                self.review_model.setCurrentIndex(row)
        if not self.models:
            self.review_model.addItem('저장된 모델이 없습니다',None)
        self.review_model.blockSignals(False)
        self._review_model_changed()
        if self.models:
            self.model_table.selectRow(0)

    def _model_selected(self,row,column,prev_row,prev_column):
        if not hasattr(self,'models') or not 0<=row<len(self.models):
            self.model_details.clear()
            self.metrics_summary.setText('모델을 선택하면 검증 결과가 표시됩니다.')
            self.metrics_table.setRowCount(0)
            return
        model=self.models[row]
        definitions=definitions_for_metadata(model)
        names={item['id']:item['name'] for item in definitions}
        metrics=model.get('metrics',{})
        by_class=metrics.get('by_class',{})
        self.metrics_table.setRowCount(len(by_class))
        for metric_row,(name,values) in enumerate(by_class.items()):
            display=[names.get(name,name),f'{values.get("mae",0):.2f}장',f'{values.get("exact",0):.1%}',f'{values.get("within_1",0):.1%}',f'{values.get("within_3",0):.1%}']
            for col,value in enumerate(display):
                self.metrics_table.setItem(metric_row,col,QTableWidgetItem(value))
        if metrics.get('validated'):
            self.metrics_summary.setText(f'그룹별 {metrics.get("folds",3)}분할 검증 · {metrics.get("folders",0)}개 폴더\n'
                f'필요 이미지 누락 {metrics.get("missed_images",0)}장 · 불필요 이미지 보존 {metrics.get("extra_images",0)}장 · '
                f'경계 수정이 필요한 폴더 {metrics.get("folders_needing_correction",0):.1%}\n'
                f'평가 보존 기준: {names.get(metrics.get("end_class"),metrics.get("end_class",""))} + {metrics.get("offset",30)}장')
        else:
            self.metrics_summary.setText('일반화 성능 미검증 · '+metrics.get('reason','검증 결과가 없습니다. 실제 SEM 데이터에서 성능을 확인하세요.'))
        lines=[f'모델: {model.get("name",model.get("model_name",model.get("id","")))}',f'클래스: {" → ".join(item["name"] for item in definitions)}',f'위치: {model.get("path","")}','',
            *[f'{index+1}. {item["name"]}: {item["description"]}' for index,item in enumerate(definitions)],'',
            '평가는 폴더 / 촬영 그룹을 분리하여 계산합니다. 높은 순위는 사용자 확인이 더 필요하다는 의미입니다.','',json.dumps({k:v for k,v in model.items() if k not in {'path','project_relative_path'}},ensure_ascii=False,indent=2)]
        self.model_details.setPlainText('\n'.join(lines))

    def import_model(self):
        if not self.project or self.busy:
            return
        filename,_=QFileDialog.getOpenFileName(self,'모델 가져오기','','SEM 모델 (*.sem-model)')
        if filename:
            try:
                self.project.import_model(filename)
                self._refresh_models()
                self.status.setText('모델을 프로젝트에 가져왔습니다.')
            except Exception as exc:
                self._show_error('모델 가져오기 실패',exc)

    def export_model(self):
        row=self.model_table.currentRow()
        if not self.project or not 0<=row<len(getattr(self,'models',[])):
            return
        model=self.models[row]
        filename,_=QFileDialog.getSaveFileName(self,'모델 내보내기',model.get('name','SEM 모델')+'.sem-model','SEM 모델 (*.sem-model)')
        if filename:
            try:
                if Path(filename).resolve()!=Path(model['path']).resolve():
                    shutil.copy2(model['path'],filename)
                self.status.setText(f'모델을 저장했습니다: {filename}')
            except Exception as exc:
                self._show_error('모델 내보내기 실패',exc)

    def _review_model_changed(self,*args):
        model=self.review_model.currentData()
        old=self.end_class.currentData()
        self.end_class.blockSignals(True)
        self.end_class.clear()
        classes=model.get('classes',[]) if isinstance(model,dict) else []
        definitions=definitions_for_metadata(model) if isinstance(model,dict) else []
        for item in definitions:
            self.end_class.addItem(item['name'],item['id'])
            self.end_class.setItemData(self.end_class.count()-1,item['description'],Qt.ItemDataRole.ToolTipRole)
        self.end_class.setCurrentIndex(max(0,self.end_class.count()-1))
        remembered=self.project.get('review_end_class') if self.project else old
        if remembered in classes:
            self.end_class.setCurrentIndex(self.end_class.findData(remembered))
        self.end_class.blockSignals(False)
        if self.project and isinstance(model,dict):
            self.project.set('review_model_id',model['id'])
        if not self._loading:
            self._rebuild_plans()
            row=self.review_table.currentRow()
            if row>=0:
                self._review_selected(row,0,-1,-1)

    def _review_options_changed(self,*args):
        if self.project and not self._loading:
            self.project.set('review_offset',self.offset.value())
            self.project.set('review_end_class',self.end_class.currentData())
            self._rebuild_plans()

    def infer(self):
        if not self.project or self.busy:
            return
        model=self.review_model.currentData()
        if not isinstance(model,dict):
            QMessageBox.information(self,'모델 선택','먼저 모델을 학습하거나 저장된 모델을 가져오세요.')
            return
        if not self.sequences:
            QMessageBox.information(self,'폴더 없음','베이스 폴더를 탐색한 뒤 실행하세요.')
            return
        self._start('infer',{'model_path':model['path'],'sequences':self.sequences,'project_dir':str(self.project.root)})

    def _rebuild_plans(self,preserve_approvals=False,exclude_relative=None):
        if not self.project:
            return
        approved=set()
        if preserve_approvals:
            for row,old_plan in enumerate(self.plans):
                item=self.review_table.item(row,1)
                if old_plan and item and item.checkState()==Qt.CheckState.Checked:
                    relative=old_plan['sequence']['relative_path']
                    if relative!=exclude_relative:
                        approved.add(relative)
        from .core import make_review_plan
        model=self.review_model.currentData()
        classes=model.get('classes',[]) if isinstance(model,dict) else []
        definitions=definitions_for_metadata(model) if isinstance(model,dict) else []
        missing_boundary_errors={f'클래스 기준점이 없거나 범위를 벗어났습니다: {item["id"]}':
            f'클래스 기준점이 없거나 범위를 벗어났습니다: {item["name"]}' for item in definitions}
        model_id=model.get('id') if isinstance(model,dict) else None
        self.plans=[]
        if not self.predictions:
            self.current_review=-1
            self.review_editor.set_sequence(None,classes,class_definitions=definitions_for_metadata(model) if isinstance(model,dict) else [])
        self.review_table.blockSignals(True)
        self.review_table.setRowCount(len(self.predictions))
        for row,prediction in enumerate(self.predictions):
            sequence=next((s for s in self.sequences if s['relative_path']==prediction['relative_path']),None)
            plan=None
            errors=[]
            if sequence and classes:
                try:
                    plan=make_review_plan(self.base_path.text(),sequence,prediction.get('landmarks',{}),classes,self.end_class.currentData(),self.offset.value(),prediction.get('model_id','manual'))
                    errors.extend(missing_boundary_errors.get(str(x),str(x)) for x in plan.get('errors',[]))
                    if prediction.get('model_id')!=model_id:
                        errors.append('선택한 모델과 분석 모델이 다릅니다. 다시 분석하세요.')
                except Exception as exc:
                    errors.append(str(exc))
            else:
                errors.append('현재 파일 목록 또는 모델에서 찾을 수 없습니다.')
            if plan:
                plan['errors']=errors
            self.plans.append(plan)
            self.review_table.setItem(row,0,QTableWidgetItem(prediction['relative_path']))
            check=QTableWidgetItem('')
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | (Qt.ItemFlag.ItemIsUserCheckable if not errors else Qt.ItemFlag.NoItemFlags))
            check.setCheckState(Qt.CheckState.Checked if not errors and prediction['relative_path'] in approved else Qt.CheckState.Unchecked)
            self.review_table.setItem(row,1,check)
            images=sequence['images'] if sequence else []
            start,end=(plan.get('start'),plan.get('end')) if plan else (None,None)
            interval=f'{images[start]["name"]} → {images[end]["name"]}' if isinstance(start,int) and isinstance(end,int) and 0<=start<len(images) and 0<=end<len(images) else '확인 필요'
            counts=f'{len(plan.get("keep",[]))} / {len(plan.get("move",[]))}' if plan else '—'
            priority=prediction.get('priority',0)
            priority_text=f'{priority:.3f}' if isinstance(priority,(int,float)) else str(priority)
            warnings=errors+[str(x) for x in prediction.get('warnings',[])]+([str(x) for x in plan.get('warnings',[])] if plan else [])
            for col,value in enumerate([interval,counts,priority_text,' · '.join(warnings) or '범위를 확인하세요'],start=2):
                item=QTableWidgetItem(value)
                item.setToolTip(value)
                if errors:
                    item.setForeground(QColor('#f4b0a4'))
                self.review_table.setItem(row,col,item)
        self.review_table.blockSignals(False)
        self._update_review_summary()
        if 0<=self.current_review<len(self.plans):
            selected_plan=self.plans[self.current_review]
            self.review_editor.set_keep_range(selected_plan.get('start') if selected_plan else None,selected_plan.get('end') if selected_plan and not selected_plan.get('errors') else None)
        else:
            self.review_editor.set_keep_range()
        if self.predictions and self.review_table.currentRow()<0:
            self.review_table.selectRow(0)

    def _review_selected(self,row,column,prev_row,prev_column):
        self.current_review=row
        if not 0<=row<len(self.predictions):
            return
        prediction=self.predictions[row]
        sequence=next((s for s in self.sequences if s['relative_path']==prediction['relative_path']),None)
        model=self.review_model.currentData()
        if not sequence or not isinstance(model,dict):
            return
        self._loading=True
        self.review_editor.set_sequence(sequence,model.get('classes',[]),prediction.get('landmarks',{}),next(iter(prediction.get('landmarks',{}).values()),0),class_definitions=definitions_for_metadata(model))
        self.review_editor.set_analysis(prediction)
        plan=self.plans[row] if row<len(self.plans) else None
        self.review_editor.set_keep_range(plan.get('start') if plan else None,plan.get('end') if plan and not plan.get('errors') else None)
        self._loading=False

    def _review_labels_changed(self,landmarks):
        if self._loading or not 0<=self.current_review<len(self.predictions):
            return
        row=self.current_review
        self.predictions[row]['landmarks']=dict(landmarks)
        self.predictions[row]['user_modified']=True
        self.project.set('review_predictions',self.predictions)
        self._rebuild_plans(preserve_approvals=True,exclude_relative=self.predictions[row]['relative_path'])
        self.review_table.setCurrentCell(row,0)

    def _review_check_changed(self,item):
        self._update_review_summary()

    def _update_review_summary(self):
        checked=0
        files=0
        for row,plan in enumerate(self.plans):
            item=self.review_table.item(row,1)
            if item and item.checkState()==Qt.CheckState.Checked and plan and not plan.get('errors'):
                checked+=1
                files+=len(plan.get('move',[]))
        self.review_summary.setText(f'{checked}개 폴더 확인 · {files}개 이미지 이동 예정')
        self.apply_button.setEnabled(checked>0 and not self.busy)

    def approve_current_review(self):
        row=self.review_table.currentRow()
        if not 0<=row<len(self.plans):
            return
        plan=self.plans[row]
        if not plan or plan.get('errors'):
            QMessageBox.warning(self,'검수 실행 보류','\n'.join(map(str,plan.get('errors',[]))) if plan else '다시 분석한 뒤 기준점을 확인하세요.')
            return
        self.review_table.item(row,1).setCheckState(Qt.CheckState.Checked)

    def approve_and_next_review(self):
        self.approve_current_review()
        row=self.review_table.currentRow()
        if row+1<self.review_table.rowCount():
            self.review_table.setCurrentCell(row+1,0)

    def _approve_valid_reviews(self):
        for row,plan in enumerate(self.plans):
            if plan and not plan.get('errors'):
                self.review_table.item(row,1).setCheckState(Qt.CheckState.Checked)

    def _uncheck_reviews(self):
        for row in range(self.review_table.rowCount()):
            self.review_table.item(row,1).setCheckState(Qt.CheckState.Unchecked)

    def apply_reviews(self):
        if not self.project or self.busy:
            return
        plans=[plan for row,plan in enumerate(self.plans) if plan and not plan.get('errors') and self.review_table.item(row,1).checkState()==Qt.CheckState.Checked]
        if not plans:
            return
        files=sum(len(p.get('move',[])) for p in plans)
        answer=QMessageBox.question(self,'확인한 결과 적용',f'{len(plans)}개 폴더에서 보존 구간 밖의 이미지 {files}개를 격리 이동합니다.\n\n대상: {self.base_path.text()}\n격리 위치: 베이스/.sem-review/quarantine\n\n파일 목록과 내용을 다시 검사한 뒤 이동합니다. 이동 기록에서 복원할 수 있습니다.',QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Cancel)
        if answer==QMessageBox.StandardButton.Yes:
            self._start('apply',{'plans':plans,'project_dir':str(self.project.root)})

    def _refresh_runs(self):
        if not self.project:
            return
        from .core import list_runs
        try:
            self.runs=list_runs(str(self.project.root))
            self.runs_table.setRowCount(len(self.runs))
            for row,run in enumerate(self.runs):
                values=[run.get('id',run.get('run_id','')),run.get('status',''),run.get('created_at',''),run.get('file_count',run.get('total',len(run.get('files',[]))))]
                for col,value in enumerate(values):
                    self.runs_table.setItem(row,col,QTableWidgetItem(str(value)))
        except Exception as exc:
            self.run_details.setPlainText(str(exc))

    def run_action(self,command):
        if not self.project or self.busy:
            return
        row=self.runs_table.currentRow()
        if not 0<=row<len(getattr(self,'runs',[])):
            QMessageBox.information(self,'실행 선택','이동 기록에서 대상 실행을 선택하세요.')
            return
        run=self.runs[row]
        run_id=run.get('id',run.get('run_id'))
        action='원래 경로로 복원' if command=='restore' else '남은 파일 이동 재개'
        answer=QMessageBox.question(self,action,f'선택한 실행의 {action} 작업을 진행합니다.\n실행: {run_id}\n이름이 충돌하는 파일은 덮어쓰지 않습니다.',QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,QMessageBox.StandardButton.Cancel)
        if answer==QMessageBox.StandardButton.Yes:
            self._start(command,{'project_dir':str(self.project.root),'run_id':run_id})

    def _start(self,command,payload):
        if self.busy:
            return
        self.operation=command
        self.progress.setRange(0,0)
        self.status.setText({'scan':'폴더와 원본 파일을 검사하고 있습니다…','train':'특징 추출 및 모델 학습을 시작합니다…','infer':'폴더별 경계점을 분석합니다…','apply':'원본 검사 후 격리 이동합니다…','restore':'원본 위치로 복원합니다…','resume':'중단된 이동을 재개합니다…'}.get(command,'작업 중…'))
        try:
            self.jobs.start(command,payload)
        except Exception as exc:
            self._failed(str(exc))

    def _busy_changed(self,state):
        self.busy=state
        self.cancel_button.setEnabled(state or self.rule_busy)
        for button in [self.scan_button,self.choose_base_button,self.settings_button,self.classes_button,self.training_classes_button,self.train_button,self.infer_button,self.resume_button,self.restore_button]:
            button.setEnabled(not state)
        self._update_review_summary()
        if self._closing and not state:
            QTimer.singleShot(0,self._finish_close)

    def _rule_busy_changed(self,state):
        self.rule_busy=state
        self.cancel_button.setEnabled(state or self.busy)
        self.rules_button.setEnabled(not state)
        if self._closing and not state:
            QTimer.singleShot(0,self._finish_close)

    def _finish_close(self):
        if self._closing and not self.jobs.busy and not self.rule_jobs.busy:
            self.close()

    def _progress(self,event):
        if not isinstance(event,dict):
            return
        total=event.get('total',0)
        completed=event.get('completed',event.get('current',0))
        if total and total>0:
            self.progress.setRange(0,int(total))
            self.progress.setValue(int(completed))
        else:
            self.progress.setRange(0,0)
        self.status.setText(str(event.get('message') or event.get('stage') or '작업 중…'))

    def _finished(self,result):
        command=self.operation
        self.progress.setRange(0,100)
        self.progress.setValue(100)
        try:
            if command=='scan':
                self.sequences=result.get('sequences',[]) if isinstance(result,dict) else result
                self.project.set('sequences',self.sequences)
                self.predictions=[]
                self.project.set('review_predictions',[])
                for sequence in self.sequences:
                    ann=self.annotations.get(sequence['relative_path'])
                    if ann and ann['sequence_signature']!=sequence_signature(sequence):
                        self.project.save_annotation(sequence['relative_path'],ann['landmarks'],ann['sequence_signature'],'invalid',ann.get('cursor',0))
                self.annotations=self.project.annotations()
                self._refresh_queue()
                if self.folder_list.count():
                    self.folder_list.setCurrentRow(0)
                    self._folder_selected(self.folder_list.currentItem())
                else:
                    self.current_sequence=None
                    self.editor.set_sequence(None,self.project.get('classes',DEFAULT_CLASSES),class_definitions=self.project.class_definitions())
                self._rebuild_plans()
                images=sum(len(s['images']) for s in self.sequences)
                self.status.setText(f'탐색 완료 · {len(self.sequences)}개 폴더 · {images}장 · 설정에서 폴더 규칙을 조정할 수 있습니다.')
            elif command=='train':
                self.project.add_model(result)
                self._refresh_models()
                self.tabs.setCurrentIndex(2)
                self.status.setText('학습 완료 · 모델과 평가 결과를 저장했습니다.')
            elif command=='infer':
                self.predictions=result.get('predictions',[]) if isinstance(result,dict) else result
                self.predictions.sort(key=lambda item:(-float(item.get('priority',0)),item['relative_path']))
                self.project.set('review_predictions',self.predictions)
                self._rebuild_plans()
                row=self.review_table.currentRow()
                if row>=0:
                    self._review_selected(row,0,-1,-1)
                self.status.setText('분석 완료 · 보존 범위를 확인하고 적용할 폴더에 체크하세요.')
            elif command in {'apply','restore','resume'}:
                self.run_details.setPlainText(json.dumps(result,ensure_ascii=False,indent=2))
                self._refresh_runs()
                self.tabs.setCurrentIndex(3)
                self._uncheck_reviews()
                self.predictions=[]
                self.project.set('review_predictions',[])
                self._rebuild_plans()
                self.status.setText('파일 작업을 마쳤습니다. 기록을 확인한 뒤 다시 탐색하세요.')
        except Exception as exc:
            self._show_error('결과 처리 실패',exc)

    def _failed(self,message):
        self.progress.setRange(0,100)
        self.progress.setValue(0)
        self.status.setText('작업이 중단되었습니다. 오류 내용을 확인하세요.')
        self._show_error('작업 실패',message)
        if self.operation in {'apply','restore','resume'}:
            self._refresh_runs()

    def _cancelled(self):
        self.progress.setRange(0,100)
        self.progress.setValue(0)
        self.status.setText('작업을 취소했습니다. 이동된 파일은 이동 기록에서 복원하거나 재개할 수 있습니다.')
        if self.operation in {'apply','restore','resume'}:
            self._refresh_runs()

    def _show_error(self,title,error):
        box=QMessageBox(QMessageBox.Icon.Warning,title,str(error),parent=self)
        box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box.exec()

    def eventFilter(self,obj,event):
        if event.type()!=QEvent.Type.KeyPress or not self.isActiveWindow() or self.stack.currentIndex()!=1:
            return super().eventFilter(obj,event)
        focus=QApplication.focusWidget()
        if isinstance(focus,(QLineEdit,QPlainTextEdit,QSpinBox,QComboBox)) or (focus and isinstance(focus.parent(),QSpinBox)):
            return super().eventFilter(obj,event)
        if QApplication.activeModalWidget():
            return super().eventFilter(obj,event)
        tab=self.tabs.currentIndex()
        if tab not in (0,1):
            return super().eventFilter(obj,event)
        editor=self.editor if tab==0 else self.review_editor
        key=event.key()
        modifiers=event.modifiers()
        if event.matches(QKeySequence.StandardKey.Undo):
            editor.undo()
            return True
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.AltModifier):
            return super().eventFilter(obj,event)
        if key in (Qt.Key.Key_Left,Qt.Key.Key_Right):
            editor.move((-1 if key==Qt.Key.Key_Left else 1)*(10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1))
            return True
        if key in (Qt.Key.Key_Up,Qt.Key.Key_Down):
            if not focus or not (focus is editor or editor.isAncestorOf(focus)) or isinstance(focus,QAbstractButton):
                return super().eventFilter(obj,event)
            editor.select_next_class(-1 if key==Qt.Key.Key_Up else 1)
            return True
        if key==Qt.Key.Key_Space:
            if not focus or not (focus is editor or editor.isAncestorOf(focus)) or isinstance(focus,QAbstractButton):
                return super().eventFilter(obj,event)
            editor.assign_selected()
            return True
        if Qt.Key.Key_1<=key<=Qt.Key.Key_9:
            editor.assign_number(key-Qt.Key.Key_1)
            return True
        if key in (Qt.Key.Key_Return,Qt.Key.Key_Enter):
            self.complete_current() if tab==0 else self.approve_and_next_review()
            return True
        return super().eventFilter(obj,event)

    def closeEvent(self,event):
        if self.jobs.busy or self.rule_jobs.busy:
            if not self._closing:
                answer=QMessageBox.question(self,'작업 중','진행 중인 작업을 취소하고 종료하시겠습니까? 이동 기록은 보존됩니다.',QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
                if answer!=QMessageBox.StandardButton.Yes:
                    event.ignore()
                    return
                self._closing=True
                self.jobs.cancel()
                self.rule_jobs.cancel()
                self.status.setText('안전한 단계가 완료되면 종료합니다…')
                self.setEnabled(False)
            event.ignore()
            return
        QApplication.instance().removeEventFilter(self)
        if self.project:
            self.project.close()
            self.project=None
        event.accept()
