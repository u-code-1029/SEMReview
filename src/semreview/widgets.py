"""Image navigation and landmark editing widgets (all TIFF I/O stays off the UI thread)."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QObject, QRunnable, QThreadPool, QPointF, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QSlider, QSpinBox, QSplitter, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QComboBox, QLineEdit, QScrollArea)

from .classdefs import CLASS_COLORS, DEFAULT_CLASS_DEFINITIONS

COLORS = CLASS_COLORS


def _definition_map(classes, definitions=None):
    """Keep stable keys separate from editable presentation, including old callers."""
    known = {item['id']: dict(item) for item in DEFAULT_CLASS_DEFINITIONS}
    known.update({item['id']: dict(item) for item in (definitions or [])})
    return {key: known.get(key, {'id': key, 'name': key, 'description': '',
                                'color': COLORS[index % len(COLORS)]})
            for index, key in enumerate(classes)}


class _PreviewSignals(QObject):
    ready = Signal(str, object, str)


class _PreviewTask(QRunnable):
    def __init__(self, key, path, size):
        super().__init__()
        self.key, self.path, self.size = key, path, size
        self.signals = _PreviewSignals()

    def run(self):
        try:
            import numpy as np
            from .imaging import read_grayscale
            arr = read_grayscale(self.path)
            sample = arr[::max(1, arr.shape[0] // 256), ::max(1, arr.shape[1] // 256)]
            low, high = np.nanpercentile(sample, [0.5, 99.5])
            if not np.isfinite(low + high):
                raise ValueError('표시할 수 없는 픽셀 값입니다.')
            if high <= low:
                high = low + 1
            data = np.ascontiguousarray(np.clip((arr.astype(np.float32) - low) * (255.0 / (high-low)), 0, 255).astype(np.uint8))
            img = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_Grayscale8).copy()
            if self.size and max(img.width(), img.height()) > self.size:
                img = img.scaled(self.size, self.size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.signals.ready.emit(self.key, img, '')
        except Exception as exc:
            self.signals.ready.emit(self.key, None, str(exc))


class PreviewCache(QObject):
    loaded = Signal(str, object, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.cache = OrderedDict()
        self.pending = {}

    @staticmethod
    def key(path, size=0):
        p = Path(path)
        try:
            stat = p.stat()
            stamp = f'{stat.st_size}:{stat.st_mtime_ns}'
        except OSError:
            stamp = 'missing'
        return f'{p}:{stamp}:{size}'

    def request(self, path, size=0):
        key = self.key(path, size)
        if key in self.cache:
            self.cache.move_to_end(key)
            return key, self.cache[key]
        if key not in self.pending:
            task = _PreviewTask(key, str(path), size)
            task.signals.ready.connect(self._done)
            self.pending[key] = task
            self.pool.start(task)
        return key, None

    def _done(self, key, image, error):
        self.pending.pop(key, None)
        if image is not None:
            self.cache[key] = image
            self.cache.move_to_end(key)
            while len(self.cache) > 30 or sum(im.sizeInBytes() for im in self.cache.values()) > 160_000_000:
                self.cache.popitem(last=False)
        self.loaded.emit(key, image, error)


class ImageView(QGraphicsView):
    viewChanged = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = self.scene().addPixmap(QPixmap())
        self.setBackgroundBrush(QColor('#11161e'))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.fit_mode = True
        self._syncing = False
        self.setMinimumSize(80, 110)
        self.message = '폴더를 선택하세요'

    def set_image(self, image):
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self.message = ''
        if self.fit_mode:
            self.fit()
        self.viewport().update()

    def clear_image(self, message='불러오는 중…'):
        self.pixmap_item.setPixmap(QPixmap())
        self.message = message
        self.viewport().update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.message:
            painter = QPainter(self.viewport())
            painter.setPen(QColor('#8c9bb0'))
            painter.drawText(self.viewport().rect().adjusted(10,10,-10,-10), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)

    def fit(self):
        self.fit_mode = True
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self):
        self.fit_mode = False
        self.resetTransform()
        self._emit_view()

    def wheelEvent(self, event):
        self.fit_mode = False
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        scale = self.transform().m11() * factor
        if 0.025 <= scale <= 15:
            self.scale(factor, factor)
            self._emit_view()
        event.accept()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.fit_mode = False
            self._emit_view()

    def _emit_view(self):
        if not self._syncing:
            self.viewChanged.emit(self.transform(), self.mapToScene(self.viewport().rect().center()))

    def sync(self, transform, center):
        self._syncing = True
        self.fit_mode = False
        self.setTransform(transform)
        self.centerOn(center)
        self._syncing = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit()


class SequenceTimeline(QWidget):
    selected = Signal(int)
    landmarkMoved = Signal(str, int)

    def __init__(self, cache, parent=None):
        super().__init__(parent)
        self.cache = cache
        self.cache.loaded.connect(self._loaded)
        self.images, self.classes, self.landmarks, self.scores = [], [], {}, []
        self.class_definitions = {}
        self.selected_class = None
        self.thumbs, self.thumb_keys = {}, {}
        self.cursor = 0
        self.drag_class = None
        self.keep_range = None
        self.setMinimumHeight(148)
        self.setMaximumHeight(180)
        self.setMouseTracking(True)
        self.setToolTip('썸네일을 클릭해 이동 · 아래 그래프를 클릭해 이미지 선택 · 색상 기준점을 드래그해 변경')

    def set_sequence(self, sequence, classes, landmarks, class_definitions=None):
        self.images = sequence.get('images', []) if sequence else []
        self.classes, self.landmarks = list(classes), dict(landmarks)
        self.class_definitions = _definition_map(classes, class_definitions)
        self.selected_class = self.classes[0] if self.classes else None
        self.scores = []
        self.keep_range = None
        self.thumbs, self.thumb_keys = {}, {}
        self.cursor = 0
        self._request_thumbs()
        self.update()

    def _request_thumbs(self):
        if not self.images:
            return
        count = max(3, min(10, self.width() // 100))
        for idx in sorted(set(round(i * (len(self.images)-1) / max(1, count-1)) for i in range(count))):
            key, image = self.cache.request(self.images[idx]['path'], 160)
            self.thumb_keys[key] = idx
            if image is not None:
                self.thumbs[idx] = QPixmap.fromImage(image)

    def _loaded(self, key, image, error):
        if key in self.thumb_keys and image is not None:
            self.thumbs[self.thumb_keys[key]] = QPixmap.fromImage(image)
            self.update()

    def set_scores(self, scores):
        try:
            self.scores = [float(x) for x in scores]
        except (TypeError, ValueError):
            self.scores = []
        self.update()

    def _x(self, index):
        return 15 + index / max(1, len(self.images)-1) * max(1, self.width()-30)

    def _index(self, x):
        return max(0, min(len(self.images)-1, round((x-15) / max(1,self.width()-30) * max(1,len(self.images)-1))))

    def class_name(self, key):
        return self.class_definitions.get(key, {}).get('name', key)

    def class_color(self, key):
        index = self.classes.index(key) if key in self.classes else 0
        return self.class_definitions.get(key, {}).get('color', COLORS[index % len(COLORS)])

    def _nearby_landmarks(self, x):
        return sorted((key for key in self.classes
                       if isinstance(self.landmarks.get(key), int)
                       and abs(self._x(self.landmarks[key])-x) < 8),
                      key=lambda key: (key != self.selected_class,
                                       abs(self._x(self.landmarks[key])-x)))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor('#151d29'))
        if not self.images:
            p.setPen(QColor('#8c9bb0'))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, '이미지 타임라인')
            return
        indices = sorted(set(self.thumb_keys.values()))
        cell = self.width() / max(1, len(indices))
        for slot, idx in enumerate(indices):
            rect = QRectF(slot*cell+4, 4, cell-8, 63)
            if idx in self.thumbs:
                pix = self.thumbs[idx]
                fitted = pix.size().scaled(int(rect.width()), int(rect.height()), Qt.AspectRatioMode.KeepAspectRatio)
                target = QRectF(rect.center().x()-fitted.width()/2, rect.center().y()-fitted.height()/2, fitted.width(), fitted.height())
                p.drawPixmap(target, pix, QRectF(pix.rect()))
            p.setPen(QColor('#8c9bb0'))
            p.drawText(QRectF(slot*cell, 68, cell, 17), Qt.AlignmentFlag.AlignCenter, str(self.images[idx].get('name',idx+1)))
        graph_top, graph_bottom = 94, self.height()-18
        if self.keep_range is not None:
            start,end=self.keep_range
            margin=max(2,(self.width()-30)/max(1,len(self.images)-1)/2)
            left=max(8,self._x(start)-margin)
            right=min(self.width()-8,self._x(end)+margin)
            p.fillRect(QRectF(left,88,max(1,right-left),graph_bottom-82),QColor(55,179,151,65))
            p.setPen(QPen(QColor('#62d6bb'),1))
            p.drawLine(QPointF(left,graph_bottom+5),QPointF(right,graph_bottom+5))
        p.setPen(QPen(QColor('#2d3b4e'),1))
        p.drawLine(15,graph_bottom,self.width()-15,graph_bottom)
        if self.scores:
            maximum = max(self.scores) or 1
            p.setPen(QPen(QColor('#7187a4'),1.5))
            for idx in range(1,min(len(self.images),len(self.scores))):
                p.drawLine(QPointF(self._x(idx-1),graph_bottom-self.scores[idx-1]/maximum*(graph_bottom-graph_top)), QPointF(self._x(idx),graph_bottom-self.scores[idx]/maximum*(graph_bottom-graph_top)))
        # Paint the selected marker last so dense class sequences remain editable.
        for name in sorted(self.classes, key=lambda key: key == self.selected_class):
            value = self.landmarks.get(name)
            if value is None:
                continue
            x = self._x(value)
            selected = name == self.selected_class
            p.setPen(QPen(QColor(self.class_color(name)),3 if selected else 2))
            p.drawLine(QPointF(x,90),QPointF(x,graph_bottom+4))
            p.setBrush(QColor(self.class_color(name)))
            p.drawEllipse(QPointF(x,94),6 if selected else 4,6 if selected else 4)
        p.setPen(QPen(QColor('#ffffff'),1,Qt.PenStyle.DashLine))
        p.drawLine(QPointF(self._x(self.cursor),88),QPointF(self._x(self.cursor),graph_bottom+6))
        p.setPen(QColor('#8493a8'))
        footer = '변화량 · 기준점에 마우스를 올리면 클래스 표시 · 드래그해서 수정'
        if self.selected_class in self.classes:
            number = self.classes.index(self.selected_class)+1
            footer = f'{number}. {self.class_name(self.selected_class)} · 선택한 기준점 · 드래그해서 수정'
        p.drawText(15,self.height()-3,p.fontMetrics().elidedText(footer, Qt.TextElideMode.ElideRight, self.width()-30))

    def mousePressEvent(self,event):
        if not self.images:
            return
        if event.position().y() < 86:
            indices=sorted(set(self.thumb_keys.values()))
            if indices:
                slot=min(len(indices)-1,int(event.position().x()/max(1,self.width())*len(indices)))
                self.selected.emit(indices[slot])
            return
        nearby = self._nearby_landmarks(event.position().x())
        self.drag_class = nearby[0] if nearby else None
        self.selected.emit(self._index(event.position().x()))

    def mouseMoveEvent(self,event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.images and event.position().y()>=86:
            self.selected.emit(self._index(event.position().x()))
        nearby = self._nearby_landmarks(event.position().x()) if event.position().y() >= 86 else []
        if nearby:
            lines = []
            for key in nearby:
                value = self.landmarks[key]
                filename = self.images[value]['name'] if 0 <= value < len(self.images) else str(value+1)
                lines.append(f'{self.classes.index(key)+1}. {self.class_name(key)} · {filename}')
            self.setToolTip('\n'.join(lines))
        else:
            self.setToolTip('썸네일을 클릭해 이동 · 아래 그래프를 클릭해 이미지 선택 · 색상 기준점을 드래그해 변경')

    def mouseReleaseEvent(self,event):
        if self.drag_class is not None:
            self.landmarkMoved.emit(self.drag_class,self._index(event.position().x()))
            self.drag_class=None

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self._request_thumbs()


class SequenceEditor(QWidget):
    changed = Signal(dict)
    cursorChanged = Signal(int)
    nextRequested = Signal()
    skipRequested = Signal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.sequence=None
        self.classes=[]
        self.class_definitions={}
        self.landmarks={}
        self.cursor=0
        self.undo_stack=[]
        self.active_keys={}
        self.analysis=None
        self.cache=PreviewCache(self)
        self.cache.loaded.connect(self._preview_loaded)
        main=QVBoxLayout(self)
        main.setContentsMargins(0,0,0,0)
        titlebar=QHBoxLayout()
        self.folder_label=QLabel('이미지 폴더를 선택하세요')
        self.folder_label.setObjectName('sectionTitle')
        titlebar.addWidget(self.folder_label,1)
        self.compare=QCheckBox('앞뒤 비교')
        self.compare.setChecked(True)
        self.compare.toggled.connect(self._toggle_comparison)
        titlebar.addWidget(self.compare)
        fit=QPushButton('화면 맞춤')
        fit.clicked.connect(lambda:[view.fit() for view in self.views])
        titlebar.addWidget(fit)
        actual=QPushButton('100%')
        actual.clicked.connect(lambda:self.views[1].actual_size())
        titlebar.addWidget(actual)
        main.addLayout(titlebar)
        split=QSplitter(Qt.Orientation.Horizontal)
        canvas=QWidget()
        canvas_layout=QVBoxLayout(canvas)
        canvas_layout.setContentsMargins(0,0,0,0)
        self.image_split=QSplitter(Qt.Orientation.Horizontal)
        self.views=[]
        self.frame_labels=[]
        self.frames=[]
        for label in ['이전','현재','다음']:
            frame=QWidget()
            layout=QVBoxLayout(frame)
            layout.setContentsMargins(2,0,2,0)
            caption=QLabel(label)
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setObjectName('muted')
            view=ImageView()
            layout.addWidget(caption)
            layout.addWidget(view,1)
            self.image_split.addWidget(frame)
            self.views.append(view)
            self.frame_labels.append(caption)
            self.frames.append(frame)
        self.image_split.setSizes([180,500,180])
        for source in self.views:
            source.viewChanged.connect(lambda transform,center,src=source:self._sync_views(src,transform,center))
        canvas_layout.addWidget(self.image_split,1)
        nav=QHBoxLayout()
        for text,delta in [('−10',-10),('이전',-1),('다음',1),('+10',10)]:
            button=QPushButton(text)
            button.setMaximumWidth(65)
            button.clicked.connect(lambda checked=False,d=delta:self.move(d))
            nav.addWidget(button)
        self.position=QLabel('0 / 0')
        nav.addWidget(self.position,1)
        nav.addWidget(QLabel('순번'))
        self.jump=QSpinBox()
        self.jump.setRange(1,1)
        self.jump.setKeyboardTracking(False)
        self.jump.valueChanged.connect(lambda value:self.set_cursor(value-1))
        nav.addWidget(self.jump)
        nav.addWidget(QLabel('파일 번호'))
        self.number_jump=QLineEdit()
        self.number_jump.setMaximumWidth(85)
        self.number_jump.setPlaceholderText('예: 100')
        self.number_jump.setToolTip('파일명에 있는 숫자를 입력하고 Enter를 누르세요.')
        self.number_jump.returnPressed.connect(self.jump_to_number)
        nav.addWidget(self.number_jump)
        canvas_layout.addLayout(nav)
        self.range_label=QLabel('보존 구간 · 기준점을 지정하세요')
        self.range_label.setObjectName('muted')
        canvas_layout.addWidget(self.range_label)
        self.slider=QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0,0)
        self.slider.valueChanged.connect(self.set_cursor)
        canvas_layout.addWidget(self.slider)
        self.timeline=SequenceTimeline(self.cache)
        self.timeline.selected.connect(self.set_cursor)
        self.timeline.landmarkMoved.connect(self.assign)
        canvas_layout.addWidget(self.timeline)
        split.addWidget(canvas)
        side=QWidget()
        side.setMinimumWidth(230)
        side.setMaximumWidth(360)
        sl=QVBoxLayout(side)
        sl.setContentsMargins(12,0,0,0)
        label=QLabel('시작 기준점')
        label.setObjectName('sectionTitle')
        sl.addWidget(label)
        hint=QLabel('클래스가 처음 시작되는 이미지에 지정합니다.\n↑ ↓: 클래스 선택 · Space: 지정 · 1–9: 바로 지정')
        hint.setWordWrap(True)
        hint.setObjectName('muted')
        sl.addWidget(hint)
        self.class_filter=QLineEdit()
        self.class_filter.setPlaceholderText('클래스 이름 / 기준 검색')
        self.class_filter.setClearButtonEnabled(True)
        self.class_filter.textChanged.connect(self._filter_classes)
        sl.addWidget(self.class_filter)
        self.class_progress=QLabel('0 / 0개 지정')
        self.class_progress.setObjectName('muted')
        sl.addWidget(self.class_progress)
        self.label_table=QTableWidget(0,2)
        self.label_table.setHorizontalHeaderLabels(['클래스','이미지'])
        self.label_table.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.label_table.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.ResizeToContents)
        self.label_table.verticalHeader().hide()
        self.label_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.label_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.label_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.label_table.cellDoubleClicked.connect(self._jump_landmark)
        self.label_table.currentCellChanged.connect(self._selected_class_changed)
        self.label_table.setMinimumHeight(120)
        self.label_table.setMaximumHeight(300)
        sl.addWidget(self.label_table,1)
        self.class_criterion=QLabel('클래스를 선택하면 라벨링 기준이 표시됩니다.')
        self.class_criterion.setWordWrap(True)
        self.class_criterion.setObjectName('muted')
        criterion_scroll=QScrollArea()
        criterion_scroll.setWidgetResizable(True)
        criterion_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        criterion_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        criterion_scroll.setMinimumHeight(55)
        criterion_scroll.setMaximumHeight(95)
        criterion_scroll.setWidget(self.class_criterion)
        sl.addWidget(criterion_scroll)
        assign=QPushButton('선택한 클래스에 현재 이미지 지정')
        assign.setObjectName('primary')
        assign.clicked.connect(self.assign_selected)
        sl.addWidget(assign)
        row=QHBoxLayout()
        remove=QPushButton('기준점 해제')
        remove.clicked.connect(self.remove_selected)
        undo=QPushButton('실행 취소')
        undo.clicked.connect(self.undo)
        row.addWidget(remove)
        row.addWidget(undo)
        sl.addLayout(row)
        sl.addWidget(QLabel('추천 후보'))
        self.candidates=QComboBox()
        self.candidates.addItem('변화량 분석 후 표시됩니다',None)
        self.candidates.activated.connect(self._candidate_selected)
        sl.addWidget(self.candidates)
        self.warning=QLabel('')
        self.warning.setWordWrap(True)
        self.warning.setObjectName('warning')
        sl.addWidget(self.warning)
        sl.addStretch()
        note=QLabel('표시 대비 자동 조절 · 원본 변경 없음\n휠: 확대 / 축소 · 드래그: 이미지 이동\n← →: 한 장 · Shift+← →: 열 장')
        note.setWordWrap(True)
        note.setObjectName('muted')
        sl.addWidget(note)
        self.next_button=QPushButton('확인하고 다음 폴더  ↵')
        self.next_button.setObjectName('primary')
        self.next_button.clicked.connect(self.nextRequested)
        sl.addWidget(self.next_button)
        self.skip_button=QPushButton('판단 불가 · 건너뛰기')
        self.skip_button.clicked.connect(self.skipRequested)
        sl.addWidget(self.skip_button)
        split.addWidget(side)
        split.setStretchFactor(0,1)
        split.setSizes([900,250])
        main.addWidget(split,1)

    def _toggle_comparison(self,checked):
        self.frames[0].setVisible(checked)
        self.frames[2].setVisible(checked)

    def _sync_views(self,source,transform,center):
        for view in self.views:
            if view is not source:
                view.sync(transform,center)

    def set_sequence(self,sequence,classes,landmarks=None,cursor=0,class_definitions=None):
        definitions = _definition_map(classes, class_definitions)
        # Renaming/recoloring a class must not discard review offsets, rules or undo.
        if (sequence == self.sequence and list(classes) == self.classes
                and dict(landmarks or {}) == self.landmarks
                and definitions != self.class_definitions):
            self.class_definitions=definitions
            self.timeline.class_definitions=definitions
            self._update_labels()
            self._update_warning()
            if self.analysis is not None:
                candidate_index=self.candidates.currentIndex()
                self.set_analysis(self.analysis)
                self.candidates.setCurrentIndex(candidate_index)
            self.timeline.update()
            return
        self.sequence=sequence
        self.classes=list(classes)
        self.class_definitions=definitions
        self.landmarks=dict(landmarks or {})
        self.undo_stack=[]
        self.analysis=None
        images=sequence.get('images',[]) if sequence else []
        self.folder_label.setText(sequence.get('relative_path','') if sequence else '이미지 폴더를 선택하세요')
        self.folder_label.setToolTip(sequence.get('path','') if sequence else '')
        self.slider.blockSignals(True)
        self.slider.setRange(0,max(0,len(images)-1))
        self.slider.blockSignals(False)
        self.jump.blockSignals(True)
        self.jump.setRange(1,max(1,len(images)))
        self.jump.blockSignals(False)
        self.timeline.set_sequence(sequence,classes,self.landmarks,class_definitions=list(definitions.values()))
        self._update_labels()
        self._training_keep_range()
        self.candidates.clear()
        self.candidates.addItem('변화량 분석 후 표시됩니다',None)
        self.set_cursor(cursor,force=True)
        self._update_warning()

    def set_cursor(self,index,force=False):
        if not self.sequence or not self.sequence.get('images'):
            self.position.setText('0 / 0')
            self.number_jump.clear()
            for view in self.views:
                view.clear_image('이미지가 없습니다')
            return
        images=self.sequence['images']
        index=max(0,min(len(images)-1,int(index)))
        if index==self.cursor and not force:
            return
        self.cursor=index
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self.jump.blockSignals(True)
        self.jump.setValue(index+1)
        self.jump.blockSignals(False)
        self.position.setText(f'{images[index]["name"]}  ·  {index+1} / {len(images)}')
        self.number_jump.setText(str(images[index].get('number','')))
        self.timeline.cursor=index
        self.timeline.update()
        self.active_keys={}
        for view_idx,delta in enumerate([-1,0,1]):
            idx=index+delta
            if idx<0 or idx>=len(images):
                self.views[view_idx].clear_image('첫 이미지 이전' if idx<0 else '마지막 이미지 이후')
                self.frame_labels[view_idx].setText(['이전','현재','다음'][view_idx])
                continue
            item=images[idx]
            self.frame_labels[view_idx].setText(f'{["이전","현재","다음"][view_idx]} · {item["name"]}')
            key,image=self.cache.request(item['path'])
            self.active_keys.setdefault(key,[]).append(view_idx)
            if image is not None:
                self.views[view_idx].set_image(image)
            else:
                self.views[view_idx].clear_image()
        self.cursorChanged.emit(index)

    def _preview_loaded(self,key,image,error):
        for idx in self.active_keys.get(key,[]):
            if image is not None:
                self.views[idx].set_image(image)
            else:
                self.views[idx].clear_image(f'이미지 읽기 오류\n{error}')

    def jump_to_number(self):
        if not self.sequence:
            return
        try:
            number=int(self.number_jump.text().strip())
        except ValueError:
            self.warning.setText('파일 번호는 정수로 입력하세요.')
            return
        matches=[i for i,item in enumerate(self.sequence['images']) if item.get('number')==number]
        if len(matches)!=1:
            self.warning.setText(f'{number}번 이미지가 없거나 중복됩니다. 목록을 확인하세요.')
            return
        self.set_cursor(matches[0])
        self._update_warning()

    def set_keep_range(self,start=None,end=None):
        images=self.sequence.get('images',[]) if self.sequence else []
        if isinstance(start,int) and isinstance(end,int) and 0<=start<=end<len(images):
            self.timeline.keep_range=(start,end)
            self.range_label.setText(f'보존 구간  {images[start]["name"]} → {images[end]["name"]}  ·  {end-start+1}장 모두 보존')
        else:
            self.timeline.keep_range=None
            self.range_label.setText('보존 구간 · 모든 기준점을 순서대로 지정하세요')
        self.timeline.update()

    def _training_keep_range(self):
        if self.classes:
            self.set_keep_range(self.landmarks.get(self.classes[0]),self.landmarks.get(self.classes[-1]))
        else:
            self.set_keep_range()

    def move(self,delta):
        self.set_cursor(self.cursor+delta)

    def assign(self,name,index=None):
        if not self.sequence or not self.sequence.get('images') or name not in self.classes:
            return
        index=self.cursor if index is None else index
        if self.landmarks.get(name)==index:
            return
        self.undo_stack.append(dict(self.landmarks))
        self.landmarks[name]=int(index)
        self._labels_changed()

    def assign_selected(self):
        key=self._class_at_row(self.label_table.currentRow())
        if key is not None:
            self.assign(key)

    def select_next_class(self,delta):
        """Move through visible classes, including those beyond numeric shortcuts."""
        rows=[row for row in range(self.label_table.rowCount()) if not self.label_table.isRowHidden(row)]
        if not rows:
            return
        current=self.label_table.currentRow()
        position=rows.index(current) if current in rows else (-1 if delta > 0 else 0)
        self.label_table.selectRow(rows[(position+delta) % len(rows)])
        self.label_table.scrollToItem(self.label_table.currentItem())

    def _class_at_row(self,row):
        item=self.label_table.item(row,0) if row >= 0 else None
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _class_name(self,key):
        return self.class_definitions.get(key,{}).get('name',key)

    def assign_number(self,number):
        if 0<=number<len(self.classes):
            if self.label_table.isRowHidden(number):
                self.class_filter.clear()
            self.label_table.selectRow(number)
            self.label_table.scrollToItem(self.label_table.currentItem())
            self.assign(self.classes[number])

    def remove_selected(self):
        key=self._class_at_row(self.label_table.currentRow())
        if key in self.landmarks:
            self.undo_stack.append(dict(self.landmarks))
            self.landmarks.pop(key)
            self._labels_changed()

    def undo(self):
        if self.undo_stack:
            self.landmarks=self.undo_stack.pop()
            self._labels_changed()

    def _labels_changed(self):
        self._update_labels()
        self.timeline.landmarks=dict(self.landmarks)
        self._training_keep_range()
        self.timeline.update()
        self._update_warning()
        self.changed.emit(dict(self.landmarks))

    def _update_labels(self):
        selected=self._class_at_row(self.label_table.currentRow())
        self.label_table.blockSignals(True)
        self.label_table.setRowCount(len(self.classes))
        images=self.sequence.get('images',[]) if self.sequence else []
        for idx,key in enumerate(self.classes):
            definition=self.class_definitions[key]
            title=QTableWidgetItem(f'{idx+1}. {definition["name"]}')
            title.setData(Qt.ItemDataRole.UserRole,key)
            title.setToolTip(definition['name']+'\n'+definition.get('description',''))
            title.setForeground(QColor(definition.get('color',COLORS[idx%len(COLORS)])))
            self.label_table.setItem(idx,0,title)
            value=self.landmarks.get(key)
            text=images[value]['name'] if isinstance(value,int) and 0<=value<len(images) else '미지정'
            self.label_table.setItem(idx,1,QTableWidgetItem(text))
        if self.classes:
            self.label_table.selectRow(self.classes.index(selected) if selected in self.classes else 0)
        self.label_table.blockSignals(False)
        self._filter_classes()

    def _filter_classes(self,*args):
        query=self.class_filter.text().strip().casefold()
        visible=[]
        for row,key in enumerate(self.classes):
            definition=self.class_definitions[key]
            haystack=f'{row+1} {definition["name"]} {definition.get("description","")}'.casefold()
            hide=bool(query and query not in haystack)
            self.label_table.setRowHidden(row,hide)
            if not hide:
                visible.append(row)
        if self.label_table.currentRow() not in visible:
            if visible:
                self.label_table.selectRow(visible[0])
            else:
                self.label_table.clearSelection()
                self.label_table.setCurrentCell(-1,-1)
        images=self.sequence.get('images',[]) if self.sequence else []
        count=sum(isinstance(self.landmarks.get(key),int) and 0<=self.landmarks[key]<len(images)
                  for key in self.classes)
        suffix=f' · 검색 {len(visible)}개' if query else ''
        self.class_progress.setText(f'{count} / {len(self.classes)}개 지정{suffix}')
        self._selected_class_changed()

    def _selected_class_changed(self,*args):
        key=self._class_at_row(self.label_table.currentRow())
        self.timeline.selected_class=key
        self.timeline.update()
        if key is None:
            self.class_criterion.setText('클래스를 선택하면 라벨링 기준이 표시됩니다.')
            return
        description=self.class_definitions[key].get('description','').strip()
        self.class_criterion.setText(f'{self.classes.index(key)+1}. {self._class_name(key)}\n'
                                     +(description or '이 클래스가 처음 시작되는 이미지를 지정하세요.'))
        self.class_criterion.setToolTip(description)

    def _jump_landmark(self,row,column):
        key=self._class_at_row(row)
        if key in self.landmarks:
            self.set_cursor(self.landmarks[key])

    def validation_errors(self):
        errors=[]
        if not self.sequence or not self.sequence.get('images'):
            return ['이미지가 없습니다.']
        values=[]
        for name in self.classes:
            value=self.landmarks.get(name)
            if not isinstance(value,int) or not 0<=value<len(self.sequence['images']):
                errors.append(f'{self._class_name(name)}: 기준점을 지정하세요.')
            else:
                values.append(value)
        if any(a>=b for a,b in zip(values,values[1:])):
            errors.append('기준점은 클래스 순서에 따라 서로 다른 이미지에 지정해야 합니다.')
        return errors

    def _update_warning(self):
        errors=self.validation_errors()
        visible=errors[:3]
        if len(errors)>3:
            visible.append(f'외 {len(errors)-3}개 · 클래스 목록을 확인하세요.')
        self.warning.setText('\n'.join(visible))
        self.warning.setToolTip('\n'.join(errors))

    def set_analysis(self,analysis):
        self.analysis=analysis
        self.timeline.set_scores(analysis.get('scores',analysis.get('rule_analysis',{}).get('scores',[])))
        self.candidates.clear()
        candidates=analysis.get('candidates',[])
        if isinstance(candidates,dict):
            flattened=[]
            for name,items in candidates.items():
                for item in (items if isinstance(items,list) else [items]):
                    flattened.append(dict(item,**{'class':name}) if isinstance(item,dict) else {'index':item,'class':name})
            candidates=flattened
        for candidate in candidates:
            if isinstance(candidate,dict):
                idx=candidate.get('index',candidate.get('position',0))
                name=candidate.get('class','변화 후보')
            else:
                idx,name=candidate,'변화 후보'
            if isinstance(idx,int) and self.sequence and 0<=idx<len(self.sequence['images']):
                self.candidates.addItem(f'{self._class_name(name)} · {self.sequence["images"][idx]["name"]}',idx)
        if not self.candidates.count():
            self.candidates.addItem('추천 후보 없음',None)
        warnings=analysis.get('warnings',[])
        if warnings:
            self.warning.setText('\n'.join(str(x) for x in warnings[:4]))

    def _candidate_selected(self,index):
        value=self.candidates.itemData(index)
        if value is not None:
            self.set_cursor(value)
