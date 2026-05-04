"""
Review Labeling & Validation Tool  (PyQt5)
==========================================
İki mod:
  1. Label Editor  — pipeline REVIEW fotoları → bbox ekle/düzelt → kaydet
  2. Review Gallery — labellanmış fotoları GT ile karşılaştır

Kurulum:  pip install PyQt5
Kullanım: python tools/review_tool.py

Kısayollar (Editor):
  1-9        → class seç
  S          → kaydet
  A / ←      → önceki foto
  D / →      → sonraki foto
  Del        → seçili box sil
  U / Ctrl+Z → son box geri al
  Scroll     → zoom
  Orta tık   → pan

Bbox düzenleme:
  Sol tık (boş alan) → sürükle = yeni box çiz
  Sol tık (box)      → seç
  Sol tık (seçili, köşe/kenar nokta) → sürükle = resize
  Sol tık (seçili, iç alan) → sürükle = taşı
"""
import sys, json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsItem,
    QSizePolicy, QFrame, QMessageBox, QGraphicsTextItem,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPen, QPainter, QFont,
)

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR    = None  # run_config.json'dan okunur; yoksa hata vermesin diye None
GT_DIR       = None  # run_config.json'dan okunur; pipeline GT_DIR=None ise gösterilmez
PIPELINE_OUT = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out")

def _latest_run_dir():
    import re, sys
    # Komut satırından spesifik run verilebilir: --run 20260430_151636
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--run" and i + 1 < len(sys.argv) - 1:
            specific = PIPELINE_OUT / sys.argv[i + 2]
            if specific.exists():
                return specific
    runs = sorted([d for d in PIPELINE_OUT.iterdir()
                   if d.is_dir() and re.match(r'^\d{8}_\d{6}', d.name)], reverse=True)
    # labels/ klasörü dolu olan en son run'ı seç (henüz bitmemiş run'ları atla)
    for r in runs:
        if (r / "labels").exists() and any((r / "labels").glob("*.txt")):
            return r
    return runs[0] if runs else PIPELINE_OUT

def _run_image_dir(run_dir: Path) -> Path:
    cfg = run_dir / "run_config.json"
    if cfg.exists():
        try:
            return Path(json.load(open(cfg, encoding="utf-8"))["IMAGE_DIR"])
        except Exception:
            pass
    return None

def _run_gt_dir(run_dir: Path):
    cfg = run_dir / "run_config.json"
    if cfg.exists():
        try:
            val = json.load(open(cfg, encoding="utf-8")).get("GT_DIR")
            return Path(val) if val else None
        except Exception:
            pass
    return None

_run = _latest_run_dir()
REVIEW_DIR      = _run / "review"
PIPELINE_LABELS = _run / "labels"

CLASS_NAMES = [
    "person", "car", "falldown", "bus", "truck",
    "bicycle", "motorcycle", "boar", "tractor", "scooter", "tiller", "cat", "dog",
]

CLASS_COLORS = [
    QColor("#FF5555"),  # person
    QColor("#5599FF"),  # car
    QColor("#FFD700"),  # falldown
    QColor("#55CC55"),  # bus
    QColor("#CC66CC"),  # truck
    QColor("#FF8800"),  # bicycle
    QColor("#00CCCC"),  # motorcycle
    QColor("#FF55FF"),  # boar
    QColor("#BB8800"),  # tractor
    QColor("#FFFFFF"),  # scooter
    QColor("#FFB6C1"),  # tiller
    QColor("#00FF88"),  # cat
    QColor("#FF9933"),  # dog
]

IOU_EVAL = 0.35

DARK = """
    QWidget          { background: #1a1a2e; color: #e0e0e0; }
    QPushButton      { background: #2a2a4a; color: #e0e0e0; border: 1px solid #444;
                       padding: 5px 10px; border-radius: 3px; }
    QPushButton:hover{ background: #3a3a6a; }
    QListWidget      { background: #12122a; border: 1px solid #333; }
    QListWidget::item:selected { background: #2a2a6a; }
    QSplitter::handle{ background: #333; width: 4px; height: 4px; }
    QTabWidget::pane { border: 1px solid #333; }
    QTabBar::tab     { background: #2a2a4a; color: #aaa; padding: 8px 18px; }
    QTabBar::tab:selected { background: #3a3a6a; color: #fff; font-weight: bold; }
"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_cv2(path):
    arr = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def cv2_to_pixmap(img_bgr):
    h, w, ch = img_bgr.shape
    rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    qi  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qi.copy())

def load_labels(path):
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try: out.append((parts[0], [float(x) for x in parts[1:5]]))
            except ValueError: pass
    return out

def norm_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]

def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter   = max(0, x2-x1) * max(0, y2-y1)
    union   = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def cls_color(name):
    if name in CLASS_NAMES:
        return CLASS_COLORS[CLASS_NAMES.index(name)]
    return QColor("#AAAAAA")


# ── BoxItem ───────────────────────────────────────────────────────────────────
class BoxItem(QGraphicsRectItem):
    HANDLE_SIZE = 5  # scene-coord half-size of each resize handle square

    def __init__(self, rect: QRectF, cls_name: str, from_pipeline=False):
        super().__init__(rect)
        self.cls_name      = cls_name
        self.from_pipeline = from_pipeline
        self._is_selected  = False
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._label = QGraphicsTextItem(cls_name, self)
        self._label.setFont(QFont("Arial", 9, QFont.Bold))
        self._apply_style(False)

    def _apply_style(self, selected: bool):
        self._is_selected = selected
        c   = cls_color(self.cls_name)
        pen = QPen(c, 3 if selected else 2)
        if self.from_pipeline and not selected:
            pen.setStyle(Qt.DashLine)
        self.setPen(pen)
        self._label.setDefaultTextColor(c)
        r = self.rect()
        self._label.setPos(r.left(), r.top() - 18)
        self.update()

    def set_selected_visual(self, on: bool):
        self._apply_style(on)

    def change_class(self, cls_name: str):
        self.cls_name      = cls_name
        self.from_pipeline = False
        self._label.setPlainText(cls_name)
        self._apply_style(self._is_selected)

    def boundingRect(self):
        s = self.HANDLE_SIZE + 2
        return self.rect().adjusted(-s, -s, s, s)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self._is_selected:
            r = self.rect()
            x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
            cx, cy = (x1+x2)/2, (y1+y2)/2
            s = self.HANDLE_SIZE
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor("#222222"), 1))
            for pt in [
                QPointF(x1, y1), QPointF(cx, y1), QPointF(x2, y1),
                QPointF(x1, cy),                   QPointF(x2, cy),
                QPointF(x1, y2), QPointF(cx, y2),  QPointF(x2, y2),
            ]:
                painter.drawRect(QRectF(pt.x()-s, pt.y()-s, s*2, s*2))

    def handle_at(self, scene_pos, view_scale):
        """Returns handle name ('nw','n','ne','w','e','sw','s','se') or None."""
        r = self.rect()
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
        cx, cy = (x1+x2)/2, (y1+y2)/2
        tol = max(self.HANDLE_SIZE + 2, 7.0 / max(view_scale, 0.01))
        handles = {
            'nw': QPointF(x1, y1), 'n':  QPointF(cx, y1), 'ne': QPointF(x2, y1),
            'w':  QPointF(x1, cy),                          'e':  QPointF(x2, cy),
            'sw': QPointF(x1, y2), 's':  QPointF(cx, y2), 'se': QPointF(x2, y2),
        }
        for name, pt in handles.items():
            if abs(scene_pos.x()-pt.x()) <= tol and abs(scene_pos.y()-pt.y()) <= tol:
                return name
        return None

    def xyxy(self):
        r = self.rect()
        return r.left(), r.top(), r.right(), r.bottom()


# ── LabelCanvas ───────────────────────────────────────────────────────────────
class LabelCanvas(QGraphicsView):
    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setStyleSheet("background: #111122; border: none;")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.current_class = CLASS_NAMES[0]
        self.boxes = []
        self._selected = None

        # draw mode
        self._drawing    = False
        self._draw_start = None
        self._temp_rect  = None

        # pan mode
        self._pan_last   = None

        # resize mode
        self._resize_handle    = None
        self._resize_item      = None
        self._resize_orig_rect = None
        self._resize_start     = None

        # move mode
        self._move_item      = None
        self._move_orig_rect = None
        self._move_start     = None

        self.img_w = self.img_h = 1

    # ── Image ─────────────────────────────────────────────────────────────────
    def load_image(self, img_bgr):
        self._scene.clear()
        self.boxes.clear()
        self._selected          = None
        self._temp_rect         = None
        self._resize_handle     = None
        self._resize_item       = None
        self._move_item         = None
        self.img_h, self.img_w  = img_bgr.shape[:2]
        pix  = cv2_to_pixmap(img_bgr)
        item = self._scene.addPixmap(pix)
        item.setZValue(0)
        self._scene.setSceneRect(0, 0, self.img_w, self.img_h)
        QTimer.singleShot(0, lambda: self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio))

    # ── Box management ────────────────────────────────────────────────────────
    def add_box(self, cls_name, x1, y1, x2, y2, from_pipeline=False) -> BoxItem:
        rect = QRectF(x1, y1, x2-x1, y2-y1)
        b = BoxItem(rect, cls_name, from_pipeline)
        b.setZValue(1)
        self._scene.addItem(b)
        self.boxes.append(b)
        return b

    def _select(self, item):
        if self._selected:
            self._selected.set_selected_visual(False)
        self._selected = item
        if item:
            item.set_selected_visual(True)

    def remove_selected(self):
        if self._selected and self._selected in self.boxes:
            self._scene.removeItem(self._selected)
            self.boxes.remove(self._selected)
            self._selected = None
            self.changed.emit()

    def undo_last(self):
        if self.boxes:
            b = self.boxes.pop()
            if b is self._selected:
                self._selected = None
            self._scene.removeItem(b)
            self.changed.emit()

    def get_labels(self):
        out = []
        for b in self.boxes:
            x1, y1, x2, y2 = b.xyxy()
            cx = (x1+x2)/2/self.img_w;  cy = (y1+y2)/2/self.img_h
            bw = (x2-x1)/self.img_w;    bh = (y2-y1)/self.img_h
            out.append((b.cls_name, cx, cy, bw, bh))
        return out

    # ── Resize / move helpers ─────────────────────────────────────────────────
    def _apply_resize(self, orig: QRectF, handle: str, dx: float, dy: float) -> QRectF:
        x1, y1, x2, y2 = orig.left(), orig.top(), orig.right(), orig.bottom()
        if 'w' in handle: x1 = min(x1+dx, x2-5)
        if 'e' in handle: x2 = max(x2+dx, x1+5)
        if 'n' in handle: y1 = min(y1+dy, y2-5)
        if 's' in handle: y2 = max(y2+dy, y1+5)
        x1 = max(0.0, x1);          y1 = max(0.0, y1)
        x2 = min(float(self.img_w), x2); y2 = min(float(self.img_h), y2)
        return QRectF(x1, y1, x2-x1, y2-y1)

    def _clamp_rect(self, rect: QRectF) -> QRectF:
        x1 = max(0.0, rect.left())
        y1 = max(0.0, rect.top())
        x2 = min(float(self.img_w), rect.right())
        y2 = min(float(self.img_h), rect.bottom())
        return QRectF(x1, y1, max(5.0, x2-x1), max(5.0, y2-y1))

    def _resize_cursor(self, handle: str):
        return {
            'nw': Qt.SizeFDiagCursor, 'se': Qt.SizeFDiagCursor,
            'ne': Qt.SizeBDiagCursor, 'sw': Qt.SizeBDiagCursor,
            'n':  Qt.SizeVerCursor,   's':  Qt.SizeVerCursor,
            'w':  Qt.SizeHorCursor,   'e':  Qt.SizeHorCursor,
        }.get(handle, Qt.SizeAllCursor)

    def _update_hover_cursor(self, scene_pos: QPointF):
        if self._selected:
            scale = self.transform().m11()
            h = self._selected.handle_at(scene_pos, scale)
            if h:
                self.setCursor(self._resize_cursor(h))
                return
            if self._selected.rect().contains(scene_pos):
                self.setCursor(Qt.SizeAllCursor)
                return
        self.setCursor(Qt.ArrowCursor)

    # ── Mouse ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            pos  = self.mapToScene(ev.pos())
            item = self._scene.itemAt(pos, self.transform())
            if isinstance(item, QGraphicsTextItem):
                item = item.parentItem()

            if isinstance(item, BoxItem):
                if item is self._selected:
                    # Already selected → check resize handle first, then move
                    scale  = self.transform().m11()
                    handle = item.handle_at(pos, scale)
                    if handle:
                        self._resize_handle    = handle
                        self._resize_item      = item
                        self._resize_orig_rect = QRectF(item.rect())
                        self._resize_start     = pos
                        self.setCursor(self._resize_cursor(handle))
                    else:
                        self._move_item      = item
                        self._move_orig_rect = QRectF(item.rect())
                        self._move_start     = pos
                        self.setCursor(Qt.SizeAllCursor)
                else:
                    self._select(item)
                    self.changed.emit()
                return

            # Click on empty area → deselect + start draw
            self._select(None)
            self._drawing    = True
            self._draw_start = pos
            self._temp_rect  = self._scene.addRect(
                QRectF(pos, pos),
                QPen(cls_color(self.current_class), 2, Qt.DashLine)
            )
            self._temp_rect.setZValue(2)

        elif ev.button() == Qt.MiddleButton:
            self._pan_last = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)

        elif ev.button() == Qt.RightButton:
            self.remove_selected()

    def mouseMoveEvent(self, ev):
        pos = self.mapToScene(ev.pos())

        if self._resize_handle and self._resize_item:
            dx = pos.x() - self._resize_start.x()
            dy = pos.y() - self._resize_start.y()
            new_rect = self._apply_resize(self._resize_orig_rect, self._resize_handle, dx, dy)
            self._resize_item.setRect(new_rect)
            self._resize_item._apply_style(True)
            self.changed.emit()
            return

        if self._move_item:
            dx = pos.x() - self._move_start.x()
            dy = pos.y() - self._move_start.y()
            r  = self._move_orig_rect
            new_rect = self._clamp_rect(QRectF(r.left()+dx, r.top()+dy, r.width(), r.height()))
            self._move_item.setRect(new_rect)
            self._move_item._apply_style(True)
            self.changed.emit()
            return

        if self._drawing and self._temp_rect and self._draw_start:
            r = QRectF(self._draw_start, pos).normalized()
            self._temp_rect.setRect(r)
            return

        if self._pan_last is not None:
            delta = ev.pos() - self._pan_last
            self._pan_last = ev.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        self._update_hover_cursor(pos)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            if self._resize_handle:
                self._resize_handle = self._resize_item = None
                self._resize_orig_rect = self._resize_start = None
                self.setCursor(Qt.ArrowCursor)
                return

            if self._move_item:
                self._move_item._apply_style(True)
                self._move_item = self._move_orig_rect = self._move_start = None
                self.setCursor(Qt.ArrowCursor)
                self.changed.emit()
                return

            if self._drawing:
                self._drawing = False
                pos  = self.mapToScene(ev.pos())
                rect = QRectF(self._draw_start, pos).normalized()
                if self._temp_rect:
                    self._scene.removeItem(self._temp_rect)
                    self._temp_rect = None
                if rect.width() > 5 and rect.height() > 5:
                    b = self.add_box(self.current_class,
                                     rect.left(), rect.top(), rect.right(), rect.bottom())
                    self._select(b)
                    self.changed.emit()

        elif ev.button() == Qt.MiddleButton:
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)

    def enterEvent(self, ev):
        self.setFocus()

    def wheelEvent(self, ev):
        f = 1.15 if ev.angleDelta().y() > 0 else 1/1.15
        self.scale(f, f)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.remove_selected()
        elif ev.key() == Qt.Key_F:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        else:
            super().keyPressEvent(ev)


# ── LabelEditor ───────────────────────────────────────────────────────────────
class LabelEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.image_files = []
        self.cur_idx = 0
        self._vlm_info = {}
        self._pipeline_labels = PIPELINE_OUT  # fallback; updated in _scan_images
        self._review_dir      = PIPELINE_OUT
        self._gt_dir          = None
        self._build_ui()
        self.setStyleSheet(DARK)
        self._scan_images()

    def _out_dir(self) -> Path:
        """Reviewed labels go to the current run's reviewed_labels subfolder."""
        d = self._pipeline_labels.parent / "reviewed_labels"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_vlm_explains(self):
        jsons = sorted(self._review_dir.glob("review_*.json"))
        for jf in jsons:
            try:
                for entry in json.load(open(jf, encoding="utf-8")):
                    name = entry.get("image", "")
                    if name and entry.get("explain"):
                        self._vlm_info[name] = entry
            except Exception:
                pass

    def _scan_images(self):
        _run = _latest_run_dir()
        self._review_dir      = _run / "review"
        self._pipeline_labels = _run / "labels"
        self._image_dir       = _run_image_dir(_run)
        self._gt_dir          = _run_gt_dir(_run)
        jsons = sorted(self._review_dir.glob("review_*.json"))
        if jsons:
            latest = jsons[-1]
            try:
                entries = json.load(open(latest, encoding="utf-8"))
                names   = [e["image"] for e in entries if "image" in e]
                self.image_files = [self._image_dir / n for n in names if (self._image_dir / n).exists()]
            except Exception:
                self.image_files = []
        else:
            self.image_files = []
        self._load_vlm_explains()
        if self.image_files:
            self.load_image(0)
        else:
            self.lbl_info.setText("REVIEW klasörü boş")

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # ── Sol: canvas + nav ────────────────────────────────────────────────
        left = QWidget()
        lv   = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)

        self.canvas = LabelCanvas()
        self.canvas.changed.connect(self._on_canvas_changed)
        lv.addWidget(self.canvas, stretch=1)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◄  Önceki  (A)")
        self.lbl_info = QLabel("0 / 0")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.btn_next = QPushButton("Sonraki  (D)  ►")
        self.btn_save = QPushButton("💾  Kaydet  (S)")
        self.btn_save.setStyleSheet(
            "QPushButton { background:#1a5c1a; color:white; font-weight:bold; }"
            "QPushButton:hover { background:#226622; }"
        )
        for w in [self.btn_prev, self.lbl_info, self.btn_next, self.btn_save]:
            nav.addWidget(w)
        lv.addLayout(nav)

        self.status = QLabel("Hazır")
        self.status.setStyleSheet("color:#888; font-size:11px; padding:2px 4px;")
        lv.addWidget(self.status)

        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_save.clicked.connect(self.save_current)

        # ── Sağ: class + hints ───────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(195)
        rv = QVBoxLayout(right)
        rv.setSpacing(4)

        rv.addWidget(QLabel("<b>Class Seç</b>"))
        self.cls_list = QListWidget()
        self.cls_list.setMaximumHeight(290)
        for i, cls in enumerate(CLASS_NAMES):
            shortcut = str(i+1) if i < 9 else "-"
            item = QListWidgetItem(f"  {shortcut}  {cls}")
            item.setForeground(CLASS_COLORS[i])
            self.cls_list.addItem(item)
        self.cls_list.setCurrentRow(0)
        self.cls_list.currentRowChanged.connect(self._cls_changed)
        rv.addWidget(self.cls_list)

        rv.addSpacing(8)
        self.fname_lbl = QLabel("—")
        self.fname_lbl.setWordWrap(True)
        self.fname_lbl.setStyleSheet("color:#999; font-size:11px;")
        rv.addWidget(self.fname_lbl)

        rv.addSpacing(8)
        rv.addWidget(QLabel("<b>VLM Açıklaması</b>"))
        self.vlm_box = QLabel("—")
        self.vlm_box.setWordWrap(True)
        self.vlm_box.setStyleSheet(
            "background:#1a1a0a; color:#ffe08a; padding:6px; "
            "border:1px solid #444; font-size:11px; border-radius:3px;"
        )
        rv.addWidget(self.vlm_box)

        rv.addStretch()
        hint = QLabel(
            "<small><b>Kısayollar</b><br>"
            "Sürükle (boş) → yeni box<br>"
            "Sol tık → seç<br>"
            "Seçili + köşe/kenar → resize<br>"
            "Seçili + iç → taşı<br>"
            "Sağ tık / Del → sil<br>"
            "U / Ctrl+Z → geri al<br>"
            "1-9 → class seç<br>"
            "S → kaydet<br>"
            "A / D → önceki/sonraki<br>"
            "Scroll → zoom<br>"
            "Orta tık → pan</small>"
        )
        hint.setStyleSheet("color:#555; padding:4px;")
        hint.setWordWrap(True)
        rv.addWidget(hint)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter)

    def _cls_changed(self, row):
        if 0 <= row < len(CLASS_NAMES):
            self.canvas.current_class = CLASS_NAMES[row]
            if self.canvas._selected:
                self.canvas._selected.change_class(CLASS_NAMES[row])

    def _on_canvas_changed(self):
        n = len(self.canvas.boxes)
        pl = sum(1 for b in self.canvas.boxes if b.from_pipeline)
        self.status.setText(f"Box: {n}  (pipeline: {pl}, kullanıcı: {n-pl})")
        sel = self.canvas._selected
        if sel and sel.cls_name in CLASS_NAMES:
            self.cls_list.blockSignals(True)
            self.cls_list.setCurrentRow(CLASS_NAMES.index(sel.cls_name))
            self.cls_list.blockSignals(False)

    def load_image(self, idx):
        if not self.image_files: return
        self.cur_idx = idx
        p = self.image_files[idx]

        orig = self._image_dir / p.name
        img  = read_cv2(orig if orig.exists() else p)
        if img is None: return

        self.canvas.load_image(img)
        h, w = img.shape[:2]

        # Load priority: reviewed label → pipeline label
        reviewed  = self._out_dir() / (p.stem + ".txt")
        label_src = reviewed if reviewed.exists() else self._pipeline_labels / (p.stem + ".txt")
        is_pipeline = not reviewed.exists()

        for cls_name, box in load_labels(label_src):
            x1, y1, x2, y2 = norm_to_xyxy(box, w, h)
            self.canvas.add_box(cls_name, x1, y1, x2, y2, from_pipeline=is_pipeline)

        name = p.name
        self.lbl_info.setText(f"{idx+1} / {len(self.image_files)}")
        self.fname_lbl.setText(name if len(name) <= 32 else "…" + name[-29:])
        self._on_canvas_changed()

        info = self._vlm_info.get(p.name, {})
        if info.get("explain"):
            py, pn = info.get("p_yes", 0), info.get("p_no", 0)
            self.vlm_box.setText(f"P(Yes)={py:.2f}  P(No)={pn:.2f}\n\n{info['explain']}")
        else:
            self.vlm_box.setText("—")

    def save_current(self):
        if not self.image_files: return
        p      = self.image_files[self.cur_idx]
        labels = self.canvas.get_labels()
        out    = self._out_dir() / (p.stem + ".txt")
        lines  = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in labels]
        out.write_text("\n".join(lines), encoding="utf-8")
        self.status.setText(f"✓ Kaydedildi  ({len(labels)} box)  → {out.parent.name}/{out.name}")

    def prev_image(self):
        self.save_current()
        if self.cur_idx > 0:
            self.load_image(self.cur_idx - 1)

    def next_image(self):
        self.save_current()
        if self.cur_idx < len(self.image_files) - 1:
            self.load_image(self.cur_idx + 1)
        else:
            self.status.setText("✓ Son resim — tüm labellar tamamlandı!")

    def keyPressEvent(self, ev):
        k = ev.key()
        mod = ev.modifiers()
        if k == Qt.Key_S and not mod:
            self.save_current()
        elif k in (Qt.Key_A, Qt.Key_Left):
            self.prev_image()
        elif k in (Qt.Key_D, Qt.Key_Right):
            self.next_image()
        elif k == Qt.Key_U or (k == Qt.Key_Z and mod == Qt.ControlModifier):
            self.canvas.undo_last()
        elif Qt.Key_1 <= k <= Qt.Key_9:
            self.cls_list.setCurrentRow(k - Qt.Key_1)
        else:
            super().keyPressEvent(ev)


# ── ReviewGallery ─────────────────────────────────────────────────────────────
class ReviewGallery(QWidget):
    def __init__(self):
        super().__init__()
        self.image_files       = []
        self._cur_img_path     = None
        self._pipeline_labels  = PIPELINE_OUT  # fallback; updated in refresh()
        self._review_dir       = PIPELINE_OUT
        self._gt_dir           = None
        self._build_ui()
        self.setStyleSheet(DARK)

    def _out_dir(self) -> Path:
        d = self._pipeline_labels.parent / "reviewed_labels"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        # ── Sol: dosya listesi ───────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(240)
        lv = QVBoxLayout(left)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("<b>Fotoğraflar</b>"))
        btn_r = QPushButton("↻")
        btn_r.setFixedWidth(32)
        btn_r.clicked.connect(self.refresh)
        hdr.addWidget(btn_r)
        lv.addLayout(hdr)

        self.file_list = QListWidget()
        self.file_list.currentRowChanged.connect(self._show)
        lv.addWidget(self.file_list)

        # ── Orta: canvas + nav + stats ───────────────────────────────────────
        center = QWidget()
        cv_layout = QVBoxLayout(center)
        cv_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = LabelCanvas()
        self.canvas.changed.connect(self._on_canvas_changed)
        cv_layout.addWidget(self.canvas, stretch=1)

        cls_bar = QHBoxLayout()
        cls_bar.addWidget(QLabel("Class:"))
        self.cls_list = QListWidget()
        self.cls_list.setFlow(QListWidget.LeftToRight)
        self.cls_list.setMaximumHeight(28)
        self.cls_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for i, cls in enumerate(CLASS_NAMES):
            item = QListWidgetItem(f" {i+1 if i<9 else '-'} {cls} ")
            item.setForeground(CLASS_COLORS[i])
            self.cls_list.addItem(item)
        self.cls_list.setCurrentRow(0)
        self.cls_list.currentRowChanged.connect(self._cls_changed)
        cv_layout.addWidget(self.cls_list)

        nav = QHBoxLayout()
        self.btn_save = QPushButton("💾  Kaydet  (S)")
        self.btn_save.setStyleSheet(
            "QPushButton { background:#1a5c1a; color:white; font-weight:bold; }"
            "QPushButton:hover { background:#226622; }"
        )
        self.btn_save.clicked.connect(self.save_current)
        self.lbl_status = QLabel("Hazır")
        self.lbl_status.setStyleSheet("color:#888; font-size:11px;")
        nav.addWidget(self.btn_save)
        nav.addWidget(self.lbl_status, stretch=1)
        cv_layout.addLayout(nav)

        self.stats = QLabel("—")
        self.stats.setStyleSheet(
            "background:#0d1f0d; color:#aaffaa; padding:6px 8px;"
            "font-family:monospace; font-size:12px;"
        )
        self.stats.setWordWrap(True)
        cv_layout.addWidget(self.stats)

        # ── Sağ: legend ──────────────────────────────────────────────────────
        right = QWidget()
        right.setFixedWidth(155)
        rv = QVBoxLayout(right)
        rv.addWidget(QLabel("<b>Legend</b>"))
        for i, cls in enumerate(CLASS_NAMES):
            lbl = QLabel(f"■  {cls}")
            lbl.setStyleSheet(f"color:{CLASS_COLORS[i].name()}; font-weight:bold;")
            rv.addWidget(lbl)
        rv.addSpacing(10)
        gt_lbl = QLabel("╌╌  GT (beyaz)")
        gt_lbl.setStyleSheet("color:#ffffff;")
        rv.addWidget(gt_lbl)
        rv.addStretch()
        hint = QLabel(
            "<small>Sürükle (boş) → yeni box<br>"
            "Seçili + köşe/kenar → resize<br>"
            "Seçili + iç → taşı<br>"
            "Sağ tık / Del → sil<br>"
            "U / Ctrl+Z → geri al<br>"
            "S → kaydet<br>"
            "A/D → önceki/sonraki</small>"
        )
        hint.setStyleSheet("color:#555; padding:4px;")
        hint.setWordWrap(True)
        rv.addWidget(hint)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        root.addWidget(splitter)

    def _cls_changed(self, row):
        if 0 <= row < len(CLASS_NAMES):
            self.canvas.current_class = CLASS_NAMES[row]
            if self.canvas._selected:
                self.canvas._selected.change_class(CLASS_NAMES[row])

    def _on_canvas_changed(self):
        sel = self.canvas._selected
        if sel and sel.cls_name in CLASS_NAMES:
            self.cls_list.blockSignals(True)
            self.cls_list.setCurrentRow(CLASS_NAMES.index(sel.cls_name))
            self.cls_list.blockSignals(False)

    def refresh(self):
        _run = _latest_run_dir()
        self._pipeline_labels = _run / "labels"
        self._review_dir      = _run / "review"
        self._image_dir       = _run_image_dir(_run)
        self._gt_dir          = _run_gt_dir(_run)

        all_files = sorted(list(self._image_dir.glob("*.jpg")) +
                           list(self._image_dir.glob("*.png")) +
                           list(self._image_dir.glob("*.jpeg")))
        has_label = [f for f in all_files if (self._pipeline_labels / (f.stem+".txt")).exists()]
        no_label  = [f for f in all_files if f not in set(has_label)]
        self.image_files = has_label + no_label

        out_dir      = self._out_dir()
        review_stems = {f.stem for f in self._review_dir.glob("*.jpg")} | \
                       {f.stem for f in self._review_dir.glob("*.png")}

        self.file_list.clear()
        for f in self.image_files:
            reviewed  = (out_dir / (f.stem + ".txt")).exists()
            is_review = f.stem in review_stems
            prefix = "✓ " if reviewed else ("⚠ " if is_review else "  ")
            self.file_list.addItem(prefix + f.name)

        if self.image_files:
            self.file_list.setCurrentRow(0)

    def _show(self, idx):
        if idx < 0 or idx >= len(self.image_files): return
        p   = self.image_files[idx]
        self._cur_img_path = p
        img = read_cv2(p)
        if img is None: return
        h, w = img.shape[:2]

        self.canvas.load_image(img)

        # Load priority: reviewed → pipeline
        out_dir     = self._out_dir()
        reviewed    = out_dir / (p.stem + ".txt")
        src         = reviewed if reviewed.exists() else self._pipeline_labels / (p.stem + ".txt")
        is_pipeline = not reviewed.exists()
        for cls_name, box in load_labels(src):
            x1, y1, x2, y2 = norm_to_xyxy(box, w, h)
            self.canvas.add_box(cls_name, x1, y1, x2, y2, from_pipeline=is_pipeline)

        # GT boxes — white dashed, non-selectable
        gt_all = load_labels(self._gt_dir / (p.stem + ".txt")) if self._gt_dir else []
        for cls_name, box in gt_all:
            x1, y1, x2, y2 = norm_to_xyxy(box, w, h)
            rect    = QRectF(x1, y1, x2-x1, y2-y1)
            gt_item = QGraphicsRectItem(rect)
            gt_item.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
            gt_item.setZValue(0.5)
            gt_item.setFlag(QGraphicsItem.ItemIsSelectable, False)
            lbl = QGraphicsTextItem(f"GT:{cls_name}", gt_item)
            lbl.setDefaultTextColor(QColor("#cccccc"))
            lbl.setFont(QFont("Arial", 8))
            lbl.setPos(x1, y2 + 2)
            self.canvas._scene.addItem(gt_item)

        preds = load_labels(src)
        self._compute_stats(preds, gt_all, w, h)
        self.lbl_status.setText(p.name)

    def _compute_stats(self, preds, gt_all, w, h):
        stats = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})
        all_cls = set(c for c,_ in preds) | set(c for c,_ in gt_all)
        for cls in all_cls:
            gts = [norm_to_xyxy(b, w, h) for c,b in gt_all if c == cls]
            prs = [norm_to_xyxy(b, w, h) for c,b in preds  if c == cls]
            matched = set()
            tp = 0
            for pb in prs:
                best, gi = 0, -1
                for i, gb in enumerate(gts):
                    if i in matched: continue
                    s = iou_xyxy(pb, gb)
                    if s > best: best, gi = s, i
                if best >= IOU_EVAL:
                    matched.add(gi); tp += 1
            stats[cls]["tp"] += tp
            stats[cls]["fp"] += len(prs) - tp
            stats[cls]["fn"] += len(gts) - len(matched)

        if stats:
            parts = []
            ttp = tfp = tfn = 0
            for cls in sorted(stats):
                s = stats[cls]
                tp, fp, fn = s["tp"], s["fp"], s["fn"]
                ttp += tp; tfp += fp; tfn += fn
                p  = tp/(tp+fp) if (tp+fp) else 0
                r  = tp/(tp+fn) if (tp+fn) else 0
                f1 = 2*p*r/(p+r) if (p+r) else 0
                parts.append(f"<b>{cls}</b> TP={tp} FP={fp} FN={fn} F1={f1:.2f}")
            p  = ttp/(ttp+tfp) if (ttp+tfp) else 0
            r  = ttp/(ttp+tfn) if (ttp+tfn) else 0
            f1 = 2*p*r/(p+r)   if (p+r)    else 0
            parts.append(f"<b>TOPLAM</b>  P={p:.2f} R={r:.2f} <b>F1={f1:.2f}</b>"
                         f"  (TP={ttp} FP={tfp} FN={tfn})")
            self.stats.setText("   |   ".join(parts))
        else:
            self.stats.setText("Label yok")

    def save_current(self):
        if not self._cur_img_path: return
        labels  = self.canvas.get_labels()
        out_dir = self._out_dir()
        out     = out_dir / (self._cur_img_path.stem + ".txt")
        lines   = [f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cls, cx, cy, bw, bh in labels]
        out.write_text("\n".join(lines), encoding="utf-8")
        self.lbl_status.setText(f"✓ Kaydedildi  ({len(labels)} box)")
        row  = self.file_list.currentRow()
        item = self.file_list.item(row)
        if item:
            item.setText("✓ " + self._cur_img_path.name)

    def keyPressEvent(self, ev):
        k   = ev.key()
        mod = ev.modifiers()
        row = self.file_list.currentRow()
        if k == Qt.Key_S and not mod:
            self.save_current()
        elif k in (Qt.Key_A, Qt.Key_Left) and row > 0:
            self.file_list.setCurrentRow(row - 1)
        elif k in (Qt.Key_D, Qt.Key_Right) and row < self.file_list.count()-1:
            self.file_list.setCurrentRow(row + 1)
        elif k == Qt.Key_U or (k == Qt.Key_Z and mod == Qt.ControlModifier):
            self.canvas.undo_last()
        elif Qt.Key_1 <= k <= Qt.Key_9:
            self.cls_list.setCurrentRow(k - Qt.Key_1)
        else:
            super().keyPressEvent(ev)


# ── Main Window ───────────────────────────────────────────────────────────────
class ReviewTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Review Labeling Tool")
        self.resize(1440, 880)
        self.setStyleSheet(DARK)

        self.tabs = QTabWidget()
        self.editor  = LabelEditor()
        self.gallery = ReviewGallery()

        self.tabs.addTab(self.editor,  "✏️  Label Editor")
        self.tabs.addTab(self.gallery, "🔍  Review Gallery")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)

    def _tab_changed(self, idx):
        if idx == 1:
            self.gallery.refresh()

    def keyPressEvent(self, ev):
        if self.tabs.currentIndex() == 0:
            self.editor.keyPressEvent(ev)
        else:
            self.gallery.keyPressEvent(ev)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ReviewTool()
    win.show()
    sys.exit(app.exec_())
