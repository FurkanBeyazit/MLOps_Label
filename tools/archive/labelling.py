import os
import glob
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ★ YOLO import
from ultralytics import YOLO

# ==========================
# 설정 영역
# ==========================

CLASS_NAMES = [
    "person",
    "car",
    "falldown",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "boar",
    "tractor",
    "scooter",
    "tiller",
    "cat",
    "dog" 
]

# 클래스별 색상 (CLASS_NAMES 순서와 동일)
CLASS_COLORS = [
    "red",       # person
    "blue",      # car
    "yellow",    # falldown
    "green",     # bus
    "purple",    # truck
    "orange",    # bicycle
    "cyan",      # motorcycle
    "magenta",   # boar
    "brown",     # tractor
    "white",     # scooter
    "pink",      # tiller
    "lime",      # cat
    "gold",      # dog
]

MAX_CANVAS_WIDTH = 1280
MAX_CANVAS_HEIGHT = 720

# ★ YOLO 가중치 경로
YOLO_WEIGHTS_PATH = r"C:\Users\admin\fur\MLOps_Label\models\falldown\best_04_16.pt"

# YOLO 모델의 class index -> 현재 CLASS_NAMES index 매핑
#  예시는 COCO 계열 YOLO 기준입니다. (필요시 수정)
YOLO_TO_OUR_CLASS = {
    0: CLASS_NAMES.index("person"),       # YOLO: person
    1: CLASS_NAMES.index("car"),          # YOLO: car
    2: CLASS_NAMES.index("falldown"),          # YOLO: bus
    3: CLASS_NAMES.index("bus"),        # YOLO: truck
    4: CLASS_NAMES.index("truck"),      # YOLO: bicycle
    5: CLASS_NAMES.index("bicycle"),   # YOLO: motorbike
    6: CLASS_NAMES.index("motorcycle"),   # YOLO: cat
    7: CLASS_NAMES.index("boar"),   # YOLO: cat
    8: CLASS_NAMES.index("tractor"),   # YOLO: cat
    9: CLASS_NAMES.index("scooter"),   # YOLO: cat  
    10: CLASS_NAMES.index("tiller"),   # YOLO: cat
    11: CLASS_NAMES.index("cat"),   # YOLO: cat
    12: CLASS_NAMES.index("dog"),   # YOLO: dog
}


def get_class_color(class_id: int) -> str:
    """class_id에 대응하는 색상 반환"""
    if 0 <= class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]
    return "red"


# ==========================
# 메인 라벨링 툴 클래스
# ==========================

class YoloLabelTool:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLO Bounding Box Label Tool (Tkinter)")

        # 이미지 / 라벨 관련 상태
        self.image_dir = None
        self.label_dir = None
        self.image_paths = []
        self.cur_index = 0

        self.cur_image = None
        self.tk_image = None
        self.orig_w = 0
        self.orig_h = 0
        self.scale = 1.0

        # 각 원소: {"class_id", "x1","y1","x2","y2","rect_id"}
        self.boxes = []

        # 마우스 드래그 상태
        self.start_x = None
        self.start_y = None
        self.temp_rect_id = None

        # 현재 선택된 박스 인덱스
        self.selected_box_index = None

        # ★ YOLO 모델: 프로그램 실행 시 바로 로드
        try:
            self.yolo_model = YOLO(YOLO_WEIGHTS_PATH)
        except Exception as e:
            messagebox.showerror("YOLO Error", f"YOLO 모델 로드 실패:\n{e}")
            self.yolo_model = None

        # 현재 선택 클래스 (UI)
        self.selected_class = tk.StringVar()
        self.selected_class.set(CLASS_NAMES[0])
        # 클래스 변경 콜백 (선택 박스에 반영)
        self.selected_class.trace_add("write", self.on_class_changed)

        # UI 구성
        self._build_ui()

        # 이미지 폴더 선택
        self._select_image_folder()

        if not self.image_paths:
            messagebox.showerror("Error", "이미지 폴더에 유효한 이미지가 없습니다.")
            self.master.destroy()
            return

        self.load_image(self.cur_index)

    # ----------------------
    # UI 구성
    # ----------------------
    def _build_ui(self):
        top_frame = tk.Frame(self.master)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.info_label = tk.Label(top_frame, text="이미지 정보", anchor="w")
        self.info_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.canvas = tk.Canvas(self.master, cursor="tcross", bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        right_frame = tk.Frame(self.master)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(right_frame, text="Class:").pack(pady=(10, 0))
        class_menu = tk.OptionMenu(right_frame, self.selected_class, *CLASS_NAMES)
        class_menu.pack(pady=5)

        btn_prev = tk.Button(right_frame, text="<< Prev (A/←)", command=self.prev_image)
        btn_prev.pack(fill=tk.X, padx=5, pady=5)

        btn_next = tk.Button(right_frame, text="Next (D/→) >>", command=self.next_image)
        btn_next.pack(fill=tk.X, padx=5, pady=5)

        btn_undo = tk.Button(right_frame, text="Undo Last Box (U)", command=self.undo_last_box)
        btn_undo.pack(fill=tk.X, padx=5, pady=5)

        btn_del = tk.Button(right_frame, text="Delete Selected (Del)", command=self.delete_selected_box)
        btn_del.pack(fill=tk.X, padx=5, pady=5)

        btn_clear = tk.Button(right_frame, text="Clear All Boxes", command=self.clear_boxes)
        btn_clear.pack(fill=tk.X, padx=5, pady=5)

        btn_save = tk.Button(right_frame, text="Save Labels (S)", command=self.save_labels)
        btn_save.pack(fill=tk.X, padx=5, pady=5)

        # ★ YOLO 자동 라벨링 버튼
        btn_yolo = tk.Button(right_frame, text="YOLO Auto Label", command=self.run_yolo_autolabel)
        btn_yolo.pack(fill=tk.X, padx=5, pady=5)

        # 키보드 단축키
        self.master.bind("<Left>", lambda e: self.prev_image())
        self.master.bind("<Right>", lambda e: self.next_image())
        self.master.bind("a", lambda e: self.prev_image())
        self.master.bind("d", lambda e: self.next_image())
        self.master.bind("u", lambda e: self.undo_last_box())
        self.master.bind("s", lambda e: self.save_labels())
        self.master.bind("<Delete>", lambda e: self.delete_selected_box())
        # YOLO 단축키 (원하면 사용)
        self.master.bind("y", lambda e: self.run_yolo_autolabel())

    # ----------------------
    # 폴더 선택
    # ----------------------
    def _select_image_folder(self):
        img_dir = filedialog.askdirectory(title="이미지 폴더 선택")
        if not img_dir:
            return

        self.image_dir = img_dir
        parent = os.path.dirname(self.image_dir)
        self.label_dir = os.path.join(parent, "labels_with_name")
        os.makedirs(self.label_dir, exist_ok=True)

        patterns = ("*.jpg", "*.jpeg", "*.png")
        paths = []
        for p in patterns:
            paths.extend(glob.glob(os.path.join(self.image_dir, p)))
        self.image_paths = sorted(paths)

    # ----------------------
    # 이미지 로드
    # ----------------------
    def load_image(self, index):
        if index < 0 or index >= len(self.image_paths):
            return

        img_path = self.image_paths[index]
        self.cur_index = index

        img = Image.open(img_path).convert("RGB")
        self.cur_image = img
        self.orig_w, self.orig_h = img.size

        scale_w = MAX_CANVAS_WIDTH / self.orig_w
        scale_h = MAX_CANVAS_HEIGHT / self.orig_h
        self.scale = min(scale_w, scale_h, 1.0)

        disp_w = int(self.orig_w * self.scale)
        disp_h = int(self.orig_h * self.scale)

        disp_img = img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(disp_img)

        self.canvas.delete("all")
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        base_name = os.path.basename(img_path)
        info_text = f"[{self.cur_index + 1}/{len(self.image_paths)}] {base_name}  (orig: {self.orig_w}x{self.orig_h}, scale: {self.scale:.3f})"
        self.info_label.config(text=info_text)

        self.boxes = []
        self.selected_box_index = None
        self.load_labels()

    # ----------------------
    # 이미지 전환
    # ----------------------
    def prev_image(self):
        self.save_labels()
        new_index = self.cur_index - 1
        if new_index < 0:
            new_index = 0
        self.load_image(new_index)

    def next_image(self):
        self.save_labels()
        new_index = self.cur_index + 1
        if new_index >= len(self.image_paths):
            new_index = len(self.image_paths) - 1
        self.load_image(new_index)

    # ----------------------
    # 선택 박스 관리
    # ----------------------
    def set_selected_box(self, idx):
        # 이전 선택 해제
        if self.selected_box_index is not None:
            if 0 <= self.selected_box_index < len(self.boxes):
                old_box = self.boxes[self.selected_box_index]
                old_rect = old_box["rect_id"]
                color = get_class_color(old_box["class_id"])
                self.canvas.itemconfig(old_rect, outline=color, width=2)

        self.selected_box_index = idx

        if idx is not None and 0 <= idx < len(self.boxes):
            box = self.boxes[idx]
            rect_id = box["rect_id"]
            color = get_class_color(box["class_id"])
            # 선택된 박스: 테두리 두께만 증가
            self.canvas.itemconfig(rect_id, outline=color, width=3)
            # 선택된 박스의 클래스가 드롭다운에도 반영되도록
            self.selected_class.set(CLASS_NAMES[box["class_id"]])

    def select_box_at(self, x, y):
        # 뒤에서부터 탐색 (위에 그려진 박스 우선)
        for i in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[i]
            x1_disp = b["x1"] * self.scale
            y1_disp = b["y1"] * self.scale
            x2_disp = b["x2"] * self.scale
            y2_disp = b["y2"] * self.scale

            if x1_disp <= x <= x2_disp and y1_disp <= y <= y2_disp:
                self.set_selected_box(i)
                return True

        self.set_selected_box(None)
        return False

    # ----------------------
    # 클래스 드롭다운 변경 시 콜백
    # ----------------------
    def on_class_changed(self, *args):
        """드롭다운이 바뀌면 선택된 박스의 class_id 및 색상 갱신"""
        if self.selected_box_index is None:
            return
        if not (0 <= self.selected_box_index < len(self.boxes)):
            self.set_selected_box(None)
            return

        class_name = self.selected_class.get()
        if class_name not in CLASS_NAMES:
            return
        new_class_id = CLASS_NAMES.index(class_name)

        box = self.boxes[self.selected_box_index]
        box["class_id"] = new_class_id

        rect_id = box["rect_id"]
        color = get_class_color(new_class_id)
        # 선택 상태 유지: width=3
        self.canvas.itemconfig(rect_id, outline=color, width=3)

    # ----------------------
    # 마우스 이벤트
    # ----------------------
    def on_mouse_down(self, event):
        # 먼저 박스 클릭인지 검사
        if self.select_box_at(event.x, event.y):
            # 박스 선택이면 새 드래그 시작 안 함
            self.start_x = None
            self.start_y = None
            if self.temp_rect_id is not None:
                self.canvas.delete(self.temp_rect_id)
                self.temp_rect_id = None
            return

        # 빈 영역 클릭 → 선택 해제 후 새 박스 시작
        self.set_selected_box(None)

        self.start_x = event.x
        self.start_y = event.y
        self.temp_rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=2,
        )

    def on_mouse_drag(self, event):
        if self.temp_rect_id is not None and self.start_x is not None and self.start_y is not None:
            self.canvas.coords(self.temp_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.temp_rect_id is None or self.start_x is None or self.start_y is None:
            return

        end_x = event.x
        end_y = event.y

        if abs(end_x - self.start_x) < 3 or abs(end_y - self.start_y) < 3:
            self.canvas.delete(self.temp_rect_id)
            self.temp_rect_id = None
            self.start_x = None
            self.start_y = None
            return

        x1_disp = min(self.start_x, end_x)
        y1_disp = min(self.start_y, end_y)
        x2_disp = max(self.start_x, end_x)
        y2_disp = max(self.start_y, end_y)

        x1_orig = x1_disp / self.scale
        y1_orig = y1_disp / self.scale
        x2_orig = x2_disp / self.scale
        y2_orig = y2_disp / self.scale

        class_name = self.selected_class.get()
        class_id = CLASS_NAMES.index(class_name)

        # 클래스 색상 적용
        color = get_class_color(class_id)
        self.canvas.itemconfig(self.temp_rect_id, outline=color, width=2)

        box_info = {
            "class_id": class_id,
            "x1": x1_orig,
            "y1": y1_orig,
            "x2": x2_orig,
            "y2": y2_orig,
            "rect_id": self.temp_rect_id,
        }
        self.boxes.append(box_info)

        # 방금 만든 박스를 선택 상태로
        self.set_selected_box(len(self.boxes) - 1)

        self.temp_rect_id = None
        self.start_x = None
        self.start_y = None

    # ----------------------
    # 박스 조작
    # ----------------------
    def undo_last_box(self):
        if not self.boxes:
            return
        last_box = self.boxes.pop()
        self.canvas.delete(last_box["rect_id"])
        self.set_selected_box(None)

    def delete_selected_box(self):
        if self.selected_box_index is None:
            return
        idx = self.selected_box_index
        if not (0 <= idx < len(self.boxes)):
            self.set_selected_box(None)
            return

        box = self.boxes.pop(idx)
        self.canvas.delete(box["rect_id"])
        self.set_selected_box(None)

    def clear_boxes(self):
        for b in self.boxes:
            self.canvas.delete(b["rect_id"])
        self.boxes = []
        self.set_selected_box(None)

    # ----------------------
    # YOLO 자동 라벨링
    # ----------------------
    def run_yolo_autolabel(self):
        """현재 이미지에 YOLO 실행해서 박스 자동 추가"""
        if self.cur_image is None:
            return

        if self.yolo_model is None:
            messagebox.showerror("YOLO Error", "YOLO 모델이 로드되지 않았습니다.")
            return

        img_path = self.image_paths[self.cur_index]

        try:
            results = self.yolo_model.predict(img_path, verbose=False)[0]
        except Exception as e:
            messagebox.showerror("YOLO Error", f"YOLO 추론 실패:\n{e}")
            return

        # 기존 박스 제거 후 YOLO 결과로 채움
        self.clear_boxes()

        if results.boxes is None or len(results.boxes) == 0:
            messagebox.showinfo("YOLO", "탐지된 객체가 없습니다.")
            return

        boxes_xyxy = results.boxes.xyxy
        classes = results.boxes.cls

        for box, cls_tensor in zip(boxes_xyxy, classes):
            cls_id_yolo = int(cls_tensor.item())
            if cls_id_yolo not in YOLO_TO_OUR_CLASS:
                continue  # 매핑 안 된 클래스는 스킵

            our_class_id = YOLO_TO_OUR_CLASS[cls_id_yolo]

            x1, y1, x2, y2 = box.tolist()  # 원본 이미지 좌표

            # 디스플레이 좌표로 변환
            x1_disp = x1 * self.scale
            y1_disp = y1 * self.scale
            x2_disp = x2 * self.scale
            y2_disp = y2 * self.scale

            color = get_class_color(our_class_id)
            rect_id = self.canvas.create_rectangle(
                x1_disp, y1_disp, x2_disp, y2_disp,
                outline=color,
                width=2,
            )

            box_info = {
                "class_id": our_class_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "rect_id": rect_id,
            }
            self.boxes.append(box_info)

        self.set_selected_box(None)

    # ----------------------
    # 라벨 저장/로드 (YOLO 포맷)
    # ----------------------
    def _get_label_path(self):
        img_path = self.image_paths[self.cur_index]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + ".txt")
        return label_path

    def save_labels(self):
        if self.cur_image is None:
            return

        label_path = self._get_label_path()
        lines = []
        for b in self.boxes:
            class_id = b["class_id"]
            x1 = b["x1"]
            y1 = b["y1"]
            x2 = b["x2"]
            y2 = b["y2"]

            bw = x2 - x1
            bh = y2 - y1
            cx = x1 + bw / 2
            cy = y1 + bh / 2

            cx_n = cx / self.orig_w
            cy_n = cy / self.orig_h
            bw_n = bw / self.orig_w
            bh_n = bh / self.orig_h

            class_name = CLASS_NAMES[class_id]
            line = f"{class_name} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}"
            lines.append(line)

        if lines:
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        else:
            if os.path.exists(label_path):
                os.remove(label_path)

    def load_labels(self):
        label_path = self._get_label_path()
        if not os.path.exists(label_path):
            return

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
        except Exception as e:
            print("라벨 파일 로드 실패:", label_path, e)
            return

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            class_name = parts[0]
            if class_name not in CLASS_NAMES:
                print(f"알 수 없는 클래스명: {class_name} (line: {line})")
                continue

            class_id = CLASS_NAMES.index(class_name)

            cx_n = float(parts[1])
            cy_n = float(parts[2])
            bw_n = float(parts[3])
            bh_n = float(parts[4])

            cx = cx_n * self.orig_w
            cy = cy_n * self.orig_h
            bw = bw_n * self.orig_w
            bh = bh_n * self.orig_h

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            x1_disp = x1 * self.scale
            y1_disp = y1 * self.scale
            x2_disp = x2 * self.scale
            y2_disp = y2 * self.scale

            color = get_class_color(class_id)
            rect_id = self.canvas.create_rectangle(
                x1_disp,
                y1_disp,
                x2_disp,
                y2_disp,
                outline=color,
                width=2,
            )

            box_info = {
                "class_id": class_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "rect_id": rect_id,
            }
            self.boxes.append(box_info)

        self.set_selected_box(None)


# ==========================
# main
# ==========================

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloLabelTool(root)
    root.mainloop()
