"""
FP/FN görsel inceleme aracı.
Her class için ayrı klasör, içinde annotated fotoğraflar.

pipeline_out/fp_fn_check/
  car/
    FP_E02_001.jpg   ← pipeline buldu ama GT'de yok (kırmızı)
    FN_공원0002.jpg   ← GT'de var ama pipeline kaçırdı (turuncu)
  falldown/
    ...

Kutu renkleri:
  Yeşil  = TP (doğru tespit)
  Kırmızı = FP (yanlış tespit)
  Turuncu = FN (kaçırılan)
"""
import cv2, numpy as np
from pathlib import Path
from collections import defaultdict

IMAGE_DIR = Path(r"C:\Users\admin\fur\0424(쓰러짐)")
GT_DIR    = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out\20260429_153850_쓰러짐_107\labels_with_name")
PRED_DIR  = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out\20260430_151636\labels")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out\20260430_151636\fp_fn_check")

IOU_EVAL  = 0.35
SAVE_TP   = True   # TP olan fotoları da kaydet (tp/ klasörüne)

# İncelemek istediğin class'lar — hepsini görmek için None bırak
TARGET_CLASSES = None  # veya {"car", "falldown", "cat", ...}

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_image(p):
    arr = np.fromfile(str(p), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter   = max(0, x2-x1) * max(0, y2-y1)
    union   = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def norm_to_xyxy(box, w, h):
    cx, cy, bw, bh = box
    return [int((cx-bw/2)*w), int((cy-bh/2)*h),
            int((cx+bw/2)*w), int((cy+bh/2)*h)]

def load_labels(path):
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try: out.append((parts[0], [float(x) for x in parts[1:5]]))
            except ValueError: pass
    return out

def draw_box(img, box, color, label, thickness=2):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    fs = 0.5
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0]
    cv2.rectangle(img, (x1, max(y1-th-6, 0)), (x1+tw+4, max(y1, th+6)), color, -1)
    cv2.putText(img, label, (x1+2, max(y1-3, th+3)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1)

def draw_dashed_box(img, box, color=(255, 255, 255), dash=10, thickness=1):
    x1, y1, x2, y2 = box
    for seg, (p1, p2) in enumerate([((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
                                     ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]):
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        length = int((dx**2 + dy**2) ** 0.5) or 1
        for i in range(0, length, dash * 2):
            t0, t1 = i / length, min(i + dash, length) / length
            s = (int(p1[0] + dx*t0), int(p1[1] + dy*t0))
            e = (int(p1[0] + dx*t1), int(p1[1] + dy*t1))
            cv2.line(img, s, e, color, thickness)

def save_image(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    _, enc = cv2.imencode(path.suffix or ".jpg", img)
    path.write_bytes(enc.tobytes())

# ── Ana işlem ─────────────────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))

# class → {fp: [img], fn: [img]} sayaçları
summary = defaultdict(lambda: {"fp": 0, "fn": 0, "tp": 0})

for img_path in image_files:
    img_bgr = read_image(img_path)
    if img_bgr is None: continue
    h, w = img_bgr.shape[:2]

    gt_all   = load_labels(GT_DIR  / (img_path.stem + ".txt"))
    pred_all = load_labels(PRED_DIR / (img_path.stem + ".txt"))

    all_classes = set(c for c, _ in gt_all) | set(c for c, _ in pred_all)
    if TARGET_CLASSES:
        all_classes &= TARGET_CLASSES

    for cls in all_classes:
        gt_boxes   = [norm_to_xyxy(b, w, h) for c, b in gt_all   if c == cls]
        pred_boxes = [norm_to_xyxy(b, w, h) for c, b in pred_all if c == cls]

        # Eşleştir
        matched_gt   = set()
        matched_pred = set()
        for pi, pb in enumerate(pred_boxes):
            best_iou, best_gi = 0, -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt: continue
                s = iou_xyxy(pb, gb)
                if s > best_iou: best_iou, best_gi = s, gi
            if best_iou >= IOU_EVAL:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

        fp_boxes = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
        fn_boxes = [gt_boxes[i]   for i in range(len(gt_boxes))   if i not in matched_gt]
        tp_boxes = [pred_boxes[i] for i in matched_pred]

        summary[cls]["tp"] += len(tp_boxes)
        summary[cls]["fp"] += len(fp_boxes)
        summary[cls]["fn"] += len(fn_boxes)

        if not fp_boxes and not fn_boxes and not (SAVE_TP and tp_boxes):
            continue

        canvas = img_bgr.copy()

        # Tüm GT kutularını (tüm class'lar) beyaz dashed arka plan olarak çiz
        for gt_cls, gt_box in gt_all:
            b = norm_to_xyxy(gt_box, w, h)
            draw_dashed_box(canvas, b, color=(255, 255, 255), dash=10, thickness=1)

        for b in tp_boxes:
            draw_box(canvas, b, (0, 200, 0), f"TP:{cls}")
        for b in fp_boxes:
            draw_box(canvas, b, (0, 0, 220), f"FP:{cls}", thickness=3)
        for b in fn_boxes:
            draw_box(canvas, b, (0, 140, 255), f"FN:{cls}", thickness=3)

        tag = []
        if fp_boxes: tag.append(f"FP{len(fp_boxes)}")
        if fn_boxes: tag.append(f"FN{len(fn_boxes)}")
        if not tag:  tag.append(f"TP{len(tp_boxes)}")
        fname = "_".join(tag) + "_" + img_path.name

        subfolder = "tp_only" if (not fp_boxes and not fn_boxes) else cls
        out_path = OUT_DIR / subfolder / cls / fname
        save_image(canvas, out_path)

# ── Özet ──────────────────────────────────────────────────────────────────────
print(f"\n{'Sınıf':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
print("-" * 60)
total_tp = total_fp = total_fn = 0
for cls in sorted(summary):
    s = summary[cls]
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    total_tp += tp; total_fp += fp; total_fn += fn
    p  = tp / (tp+fp) if (tp+fp) > 0 else 0
    r  = tp / (tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
    marker = " ★" if cls == "falldown" else ""
    print(f"{cls:<20} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}{marker}")

print("-" * 60)
p  = total_tp / (total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
r  = total_tp / (total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
print(f"{'TOPLAM':<20} {total_tp:>5} {total_fp:>5} {total_fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
print(f"\nKlasör: {OUT_DIR}")
