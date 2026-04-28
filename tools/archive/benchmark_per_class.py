"""
Her modeli YOLO_ALLOWED_CLASSES üzerinde bağımsız olarak karşılaştırır.
Hangi model hangi class'ta daha iyi? Per-class TP/FP/FN tablosu.

Test edilen modeller:
  - Tüm coco_yolo/*.pt
  - models/falldown/best_04_15.pt (aynı class'lara bakıyor mu diye)

Kullanım: python benchmark_per_class.py
"""
import cv2, json, time
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
GT_DIR    = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\labels_with_name")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out")
MODEL_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\models")

TARGET_CLASSES = {"person", "car", "bus", "truck", "bicycle", "motorcycle", "dog", "cat"}
CONF      = 0.25
IOU_EVAL  = 0.35
BATCH     = 8

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
    return [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]

def load_gt(path):
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try: out.append((parts[0], [float(x) for x in parts[1:5]]))
            except ValueError: pass
    return out

def run_benchmark(model, image_files):
    """Modeli tüm görüntülerde çalıştır, per-class TP/FP/FN döndür."""
    valid = [p for p in image_files if read_image(p) is not None]

    # Hangi class'lar bu modelde var?
    model_classes = set(model.names.values())
    testable = TARGET_CLASSES & model_classes
    skipped  = TARGET_CLASSES - model_classes
    if skipped:
        print(f"    [!] Bu modelde yok, atlanacak: {skipped}")

    # Batch inference
    pred_by_img = {}
    for start in range(0, len(valid), BATCH):
        batch = valid[start:start+BATCH]
        imgs  = [read_image(p) for p in batch]
        results = model(imgs, conf=CONF, verbose=False)
        for img_path, result in zip(batch, results):
            boxes = []
            if len(result.boxes):
                for box, cls, c in zip(
                    result.boxes.xyxy.cpu().numpy().tolist(),
                    result.boxes.cls.cpu().numpy().tolist(),
                    result.boxes.conf.cpu().numpy().tolist()
                ):
                    cls_name = model.names[int(cls)]
                    if cls_name in TARGET_CLASSES:
                        boxes.append({"class": cls_name, "box": box, "conf": float(c)})
            pred_by_img[img_path] = boxes

    # Per-class TP/FP/FN
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for img_path in valid:
        img_bgr  = read_image(img_path)
        h, w     = img_bgr.shape[:2]
        gt_all   = load_gt(GT_DIR / (img_path.stem + ".txt"))
        pred_all = pred_by_img.get(img_path, [])

        for cls in TARGET_CLASSES:
            gt_boxes   = [norm_to_xyxy(b, w, h) for c, b in gt_all   if c == cls]
            pred_boxes = [pb["box"]              for pb in pred_all    if pb["class"] == cls]

            matched = set()
            tp = 0
            for pb in pred_boxes:
                best_iou, best_gi = 0, -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched: continue
                    s = iou_xyxy(pb, gb)
                    if s > best_iou: best_iou, best_gi = s, gi
                if best_iou >= IOU_EVAL:
                    matched.add(best_gi); tp += 1

            stats[cls]["tp"] += tp
            stats[cls]["fp"] += len(pred_boxes) - tp
            stats[cls]["fn"] += len(gt_boxes) - len(matched)

    # Skipped class'ları FN olarak say
    for cls in skipped:
        gt_count = sum(
            sum(1 for c, _ in load_gt(GT_DIR / (p.stem + ".txt")) if c == cls)
            for p in valid
        )
        stats[cls]["fn"] = gt_count

    return dict(stats)

# ── Model listesi ─────────────────────────────────────────────────────────────
models_to_test = []
for p in sorted((MODEL_DIR / "coco_yolo").glob("*.pt")):
    models_to_test.append(("coco", p))
for p in sorted((MODEL_DIR / "falldown").glob("*.pt")):
    models_to_test.append(("own", p))

# ── Çalıştır ──────────────────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_DIR / f"benchmark_perclass_{ts}.txt"
LOG      = open(log_path, "w", encoding="utf-8", buffering=1)

def log(line=""):
    print(line)
    LOG.write(line + "\n")

log("=" * 90)
log(f"PER-CLASS MODEL BENCHMARK  —  {ts}")
log(f"Target classes : {sorted(TARGET_CLASSES)}")
log(f"IOU_EVAL={IOU_EVAL}  CONF={CONF}  Görüntü={len(image_files)}")
log("=" * 90)

all_stats = {}  # model_name → {class → {tp,fp,fn}}

classes_sorted = sorted(TARGET_CLASSES)

for source, model_path in models_to_test:
    log(f"\n{'─'*90}")
    log(f"Model: {model_path.name}  [{source}]")
    log(f"{'─'*90}")

    t0    = time.time()
    model = YOLO(str(model_path))
    stats = run_benchmark(model, image_files)
    elapsed = time.time() - t0
    del model

    all_stats[model_path.name] = stats

    # Per-class tablo
    log(f"\n  {'Sınıf':<14} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7} {'GT':>5}")
    log(f"  {'-'*62}")
    total_tp = total_fp = total_fn = 0
    for cls in classes_sorted:
        s  = stats.get(cls, {"tp":0,"fp":0,"fn":0})
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        total_tp += tp; total_fp += fp; total_fn += fn
        p  = tp/(tp+fp) if (tp+fp) > 0 else 0
        r  = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        gt = tp + fn
        log(f"  {cls:<14} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {gt:>5}")
    log(f"  {'-'*62}")
    p  = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0
    r  = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0
    f1 = 2*p*r/(p+r) if (p+r)>0 else 0
    log(f"  {'TOPLAM':<14} {total_tp:>5} {total_fp:>5} {total_fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
    log(f"  Süre: {elapsed:.0f}s")

# ── Karşılaştırma özeti: her class için en iyi model ─────────────────────────
log(f"\n\n{'='*90}")
log("CLASS BAZINDA EN İYİ MODEL (TP ve F1'e göre)")
log(f"{'='*90}")

# Header
model_names = [p.name for _, p in models_to_test]
short = {n: n.replace(".pt","")[:14] for n in model_names}
header = f"  {'Sınıf':<14} " + " ".join(f"{short[n]:>10}" for n in model_names) + "   EN İYİ"
log(header + "  (F1)")
log("  " + "-" * (len(header)+5))

for cls in classes_sorted:
    row_f1  = {}
    row_tp  = {}
    for mname in model_names:
        s  = all_stats[mname].get(cls, {"tp":0,"fp":0,"fn":0})
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        row_f1[mname]  = f1
        row_tp[mname]  = tp

    best_model = max(row_f1, key=row_f1.get)
    cells = " ".join(
        f"{'>>'+f'{row_f1[n]:.3f}':>10}" if n == best_model else f"{row_f1[n]:>10.3f}"
        for n in model_names
    )
    log(f"  {cls:<14} {cells}   {short[best_model]} ({row_f1[best_model]:.3f})")

log(f"\nLog → {log_path}")
LOG.close()

# JSON
json_out = OUT_DIR / f"benchmark_perclass_{ts}.json"
json_out.write_text(
    json.dumps({m: {c: s for c, s in v.items()} for m, v in all_stats.items()},
               ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"JSON → {json_out}")
