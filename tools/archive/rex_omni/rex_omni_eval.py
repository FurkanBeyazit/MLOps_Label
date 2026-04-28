"""
Rex-Omni tüm dataset F1 değerlendirmesi — Florence karşılaştırmalı.
"""
import cv2, numpy as np, time
from pathlib import Path
from PIL import Image
from rex_omni import RexOmniWrapper, RexOmniVisualize

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
GT_DIR    = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\labels_with_name")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\tools\rex_omni_eval_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_EVAL = 0.35

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

def load_gt(label_path):
    if not label_path.exists(): return []
    out = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try: out.append((parts[0], [float(x) for x in parts[1:5]]))
            except ValueError: pass
    return out

def eval_boxes(pred_boxes, gt_boxes):
    matched = set()
    tp = 0
    for pb in pred_boxes:
        best_iou, best_i = 0, -1
        for i, gb in enumerate(gt_boxes):
            if i in matched: continue
            s = iou_xyxy(pb, gb)
            if s > best_iou: best_iou, best_i = s, i
        if best_iou >= IOU_EVAL:
            matched.add(best_i); tp += 1
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn

def preds_to_boxes(preds):
    """extracted_predictions dict → flat xyxy box listesi (fallen person only)"""
    boxes = []
    for category, annotations in preds.items():
        if "fallen" in category.lower():
            for ann in annotations:
                if ann["type"] == "box":
                    boxes.append(ann["coords"])
    return boxes

# ── Model ─────────────────────────────────────────────────────────────────────
print("Rex-Omni yükleniyor...")
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers",
    attn_implementation="sdpa",
    device_map="cuda:0",
)
print("Hazır.\n")

# ── Tüm dataset ───────────────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
print(f"{len(image_files)} görsel işlenecek.\n")

total_tp = total_fp = total_fn = 0
per_image_log = []

for i, img_path in enumerate(image_files, 1):
    img_bgr = read_image(img_path)
    if img_bgr is None: continue
    img_h, img_w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    t0 = time.time()
    results = rex.inference(
        images=img_pil,
        task="detection",
        categories=["fallen person"]
    )
    elapsed = time.time() - t0

    preds      = results[0]["extracted_predictions"]
    pred_boxes = preds_to_boxes(preds)

    gt_labels  = load_gt(GT_DIR / (img_path.stem + ".txt"))
    gt_boxes   = [norm_to_xyxy(b, img_w, img_h) for c, b in gt_labels if c == "falldown"]

    tp, fp, fn = eval_boxes(pred_boxes, gt_boxes)
    total_tp += tp; total_fp += fp; total_fn += fn

    # Annotate ve kaydet
    canvas = img_bgr.copy()
    for j, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 80, 255), 2)
        cv2.putText(canvas, f"rex_{j+1}", (x1, max(y1-5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 1)

    suffix = img_path.suffix if img_path.suffix else ".jpg"
    _, enc = cv2.imencode(suffix, canvas)
    (OUT_DIR / img_path.name).write_bytes(enc.tobytes())

    marker = " ★" if (gt_boxes or pred_boxes) else ""
    print(f"[{i:3d}/{len(image_files)}] {elapsed:5.1f}s  pred={len(pred_boxes)} gt={len(gt_boxes)} tp={tp} fp={fp} fn={fn}{marker}  {img_path.name[:40]}")
    per_image_log.append({"file": img_path.name, "pred": len(pred_boxes), "gt": len(gt_boxes), "tp": tp, "fp": fp, "fn": fn, "elapsed": round(elapsed, 2)})

# ── F1 özeti ──────────────────────────────────────────────────────────────────
import json
p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2*p*r / (p+r) if (p+r) > 0 else 0

summary = {
    "model": "Rex-Omni only",
    "TP": total_tp, "FP": total_fp, "FN": total_fn,
    "P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
    "per_image": per_image_log,
}
results_path = OUT_DIR / "results.json"
results_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"\n{'='*50}")
print(f"REX-OMNI — falldown F1 sonucu")
print(f"{'='*50}")
print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
print(f"  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
print(f"\n  Florence karşılaştırma (pipeline D):")
print(f"  TP=27  FP=5  FN=12  P=0.844  R=0.692  F1=0.761")
print(f"\nBitti → {OUT_DIR}")
print(f"Sonuçlar kaydedildi → {results_path}")
