"""
4 Konfigürasyon Karşılaştırması
================================
Aynı falldown (VLM cache + Florence labels) üzerine 4 farklı YOLO stratejisini karşılaştırır:

  1. Tek model: yolo11x (tüm YOLO class'ları)
  2. Tek model: best_04_16 (tüm YOLO class'ları)
  3. Tek model: best_sgd_04_16 (tüm YOLO class'ları)
  4. CLASS_MODEL_MAP: her class kendi benchmark lideri

Falldown box'ları her konfig için aynı → pipeline_out/labels/*.txt'den alınır
(bu dosyalar son label_pipeline.py çalışmasının çıktısıdır).

Kullanım:
  python benchmark_configs.py
"""
import cv2, numpy as np, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
GT_DIR     = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\labels_with_name")
OUT_DIR    = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out")
FD_LABELS  = OUT_DIR / "labels"   # falldown bbox'ları buradan alınır

_M = Path(r"C:\Users\admin\fur\MLOps_Label\models")

YOLO_CLASSES  = {"person", "car", "bus", "truck", "bicycle", "motorcycle", "dog", "cat"}
CUSTOM_CLASSES = {"boar", "tractor", "tiller", "scooter"}
ALL_CLASSES    = YOLO_CLASSES | CUSTOM_CLASSES

CONF         = 0.25
IOU_MERGE    = 0.60   # falldown ile overlap → model box atılır
IOU_EVAL     = 0.35
YOLO_BATCH   = 8

# Konfigürasyon tanımları
CONFIGS = [
    {
        "name": "1) Tek: yolo11x",
        "map": {cls: _M / "coco_yolo/yolo11x.pt" for cls in ALL_CLASSES},
    },{
        "name": "aaa Ee9",
        "map": {cls: _M / "coco_yolo/yolov9e.pt" for cls in ALL_CLASSES},
    },
    {
        "name": "2) Tek: best_04_15",
        "map": {cls: _M / "falldown/best_04_15.pt" for cls in ALL_CLASSES},
    },
    {
        "name": "3) Tek: best_04_16",
        "map": {cls: _M / "falldown/best_04_16.pt" for cls in ALL_CLASSES},
    },
    {
        "name": "4) Tek: best_sgd_04_16",
        "map": {cls: _M / "falldown/best_sgd_04_16.pt" for cls in ALL_CLASSES},
    },
]

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

def xyxy_to_norm(box, w, h):
    x1, y1, x2, y2 = box
    return [((x1+x2)/2)/w, ((y1+y2)/2)/h, (x2-x1)/w, (y2-y1)/h]

def load_labels(path):
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try: out.append((parts[0], [float(x) for x in parts[1:5]]))
            except ValueError: pass
    return out

def _parse_batch(results, model, paths):
    out = {}
    for result, img_path in zip(results, paths):
        boxes = []
        if len(result.boxes):
            for box, cls, c in zip(
                result.boxes.xyxy.cpu().numpy().tolist(),
                result.boxes.cls.cpu().numpy().tolist(),
                result.boxes.conf.cpu().numpy().tolist()
            ):
                boxes.append({"class": model.names[int(cls)], "box": box, "conf": float(c)})
        out[img_path] = boxes
    return out

def run_config(cfg, image_files):
    class_map = cfg["map"]
    t0 = time.time()

    # Unique modelleri yükle
    unique_models = {}
    for cls, mpath in class_map.items():
        key = str(mpath)
        if key not in unique_models:
            # Var olan .pt kontrolü
            if not mpath.exists():
                print(f"    [!] Model bulunamadı, atlandı: {mpath}")
                continue
            unique_models[key] = YOLO(str(mpath))

    valid_paths = [p for p in image_files if read_image(p) is not None]

    # Batch inference — her model
    raw_by_model = {}
    for model_key, model in unique_models.items():
        raw_by_model[model_key] = {}
        for start in range(0, len(valid_paths), YOLO_BATCH):
            batch_paths = valid_paths[start:start + YOLO_BATCH]
            batch_imgs  = [read_image(p) for p in batch_paths]
            results     = model(batch_imgs, conf=CONF, verbose=False)
            raw_by_model[model_key].update(_parse_batch(results, model, batch_paths))

    # Class bazlı birleştir
    model_boxes = {}
    for img_path in valid_paths:
        boxes = []
        for cls, mpath in class_map.items():
            key = str(mpath)
            if key not in raw_by_model:
                continue
            for box in raw_by_model[key].get(img_path, []):
                if box["class"] == cls:
                    boxes.append(box)
        model_boxes[img_path] = boxes

    # GT karşılaştırma
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for img_path in image_files:
        img_bgr = read_image(img_path)
        if img_bgr is None: continue
        img_h, img_w = img_bgr.shape[:2]

        # Falldown: pipeline'ın kaydettiği son çıktıdan al
        fd_labels = load_labels(FD_LABELS / (img_path.stem + ".txt"))
        fd_boxes  = [norm_to_xyxy(b, img_w, img_h) for c, b in fd_labels if c == "falldown"]

        # Falldown ile çakışan model box'larını kaldır
        m_boxes = model_boxes.get(img_path, [])
        kept    = [mb for mb in m_boxes
                   if not any(iou_xyxy(mb["box"], fb) >= IOU_MERGE for fb in fd_boxes)]

        # Pred listesi
        pred_all  = [("falldown", xyxy_to_norm(fb, img_w, img_h)) for fb in fd_boxes]
        pred_all += [(mb["class"], xyxy_to_norm(mb["box"], img_w, img_h)) for mb in kept]

        # GT
        gt_all = load_labels(GT_DIR / (img_path.stem + ".txt"))

        all_classes = set(c for c, _ in gt_all) | set(c for c, _ in pred_all)
        for cls in all_classes:
            gt_boxes   = [norm_to_xyxy(b, img_w, img_h) for c, b in gt_all   if c == cls]
            pred_boxes = [norm_to_xyxy(b, img_w, img_h) for c, b in pred_all if c == cls]

            matched_gt = set()
            tp = 0
            for pb in pred_boxes:
                best_iou, best_gi = 0, -1
                for gi, gb in enumerate(gt_boxes):
                    if gi in matched_gt: continue
                    s = iou_xyxy(pb, gb)
                    if s > best_iou: best_iou, best_gi = s, gi
                if best_iou >= IOU_EVAL:
                    matched_gt.add(best_gi); tp += 1

            class_stats[cls]["tp"] += tp
            class_stats[cls]["fp"] += len(pred_boxes) - tp
            class_stats[cls]["fn"] += len(gt_boxes) - len(matched_gt)

    # Modelleri temizle
    for model in unique_models.values():
        del model

    elapsed = time.time() - t0

    total_tp = sum(s["tp"] for s in class_stats.values())
    total_fp = sum(s["fp"] for s in class_stats.values())
    total_fn = sum(s["fn"] for s in class_stats.values())
    p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2*p*r / (p+r) if (p+r) > 0 else 0

    fd = class_stats.get("falldown", {"tp":0,"fp":0,"fn":0})
    fdp  = fd["tp"]/(fd["tp"]+fd["fp"]) if (fd["tp"]+fd["fp"])>0 else 0
    fdr  = fd["tp"]/(fd["tp"]+fd["fn"]) if (fd["tp"]+fd["fn"])>0 else 0
    fdf1 = 2*fdp*fdr/(fdp+fdr) if (fdp+fdr)>0 else 0

    return {
        "class_stats": dict(class_stats),
        "total": {"tp": total_tp, "fp": total_fp, "fn": total_fn,
                  "p": p, "r": r, "f1": f1},
        "falldown": {"p": fdp, "r": fdr, "f1": fdf1,
                     "tp": fd["tp"], "fp": fd["fp"], "fn": fd["fn"]},
        "elapsed_s": elapsed,
    }

# ── Ana çalışma ───────────────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = OUT_DIR / f"benchmark_configs_{ts}.txt"
LOG = open(log_path, "w", encoding="utf-8", buffering=1)

def log(line=""):
    print(line)
    LOG.write(line + "\n")

log("=" * 80)
log(f"4 KONFİGÜRASYON KARŞILAŞTIRMASI  —  {ts}")
log(f"Falldown labels: {FD_LABELS}")
log(f"IOU_EVAL={IOU_EVAL}  CONF={CONF}  Görüntü={len(image_files)}")
log("=" * 80)

all_results = {}
all_classes_seen = set()

for cfg in CONFIGS:
    log(f"\n{'─'*80}")
    log(f"Çalışıyor: {cfg['name']}")
    log(f"{'─'*80}")
    result = run_config(cfg, image_files)
    all_results[cfg["name"]] = result
    all_classes_seen |= set(result["class_stats"].keys())

    # Per-class tablo
    classes = sorted(result["class_stats"].keys())
    log(f"\n  {'Sınıf':<14} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
    log(f"  {'-'*56}")
    for cls in classes:
        s  = result["class_stats"][cls]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        log(f"  {cls:<14} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
    log(f"  {'-'*56}")
    t = result["total"]
    log(f"  {'TOPLAM':<14} {t['tp']:>5} {t['fp']:>5} {t['fn']:>5} {t['p']:>7.3f} {t['r']:>7.3f} {t['f1']:>7.3f}")
    fd = result["falldown"]
    log(f"  {'falldown*':<14} {fd['tp']:>5} {fd['fp']:>5} {fd['fn']:>5} {fd['p']:>7.3f} {fd['r']:>7.3f} {fd['f1']:>7.3f}  (* aynı FD bbox)")
    log(f"  Süre: {result['elapsed_s']:.0f}s")

# ── Özet karşılaştırma tablosu ────────────────────────────────────────────────
log(f"\n\n{'='*80}")
log("ÖZET: Per-class F1 karşılaştırması")
log(f"{'='*80}")

cfg_names = [cfg["name"] for cfg in CONFIGS]
col = 11

# Header
header = f"  {'Sınıf':<14}" + "".join(f"  {n[:col]:>{col}}" for n in cfg_names)
log(header)
log("  " + "-" * (14 + (col+2)*len(cfg_names)))

for cls in sorted(all_classes_seen):
    f1s = []
    for cname in cfg_names:
        s  = all_results[cname]["class_stats"].get(cls, {"tp":0,"fp":0,"fn":0})
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p  = tp/(tp+fp) if (tp+fp)>0 else 0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        f1s.append(f1)
    best_idx = f1s.index(max(f1s))
    cells = []
    for i, f1 in enumerate(f1s):
        tag = f"**{f1:.3f}**" if i == best_idx else f"  {f1:.3f}  "
        cells.append(f"{tag:>{col}}")
    log(f"  {cls:<14}" + "".join(f"  {c}" for c in cells))

log("  " + "-" * (14 + (col+2)*len(cfg_names)))

# Toplam satırı
log(f"\n  {'Genel F1':<14}" + "".join(
    f"  {all_results[n]['total']['f1']:>{col}.3f}" for n in cfg_names
))
log(f"  {'Genel P':<14}" + "".join(
    f"  {all_results[n]['total']['p']:>{col}.3f}" for n in cfg_names
))
log(f"  {'Genel R':<14}" + "".join(
    f"  {all_results[n]['total']['r']:>{col}.3f}" for n in cfg_names
))
log(f"  {'Falldown F1':<14}" + "".join(
    f"  {all_results[n]['falldown']['f1']:>{col}.3f}" for n in cfg_names
))

log(f"\nLog → {log_path}")
LOG.close()
print(f"\nTamamlandı.")
