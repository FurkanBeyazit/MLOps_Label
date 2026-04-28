"""
Otonom Label Pipeline
=====================
Akış:
  1. VLM (Qwen3VL, tam foto) → AUTO-NO / AUTO-YES / REVIEW
  2. AUTO-YES → Florence → falldown bbox
  3. YOLO 11x + kendi model → tüm fotolar için background class'lar
  4. Merge: Florence falldown + model box'ları
  5. labels_with_name formatında kaydet
  6. GT ile karşılaştır → per-class F1 raporu

Hybrid sınıf stratejisi:
  - YOLO_ALLOWED_CLASSES → YOLO11x kazanır (person, car, bus, ...)
  - OWN_CUSTOM_CLASSES   → kendi model kazanır (boar, tractor, tiller, scooter)
  - falldown             → Florence+VLM kazanır
  - Geometrik overlap (IoU ≥ IOU_MODEL_MERGE): YOLO her zaman kazanır

Çıktı:
  pipeline_out/labels/          ← final txt (YOLO format, class isimli)
  pipeline_out/review/          ← REVIEW fotoları + json
  pipeline_out/pipeline_log.txt
"""

import torch, torch.nn.functional as F
import cv2, numpy as np, json, gc, time
from datetime import datetime
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from transformers import (
    AutoModelForCausalLM, AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR  = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
GT_DIR     = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\labels_with_name")
OUT_DIR    = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out")
RUN_TS     = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH   = OUT_DIR / f"pipeline_log_{RUN_TS}.txt"

# VLM sonuçlarını yeniden kullanmak için: mevcut bir vlm_buckets_*.json dosyasının
# yolunu buraya yaz. None ise VLM sıfırdan çalışır ve yeni dosya kaydeder.
VLM_CACHE  = OUT_DIR / "vlm_buckets_20260422_165231.json"  # D prompt: hayvan + karanlık cümlesi, thr=0.75

CONF            = 0.25
IOU_MODEL_MERGE = 0.50   # aynı nesneyi iki model de bulduysa yüksek conf kazanır

_M = Path(r"C:\Users\admin\fur\MLOps_Label\models")

COCO_MODEL    = _M / "coco_yolo/yolov9e.pt"       # person, car, bus, truck, ...
OWN_MODEL     = _M / "falldown/best_04_15.pt"      # boar, tractor, tiller, scooter, falldown

YOLO_CLASSES  = {"person", "car", "bus", "truck", "bicycle", "motorcycle", "dog", "cat"}
CUSTOM_CLASSES = {"boar", "tractor", "tiller", "scooter"}

VLM_THR   = 0.75   # P(Yes) veya P(No) eşiği — altı REVIEW
IOU_MERGE = 0.60   # Florence falldown ile model box overlap → model atılır
IOU_EVAL  = 0.35   # GT eşleşme eşiği (F1 hesabı) — falldown bbox'ı bazen eksik

FLORENCE_PROMPT = "a person lying on the ground"

# Odak: yatay/düşmüş = YES, dikey/yürüyen = NO
VLM_QUESTION = (
    "This is a CCTV security camera image. "
    "Has a person FALLEN DOWN or is LYING on the ground? "
    "Answer Yes ONLY if someone is clearly horizontal: lying flat, collapsed, or sprawled. "
    "Answer No if all people are upright — standing, walking, running "
    "even if they are on concrete, asphalt, or any outdoor surface. "
    "A dog, cat or other animal lying on the ground is NOT a fallen person. "
    "If the image is too dark or the shape is unclear, answer No. "
    "Ignore animals. "
    "Answer only Yes or No."
)



VLM_QUESTION_EXPLAIN = (
    "This is a CCTV security camera image. "
    "Has a person FALLEN DOWN or is LYING on the ground? "
    "Answer Yes ONLY if someone is clearly horizontal: lying flat, collapsed, or sprawled. "
    "Answer No if all people are upright — standing, walking, running "
    "Ignore animals. "
    "Answer Yes or No, then in one sentence describe the person's posture and location."
)

# ── Setup ─────────────────────────────────────────────────────────────────────
(OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "review").mkdir(parents=True, exist_ok=True)
LOG = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

def log(line, quiet=False):
    if not quiet: print(line)
    LOG.write(line + "\n")

VLM_MAX_PX = None  #int max = 640 (Qwen3VL için önerilen boyut), None ise orijinal boyutta çalışır

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_image(p):
    arr = np.fromfile(str(p), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def resize_for_vlm(img_bgr):
    if VLM_MAX_PX is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    scale = VLM_MAX_PX / min(h, w)
    if scale >= 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter   = max(0, x2-x1) * max(0, y2-y1)
    union   = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def xyxy_to_norm(box, w, h):
    x1, y1, x2, y2 = box
    return [((x1+x2)/2)/w, ((y1+y2)/2)/h, (x2-x1)/w, (y2-y1)/h]

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

def draw_box(canvas, box_xyxy, color, label):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, label, (x1, max(y1-5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def run_yolo(model, img_bgr, conf):
    result = model(img_bgr, conf=conf, verbose=False)[0]
    boxes  = []
    if len(result.boxes):
        for box, cls, c in zip(
            result.boxes.xyxy.cpu().numpy().tolist(),
            result.boxes.cls.cpu().numpy().tolist(),
            result.boxes.conf.cpu().numpy().tolist()
        ):
            boxes.append({"class": model.names[int(cls)], "box": box, "conf": float(c)})
    return boxes

image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))

# ══════════════════════════════════════════════════════════════════════════════
# AŞAMA 1 — VLM Tarama
# ══════════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("AŞAMA 1: VLM Tarama (Qwen3VL)")
log("=" * 60)

if VLM_CACHE and Path(VLM_CACHE).exists():
    log(f"VLM_CACHE yükleniyor: {VLM_CACHE}")
    raw = json.loads(Path(VLM_CACHE).read_text(encoding="utf-8"))
    raw_buckets    = raw.get("buckets", raw)
    buckets        = {k: [Path(p) for p in v] for k, v in raw_buckets.items()}
    review_explains = raw.get("review_explains", {})
    log(f"VLM sonuç (cache): YES={len(buckets['yes'])}  NO={len(buckets['no'])}  REVIEW={len(buckets['review'])}")
    log(f"  review_explains: {len(review_explains)} açıklama yüklendi\n")
else:
    log(f"{len(image_files)} resim taranacak.\n")

    vlm      = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
    )
    vlm_proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    vocab    = vlm_proc.tokenizer.get_vocab()
    yes_ids  = [vocab[t] for t in ("Yes", "yes", "YES") if t in vocab]
    no_ids   = [vocab[t] for t in ("No",  "no",  "NO")  if t in vocab]

    def _make_input(img_pil, question):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img_pil},
            {"type": "text",  "text":  question},
        ]}]
        inp = vlm_proc.apply_chat_template(msgs, tokenize=True,
              add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inp.pop("token_type_ids", None)
        return {k: v.to("cuda") for k, v in inp.items()}

    def vlm_logit(img_pil, question):
        inp = _make_input(img_pil, question)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=5,
                               output_scores=True, return_dict_in_generate=True,
                               do_sample=False)
        logits  = out.scores[0][0]
        y_logit = torch.stack([logits[i] for i in yes_ids]).max()
        n_logit = torch.stack([logits[i] for i in no_ids ]).max()
        probs   = F.softmax(torch.stack([y_logit, n_logit]), dim=0)
        return probs[0].item(), probs[1].item()

    def vlm_explain(img_pil):
        inp = _make_input(img_pil, VLM_QUESTION_EXPLAIN)
        with torch.no_grad():
            out = vlm.generate(**inp, max_new_tokens=60, do_sample=False)
        return vlm_proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def vlm_classify(img_pil):
        py, pn = vlm_logit(img_pil, VLM_QUESTION)
        if py >= VLM_THR: return "yes",    py, pn, ""
        if pn >= VLM_THR: return "no",     py, pn, ""
        return "review", py, pn, vlm_explain(img_pil)

    buckets = {"yes": [], "no": [], "review": []}
    review_explains = {}

    for i, img_path in enumerate(image_files, 1):
        img_bgr = read_image(img_path)
        if img_bgr is None: continue
        img_small = resize_for_vlm(img_bgr)
        img_pil   = Image.fromarray(cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        t0        = time.time()
        decision, py, pn, expl = vlm_classify(img_pil)
        elapsed = time.time() - t0
        buckets[decision].append(img_path)
        if decision == "review":
            review_explains[img_path.name] = {"p_yes": round(py,3), "p_no": round(pn,3), "explain": expl}
        icon = {"yes": "🔴", "no": "🟢", "review": "🟡"}[decision]
        log(f"[{i:3d}/{len(image_files)}] {icon} {decision:6s}  P(Y)={py:.3f} P(N)={pn:.3f}  {elapsed:.1f}s  {img_path.name}")
        if expl:
            log(f"           → {expl}")

    log(f"\nVLM sonuç: YES={len(buckets['yes'])}  NO={len(buckets['no'])}  REVIEW={len(buckets['review'])}\n")

    # Sonraki run'larda yeniden kullanmak için kaydet
    vlm_cache_path = OUT_DIR / f"vlm_buckets_{RUN_TS}.json"
    vlm_cache_path.write_text(
        json.dumps({
            "buckets":         {k: [str(p) for p in v] for k, v in buckets.items()},
            "review_explains": review_explains,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log(f"VLM buckets kaydedildi: {vlm_cache_path.name}")

if not VLM_CACHE:
    del vlm, vlm_proc
    gc.collect(); torch.cuda.empty_cache()
    log("Qwen VRAM'dan temizlendi.")

# ══════════════════════════════════════════════════════════════════════════════
# AŞAMA 2 — Florence bbox (sadece YES olanlar)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("AŞAMA 2: Florence bbox (AUTO-YES fotolar)")
log("=" * 60)

florence = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    dtype=torch.float16, trust_remote_code=True, attn_implementation="eager"
).to("cuda")
flo_proc = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

def florence_detect(img_pil):
    task = "<OPEN_VOCABULARY_DETECTION>"
    inp  = flo_proc(text=task + FLORENCE_PROMPT, images=img_pil, return_tensors="pt")
    with torch.no_grad():
        out = florence.generate(
            input_ids=inp["input_ids"].to("cuda"),
            pixel_values=inp["pixel_values"].to("cuda", torch.float16),
            max_new_tokens=1024, num_beams=1, do_sample=False, use_cache=False
        )
    txt    = flo_proc.batch_decode(out, skip_special_tokens=False)[0]
    parsed = flo_proc.post_process_generation(txt, task=task, image_size=(img_pil.width, img_pil.height))
    return parsed.get(task, {}).get("bboxes", [])

falldown_boxes = {}  # img_path → [(x1,y1,x2,y2), ...]

for img_path in buckets["yes"]:
    img_bgr = read_image(img_path)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    boxes   = florence_detect(img_pil)
    falldown_boxes[img_path] = boxes
    log(f"  {img_path.name[:55]}  → {len(boxes)} falldown box")

del florence, flo_proc
gc.collect(); torch.cuda.empty_cache()
log("Florence VRAM'dan temizlendi.")

# ══════════════════════════════════════════════════════════════════════════════
# AŞAMA 3 — YOLO: iki model, tüm fotolar
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("AŞAMA 3: YOLO inference (coco + own model)")
log("=" * 60)

coco_model = YOLO(str(COCO_MODEL))
own_model  = YOLO(str(OWN_MODEL))
log(f"  COCO : {COCO_MODEL.name}")
log(f"  Own  : {OWN_MODEL.name}\n")

YOLO_BATCH  = 8
model_boxes = {}  # img_path → [{"class", "box", "conf"}, ...]

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

valid_paths = [p for p in image_files if read_image(p) is not None]

coco_raw = {}
own_raw  = {}
for start in range(0, len(valid_paths), YOLO_BATCH):
    batch_paths = valid_paths[start:start + YOLO_BATCH]
    batch_imgs  = [read_image(p) for p in batch_paths]
    coco_raw.update(_parse_batch(coco_model(batch_imgs, conf=CONF, verbose=False), coco_model, batch_paths))
    own_raw.update( _parse_batch(own_model( batch_imgs, conf=CONF, verbose=False), own_model,  batch_paths))

del coco_model, own_model

for img_path in valid_paths:
    coco_boxes = [b for b in coco_raw.get(img_path, []) if b["class"] in YOLO_CLASSES]
    own_boxes  = [b for b in own_raw.get(img_path,  []) if b["class"] in CUSTOM_CLASSES
                  and not any(iou_xyxy(b["box"], cb["box"]) >= IOU_MODEL_MERGE for cb in coco_boxes)]
    model_boxes[img_path] = coco_boxes + own_boxes

log("YOLO inference tamamlandı.")

# ══════════════════════════════════════════════════════════════════════════════
# AŞAMA 4 — Merge + Kaydet
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("AŞAMA 4: Merge ve kaydet")
log("=" * 60)

yes_set    = set(buckets["yes"])
review_set = set(buckets["review"])
review_queue = []
if "review_explains" not in dir():
    review_explains = {}

for img_path in image_files:
    img_bgr = read_image(img_path)
    if img_bgr is None: continue
    img_h, img_w = img_bgr.shape[:2]

    fd_boxes = falldown_boxes.get(img_path, [])
    m_boxes  = model_boxes.get(img_path, [])
    is_yes   = img_path in yes_set

    # Florence falldown ile overlap'i olan model box'larını sil
    kept_model = [mb for mb in m_boxes
                  if not any(iou_xyxy(mb["box"], fb) >= IOU_MERGE for fb in fd_boxes)]

    # Cross-class NMS: aynı bölgede birden fazla class varsa yüksek conf kazanır
    kept_model.sort(key=lambda x: x["conf"], reverse=True)
    nms_kept = []
    for mb in kept_model:
        if not any(iou_xyxy(mb["box"], kb["box"]) >= IOU_MODEL_MERGE for kb in nms_kept):
            nms_kept.append(mb)
    kept_model = nms_kept

    removed = len(m_boxes) - len(kept_model)

    # NO/REVIEW: kendi modelin falldown tespitini yazma (VLM hayır dedi)
    if not is_yes:
        kept_model = [mb for mb in kept_model if mb["class"] != "falldown"]

    lines = []
    for fb in fd_boxes:
        norm = xyxy_to_norm(fb, img_w, img_h)
        lines.append(f"falldown {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")
    for mb in kept_model:
        norm = xyxy_to_norm(mb["box"], img_w, img_h)
        lines.append(f"{mb['class']} {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")

    label_out = OUT_DIR / "labels" / (img_path.stem + ".txt")
    label_out.write_text("\n".join(lines), encoding="utf-8")

    decision = "yes" if is_yes else ("review" if img_path in review_set else "no")
    tag = " [BACKGROUND]" if (decision == "no" and len(m_boxes) == 0) else ""
    log(f"  {decision:6s}  fd={len(fd_boxes)}  model={len(m_boxes)}  dup_rm={removed}{tag}  {img_path.name[:40]}", quiet=True)

    if img_path in review_set:
        canvas = img_bgr.copy()
        for fb in fd_boxes:
            draw_box(canvas, fb, (0, 200, 255), "FD?")
        for mb in kept_model:
            draw_box(canvas, mb["box"], (200, 200, 0), mb["class"])
        out_img = OUT_DIR / "review" / img_path.name
        _, enc  = cv2.imencode(img_path.suffix, canvas)
        out_img.write_bytes(enc.tobytes())
        vlm_info = review_explains.get(img_path.name, {})
        review_queue.append({"image": img_path.name, "label_file": label_out.name, **vlm_info})

REVIEW_PATH = OUT_DIR / "review" / f"review_{RUN_TS}.json"
REVIEW_PATH.write_text(
    json.dumps(review_queue, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"\nMerge tamamlandı. REVIEW={len(review_queue)} resim → {REVIEW_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# AŞAMA 5 — GT Karşılaştırma (F1 per-class)
# ══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log(f"AŞAMA 5: GT Karşılaştırma (IOU_EVAL={IOU_EVAL})")
log("=" * 60)

from collections import defaultdict
class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

for img_path in image_files:
    img_bgr = read_image(img_path)
    if img_bgr is None: continue
    img_h, img_w = img_bgr.shape[:2]

    gt_all   = load_gt(GT_DIR / (img_path.stem + ".txt"))
    pred_txt = OUT_DIR / "labels" / (img_path.stem + ".txt")
    pred_all = load_gt(pred_txt) if pred_txt.exists() else []

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

log(f"\n{'Sınıf':<20} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
log("-" * 60)

total_tp = total_fp = total_fn = 0
for cls in sorted(class_stats):
    s  = class_stats[cls]
    tp, fp, fn = s["tp"], s["fp"], s["fn"]
    total_tp += tp; total_fp += fp; total_fn += fn
    p  = tp / (tp+fp) if (tp+fp) > 0 else 0
    r  = tp / (tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
    marker = " ★" if cls == "falldown" else ""
    log(f"{cls:<20} {tp:>5} {fp:>5} {fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}{marker}")

log("-" * 60)
p  = total_tp / (total_tp+total_fp) if (total_tp+total_fp) > 0 else 0
r  = total_tp / (total_tp+total_fn) if (total_tp+total_fn) > 0 else 0
f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
log(f"{'TOPLAM':<20} {total_tp:>5} {total_fp:>5} {total_fn:>5} {p:>7.3f} {r:>7.3f} {f1:>7.3f}")
log(f"\n{'='*60}")
log(f"Log → {LOG_PATH}")

# ── Review tool otomatik aç ───────────────────────────────────────────────────
if review_queue:
    import subprocess, sys
    tool = Path(__file__).parent.parent / "tools" / "review_tool.py"
    log(f"\n{len(review_queue)} REVIEW foto var — review_tool açılıyor...")
    subprocess.Popen([sys.executable, str(tool)])
else:
    log("\nREVIEW foto yok, review_tool atlanıyor.")

LOG.close()
