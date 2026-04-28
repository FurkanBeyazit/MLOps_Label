"""
Qwen grounding prompt karşılaştırması.
VLM YES görsellerinde farklı prompt'ları test eder, F1 yan yana verir.
"""
import torch, cv2, numpy as np, re, json, time
from pathlib import Path
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
GT_DIR    = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\labels_with_name")
VLM_CACHE = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out\vlm_buckets_20260422_165231.json")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\tools\qwen_prompt_compare_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_EVAL = 0.35

PROMPTS = {
    "A_original":   "Detect all persons lying on the ground or fallen down. Output their bounding boxes.",
    "B_cctv":       "This is a CCTV security camera image. Detect all fallen persons — people lying flat, collapsed, or sprawled on the ground. Output their bounding boxes.",
    "C_exclude":    "Detect persons who have fallen — lying horizontally or collapsed on the ground. Exclude people who are standing, walking, sitting, or crouching. Output bounding boxes.",
    "D_short":      "Find all fallen people lying on the ground. Output bounding boxes.",
    "E_horizontal": "Locate all people lying horizontally on the ground. Do not include people who are upright. Output bounding boxes.",
    "F_detail":     "This is a CCTV image. Find people who appear injured or unconscious — lying flat on the ground, not moving. Exclude animals, sitting persons, and standing persons. Output bounding boxes.",
}

# ── VLM cache ─────────────────────────────────────────────────────────────────
raw     = json.loads(VLM_CACHE.read_text(encoding="utf-8"))
buckets = raw.get("buckets", raw)
vlm_yes = {Path(p).stem for p in buckets.get("yes", [])}
vlm_rev = {Path(p).stem for p in buckets.get("review", [])}
vlm_pass = vlm_yes | vlm_rev
print(f"VLM YES: {len(vlm_yes)}  REVIEW: {len(vlm_rev)}  toplam: {len(vlm_pass)} görsel\n")

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

def parse_boxes(text, img_w, img_h):
    boxes = []
    for m in re.finditer(r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text):
        x1, y1, x2, y2 = [int(v) for v in m.groups()]
        boxes.append([x1/1000*img_w, y1/1000*img_h, x2/1000*img_w, y2/1000*img_h])
    if not boxes:
        for m in re.finditer(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', text):
            x1, y1, x2, y2 = [int(v) for v in m.groups()]
            boxes.append([x1/1000*img_w, y1/1000*img_h, x2/1000*img_w, y2/1000*img_h])
    return boxes

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

# ── Model ─────────────────────────────────────────────────────────────────────
print("Qwen3-VL yükleniyor...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
print("Hazır.\n")

def run_grounding(img_pil, prompt):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img_pil},
        {"type": "text",  "text":  prompt},
    ]}]
    inp = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    inp.pop("token_type_ids", None)
    inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False)
    return processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)

# ── YES görsellerini yükle ────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
pass_files  = [p for p in image_files if p.stem in vlm_pass]
print(f"YES+REVIEW görseller: {len(pass_files)} / {len(image_files)}\n")

# GT'yi bir kere yükle
gt_cache = {}
for img_path in pass_files:
    gt_labels = load_gt(GT_DIR / (img_path.stem + ".txt"))
    img_bgr   = read_image(img_path)
    if img_bgr is None: continue
    img_h, img_w = img_bgr.shape[:2]
    gt_cache[img_path.stem] = {
        "path": img_path,
        "bgr": img_bgr,
        "w": img_w, "h": img_h,
        "gt_boxes": [norm_to_xyxy(b, img_w, img_h) for c, b in gt_labels if c == "falldown"],
    }

# VLM NO FN katkısı (sabit, prompt değişmez)
no_fn = sum(
    len([c for c, b in load_gt(GT_DIR / (p.stem + ".txt")) if c == "falldown"])
    for p in image_files if p.stem not in vlm_pass
)

# ── Her prompt için eval ──────────────────────────────────────────────────────
results = {}

for prompt_name, prompt_text in PROMPTS.items():
    print(f"\n{'='*60}")
    print(f"PROMPT {prompt_name}: {prompt_text[:70]}...")
    print(f"{'='*60}")

    total_tp = total_fp = 0
    total_fn = no_fn  # NO görsellerinden gelen FN

    vis_dir = OUT_DIR / prompt_name
    vis_dir.mkdir(exist_ok=True)

    for stem, data in gt_cache.items():
        img_pil    = Image.fromarray(cv2.cvtColor(data["bgr"], cv2.COLOR_BGR2RGB))
        text       = run_grounding(img_pil, prompt_text)
        pred_boxes = parse_boxes(text, data["w"], data["h"])
        tp, fp, fn = eval_boxes(pred_boxes, data["gt_boxes"])
        total_tp += tp; total_fp += fp; total_fn += fn

        # Görsel kaydet
        canvas = data["bgr"].copy()
        for j, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 80, 255), 2)
            cv2.putText(canvas, f"q{j+1}", (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
        for box in data["gt_boxes"]:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 1)
        suffix = data["path"].suffix or ".jpg"
        _, enc = cv2.imencode(suffix, canvas)
        (vis_dir / data["path"].name).write_bytes(enc.tobytes())

        marker = " ★" if (data["gt_boxes"] or pred_boxes) else ""
        print(f"  pred={len(pred_boxes)} gt={len(data['gt_boxes'])} tp={tp} fp={fp} fn={fn}{marker}  {stem[:40]}")

    p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
    results[prompt_name] = {"TP": total_tp, "FP": total_fp, "FN": total_fn,
                             "P": round(p,4), "R": round(r,4), "F1": round(f1,4)}
    print(f"  → TP={total_tp} FP={total_fp} FN={total_fn}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

# ── Özet tablo ────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'PROMPT':<18} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}")
print(f"{'='*60}")
for name, m in results.items():
    print(f"{name:<18} {m['TP']:>4} {m['FP']:>4} {m['FN']:>4} {m['P']:>6.3f} {m['R']:>6.3f} {m['F1']:>6.3f}")
print(f"\n  Florence baseline:   TP=27  FP=5   FN=12  P=0.844  R=0.692  F1=0.761")

(OUT_DIR / "prompt_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"\nSonuçlar kaydedildi → {OUT_DIR}/prompt_results.json")
