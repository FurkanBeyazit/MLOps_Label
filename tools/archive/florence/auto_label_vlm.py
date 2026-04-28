"""
Otonom VLM Label Scripti
========================
Pipeline:
  1. Florence OPEN_VOC → candidate box'lar
  2. Her box → büyük crop → VLMFilter.classify()
      "no"     → at (otomatik)
      "yes"    → tut, label olarak kaydet (otomatik)
      "review" → manuel_queue'ye ekle
  3. Sonunda manuel_queue'deki resimleri listele / kaydet

Çıktı:
  - auto_labels/   → otomatik üretilen label'lar (YOLO format)
  - review_queue/  → kullanıcının bakması gereken crop'lar + info
  - auto_label_log.txt
"""

import torch
import cv2
import numpy as np
import json
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_LABELS  = Path(r"C:\Users\admin\fur\MLOps_Label\tools\auto_labels")
REVIEW_DIR  = Path(r"C:\Users\admin\fur\MLOps_Label\tools\review_queue")
LOG_PATH    = Path(r"C:\Users\admin\fur\MLOps_Label\logs\auto_label_vlm_log.txt")

FLORENCE_PROMPT = "a person lying on the ground"
CROP_PAD        = 2.0

# VLM güven eşikleri (vlm_filter.py ile aynı)
AUTO_YES_THR = 0.85
AUTO_NO_THR  = 0.85

# ── Model yükle ───────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from vlm_filter import VLMFilter, get_crop

print("Florence-2 yükleniyor...")
florence = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager"
).to("cuda")
flo_proc = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

print("Qwen3VL yükleniyor...")
vlm = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0"
)
vlm_proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
vf = VLMFilter(vlm, vlm_proc)
print("Hazır.\n")

OUT_LABELS.mkdir(parents=True, exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

def log(line):
    print(line)
    LOG_FILE.write(line + "\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_image(p):
    arr = np.fromfile(str(p), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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

def xyxy_to_cxcywh_norm(box, w, h):
    x1, y1, x2, y2 = box
    return [((x1+x2)/2)/w, ((y1+y2)/2)/h, (x2-x1)/w, (y2-y1)/h]

def draw_box_on_image(img_bgr, box, color, label):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img_bgr, label, (x1, max(y1-5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# ── Main ──────────────────────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
log(f"{len(image_files)} resim işlenecek.")
log(f"Auto-yes: P(Yes)≥{AUTO_YES_THR}  Auto-no: P(No)≥{AUTO_NO_THR}  Arası: review\n")

stats = {"auto_yes": 0, "auto_no": 0, "review": 0, "no_box": 0}
review_queue = []   # manuel bakılacaklar

for i, img_path in enumerate(image_files, 1):
    img_bgr = read_image(img_path)
    if img_bgr is None: continue
    img_h, img_w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    boxes = florence_detect(img_pil)
    if not boxes:
        stats["no_box"] += 1
        log(f"[{i}/{len(image_files)}] {img_path.name[:55]} | box=0 → atlandı")
        continue

    kept_labels = []    # otomatik "yes" → label olarak kaydet
    review_boxes = []   # belirsiz → review kuyruğu

    log(f"[{i}/{len(image_files)}] {img_path.name[:55]} | flo={len(boxes)} box")

    for bi, box in enumerate(boxes):
        crop = get_crop(img_pil, box, pad=CROP_PAD)
        decision, p_yes, p_no = vf.classify(crop)

        icon = {"yes": "🔴 YES (tut)", "no": "🟢 No  (at)", "review": "🟡 REVIEW"}[decision]
        log(f"  box[{bi}]  {icon}  P(Yes)={p_yes:.3f}  P(No)={p_no:.3f}")

        if decision == "yes":
            norm = xyxy_to_cxcywh_norm(box, img_w, img_h)
            kept_labels.append(f"falldown {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}")
            stats["auto_yes"] += 1

        elif decision == "no":
            stats["auto_no"] += 1

        else:  # review
            stats["review"] += 1
            review_boxes.append({
                "box_idx": bi,
                "box_xyxy": box,
                "p_yes": round(p_yes, 4),
                "p_no": round(p_no, 4),
            })

    # Label dosyası kaydet (sadece auto-yes olanlar)
    if kept_labels:
        label_out = OUT_LABELS / (img_path.stem + ".txt")
        label_out.write_text("\n".join(kept_labels), encoding="utf-8")
        log(f"  → {len(kept_labels)} label kaydedildi: {label_out.name}")

    # Review: annotated resim kaydet
    if review_boxes:
        canvas = img_bgr.copy()
        for rb in review_boxes:
            draw_box_on_image(canvas, rb["box_xyxy"],
                              (0, 200, 255),
                              f"REVIEW P(Y)={rb['p_yes']:.2f}")
        out_img = REVIEW_DIR / img_path.name
        _, enc = cv2.imencode(img_path.suffix, canvas)
        out_img.write_bytes(enc.tobytes())

        review_queue.append({
            "image": img_path.name,
            "boxes": review_boxes,
        })
        log(f"  → {len(review_boxes)} review box → {out_img.name}")

# ── Review Queue JSON ──────────────────────────────────────────────────────────
review_json = REVIEW_DIR / "review_queue.json"
review_json.write_text(json.dumps(review_queue, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Özet ──────────────────────────────────────────────────────────────────────
total_boxes = stats["auto_yes"] + stats["auto_no"] + stats["review"]
log(f"""
{'='*60}
SONUÇ
{'='*60}
Toplam box     : {total_boxes}
Auto-YES (tut) : {stats['auto_yes']}  ({stats['auto_yes']/max(total_boxes,1)*100:.1f}%)
Auto-NO  (at)  : {stats['auto_no']}   ({stats['auto_no']/max(total_boxes,1)*100:.1f}%)
Review         : {stats['review']}    ({stats['review']/max(total_boxes,1)*100:.1f}%)
Box yok        : {stats['no_box']} resim

Label çıktısı  → {OUT_LABELS}
Review resimleri → {REVIEW_DIR}
Review listesi → {review_json}
{'='*60}
""")

LOG_FILE.close()
