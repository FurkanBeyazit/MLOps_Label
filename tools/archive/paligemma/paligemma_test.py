"""
PaliGemma 2 3B-pt-448 detection testi — falldown odaklı.
Çıktı formatı: <loc0123><loc0456><loc0789><loc0987> fallen person
loc sırası: y_min, x_min, y_max, x_max (0-1023 normalize)
"""
import re, cv2, numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\tools\paligemma_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGES = [
    "E02_005.mp4_1955.jpg",   # iki yan yana düşmüş çocuk
    "E02_021.mp4_3464.jpg",   # iki ayrı yerde insan
    "E02_003.mp4_2072.jpg",   # review — düşmüş insan
    "E02_035.mp4_4503.jpg",   # review — çöp yığını FP
    "어린이 0065 하안1동 밤일안로42번길 37 산들유치원2_2026 2 3 15 1 22.jpg",  # çömelmiş insan
    "어린이 0065 하안1동 밤일안로42번길 37 산들유치원2_2026 1 24 18 49 59.jpg",  # kedi FP
]

PROMPTS = {
    "detect":        "<image>detect fallen person\n",
    "detect_lying":  "<image>detect person lying on the ground\n",
    "detect_multi":  "<image>detect fallen person ; detect person lying down\n",
}

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_paligemma_boxes(text, img_w, img_h):
    """
    <loc0123><loc0456><loc0789><loc0987> label ; ...
    sırası: y_min, x_min, y_max, x_max (0-1023)
    """
    boxes = []
    pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>"
    for m in re.finditer(pattern, text):
        y1, x1, y2, x2 = [int(v) / 1023.0 for v in m.groups()]
        boxes.append([
            x1 * img_w, y1 * img_h,
            x2 * img_w, y2 * img_h,
        ])
    return boxes

# ── Model ─────────────────────────────────────────────────────────────────────
print("PaliGemma 2 yükleniyor...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    r"C:\Users\danusys\models\paligemma2-3b-pt-448",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()
processor = PaliGemmaProcessor.from_pretrained(r"C:\Users\danusys\models\paligemma2-3b-pt-448")
print("Hazır.\n")

def run_detection(img_pil, prompt):
    inputs = processor(text=prompt, images=img_pil, return_tensors="pt")
    inputs = {k: v.to(torch.bfloat16).to(model.device) if v.dtype == torch.float32
              else v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return processor.decode(out[0][input_len:], skip_special_tokens=False)

# ── Test ──────────────────────────────────────────────────────────────────────
for fname in TEST_IMAGES:
    img_path = IMAGE_DIR / fname
    if not img_path.exists():
        print(f"[!] Bulunamadı: {fname}")
        continue

    arr    = np.fromfile(str(img_path), np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_h, img_w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    print(f"\n{'─'*55}")
    print(f"{fname[:55]}")

    canvas = img_bgr.copy()
    colors = {"detect": (0,80,255), "detect_lying": (0,180,80), "detect_multi": (180,0,255)}

    for pname, prompt in PROMPTS.items():
        raw    = run_detection(img_pil, prompt)
        boxes  = parse_paligemma_boxes(raw, img_w, img_h)
        print(f"  [{pname}] {len(boxes)} bbox")
        print(f"    ham: {raw[:150]}")

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            color = colors[pname]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, f"{pname[0]}{j+1}", (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out_path = OUT_DIR / fname
    _, enc = cv2.imencode(img_path.suffix or ".jpg", canvas)
    out_path.write_bytes(enc.tobytes())

print(f"\nBitti → {OUT_DIR}")
print("Renk kodu: Mavi=detect  Yeşil=detect_lying  Mor=detect_multi")
