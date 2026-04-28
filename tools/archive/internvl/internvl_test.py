"""
InternVL3.5-2B-Flash detection testi — falldown odaklı.
Çıktı formatı: <ref>...</ref><box>[[x1, y1, x2, y2]]</box> (0-1000 normalize)
"""
import re, cv2, numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\tools\internvl_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "OpenGVLab/InternVL3_5-2B-Flash"

TEST_IMAGES = [
    "E02_005.mp4_1955.jpg",   # iki yan yana düşmüş çocuk
    "E02_021.mp4_3464.jpg",   # iki ayrı yerde insan
    "E02_003.mp4_2072.jpg",   # review — düşmüş insan
    "E02_035.mp4_4503.jpg",   # review — çöp yığını FP
    "어린이 0065 하안1동 밤일안로42번길 37 산들유치원2_2026 2 3 15 1 22.jpg",  # çömelmiş insan
    "어린이 0065 하안1동 밤일안로42번길 37 산들유치원2_2026 1 24 18 49 59.jpg",  # kedi FP
]

PROMPTS = {
    "ref_fallen":  "Please provide the bounding box coordinate of the region this sentence describes: <ref>fallen person lying on the ground</ref>",
    "ref_lying":   "Please provide the bounding box coordinate of the region this sentence describes: <ref>person lying flat on the ground</ref>",
    "detect":      "Detect all fallen persons in this CCTV image. Output their bounding boxes.",
}

# ── Parser ────────────────────────────────────────────────────────────────────
def parse_internvl_boxes(text, img_w, img_h):
    """text[[x1, y1, x2, y2], ...] veya <box>[[...]]</box> — 0-1000 normalize"""
    boxes = []
    for m in re.finditer(r"\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]", text):
        x1, y1, x2, y2 = [int(v) / 1000.0 for v in m.groups()]
        boxes.append([x1*img_w, y1*img_h, x2*img_w, y2*img_h])
    return boxes

# ── Model ─────────────────────────────────────────────────────────────────────
print("InternVL3.5 yükleniyor...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Hazır.\n")

_transform = T.Compose([
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def run_detection(img_pil, prompt):
    pixel_values = _transform(img_pil).unsqueeze(0).to(torch.bfloat16).to(model.device)
    generation_config = {"max_new_tokens": 256, "do_sample": False}
    response = model.chat(
        tokenizer,
        pixel_values=pixel_values,
        question=f"<image>\n{prompt}",
        generation_config=generation_config,
    )
    return response

# ── Test ──────────────────────────────────────────────────────────────────────
for fname in TEST_IMAGES:
    img_path = IMAGE_DIR / fname
    if not img_path.exists():
        print(f"[!] Bulunamadı: {fname}")
        continue

    arr     = np.fromfile(str(img_path), np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_h, img_w = img_bgr.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    print(f"\n{'─'*55}")
    print(f"{fname[:55]}")

    canvas = img_bgr.copy()
    colors = {"ref_fallen": (0,80,255), "ref_lying": (0,180,80), "detect": (180,0,255)}

    for pname, prompt in PROMPTS.items():
        try:
            raw   = run_detection(img_pil, prompt)
            boxes = parse_internvl_boxes(raw, img_w, img_h)
            print(f"  [{pname}] {len(boxes)} bbox")
            print(f"    ham: {raw[:150]}")

            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                color = colors[pname]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"{pname[0]}{j+1}", (x1, max(y1-5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        except Exception as e:
            print(f"  [{pname}] HATA: {e}")

    out_path = OUT_DIR / fname
    _, enc = cv2.imencode(img_path.suffix or ".jpg", canvas)
    out_path.write_bytes(enc.tobytes())

print(f"\nBitti → {OUT_DIR}")
print("Renk: Mavi=ref_fallen  Yeşil=ref_lying  Mor=detect")
