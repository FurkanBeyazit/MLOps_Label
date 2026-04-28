"""
Florence-2 en iyi prompt ile tüm resimleri annotate eder.
GT box'ları yeşil, Florence-2 tahminleri kırmızı çizer.
"""

import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

IMAGE_DIR  = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_DIR    = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\annotated_flo_3")

PROMPT     = "lying person, cat, standing person"

# Her class için renk — hangi label hangi renkle çizilsin
LABEL_COLORS = {
    "lying":     (0, 0, 255),    # kırmızı  → falldown
    "fallen":    (0, 0, 255),
    "collapsed": (0, 0, 255),
    "ground":    (0, 0, 255),
    "floor":     (0, 0, 255),
    "cat":       (255, 165, 0),  # turuncu
    "dog":       (255, 200, 0),  # sarı
    "person":    (0, 255, 0),    # yeşil → ayakta duran kişi
}

def get_color(label):
    label_lower = label.lower()
    for key, color in LABEL_COLORS.items():
        if key in label_lower:
            return color
    return (180, 180, 180)  # gri → tanımsız

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
print("Florence-2 yükleniyor...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large", torch_dtype=TORCH_DTYPE, trust_remote_code=True
).to(DEVICE)
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-large", trust_remote_code=True
)
print("Hazır.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_image(image_path):
    img_array = np.fromfile(str(image_path), np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def inference(image_pil):
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    inputs = processor(
        text=task_prompt + PROMPT,
        images=image_pil,
        return_tensors="pt",
    ).to(DEVICE, TORCH_DTYPE)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image_pil.width, image_pil.height)
    )
    result = parsed.get("<CAPTION_TO_PHRASE_GROUNDING>", {"bboxes": [], "labels": []})
    # label key normalize et
    result["bboxes_labels"] = result.get("labels", [])
    return result

def save_image(image_bgr, out_path):
    ext = out_path.suffix
    res, encoded = cv2.imencode(ext, image_bgr)
    if res:
        out_path.write_bytes(encoded.tobytes())

# ── Main ──────────────────────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
print(f"{len(image_files)} resim işlenecek. Prompt: '{PROMPT}'\n")

for i, image_path in enumerate(image_files, 1):
    image_bgr = read_image(image_path)
    if image_bgr is None:
        print(f"[{i}] OKUNAMADI: {image_path.name}")
        continue

    img_h, img_w = image_bgr.shape[:2]
    image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    result   = inference(image_pil)
    canvas = image_bgr.copy()

    # Florence-2 tahminleri — her class farklı renk
    for box, label in zip(result["bboxes"], result["bboxes_labels"]):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = get_color(label)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label[:25], (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    out_path = OUT_DIR / image_path.name
    save_image(canvas, out_path)
    print(f"[{i}/{len(image_files)}] {image_path.name} → Pred:{len(result['bboxes'])}")

print(f"\nBitti → {OUT_DIR}")
