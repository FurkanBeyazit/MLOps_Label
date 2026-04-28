"""Adım 1: Florence box'ları bul, JSON'a kaydet. Sonra bu process kapanır."""
import torch, cv2, numpy as np, json
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_JSON  = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\fp_boxes.json")

FP_CASES = [
    "어린이 0065 하안1동 밤일안로42번길 37 산들유치원2_2026 2 1 14 14 27.jpg",
    "어린이 0067 하안2동 안현로16 하안초등학교 정문 앞(고정1)_2026 2 3 20 21 16.jpg",
]

florence = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    dtype=torch.float16, trust_remote_code=True, attn_implementation="eager"
).to("cuda")
proc = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

results = {}
for fname in FP_CASES:
    img_bgr = cv2.imdecode(np.fromfile(str(IMAGE_DIR/fname), np.uint8), cv2.IMREAD_COLOR)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    W, H    = img_pil.size
    task    = "<OPEN_VOCABULARY_DETECTION>"
    inp     = proc(text=task+"a person lying on the ground", images=img_pil, return_tensors="pt")
    with torch.no_grad():
        out = florence.generate(
            input_ids=inp["input_ids"].to("cuda"),
            pixel_values=inp["pixel_values"].to("cuda", torch.float16),
            max_new_tokens=1024, num_beams=1, do_sample=False, use_cache=False
        )
    txt    = proc.batch_decode(out, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))
    boxes  = parsed.get(task, {}).get("bboxes", [])
    results[fname] = {"size": [W, H], "boxes": boxes}
    print(f"{fname[:55]} → {len(boxes)} box  {boxes}")

OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nKaydedildi → {OUT_JSON}")
