"""
Florence prompt karşılaştırma — belirli fotolar üzerinde.
Her prompt için ayrı klasöre annotated çıktı kaydeder.
"""
import torch, cv2, numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\tools\prompt_test_out")

VLM_BUCKETS = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out\vlm_buckets_20260421_112607.json")

TEST_STEMS = None  # aşağıda IMAGE_DIR yüklendikten sonra doldurulur

# (task, prompt, klasör adı)  — prompt=None → VLM explain kullan
VARIANTS = [
    ("<OPEN_VOCABULARY_DETECTION>",    "a person lying on the ground",                        "OVD_a_person_lying"),
    ("<OPEN_VOCABULARY_DETECTION>",    "people lying on the ground",                          "OVD_people_lying"),
    ("<OPEN_VOCABULARY_DETECTION>",    "two people lying on the ground",                      "OVD_two_people"),
    ("<OPEN_VOCABULARY_DETECTION>",    "person lying on the ground, person lying on the ground", "OVD_repeated"),
    ("<CAPTION_TO_PHRASE_GROUNDING>",  "person lying on the ground, person lying on the ground", "CPG_repeated"),
    ("<CAPTION_TO_PHRASE_GROUNDING>",  "lying person",                                        "CPG_lying_person"),
    ("<OPEN_VOCABULARY_DETECTION>",    None,                                                  "OVD_vlm_explain"),
]

# ── Model ─────────────────────────────────────────────────────────────────────
print("Florence-2 yükleniyor...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager",
).to("cuda")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
print("Hazır.\n")

def read_image(p):
    arr = np.fromfile(str(p), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def run(img_pil, task, prompt):
    inp = processor(text=task + prompt, images=img_pil, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            input_ids=inp["input_ids"].to("cuda"),
            pixel_values=inp["pixel_values"].to("cuda", torch.float16),
            max_new_tokens=1024, num_beams=1, do_sample=False, use_cache=False,
        )
    txt    = processor.batch_decode(out, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(txt, task=task, image_size=(img_pil.width, img_pil.height))
    result = parsed.get(task, {})
    bboxes = result.get("bboxes", [])
    labels = result.get("labels", ["box"] * len(bboxes))
    return bboxes, labels

# ── Test görselleri bul ───────────────────────────────────────────────────────
test_paths = []
test_paths = sorted([
    p for p in IMAGE_DIR.glob("E02_0*.jpg")
    if "E02_001" <= p.stem[:7] <= "E02_035"
])

print(f"{len(test_paths)} test görseli bulundu.\n")

# VLM açıklamalarını yükle
import json
review_explains = {}
if VLM_BUCKETS.exists():
    raw = json.loads(VLM_BUCKETS.read_text(encoding="utf-8"))
    review_explains = raw.get("review_explains", {})
    print(f"VLM explains: {len(review_explains)} açıklama yüklendi")

# ── Varyantları çalıştır ──────────────────────────────────────────────────────
for task, prompt, folder in VARIANTS:
    out_folder = OUT_DIR / folder
    out_folder.mkdir(parents=True, exist_ok=True)
    print(f"\n{'─'*60}")
    print(f"Task : {task}")
    print(f"Prompt: {prompt}")
    print(f"{'─'*60}")

    for img_path in test_paths:
        img_bgr = read_image(img_path)
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # VLM explain modunda per-image prompt
        if prompt is None:
            info = review_explains.get(img_path.name, {})
            actual_prompt = info.get("explain", "a person lying on the ground")
            if not actual_prompt or actual_prompt == "—":
                actual_prompt = "a person lying on the ground"
        else:
            actual_prompt = prompt

        bboxes, labels = run(img_pil, task, actual_prompt)

        canvas = img_bgr.copy()
        for box, label in zip(bboxes, labels):
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 80, 255), 2)
            cv2.putText(canvas, label[:30], (x1, max(y1-5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

        out_path = out_folder / img_path.name
        _, enc = cv2.imencode(img_path.suffix, canvas)
        out_path.write_bytes(enc.tobytes())

        used = actual_prompt if prompt is None else ""
        print(f"  {img_path.stem[:50]}  → {len(bboxes)} box  {labels}" + (f"  [{used[:60]}]" if used else ""))

print(f"\n\nBitti → {OUT_DIR}")
