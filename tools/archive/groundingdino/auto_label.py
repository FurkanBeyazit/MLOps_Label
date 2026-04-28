from groundeddino_vl import load_model, predict, annotate
from pathlib import Path
import cv2
import numpy as np
# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH     = r"C:\Users\admin\fur\MLOps_Label\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = r"C:\Users\admin\fur\MLOps_Label\GroundingDINO\weights\groundingdino_swint_ogc.pth"
DEVICE          = "cuda"

IMAGE_DIR       = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_LABELS_DIR  = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\labels_with_name_dino_5")
OUT_IMAGES_DIR  = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\annotated_dino_dino_5")

BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25

# Sadece zor / custom class'lar — standartlar zaten labelli
NAME_TO_ID = {
    #"person":     0,
    #"car":        1,
    "person fallen":   2,
    #"bus":        3,
    #"small truck":      4,
    #"bicycle":    5,
    #"motorcycle": 6,
    #"wild boar":       7,
    #"tractor":    8,
    #"4 wheels scooter":    9,
    #"power tiller":     10,
    "cat":        11,
    #"dog":        12,
    #"mini truck":    13,
}
TEXT_PROMPT = " . ".join(NAME_TO_ID.keys())

# ── Setup ─────────────────────────────────────────────────────────────────────
OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

print("Model yükleniyor...")
model = load_model(config_path=CONFIG_PATH, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
print("Model hazır.\n")

# ── Process images ────────────────────────────────────────────────────────────
image_files = sorted(list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")))
print(f"Toplam {len(image_files)} resim bulundu.\n")

for i, image_path in enumerate(image_files, 1):
    try:
        # Dosyayı byte olarak okuyoruz
        img_array = np.fromfile(str(image_path), np.uint8)
        # Byte dizisini resme dönüştürüyoruz
        image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        image_bgr = None

    if image_bgr is None:
        print(f"[{i}/{len(image_files)}] OKUNAMADI: {image_path.name}")
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = predict(
        model=model,
        image=image_rgb,
        text_prompt=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    # ── YOLO label (class_id cx cy w h) ──────────────────────────────────────
    label_txt = OUT_LABELS_DIR / (image_path.stem + ".txt")
    lines = []
    matched_key = next(iter(NAME_TO_ID))  # tek prompt class, ne isim dönerse o
    for label, box in zip(result.labels, result.boxes):
        cx, cy, w, h = box
        lines.append(f"{matched_key} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_txt.write_text("\n".join(lines))

    # ── Annotated image ───────────────────────────────────────────────────────
    annotated_bgr = annotate(image=image_rgb, result=result)
    
    # Uzantıyı alıyoruz (.jpg, .png vb.)
    extension = image_path.suffix 
    
    # 1. Resmi hafızada encode et (Disk yoluyla işi yok, o yüzden Korece hatası vermez)
    res, img_encoded = cv2.imencode(extension, annotated_bgr)

    if res:
        # 2. Python'ın kendi kütüphanesiyle binary olarak yaz (Korece karakterleri destekler)
        # pathlib kullandığın için .write_bytes() en kısa ve temiz yoldur
        (OUT_IMAGES_DIR / image_path.name).write_bytes(img_encoded)
    else:
        print(f"HATA: {image_path.name} encode edilemedi.")

    print(f"[{i}/{len(image_files)}] {image_path.name} → {len(result)} detection")

print("\nBitti.")
print(f"Labels : {OUT_LABELS_DIR}")
print(f"Annotated: {OUT_IMAGES_DIR}")
