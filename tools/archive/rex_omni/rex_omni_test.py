"""
Rex-Omni detection testi — falldown odaklı.
"""
from PIL import Image
from pathlib import Path
from rex_omni import RexOmniWrapper, RexOmniVisualize

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\tools\rex_omni_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_IMAGES = [
    "E02_005.mp4_1955.jpg",   # iki yan yana düşmüş çocuk
    "E02_021.mp4_3464.jpg",   # iki ayrı yerde insan
    "E02_003.mp4_2072.jpg",   # review — düşmüş insan
    "E02_035.mp4_4503.jpg",   # review — çöp yığını FP
    "어린이 0065 하안1동 밤일안로42번길 37 산들유치원2_2026 2 3 15 1 22.jpg",  # çömelmiş insan
    "어린이 0065 하안1동 밤일안로42번길 37 산들유치원2_2026 1 24 18 49 59.jpg",  # kedi FP
]

print("Rex-Omni yükleniyor...")
rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers",
    attn_implementation="sdpa",
    device_map="cuda:0",
)
print("Hazır.\n")


def count_and_print_preds(preds):
    """preds dict formatında: {category: [annotation, ...]}"""
    total = sum(len(v) for v in preds.values())
    print(f"  → {total} bbox bulundu")
    for category, annotations in preds.items():
        for ann in annotations:
            coords = [f"{c:.1f}" for c in ann["coords"]]
            print(f"     [{category}] {ann['type']} {coords}")
    return total


# ── Test 1: falldown tespiti ───────────────────────────────────────────────────
print("=" * 50)
print("TEST 1 — fallen person tespiti")
print("=" * 50)

for fname in TEST_IMAGES:
    img_path = IMAGE_DIR / fname
    if not img_path.exists():
        print(f"[!] Bulunamadı: {fname}")
        continue

    image = Image.open(str(img_path))
    results = rex.inference(
        images=image,
        task="detection",
        categories=["fallen person"]
    )
    preds = results[0]["extracted_predictions"]
    print(f"\n{fname[:55]}")
    count_and_print_preds(preds)
    print(f"  ham çıktı: {results[0].get('raw_output', results[0].get('output', '?'))[:200]}")

    vis = RexOmniVisualize(image=image, predictions=preds)
    out_path = OUT_DIR / f"t1_{fname}"
    vis.save(str(out_path))

# ── Test 2: çoklu kategori ────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TEST 2 — fallen person + person + cat")
print("=" * 50)

for fname in TEST_IMAGES[:4]:
    img_path = IMAGE_DIR / fname
    if not img_path.exists(): continue

    image = Image.open(str(img_path))
    results = rex.inference(
        images=image,
        task="detection",
        categories=["fallen person", "person", "cat"]
    )
    preds = results[0]["extracted_predictions"]
    print(f"\n{fname[:55]}")
    count_and_print_preds(preds)

    vis = RexOmniVisualize(image=image, predictions=preds)
    out_path = OUT_DIR / f"t2_{fname}"
    vis.save(str(out_path))

print(f"\nBitti → {OUT_DIR}")
