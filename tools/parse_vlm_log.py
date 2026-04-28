"""
VLM log dosyasından vlm_buckets JSON üretir.
Kullanım: python parse_vlm_log.py pipeline_log_YYYYMMDD_HHMMSS.txt
"""
import re, json, sys
from pathlib import Path

IMAGE_DIR = Path(r"C:\Users\admin\fur\MLOps_Label\data\full dataset\val\images")
OUT_DIR   = Path(r"C:\Users\admin\fur\MLOps_Label\pipeline\pipeline_out")

if len(sys.argv) < 2:
    # argüman verilmezse en son log dosyasını bul
    logs = sorted(OUT_DIR.glob("pipeline_log_*.txt"))
    if not logs:
        print("Hata: log dosyası bulunamadı.")
        sys.exit(1)
    log_path = logs[-1]
    print(f"Argüman verilmedi, en son log kullanılıyor: {log_path.name}")
else:
    log_path = Path(sys.argv[1])
    if not log_path.is_absolute():
        log_path = OUT_DIR / log_path

if not log_path.exists():
    print(f"Hata: {log_path} bulunamadı.")
    sys.exit(1)

lines   = log_path.read_text(encoding="utf-8").splitlines()
pattern = re.compile(
    r'\[\s*\d+/\d+\].*?(yes|no|review)\s+P\(Y\)=([\d.]+)\s+P\(N\)=([\d.]+)\s+([\d.]+)s\s+(.+)'
)

# prompt ve vlm_max_px bilgisini logdan okumaya çalış
vlm_max_px = "bilinmiyor"
prompt_line = ""
for line in lines:
    if "VLM_MAX_PX" in line:
        vlm_max_px = line.split("VLM_MAX_PX")[-1].strip()
    if "FALLEN DOWN" in line:
        prompt_line = line.strip()

buckets = {"yes": [], "no": [], "review": []}
details = []

# IMAGE_DIR içindeki tüm dosyaları önceden indeksle (prefix → tam yol)
image_index = {}
for p in IMAGE_DIR.glob("*.jpg"):
    image_index[p.name] = p
for p in IMAGE_DIR.glob("*.png"):
    image_index[p.name] = p

def resolve_path(fname):
    fname = fname.strip()
    # Tam eşleşme
    if fname in image_index:
        return str(image_index[fname])
    # Prefix eşleşme (log 45 karakterde kesiyor)
    for name, path in image_index.items():
        if name.startswith(fname):
            return str(path)
    return str(IMAGE_DIR / fname)  # bulunamazsa olduğu gibi

for line in lines:
    m = pattern.search(line)
    if not m:
        continue
    decision, py, pn, elapsed, fname = m.groups()
    img_path = resolve_path(fname)
    buckets[decision].append(img_path)
    details.append({
        "image":     Path(img_path).name,
        "decision":  decision,
        "p_yes":     float(py),
        "p_no":      float(pn),
        "elapsed_s": float(elapsed),
    })

avg = round(sum(d["elapsed_s"] for d in details) / len(details), 1) if details else 0

ts  = log_path.stem.replace("pipeline_log_", "")
out = OUT_DIR / f"vlm_buckets_{ts}.json"

result = {
    "meta": {
        "log_file":      log_path.name,
        "vlm_model":     "Qwen/Qwen3-VL-2B-Instruct",
        "vlm_thr":       0.70,
        "vlm_max_px":    vlm_max_px,
        "prompt":        (
            "Has a person FALLEN DOWN or is LYING on the ground? "
            "Answer Yes ONLY if someone is clearly horizontal: lying flat, collapsed, or sprawled. "
            "Answer No if all people are upright — standing, walking, running "
            "even if they are on concrete, asphalt, or any outdoor surface. "
            "Ignore animals. Answer only Yes or No."
        ),
        "total_images":  len(details),
        "yes":           len(buckets["yes"]),
        "no":            len(buckets["no"]),
        "review":        len(buckets["review"]),
        "avg_elapsed_s": avg,
    },
    "buckets": buckets,
    "details": details,
}

out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Kaydedildi : {out.name}")
print(f"YES={len(buckets['yes'])}  NO={len(buckets['no'])}  REVIEW={len(buckets['review'])}")
print(f"Ort. sure  : {avg}s/goruntu")
print(f"\nBu JSON'u kullanmak icin label_pipeline.py icinde:")
print(f'  VLM_CACHE = OUT_DIR / "{out.name}"')
