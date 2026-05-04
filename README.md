# MLOps Label Pipeline

VLM + Qwen Grounding tabanlı otomatik etiketleme pipeline'ı.  
**Amaç:** Falldown tespit modeli için fine-tune verisi hazırlamak. Manuel etiketleme yerine otomatik label çıkar, insan sadece belirsiz vakaları review etsin.

---

## Hızlı Başlangıç

```bash
# 1. Pipeline çalıştır
python pipeline/label_pipeline.py

# 2. Bitince review_tool otomatik açılır (veya manuel)
python tools/review_tool.py

# 3. F1 + hata analizi
python pipeline/inspect_fp_fn.py
```

---

## Pipeline Akışı

```
Tüm fotoğraflar
      ↓
[Aşama 1] VLM Tarama (Qwen3-VL-2B)
      ├── P(Yes) ≥ VLM_THR_YES  →  YES   (kesin düşme var)
      ├── P(No)  ≥ VLM_THR_NO   →  NO    (düşme yok, atla)
      └── diğer                 →  REVIEW (belirsiz, insan bakacak)
              ↓
[Aşama 2] Qwen Grounding  ← sadece YES + REVIEW
      falldown bbox üret
              ↓
[Aşama 3] YOLO Inference  ← tüm fotoğraflar
      person, car, bus, truck, bicycle, motorcycle,
      dog, cat, boar, tractor, tiller, scooter
              ↓
[Aşama 4] Merge & Kaydet → pipeline_out/{RUN_TS}/labels/
              ↓
[Aşama 5] GT Karşılaştırma → F1 raporu  ← sadece test modunda
              ↓
[Aşama 6] Review Tool açılır → insan onayı → reviewed_labels/
```

---

## Dizin Yapısı

```
MLOps_Label/
├── pipeline/
│   ├── label_pipeline.py        # ← ana pipeline
│   ├── inspect_fp_fn.py         # ← FP/FN görsel analiz + F1
│   └── pipeline_out/
│       └── 20260504_143000/     # her run kendi klasörüne
│           ├── labels/           # pipeline çıktısı (.txt, YOLO format)
│           ├── reviewed_labels/  # review sonrası düzeltilmiş
│           ├── review/           # REVIEW fotoları + json
│           ├── run_config.json   # IMAGE_DIR, GT_DIR (review_tool okur)
│           ├── pipeline_log.txt  # F1 raporu dahil tüm log
│           └── fp_fn_check/      # inspect_fp_fn çıktısı
├── tools/
│   ├── review_tool.py           # ← bbox editor GUI
│   └── archive/florence/        # Florence yedek versiyonu
└── models/
    ├── coco_yolo/yolov9e.pt
    └── falldown/best_04_15.pt
```

---

## `label_pipeline.py` — Config

Dosyanın başındaki config bloğunu düzenle:

```python
# ── Hangi fotoğrafları işleyeceğin
IMAGE_DIR = Path(r"C:\...\0424(쓰러짐)")

# ── GT etiketleri (sadece test/F1 için — production'da None yap)
GT_DIR = Path(r"C:\...\labels_with_name")   # yoksa: GT_DIR = None

# ── VLM cache: aynı dataset üzerinde tekrar çalışıyorsan süreci hızlandırır
# Prompt değiştiyse veya yeni dataset ise: VLM_CACHE = None
VLM_CACHE = PIPELINE_OUT / "vlm_buckets_20260430_142455.json"

# ── Threshold: YES kolay, NO zor → belirsizler REVIEW'e düşsün
VLM_THR_YES = 0.65
VLM_THR_NO  = 0.90
```

### Ne zaman ne değiştirilir?

| Durum | Değiştirilecek |
|---|---|
| Yeni dataset | `IMAGE_DIR` |
| GT karşılaştırması istiyorsun | `GT_DIR` → klasör yolu |
| Production (GT yok) | `GT_DIR = None` |
| Aynı dataset, prompt aynı | `VLM_CACHE` → mevcut json |
| Prompt değişti / yeni dataset | `VLM_CACHE = None` |
| Çok az REVIEW geliyor | `VLM_THR_NO` düşür (örn. 0.85) |
| Çok fazla alakasız REVIEW | `VLM_THR_NO` yükselt (örn. 0.92) |

---

## `review_tool.py` — Kullanım

```bash
# En son pipeline run'ını aç
python tools/review_tool.py

# Spesifik bir run'ı aç
python tools/review_tool.py --run 20260430_151636
```

### Arayüz

| Tab | İçerik |
|---|---|
| **Label Editor** | REVIEW fotoları — bbox ekle / sil / taşı / boyutlandır |
| **Gallery** | Tüm pipeline labelları — GT kutuları beyaz dashed |

### Kısayollar
- `A / D` — önceki / sonraki foto
- `Del` — seçili box sil
- `S` — kaydet
- `F` — ekrana sığdır

### Kayıt yeri
Onaylanan labellar `pipeline_out/{RUN_TS}/reviewed_labels/` altına kaydedilir.  
> Pipeline çıktısı `labels/` klasöründe kalır, üzerine yazılmaz.

---

## `inspect_fp_fn.py` — Hata Analizi & F1

```bash
python pipeline/inspect_fp_fn.py
```

### Config (dosya başında)

```python
IMAGE_DIR = Path(r"C:\...\0424(쓰러짐)")
GT_DIR    = Path(r"C:\...\labels_with_name")

# Pipeline sonrası F1
PRED_DIR  = Path(r"C:\...\pipeline_out\{RUN_TS}\labels")

# Review sonrası F1 görmek için:
# PRED_DIR = Path(r"C:\...\pipeline_out\{RUN_TS}\reviewed_labels")

OUT_DIR   = Path(r"C:\...\pipeline_out\{RUN_TS}\fp_fn_check")
SAVE_TP   = True   # TP fotoları da kaydet
```

### Çıktı

Terminal'de per-class F1 tablosu:

```
Sınıf                   TP    FP    FN       P       R      F1
------------------------------------------------------------
falldown                10     3     3   0.769   0.769   0.769 ★
...
```

Görsel çıktı `fp_fn_check/` altında class klasörlerine ayrılır:

| Renk | Anlam |
|---|---|
| 🟢 Yeşil | TP — doğru tespit |
| 🔴 Kırmızı | FP — yanlış tespit |
| 🟠 Turuncu | FN — kaçırılan |
| ⬜ Beyaz dashed | GT referans |

---

## Test Sonuçları

### CCTV val dataset (145 foto)

| Aşama | falldown F1 | Toplam F1 |
|---|---|---|
| Pipeline (pre-review) | 0.40 | 0.876 |
| **Pipeline + review** | **0.769** | **0.908** |

### 쓰러짐 dataset (107 foto — gerçek dünya)

| Aşama | falldown F1 | Toplam F1 |
|---|---|---|
| Pipeline (pre-review) | 0.412 | 0.876 |

> **Not:** Val dataset pipeline geliştirilirken kullanıldı → skoru referans alma.  
> Gerçek performans ölçütü: yeni, görülmemiş veri (쓰러짐 gibi).

---

## Karar & Sonraki Adımlar

### Durulan Noktalar
- Prompt mühendisliği denemeleri → **durduruldu** (marginal kazanç ~%5)
- Yeni model arama → **durduruldu** (Qwen3-VL yeterli)
- Florence karşılaştırması → **yapıldı**, benzer sonuç, yedekte duruyor

### Sıradaki Adımlar
1. **2000 yeni falldown verisi** pipeline'dan geçir
   - Frame dedup: aynı olay için max 3–5 frame seç
   - FP'leri negatif örnek olarak ekle (boş label dosyası)
2. Optimal config ile son test → paylaş
3. **Fine-tune döngüsü:**

```
Auto-label → Review → Fine-tune → Daha iyi model → Daha az Review
```

---

## Mevcut Prompt (VLM_QUESTION)

```python
"This is a CCTV security camera image. "
"Has a person FALLEN DOWN or is LYING on the ground? "
"Answer Yes ONLY if someone is clearly horizontal: lying flat, collapsed, or sprawled. "
"Answer No ONLY if ALL people are clearly upright — standing, walking, or running normally. "
"Do NOT answer No if anyone is severely bent forward, crouching low, stumbling, "
"leaning heavily, or in an abnormal posture suggesting a fall or loss of balance. "
"A dog, cat or other animal lying on the ground is NOT a fallen person. "
"If the image is too dark or the shape is unclear, answer No. "
"Answer only Yes or No."
```
