# Otonom Label Pipeline — label_pipeline.py Detaylı Kılavuz

> **Falldown F1 = 0.761** (TP=27, FP=5, FN=12) @ IoU=0.35  
> **Genel F1 = 0.689** (tüm class'lar)  
> Model: Qwen3-VL-2B + Florence-2-large + YOLO11x + kendi modelimiz (best_04_15.pt)

---

## Pipeline Genel Akışı

```
Her görüntü için:

┌─────────────────────────────────────────────────────┐
│  Aşama 1: VLM Tarama (Qwen3-VL-2B)                 │
│  Prompt: "Has a person FALLEN DOWN or is LYING..."  │
│  Logit tabanlı P(Yes) / P(No) hesabı               │
│  P(Yes) ≥ 0.70 → AUTO-YES                          │
│  P(No)  ≥ 0.70 → AUTO-NO                           │
│  İkisi de < 0.70 → REVIEW (açıklama üret)          │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
      AUTO-YES           AUTO-NO / REVIEW
        │                    │
        ▼                    │
┌───────────────┐            │
│  Aşama 2:     │            │ (falldown bbox üretilmez)
│  Florence     │            │
│  bbox tespiti │            │
│  "a person    │            │
│  lying on     │            │
│  the ground"  │            │
└───────┬───────┘            │
        │                    │
        └──────────┬─────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│  Aşama 3: YOLO11x + Kendi Modelimiz (tüm fotolar)  │
│  YOLO → sadece YOLO_ALLOWED_CLASSES                 │
│  Kendi model → sadece OWN_CUSTOM_CLASSES            │
│  Overlap (IoU ≥ 0.50): custom atılır, YOLO kazanır │
└──────────────────┬──────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│  Aşama 4: Merge + Kaydet                            │
│  Florence falldown + model box'ları birleştir       │
│  Overlap (IoU ≥ 0.60): model box atılır             │
│  AUTO-NO/REVIEW: kendi modelin falldown çıktısı da  │
│  atılır (VLM hayır dedi)                            │
│  REVIEW fotoları → review/review_YYYYMMDD_HHMMSS.json │
└──────────────────┬──────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────┐
│  Aşama 5: GT Karşılaştırma (per-class F1)           │
│  IoU eşiği: 0.35                                    │
│  Log → pipeline_log_YYYYMMDD_HHMMSS.txt            │
└─────────────────────────────────────────────────────┘
```

---

## Neden VLM önce geliyor?

Eski yaklaşımda (eval_hybrid_crop.py) Florence her görüntüde falldown adayı box üretiyordu,
sonra caption filtresiyle eleme yapılıyordu. Sorun: caption filtresi karanlık/bulanık
görüntülerde hallüsinasyon üretiyordu (yatan insan → "panda bear lying").

Yeni yaklaşımda VLM **tam görüntüyü** görür ve "kimse yatmıyor" derse Florence hiç çalışmaz.
Bu sayede:
- FP sayısı dramatik azaldı (YES=62 → YES=32 aynı veri setinde)
- Karanlık görüntülerde daha tutarlı karar
- Florence sadece gerçekten şüpheli görüntülerde çalışıyor → hız avantajı

---

## Konfigürasyon

```python
# Dosya: pipeline/label_pipeline.py

IMAGE_DIR  = Path(r"C:\Users\admin\fur\model_eval\data\full dataset\val\images")
GT_DIR     = Path(r"C:\Users\admin\fur\model_eval\data\full dataset\val\labels_with_name")
OUT_DIR    = Path(r"C:\Users\admin\fur\model_eval\pipeline\pipeline_out")

OWN_MODEL_PATH  = Path(r"...\models\falldown\best_04_15.pt")
OWN_MODEL_CONF  = 0.25
YOLO_MODEL_PATH = Path(r"...\models\coco_yolo\yolo11x.pt")
YOLO_CONF       = 0.25

# Hybrid sınıf stratejisi
YOLO_ALLOWED_CLASSES = {"person", "car", "bus", "truck", "bicycle", "motorcycle", "dog"}
OWN_CUSTOM_CLASSES   = {"boar", "tractor", "tiller", "scooter", "cat"}

IOU_MODEL_MERGE = 0.50   # custom box ↔ YOLO overlap → custom atılır
IOU_MERGE       = 0.60   # Florence falldown ↔ model box overlap → model atılır
IOU_EVAL        = 0.35   # GT eşleşme eşiği — falldown bbox bazen eksik, düşük tutuldu
VLM_THR         = 0.70   # P(Yes) veya P(No) eşiği — altı REVIEW

FLORENCE_PROMPT = "a person lying on the ground"
```

---

## VLM Prompt

```python
VLM_QUESTION = (
    "This is a CCTV security camera image. "
    "Has a person FALLEN DOWN or is LYING on the ground? "
    "Answer Yes ONLY if someone is clearly horizontal: lying flat, collapsed, or sprawled. "
    "Answer No if all people are upright — standing, walking, running "
    "even if they are on concrete, asphalt, or any outdoor surface. "
    "Ignore animals. "
    "Answer only Yes or No."
)
```

**Neden "on the ground" değil "FALLEN DOWN / LYING / horizontal"?**

Eski prompt ("on the ground") bağlamı: VLM asfaltta yürüyen insanları da YES yapıyordu
çünkü onlar da "on the ground" üzerinde. Yeni prompt açıkça yatay/düşmüş pozisyon istiyor.

---

## VLM Logit Tabanlı Sınıflandırma

VLM üretici (generative) moddan değil, ilk token'ın logit'lerinden karar alır:

```python
def vlm_logit(img_pil, question):
    # max_new_tokens=5, output_scores=True
    logits  = out.scores[0][0]           # ilk üretilen token'ın logit'leri
    y_logit = max(logits["Yes"], logits["yes"], logits["YES"])
    n_logit = max(logits["No"],  logits["no"],  logits["NO"])
    probs   = softmax([y_logit, n_logit])
    return probs[0], probs[1]            # P(Yes), P(No)
```

**Avantajı:** Tam cümle üretmekten 10x hızlı. Sadece Yes/No token'ları karşılaştırılıyor.

---

## Hybrid Sınıf Stratejisi

| Sınıf | Kaynak | Neden |
|-------|--------|-------|
| person, car, bus, truck, bicycle, motorcycle, dog | YOLO11x | COCO'da bol veri, güçlü |
| boar, tractor, tiller, scooter, cat | Kendi modelimiz | YOLO11x kaçırıyor / COCO'da yok |
| falldown | Florence + VLM | Poz tespiti gerekiyor, bbox model yeterli değil |

**Çakışma kuralı:**
- Custom box ile YOLO box aynı nesneyi gösteriyorsa (IoU ≥ 0.50) → YOLO kazanır
- Florence falldown box ile model box çakışıyorsa (IoU ≥ 0.60) → Florence kazanır (model falldown'u person olarak etiketlemiş olabilir)

**Cat neden kendi modelde?**
YOLO11x cat için 60 FN veriyordu. Kendi modelimiz cat'i daha iyi buluyor.

---

## Çıktı Yapısı

```
pipeline_out/
├── labels/                          ← final label txt'leri (YOLO format, class isimli)
│   ├── E02_001.txt
│   └── ...
├── review/
│   ├── review_20260417_143022.json  ← her run için ayrı dosya (tarih/saat)
│   ├── E02_003.jpg                  ← REVIEW görüntüsü (bbox overlay)
│   └── ...
└── pipeline_log_20260417_143022.txt ← her run için ayrı log
```

Her run kendi log ve review dosyasını oluşturur — önceki run'lar üzerine yazılmaz.

---

## Label Formatı

```
# class_ismi  cx      cy      width   height
falldown       0.316000 0.799000 0.165000 0.165000
person         0.512570 0.261509 0.027946 0.120450
car            0.582404 0.030992 0.030850 0.042020
cat            0.674000 0.469000 0.088000 0.095000
```

Standart YOLO formatı, `class_id` yerine doğrudan `class_ismi` kullanılıyor
(labels_with_name formatı — labeling aracıyla uyumlu).

---

## Mevcut Sonuçlar (Son Run)

| Sınıf | TP | FP | FN | P | R | F1 |
|-------|----|----|-----|-------|-------|-------|
| bicycle | 9 | 3 | 12 | 0.750 | 0.429 | 0.545 |
| bus | 1 | 0 | 3 | 1.000 | 0.250 | 0.400 |
| car | 121 | 50 | 28 | 0.708 | 0.812 | 0.756 |
| cat | 13 | 8 | 60 | 0.619 | 0.178 | 0.277 |
| dog | 29 | 16 | 10 | 0.644 | 0.744 | 0.690 |
| **falldown** | **27** | **5** | **12** | **0.844** | **0.692** | **0.761** ★ |
| motorcycle | 4 | 0 | 8 | 1.000 | 0.333 | 0.500 |
| person | 102 | 25 | 21 | 0.803 | 0.829 | 0.816 |
| truck | 15 | 9 | 16 | 0.625 | 0.484 | 0.545 |
| **TOPLAM** | **321** | **119** | **171** | **0.730** | **0.652** | **0.689** |

> IoU_EVAL=0.35 — falldown bbox bazen bacakları kapsamıyor, düşük eşik ile borderline TP'ler kurtarıldı.

> FP 5-3=2 1 bbox tam eslesmiyor 2 tane Gtsi olmayan TP var

> 12 FN icinse 7 tanesini reviewdan dusebiliriz

> Son Falldown 34 2 5 0.95 0.87
---

## Falldown FP/FN Analizi

### FP=5 (Gerçek hata sayısı: 2)

| Foto | Sorun | Değerlendirme |
|------|-------|---------------|
| E02_002 | IoU≈0.32, bbox büyük ama aynı kişi | Kabul edilebilir |
| E02_022 | GT etiketi eksik, pipeline doğru bulmuş | Gerçekte TP |
| E02_024 | GT etiketi eksik, pipeline doğru bulmuş | Gerçekte TP |
| E02_035 | Çöp torbaları falldown zannedildi | **Gerçek FP** |
| 어린이 0065 유치원2 | Yatan kedi → Florence "person lying" sandı | **Gerçek FP** |

### FN=12 (Temel nedenler)

| Neden | Fotoğraflar | Açıklama |
|-------|-------------|----------|
| VLM NO dedi, Florence çalışmadı | E02_003, E02_008 | Gerçek falldown atlandı (ikiside Reviewda)| 
| Florence yanlış bölgeye baktı | E02_019, E02_021, E02_035 | Sahnede başka nesne |
| Çok küçük / uzak | 공원 0002 x2, 공원 0008 | Gece/uzak, tespit zor |
| IoU düşük (farklı bbox boyutu) | E02_005, E02_014 | Pred var ama eşleşmedi |

---

## Bilinen Kısıtlamalar

**Cat FN=60 (yüksek):**
Kendi modelimiz de cat'i tam yakalayamıyor. Olası çözüm: YOLO9e (cat/dog performansı daha iyi).

**Bus FN=3:**
GT'deki bus'ların büyük kısmı görüntü kenarında/uzakta, YOLO kaçırıyor.

**Bisiklet ve Truck:**
Çoğu uzak/küçük nesne — GT doğru ama model görmüyor. Öncelik değil.

**VLM hızı:**
2B model kullanılıyor, ~1-2 dk/görüntü. Büyük dataset'lerde yavaş olabilir.

---

## F1 Geliştirme Tarihi

| Aşama | Yöntem | Falldown F1 | Notlar |
|-------|--------|-------------|--------|
| Başlangıç | Florence OVD only | ~0.30 | FP çok yüksek |
| + YOLO cat/dog filtresi | eval_hybrid.py | 0.380 | Ayakta insanlar hâlâ FP |
| + Crop caption filtresi | eval_hybrid_crop.py | 0.685 | Seçilen eski yöntem |
| **+ VLM ön tarama** | **label_pipeline.py v1** | **~0.72** | YES=62, eski prompt |
| **+ Prompt düzeltme** | **label_pipeline.py v2** | **0.761** | "FALLEN DOWN/LYING/horizontal" |
