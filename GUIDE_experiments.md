# Zero-Shot Falldown Tespiti — Tüm Denemeler ve Sonuçlar

> **Amaç:** CCTV görüntülerinde düşme (falldown) tespiti için auto-labeling pipeline geliştirmek.
> GT label formatı: `2 cx cy w h` veya `falldown cx cy w h` (normalized)
> Değerlendirme metriği: **Falldown F1 @ IoU=0.35** (genel pipeline: tüm class F1 de ölçülüyor)

---

## Kısa Karşılaştırma Tablosu

| Sıra | Dosya | Yöntem | Falldown F1 | Notlar |
|------|-------|--------|-------------|--------|
| 1 | `eval_florence.py` | Florence OPEN_VOC — tek prompt | ~0.30 | Yüksek recall, çok fazla FP |
| 2 | `eval_florence2.py` | Florence CAPTION_TO_PHRASE_GROUNDING | ~0.218 | Model ayrım yapamıyor |
| 3 | `eval_florence3.py` | Scene caption → keyword → OPEN_VOC | 0.395 | Çok fazla SKIP, FN yüksek |
| 4 | `eval_florence4.py` | CTPG + hint caption | ~0.218 | Her kareye box atıyor |
| 5 | `eval_hybrid.py` | Florence OVD + YOLO cat/dog filtresi | 0.380 | Ayakta insanlar hâlâ FP |
| 6 | `eval_hybrid_grounding.py` | Florence OVD + CTPG "standing" filtresi | ~0.218 | TP'leri de eliyor |
| 7 | `eval_dino.py` | GroundingDINO SwinB — 3 prompt | 0.531 | Precision iyi, recall düşük |
| 8 | `eval_hybrid_crop.py` | Florence OVD + YOLO + crop caption | 0.685 | Eski seçilen yöntem |
| **9** | **`label_pipeline.py`** | **VLM (Qwen3VL) + Florence + YOLO11x + kendi model** | **0.761** | **Güncel yöntem — tüm class'lar** |

---

## Florence-2 Denemeleri

### Deneme 1 — OPEN_VOCABULARY_DETECTION (`eval_florence.py`)

**Ne yapıyor:** Florence-2'ye doğrudan "a person lying on the ground" gibi bir prompt verip
`<OPEN_VOCABULARY_DETECTION>` taskı ile box çıkarıyoruz. Tüm dönen box'lar falldown sayılıyor.

**Test edilen promptlar:**
```
"a person lying on the ground"           ← en iyi
"a person fallen on the floor"
"person collapsed on the ground"
"a person lying motionless on the ground"
"human lying on ground"
"a person who has fallen down"
"a person lying flat on the floor"
# Multi-class (sadece falldown keyword'lü label'lar alınıyor):
"a person lying on the ground, cat, standing person"
"a person lying on the ground, cat, person standing, person walking"
"a person fallen, cat, person standing upright, person walking"
```

**Sonuç:** F1 ≈ 0.30  
**Sorun:** Florence her görüntüde "lying person" bulma eğiliminde — ayakta duran, oturan,
hatta hayvan olan her şeyi falldown olarak etiketliyor. FP çok yüksek.

---

### Deneme 2 — CAPTION_TO_PHRASE_GROUNDING (`eval_florence2.py`)

**Ne yapıyor:** Florence'a bir caption metin veriyoruz (`"lying person, cat, standing person"`),
model bu metindeki phrase'leri görüntüdeki bölgelere ground ediyor.

**Mantık:** Multi-class prompt verince "standing" phrase'i standing kişiye ground edilirse,
o box'ları FALLDOWN_KEYWORDS ile filtreleyip sadece "lying/fallen" label'lı box'ları tutabiliriz.

**Sorun:** CAPTION_TO_PHRASE_GROUNDING **sliding-window gibi davranıyor** — her phrase'i
görüntünün bir yerine eşliyor, gerçek içerikten bağımsız. "standing person" phrase'i
yatan kişiye ground ediliyor, "lying person" ayaktakine eşlenebiliyor.

**Sonuç:** F1 ≈ 0.218

---

### Deneme 3 — İki Aşamalı: Scene Caption → OVD (`eval_florence3.py`)

**Ne yapıyor:**
1. `<MORE_DETAILED_CAPTION>` ile tüm sahneye bakıp caption üret
2. Caption'da falldown keyword var mı kontrol et (`lying`, `fallen`, `on the ground`...)
3. Varsa → `<OPEN_VOCABULARY_DETECTION>` ile box bul; yoksa → bu frame'i atla (SKIP)

**Mantık:** Önce "bu sahnede düşme var mı?" filtresi uygula, varsa box bul — yoksa
hiç tahmin yapma (FP'yi azaltır).

**Sonuç:** TP=16, FP=25, FN=24, **F1=0.395**

**Sorun:** Pek çok görüntüde caption "lying" içermiyor (Florence sahneye baktığında
bazen yatan kişiyi görmüyor), bu frame'ler SKIP geçiliyor → FN çok yüksek.
Log: `eval_detail.txt`

---

### Deneme 4 — CTPG Hint Caption (`eval_florence4.py`)

**Ne yapıyor:** Gerçek caption üretmek yerine sabit bir "hint" veriyoruz:
`"there might be a lying person on the ground"` → CTPG bunu görüntüye ground etmeye çalışıyor.

**Test edilen hint'ler:**
```
"there might be a lying person on the ground"
"a lying person on the ground"
"a person has fallen on the ground"
"a person is lying on the floor unconscious"
"there is a fallen person lying on the ground near other people"
```

**Sorun:** CTPG her frame'de bir şeyler bulunca hint'e ground ediyor → her kareye box atıyor,
precision çöküyor.

**Sonuç:** F1 ≈ 0.218

---

### Deneme 5 — Hybrid: Florence OVD + YOLO Cat/Dog Filtresi (`eval_hybrid.py`)

**Ne yapıyor:**
1. Florence OPEN_VOC → falldown candidate box'lar
2. YOLO (`yolo26x.pt`, conf=0.5, classes=[cat=15, dog=16]) çalıştır
3. Florence box'ı ile YOLO box'ı IoU≥0.30 örtüşüyorsa → FP say, at

**Mantık:** Florence bazen kedi/köpeği yatan insan sanıyor. YOLO COCO modeliyle
cat/dog tespit et, Florence'ın bunlarla örtüşen box'larını ele.

**Sonuç:** **F1=0.380**

**Sorun:** Cat/dog FP'leri azaldı ama ayakta duran insanları hâlâ falldown sanıyor.
YOLO person filtrelemesi (class=0) eklenince TP'ler de kayboluyor.

---

### Deneme 6 — Hybrid: Florence + CTPG "Standing" Filtresi (`eval_hybrid_grounding.py`)

**Ne yapıyor:**
1. Florence OPEN_VOC → candidate
2. YOLO cat/dog filtresi
3. Full image üzerinde CTPG ile `"standing person, lying person on the ground"` → "standing" label'lı box'lar
4. Florence box'larından standing box'larla IoU≥0.40 olanları at

**Mantık:** CTPG'yi falldown detection için değil, ayakta duranları tespit etmek için kullan.

**Sorun:** CTPG burada da tutarsız — gerçek yatan kişileri de "standing" olarak label'lıyor,
bunlar da eleniyor → TP kaybı.

**Sonuç:** F1 ≈ 0.218 (daha kötü)

---

## GroundingDINO Denemeleri (`eval_dino.py`)

**Model:** GroundingDINO SwinB (`groundingdino_swinb_cogcoor.pth`)  
**Kütüphane:** `groundeddino_vl`  
**Ayarlar:** box_thr=0.6, text_thr=0.6, iou_thr=0.45

GroundingDINO text-conditioned detection yapıyor — verilen metin prompt'una göre box bulur.
Negative class desteği: `"class_a . class_b . class_c"` formatı ile birden fazla class.

### Prompt 1: Sadece Falldown
```
"a person lying on the ground"
```
| TP | FP | FN | Precision | Recall | F1 |
|----|----|----|-----------|--------|----|
| 19 | 14 | 20 | 0.576 | 0.487 | **0.528** |

### Prompt 2: + Cat + Standing Person
```
"a person lying on the ground . cat . standing person"
```
| TP | FP | FN | Precision | Recall | F1 |
|----|----|----|-----------|--------|----|
| 17 | 8 | 22 | 0.680 | 0.436 | **0.531** |

Ek istatistikler:
- [cat] TP=34, FP=6, FN=39
- [standing person] TP=40, FP=1, FN=83

### Prompt 3: + Cat + Person Standing + Person Walking
```
"a person lying on the ground . cat . person standing . person walking"
```
| TP | FP | FN | Precision | Recall | F1 |
|----|----|----|-----------|--------|----|
| 14 | 7 | 25 | 0.667 | 0.359 | **0.467** |

**Değerlendirme:**
- DINO Prompt 2 en iyi: F1=0.531
- Precision iyi (0.680) ama recall çok düşük (0.436) — birçok falldown'ı atlıyor
- "standing person" eklenmesi FP'yi 14→8'e düşürdü ama TP'yi de 19→17'ye düşürdü
- "person walking" eklenmesi işleri daha da kötüleştirdi — standingden çok TP kaybediyor
- DINO tek başına F1=0.531, Florence hybrid pipeline F1=0.685 → **Florence kazandı**

---

## Sonuç: Seçilen Yöntem

**`eval_hybrid_crop.py`** — F1=**0.685**

Florence OPEN_VOC yüksek recall sağlarken, crop caption filtresi FP'leri semantik olarak
eliyor. Detaylar için `GUIDE_pipeline.md`.
