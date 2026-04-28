# Rex-Omni Kurulum Notları

## Ortam
- Windows 10
- Python 3.10
- CUDA 12.4 / PyTorch 2.6.0+cu124

## Kurulum Adımları

### 1. Python 3.10 kur
python.org/downloads/release/python-3109 → Windows installer (64-bit)

### 2. Venv oluştur
```bash
py -3.10 -m venv rex_env
rex_env\Scripts\activate
```

### 3. PyTorch kur
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

### 4. flash-attn kur (prebuilt wheel — source'dan build etme, PC donar)
GitHub Releases sayfasından `flash_attn-2.7.4-...-win_amd64.whl` indir:
https://github.com/Dao-AILab/flash-attention/releases

```bash
pip install packaging wheel
pip install flash_attn-2.7.4*.whl
```

> ⚠️ `pip install flash-attn` veya `--no-build-isolation` ile source'dan build etme —
> Windows'ta MSVC/CUTLASS uyumsuzluğu nedeniyle derleme patlar ve PC donar.

### 5. Rex-Omni kur
```bash
git clone https://github.com/IDEA-Research/Rex-Omni.git
cd Rex-Omni
pip install qwen-vl-utils
pip install accelerate==1.10.1 gradio==4.44.1 gradio_image_prompter==0.1.0 matplotlib==3.10.6 pydantic==2.10.6 transformers==4.51.3 numpy==1.26.4 "Pillow==10.4.0" triton-windows
pip install -v -e . --no-deps
```

> Not: rex-omni flash-attn==2.7.4.post1 ister ama 2.7.4 ile çalışıyor.

### 6. Sonraki kullanımda aktifleştirme
```bash
rex_env\Scripts\activate
```

## Tam bağımlılık listesi
Bkz: `rex_requirements.txt`
