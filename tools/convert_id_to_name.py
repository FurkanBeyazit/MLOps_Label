from pathlib import Path
from tqdm import tqdm

NAME_TO_ID = {
    "person": 0,
    "car": 1,
    "falldown": 2,
    "bus": 3,
    "truck": 4,
    "bicycle": 5,
    "motorcycle": 6,
    "boar": 7,
    "tractor": 8,
    "scooter": 9,
    "tiller": 10,
    "cat": 11,
    "dog":12
}

# YENİLİK 1: Sözlüğü tam tersine (ID -> İsim) çeviriyoruz. 
# Dosyadan okunacağı için ID'leri string ("0", "1" vb.) olarak ayarlıyoruz.
ID_TO_NAME = {str(v): k for k, v in NAME_TO_ID.items()}

TARGET_DATASETS = ["full dataset"]

def convert_ids_to_names(root: Path, mapping: dict):
    for dataset in TARGET_DATASETS:
        print(f"[{dataset}]")
        for subset in["train", "val"]:
            
            # YENİLİK 2: Kaynak (src) ve Hedef (dst) klasörler yer değiştirdi.
            src_label_dir = root / dataset / subset / "labels"
            dst_label_dir = root / dataset / subset / "labels_with_name_new"

            if not src_label_dir.is_dir():
                continue
            
            # Hedef klasörü yoksa oluştur
            dst_label_dir.mkdir(parents=True, exist_ok=True)

            files = list(src_label_dir.glob("*.txt"))
            if not files:
                continue

            for file in tqdm(files, desc=f"  {subset}", unit="file"):
                dst_path = dst_label_dir / file.name
                with file.open("r") as fr, dst_path.open("w") as fw:
                    for line in fr:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        # YENİLİK 3: Satırın başındaki ID'yi alıp isme çeviriyoruz
                        cls_id = parts[0] # Örn: "7"
                        name = mapping.get(cls_id) # Ters sözlükten bul: "boar"
                        
                        # Eğer ID sözlükte yoksa (örn: yorum satırı olmuş helmet) atla
                        if name is None:
                            continue
                        
                        # Rakam yerine isimi koyarak satırı yeni dosyaya yaz (Örn: boar 0.5 0.5 0.2 0.2)
                        fw.write(" ".join([name] + parts[1:]) + "\n")
        print()

if __name__ == "__main__":
    root = Path(".")
    # YENİLİK 4: Fonksiyonu yeni ters sözlüğümüz ile çağırıyoruz
    convert_ids_to_names(root, ID_TO_NAME)