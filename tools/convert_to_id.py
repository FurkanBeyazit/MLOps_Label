from pathlib import Path
from tqdm import tqdm
import os
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

TARGET_DATASETS = [
    "full dataset",
]
def get_long_path(p: Path):
    
    path_str = str(p.resolve())
    prefix = "\\\\?\\"
    # Sadece Windows (nt) sistemlerinde ve zaten eklenmemişse ekle
    if os.name == 'nt' and not path_str.startswith(prefix):
        path_str = prefix + path_str
    return path_str
def convert_names_to_ids(root: Path, mapping: dict):
    for dataset in TARGET_DATASETS:
        print(f"[{dataset}]")
        for subset in ["train", "val"]:
            src_label_dir = root / dataset / subset / "labels_with_name"
            dst_label_dir = root / dataset / subset / "labels"
            dst_label_dir.mkdir(parents=True, exist_ok=True)

            if not src_label_dir.is_dir():
                continue

            files = list(src_label_dir.glob("*.txt"))
            if not files:
                continue

            for file in tqdm(files, desc=f"  {subset}", unit="file"):
                dst_path = dst_label_dir / file.name
                safe_src=get_long_path(file)
                safe_dst=get_long_path(dst_path)
                with open(safe_src,"r",encoding="utf-8") as fr, open(safe_dst,"w",encoding="utf-8") as fw:
                    for line in fr:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        name = parts[0]
                        cls_id = mapping.get(name)
                        if cls_id is None:
                            continue
                        fw.write(" ".join([str(cls_id)] + parts[1:]) + "\n")
        print()

if __name__ == "__main__":
    root = Path(".")
    convert_names_to_ids(root, NAME_TO_ID)