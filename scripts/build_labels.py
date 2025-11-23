# scripts/build_labels.py
import os, csv, shutil
from pathlib import Path
import pandas as pd

RAW = Path('data/raw')
OUT_DIR = Path('data/processed/images')
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

# APTOS (example: train.csv with id_code,diagnosis)
aptos_csv = RAW / 'aptos' / 'train.csv'
if aptos_csv.exists():
    df = pd.read_csv(aptos_csv)
    for _, r in df.iterrows():
        img_name = f"{r['id_code']}.png"  # or .jpg depending on files
        src = RAW / 'aptos' / 'train_images' / img_name
        if not src.exists():  # try alternative structure
            src = RAW / 'aptos' / img_name
        if src.exists():
            dst = OUT_DIR / img_name
            shutil.copy(src, dst)
            rows.append({'image': str(dst.relative_to('.')), 'label': int(r['diagnosis'])})

# EyePACS (trainLabels.csv has image,level)
eyepacs_csv = RAW / 'eyepacs' / 'trainLabels.csv'
if eyepacs_csv.exists():
    df = pd.read_csv(eyepacs_csv)
    for _, r in df.iterrows():
        img_name = r['image'] if isinstance(r['image'], str) else f"{r['image']}.png"
        src = RAW / 'eyepacs' / 'train' / img_name
        if not src.exists():
            src = RAW / 'eyepacs' / img_name
        if src.exists():
            dst = OUT_DIR / img_name
            shutil.copy(src, dst)
            rows.append({'image': str(dst.relative_to('.')), 'label': int(r['level'])})

# Write unified CSV
out_csv = Path('data/labels.csv')
pd.DataFrame(rows).to_csv(out_csv, index=False)
print("Wrote", out_csv, "with", len(rows), "rows")
