# src/training/preprocess.py
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
RAW_DIR = os.path.expanduser("D:/drd/data/raw/aptos/aptos2019-blindness-detection")
OUT_DIR = os.path.expanduser("D:/drd/data/processed/aptos/images")
OUT_CSV = os.path.expanduser("D:/drd/data/processed/aptos/train_labels.csv")
IMG_SIZE = 224
# ----------------

os.makedirs(OUT_DIR, exist_ok=True)

# helper: circular crop + resize + CLAHE
def preprocess_image(in_path, out_path, size=IMG_SIZE):
    img = cv2.imread(in_path)
    if img is None:
        return False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Find center and radius for circular crop
    cx, cy = w//2, h//2
    r = min(cx, cy)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), int(r*0.95), 255, -1)
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    # crop to square bounding box around circle
    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r
    crop = img_masked[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

    if crop.size == 0:
        crop = cv2.resize(img, (size, size))
    else:
        crop = cv2.resize(crop, (size, size))

    # Convert to YCrCb and apply CLAHE on the luminance channel
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2,a,b))
    final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # save as PNG
    final_bgr = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, final_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return True

def run_preprocess():
    train_csv = os.path.join(RAW_DIR, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} not found. Check dataset path.")
    df = pd.read_csv(train_csv)
    # aptos id_code column typically contains image name without extension
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        img_id = r['id_code']
        label = int(r['diagnosis'])
        possible = os.path.join(RAW_DIR, "train_images", f"{img_id}.png")
        if not os.path.exists(possible):
            # sometimes extension is .jpg
            possible = os.path.join(RAW_DIR, "train_images", f"{img_id}.jpg")
        if not os.path.exists(possible):
            # skip if missing
            print("missing:", img_id)
            continue
        out_name = f"{img_id}.png"
        out_path = os.path.join(OUT_DIR, out_name)
        ok = preprocess_image(possible, out_path)
        if ok:
            rows.append({"image": out_name, "label": label})
    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)
    print("Processed images in:", OUT_DIR)

if __name__ == "__main__":
    run_preprocess()
