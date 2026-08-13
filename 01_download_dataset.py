# ============================================================
# STEP 1: Setup — run this cell first in Google Colab
# ============================================================
# Colab has open internet access, unlike a locked-down sandbox,
# so this is where the actual image downloading happens.

!pip install -q open_clip_torch torch torchvision pillow requests chromadb tqdm

import csv
import io
import os
import requests
from PIL import Image
from tqdm import tqdm

# Upload sarees.csv to Colab's file browser (left sidebar) before running this.
CSV_PATH = "sarees.csv"
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

rows = []
with open(CSV_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r["image_url"].strip():
            rows.append(r)

print(f"Total products: {len(rows)}")

# Download every image locally. We keep a local copy because:
# 1. Embedding needs the actual pixels, not just the URL
# 2. Re-downloading 1000+ images every time you test is slow and unreliable
downloaded = []
failed = []

for row in tqdm(rows, desc="Downloading images"):
    sku = row["SKU"]
    url = row["image_url"]
    local_path = os.path.join(IMAGE_DIR, f"{sku}.webp")

    if os.path.exists(local_path):
        downloaded.append(row)
        continue

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        # Validate it's actually a readable image before saving
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(local_path, "WEBP")
        downloaded.append(row)
    except Exception as e:
        failed.append((sku, str(e)))

print(f"Downloaded: {len(downloaded)}")
print(f"Failed: {len(failed)}")
if failed:
    print("First few failures:", failed[:5])
