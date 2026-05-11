"""
Creates a test set where every image has a single full-image polygon
assigned to either class 0 (black region) or class 1 (white region).
"""
import os
import shutil
import random
import numpy as np
import cv2
from pathlib import Path

SRC_IMAGES   = "masked_rscd_dataset_coco/images"
OUTPUT_DIR   = "solid_label_testset"
NUM_IMAGES   = 100
random.seed(42)

(Path(OUTPUT_DIR) / "images").mkdir(parents=True, exist_ok=True)
(Path(OUTPUT_DIR) / "labels").mkdir(parents=True, exist_ok=True)

img_files = random.sample(list(Path(SRC_IMAGES).glob("*.jpg")), NUM_IMAGES)

for img_path in img_files:
    img = cv2.imread(str(img_path))
    if img is None: continue
    h, w = img.shape[:2]

    # Copy image
    shutil.copy(str(img_path), f"{OUTPUT_DIR}/images/{img_path.name}")

    # Create label: single polygon = full image rectangle, class 0
    # Normalized corners of the full image
    polygon = "0 0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0"
    label_path = f"{OUTPUT_DIR}/labels/{img_path.stem}.txt"
    with open(label_path, "w") as f:
        f.write(polygon + "\n")

print(f"Created {NUM_IMAGES} solid-label images in {OUTPUT_DIR}/")