"""
Generates binary masks from ALL COCO train2017 annotations (no image limit).
Skips already-generated files so the script is safe to resume after interruption.

Output: datasets/coco_binary_masks_full/
"""
import json
import numpy as np
import cv2
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

COCO_ANNOTATIONS = "/home/talt_wireten_c/road-segmentation/datasets/coco/annotations/instances_train2017.json"
OUTPUT_MASK_DIR  = "/home/talt_wireten_c/road-segmentation/datasets/coco_binary_masks_full"
TARGET_SIZE      = (480, 360)

Path(OUTPUT_MASK_DIR).mkdir(parents=True, exist_ok=True)

coco = COCO(COCO_ANNOTATIONS)
img_ids = coco.getImgIds()  # all ~118K images, no limit

existing = {p.stem for p in Path(OUTPUT_MASK_DIR).glob("*.png")}
img_ids_todo = [
    iid for iid in img_ids
    if Path(coco.loadImgs(iid)[0]["file_name"]).stem not in existing
]

print(f"Total COCO images : {len(img_ids)}")
print(f"Already generated : {len(existing)}")
print(f"To process        : {len(img_ids_todo)}")

iterator = tqdm(img_ids_todo, unit="img") if USE_TQDM else img_ids_todo
skipped = 0

for img_id in iterator:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids  = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns     = coco.loadAnns(ann_ids)

    if not anns:
        skipped += 1
        continue

    h, w = img_info["height"], img_info["width"]
    canvas = np.full((h, w), 128, dtype=np.uint8)  # 128 = ignored region

    for ann in anns:
        class_id   = ann["category_id"]
        binary_val = 255 if (class_id % 2 == 1) else 0

        if isinstance(ann["segmentation"], list):
            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        else:
            rle  = coco.annToRLE(ann)
            mask = maskUtils.decode(rle)

        canvas[mask == 1] = binary_val

    canvas_resized = cv2.resize(canvas, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    out_name = Path(img_info["file_name"]).stem + ".png"
    cv2.imwrite(str(Path(OUTPUT_MASK_DIR) / out_name), canvas_resized)

total_generated = len(list(Path(OUTPUT_MASK_DIR).glob("*.png")))
print(f"\nDone. Masks in {OUTPUT_MASK_DIR}: {total_generated} (skipped no-annotation: {skipped})")
