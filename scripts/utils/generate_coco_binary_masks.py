# generate_coco_binary_masks.py
import json
import numpy as np
import cv2
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

COCO_ANNOTATIONS = "/home/talt_wireten_c/road-segmentation/datasets/coco/annotations/instances_train2017.json"
OUTPUT_MASK_DIR  = "/home/talt_wireten_c/road-segmentation/datasets/coco_binary_masks"
TARGET_SIZE      = (480, 360)   # match Bad Apple frame size for fair comparison
MAX_IMAGES       = 6572         # match Bad Apple frame count for fair comparison

Path(OUTPUT_MASK_DIR).mkdir(parents=True, exist_ok=True)
coco = COCO(COCO_ANNOTATIONS)
img_ids = coco.getImgIds()[:MAX_IMAGES]

print(f"Processing {len(img_ids)} COCO images...")

for img_id in img_ids:
    img_info = coco.loadImgs(img_id)[0]
    ann_ids  = coco.getAnnIds(imgIds=img_id, iscrowd=False)
    anns     = coco.loadAnns(ann_ids)

    if not anns:
        continue

    h, w = img_info["height"], img_info["width"]
    canvas = np.full((h, w), 128, dtype=np.uint8)  # 128=ignored

    for ann in anns:
        class_id  = ann["category_id"]
        binary_val = 255 if (class_id % 2 == 1) else 0  # mod(class_id, 2)

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

print(f"Done. Masks saved to {OUTPUT_MASK_DIR}")