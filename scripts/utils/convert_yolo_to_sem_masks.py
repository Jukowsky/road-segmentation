"""
Convert YOLO polygon label files → single-channel semantic PNG masks.

Source layout:
  <dataset>/images/              ← train images (.jpg)
  <dataset>/labels/              ← train labels (.txt, YOLO polygon format)
  <dataset>/val_split/images/    ← val images
  <dataset>/val_split/labels/    ← val labels

Output (Ultralytics auto-derives mask path by replacing 'images' → 'masks'):
  <dataset>/masks/               ← train semantic masks (.png)
  <dataset>/val_split/masks/     ← val semantic masks   (.png)

Mask encoding:
  pixel value = class_id (0–26 for the 27 RSCD classes)
  pixel value = 255       → ignore / background (no annotation)

Resume-safe: skips files that already have a .png in the output dir.
"""
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

DATASET_ROOT = Path("/home/talt_wireten_c/road-segmentation/masked_rscd_dataset_coco_full")

SPLITS = [
    {
        "images_dir": DATASET_ROOT / "images",
        "labels_dir": DATASET_ROOT / "labels",
        "masks_dir":  DATASET_ROOT / "masks",
    },
    {
        "images_dir": DATASET_ROOT / "val_split" / "images",
        "labels_dir": DATASET_ROOT / "val_split" / "labels",
        "masks_dir":  DATASET_ROOT / "val_split" / "masks",
    },
]

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def process_one(args):
    label_path, images_dir, masks_dir = args
    stem = label_path.stem
    mask_out = masks_dir / (stem + ".png")
    if mask_out.exists():
        return "skip"

    img_path = None
    for ext in IMG_EXTS:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        return "no_image"

    img = cv2.imread(str(img_path))
    if img is None:
        return "unreadable"
    h, w = img.shape[:2]

    canvas = np.full((h, w), 255, dtype=np.uint8)

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            cv2.fillPoly(canvas, [pts.astype(np.int32)], class_id)

    cv2.imwrite(str(mask_out), canvas)
    return "ok"


def convert_split(split_cfg):
    labels_dir = split_cfg["labels_dir"]
    images_dir = split_cfg["images_dir"]
    masks_dir  = split_cfg["masks_dir"]
    masks_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"\n[{labels_dir}]  {len(label_files):,} files → {masks_dir}")
    if not label_files:
        print("  No label files found, skipping.")
        return

    args = [(lf, images_dir, masks_dir) for lf in label_files]
    workers = max(1, cpu_count() - 2)
    counts = {"ok": 0, "skip": 0, "no_image": 0, "unreadable": 0}

    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, args, chunksize=500)):
            counts[result] = counts.get(result, 0) + 1
            if (i + 1) % 50_000 == 0:
                print(f"  {i+1:,}/{len(label_files):,} …")

    print(f"  ok={counts['ok']:,}  skipped={counts['skip']:,}  "
          f"no_image={counts['no_image']:,}  unreadable={counts['unreadable']:,}")


if __name__ == "__main__":
    for split in SPLITS:
        convert_split(split)
    print("\nAll splits done.")
