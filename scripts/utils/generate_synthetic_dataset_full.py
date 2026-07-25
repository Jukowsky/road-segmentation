"""
Generates ~1M synthetic RSCD segmentation images from full COCO binary masks.

Pipeline:
  coco_binary_masks_full/  (118K masks)
  x SAMPLES_PER_MASK variations each
  → masked_rscd_dataset_coco_full/images/   (JPEG)
  → masked_rscd_dataset_coco_full/labels/   (YOLO segmentation .txt)

Resume-safe: skips mask+sample combos whose output files already exist.

Usage:
  cd /home/talt_wireten_c/road-segmentation
  python scripts/utils/generate_synthetic_dataset_full.py

  # Override samples per mask:
  SAMPLES_PER_MASK=5 python scripts/utils/generate_synthetic_dataset_full.py
"""
import os
import sys
import cv2
import random
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).parent))
from main import (
    find_material_folders,
    get_material_file_paths,
    binarize_and_make_gt_mask,
    mix_gt_mask_for_semseg,
    export_to_yolo,
)

# ── Config ────────────────────────────────────────────────────────────────────
MASK_FOLDER    = "/home/talt_wireten_c/road-segmentation/datasets/coco_binary_masks_full"
TEXTURE_ROOT   = "/home/talt_wireten_c/road-segmentation/datasets/rscd/train"
OUTPUT_DIR     = "/home/talt_wireten_c/road-segmentation/masked_rscd_dataset_coco_full"
TARGET_SIZE    = (512, 512)
SAMPLES_PER_MASK = int(os.environ.get("SAMPLES_PER_MASK", 9))
NUM_WORKERS    = max(1, cpu_count() - 1)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False


def _process_mask(args):
    """Worker: generates SAMPLES_PER_MASK variations for a single mask file."""
    mask_path, material_folders, materials_paths, output_dir, samples, target_size = args
    output_path = Path(output_dir)
    generated = 0

    for s_idx in range(samples):
        file_id  = f"{mask_path.stem}_s{s_idx}"
        img_file = output_path / "images" / f"{file_id}.jpg"
        txt_file = output_path / "labels" / f"{file_id}.txt"

        if img_file.exists() and txt_file.exists():
            continue  # resume: already done

        gt_mask, _ = binarize_and_make_gt_mask(mask_path, material_folders, target_size)
        if gt_mask is None:
            continue

        synth_img = mix_gt_mask_for_semseg(gt_mask, materials_paths)
        cv2.imwrite(str(img_file), synth_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        export_to_yolo(gt_mask, txt_file, target_size[0], target_size[1])
        generated += 1

    return generated


def main():
    output_path = Path(OUTPUT_DIR)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)

    if not Path(MASK_FOLDER).exists():
        print(f"[ERROR] Mask folder not found: {MASK_FOLDER}")
        print("Run generate_coco_binary_masks_full.py first.")
        sys.exit(1)

    if not Path(TEXTURE_ROOT).exists():
        print(f"[ERROR] Texture root not found: {TEXTURE_ROOT}")
        print("Transfer the full RSCD dataset before running this script.")
        sys.exit(1)

    material_folders = find_material_folders(TEXTURE_ROOT)
    materials_paths  = get_material_file_paths(material_folders)
    mask_files = sorted(Path(MASK_FOLDER).glob("*.png"))

    already_done = len(list((output_path / "images").glob("*.jpg")))
    total_expected = len(mask_files) * SAMPLES_PER_MASK
    print(f"Masks            : {len(mask_files)}")
    print(f"Samples per mask : {SAMPLES_PER_MASK}")
    print(f"Expected total   : {total_expected:,}")
    print(f"Already generated: {already_done:,}")
    print(f"Workers          : {NUM_WORKERS}")
    print()

    args_list = [
        (mf, material_folders, materials_paths, OUTPUT_DIR, SAMPLES_PER_MASK, TARGET_SIZE)
        for mf in mask_files
    ]

    total_new = 0
    with Pool(NUM_WORKERS) as pool:
        iterator = pool.imap_unordered(_process_mask, args_list, chunksize=16)
        if USE_TQDM:
            iterator = tqdm(iterator, total=len(mask_files), unit="mask")
        for count in iterator:
            total_new += count

    final_count = len(list((output_path / "images").glob("*.jpg")))
    print(f"\nDone. New images this run : {total_new:,}")
    print(f"Total images in dataset  : {final_count:,}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
