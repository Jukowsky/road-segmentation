"""
Verifies that masked_rscd_dataset_coco_full's train/val split is disjoint.

Checks two things:
  1. Exact-file overlap between train.txt and val_split.txt (must be 0 —
     no image is used for both training and validation).
  2. Mask-shape overlap: how many of the underlying COCO mask geometries
     (the part of the filename before "_sN") appear on both sides. Since
     each sample's class assignment and texture crop are re-randomized
     independently (see main.py: binarize_and_make_gt_mask /
     mix_gt_mask_for_semseg), a shared shape is not a shared image — this
     is reported for transparency, not as a failure condition.

Usage:
  cd /home/talt_wireten_c/road-segmentation
  python scripts/utils/verify_dataset_split.py > docs/split_verification_report.txt
"""
import re
from collections import defaultdict
from pathlib import Path

DATASET_ROOT = Path("/home/talt_wireten_c/road-segmentation/masked_rscd_dataset_coco_full")
TRAIN_LIST = DATASET_ROOT / "train.txt"
VAL_LIST = DATASET_ROOT / "val_split.txt"

STEM_RE = re.compile(r"(\d+)_s(\d+)\.jpg")


def load(list_path, strip_prefix=None):
    files, stems = set(), defaultdict(set)
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name = line
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix):]
            files.add(name)
            m = STEM_RE.search(name)
            if m:
                stems[m.group(1)].add(int(m.group(2)))
    return files, stems


def main():
    train_files, train_stems = load(TRAIN_LIST)
    val_files, val_stems = load(VAL_LIST, strip_prefix="./val_split/")

    print("=== Exact-file overlap (train.txt vs val_split.txt) ===")
    print(f"train.txt images     : {len(train_files):,}")
    print(f"val_split.txt images : {len(val_files):,}")
    shared_files = train_files & val_files
    print(f"Shared files         : {len(shared_files):,}")
    print("PASS: 0 shared files" if not shared_files else "FAIL: shared files found")

    print()
    print("=== Mask-shape overlap (disclosure, not a pass/fail check) ===")
    train_shape_ids = set(train_stems)
    val_shape_ids = set(val_stems)
    shared_shapes = train_shape_ids & val_shape_ids
    print(f"Unique mask shapes in train.txt     : {len(train_shape_ids):,}")
    print(f"Unique mask shapes in val_split.txt : {len(val_shape_ids):,}")
    print(f"Shapes present in both              : {len(shared_shapes):,}")

    same_sample_index = 0
    for shape_id in shared_shapes:
        if train_stems[shape_id] & val_stems[shape_id]:
            same_sample_index += 1
    print(f"Shared shapes with the SAME _sN sample index on both sides: {same_sample_index:,}")
    print("(should be 0 — each sample index is used on exactly one side of the split)")


if __name__ == "__main__":
    main()
