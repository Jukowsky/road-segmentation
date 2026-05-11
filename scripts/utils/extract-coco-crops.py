"""
COCO Instance Extractor for Classification
-------------------------------------------
Crops each annotated object from COCO images using ground truth bounding boxes
and organizes them into a classification dataset structure:

output_dir/
├── train/
│   ├── person/
│   ├── car/
│   └── ... (80 classes)
└── val/
    ├── person/
    └── ...

Usage:
    python extract_coco_crops.py \
        --images_dir /path/to/coco/images \
        --annotations_dir /path/to/coco/annotations \
        --output_dir /path/to/output \
        --padding 0.1 \
        --min_size 32
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image


def load_coco_annotations(ann_file):
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # Build category id -> name mapping
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Build image id -> file_name mapping
    images = {img["id"]: img["file_name"] for img in coco["images"]}

    print(f"  Found {len(images)} images, {len(coco['annotations'])} annotations, {len(categories)} categories")
    return coco["annotations"], images, categories


def extract_crops(
    images_dir,
    annotations,
    images_map,
    categories_map,
    output_split_dir,
    padding=0.1,
    min_size=32,
):
    """
    Crop each annotated bounding box from COCO images and save by class.

    Args:
        padding: Fractional padding to add around bbox (0.1 = 10%)
        min_size: Minimum width/height in pixels to keep a crop
    """
    skipped = 0
    saved = 0
    errors = 0

    # Track per-class counts for naming
    class_counters = {}

    total = len(annotations)
    print(f"Processing {total} annotations...")

    for i, ann in enumerate(annotations):
        if i % 10000 == 0:
            print(f"  [{i}/{total}] saved={saved}, skipped={skipped}")

        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]  # [x, y, width, height] in COCO format

        # Skip crowd annotations
        if ann.get("iscrowd", 0):
            skipped += 1
            continue

        # Get class name and image filename
        class_name = categories_map.get(category_id)
        file_name = images_map.get(image_id)

        if not class_name or not file_name:
            skipped += 1
            continue

        # Parse bbox
        x, y, w, h = bbox
        if w < min_size or h < min_size:
            skipped += 1
            continue

        # Load image
        img_path = Path(images_dir) / file_name
        if not img_path.exists():
            errors += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size

            # Add padding around bbox
            pad_x = w * padding
            pad_y = h * padding

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_w, x + w + pad_x)
            y2 = min(img_h, y + h + pad_y)

            # Skip if crop is too small after padding clamp
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                skipped += 1
                continue

            crop = img.crop((x1, y1, x2, y2))

            # Build output path: output_dir/split/class_name/
            class_dir = Path(output_split_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Unique filename using annotation id
            ann_id = ann.get("id", i)
            out_path = class_dir / f"{ann_id}.jpg"

            crop.save(out_path, "JPEG", quality=95)
            saved += 1

        except Exception as e:
            errors += 1
            continue

    return saved, skipped, errors


def print_dataset_summary(output_dir):
    print("\n📊 Dataset Summary:")
    for split in ["train", "val"]:
        split_dir = Path(output_dir) / split
        if not split_dir.exists():
            continue
        total = 0
        class_counts = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.jpg")))
                class_counts[class_dir.name] = count
                total += count
        print(f"\n  {split}: {total} total crops across {len(class_counts)} classes")
        print(f"  {'Class':<20} {'Count':>8}")
        print(f"  {'-'*30}")
        for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {cls:<20} {cnt:>8}")
        if len(class_counts) > 10:
            print(f"  ... and {len(class_counts) - 10} more classes")


def main():
    parser = argparse.ArgumentParser(description="Extract COCO crops for classification")
    parser.add_argument("--images_dir", required=True, help="Path to COCO images root (contains train2017/, val2017/)")
    parser.add_argument("--annotations_dir", required=True, help="Path to COCO annotations dir (contains instances_train2017.json etc.)")
    parser.add_argument("--output_dir", required=True, help="Output directory for cropped classification dataset")
    parser.add_argument("--padding", type=float, default=0.1, help="Fractional bbox padding (default: 0.1)")
    parser.add_argument("--min_size", type=int, default=32, help="Minimum crop size in pixels (default: 32)")
    args = parser.parse_args()

    splits = [
        ("train", "train2017", "instances_train2017.json"),
        ("val",   "val2017",   "instances_val2017.json"),
    ]

    for split_name, images_subdir, ann_file in splits:
        ann_path = Path(args.annotations_dir) / ann_file
        images_path = Path(args.images_dir) / images_subdir
        output_split = Path(args.output_dir) / split_name

        if not ann_path.exists():
            print(f"⚠️  Annotation file not found, skipping {split_name}: {ann_path}")
            continue

        if not images_path.exists():
            print(f"⚠️  Images directory not found, skipping {split_name}: {images_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing split: {split_name}")
        print(f"{'='*50}")

        annotations, images_map, categories_map = load_coco_annotations(ann_path)

        saved, skipped, errors = extract_crops(
            images_dir=images_path,
            annotations=annotations,
            images_map=images_map,
            categories_map=categories_map,
            output_split_dir=output_split,
            padding=args.padding,
            min_size=args.min_size,
        )

        print(f"\n✅ {split_name} done: saved={saved}, skipped={skipped}, errors={errors}")

    print_dataset_summary(args.output_dir)
    print(f"\n🎉 Dataset ready at: {args.output_dir}")
    print("\nTo train with YOLO:")
    print(f"  model = YOLO('yolo26m-cls.yaml')")
    print(f"  model.train(data='{args.output_dir}', epochs=100, imgsz=224, batch=64, pretrained=False)")


if __name__ == "__main__":
    main()