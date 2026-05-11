from ultralytics import YOLO
import os

# ─── Model ────────────────────────────────────────────────────────────────────
# Fine-tune from Exp 4 best weights (Top-1=84.8% on COCO-cls)
# This mirrors the Exp2→Exp4 transfer that worked well
model = YOLO("/home/talt_wireten_c/road-segmentation/runs/classify/yolo26m_coco_cls_finetuned/weights/best.pt")

# ─── Alternatively: train from scratch (mirrors Exp 5, use as baseline)
# model = YOLO("yolo26m-cls.yaml")

print("Model loaded OK")
print(model.info())

# ─── Data ─────────────────────────────────────────────────────────────────────
# RSCD folder structure:
#   train/    → 32 subfolders (class names)
#   vali_20k/ → flat, labels in filenames
#   test_50k/ → flat, labels in filenames
#
# YOLO cls expects train/ and val/ — symlink vali_20k → val if needed:
#   ln -s /home/talt_wireten_c/road-segmentation/datasets/rscd/vali_20k \
#         /home/talt_wireten_c/road-segmentation/datasets/rscd/val

RSCD_PATH = "/home/talt_wireten_c/road-segmentation/datasets/rscd"

# ─── Train ────────────────────────────────────────────────────────────────────
model.train(
    data=RSCD_PATH,
    epochs=100,                 # Exp 4 converged at ~50; 100 gives headroom
    imgsz=224,                  # RSCD native is 240x360 → YOLO resizes to square
    batch=64,
    device=0,

    # Optimizer — low LR for finetuning (Exp 4 used 0.0001)
    # Switch to lr0=0.001 + optimizer="AdamW" if training from scratch
    optimizer="SGD",
    lr0=0.0001,
    lrf=0.0001,                 # flat cosine (same pattern as Exp 6 seg)
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    amp=True,
    pretrained=True,            # True = keep loaded weights; False = reinit

    # Early stopping — RSCD is smaller than COCO-cls, overfitting risk is higher
    patience=20,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    mosaic=0.0,                 # CRITICAL: must be 0 for classification (lesson from Exp 2)
    mixup=0.2,

    # Output
    project="runs/classify",
    name="exp7_rscd_finetune",
)

# ─── Validate ─────────────────────────────────────────────────────────────────
metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1}")
print(f"Top-5 Accuracy: {metrics.top5}")