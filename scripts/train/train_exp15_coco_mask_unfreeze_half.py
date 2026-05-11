# exp15_coco_mask_unfreeze_half.py
# COCO masks + freeze=5 (half backbone unfrozen)
# Continues from Exp 12 best weights

from ultralytics import YOLO

model = YOLO(
    "/home/talt_wireten_c/road-segmentation/runs/segment/"
    "exp12_synthetic_frozen_backbone_coco_mask-2026-04-08/weights/best.pt"
)

print("Model loaded OK")
print(model.info())

model.train(
    task="segment",
    data="/home/talt_wireten_c/road-segmentation/config/masked_dataset_coco.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=1,
    name="exp15_coco_mask_unfreeze_half",

    # ── Freeze only first half of backbone (layers 0-4) ──────────────
    freeze=5,

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer="SGD",
    lr0=0.0005,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    amp=True,
    pretrained=True,
    patience=20,

    # ── Augmentation — identical to Exp 12/13 ────────────────────────
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    scale=0.5,
    erasing=0.4,
)

metrics = model.val()
print(f"mAP50(B):    {metrics.box.map50:.4f}")
print(f"mAP50-95(B): {metrics.box.map:.4f}")
print(f"mAP50(M):    {metrics.seg.map50:.4f}")
print(f"mAP50-95(M): {metrics.seg.map:.4f}")