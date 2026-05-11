from ultralytics import YOLO

# Continue from Exp 11 best weights (already adapted to 27 RSCD classes)
model = YOLO(
    "/home/talt_wireten_c/road-segmentation/runs/segment/"
    "exp11_synthetic_frozen_backbone-2026-04-05/weights/best.pt"
)

print("Model loaded OK")
print(model.info())

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/masked_dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    name="exp13_unfreeze_half_backbone",

    # ── Freeze only first half of backbone (layers 0-4) ──────────────
    # Layers 0-4: Conv, Conv, C3k2, Conv, C3k2 (early low-level features)
    # Layers 5-9: Conv, C3k2, Conv, C3k2, SPPF (now trainable)
    # Layers 10+: neck + head (trainable as before)
    freeze=5,

    # ── Lower LR since more layers are now updating ───────────────────
    optimizer="SGD",
    lr0=0.0005,       # lower than Exp 11 (0.001) — more layers active
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    amp=True,
    pretrained=True,
    patience=20,

    # ── Same augmentation as Exp 11 ───────────────────────────────────
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