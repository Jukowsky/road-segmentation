from ultralytics import YOLO

# ─── Model ────────────────────────────────────────────────────────────────────
# Fine-tune from official Ultralytics pretrained weights (ImageNet Top-1=78.1%)
model = YOLO("yolo26m-cls.pt")  # auto-downloads if not cached

print("Model loaded OK")
print(model.info())

# ─── Data ─────────────────────────────────────────────────────────────────────
# RSCD folder structure (ready):
#   train/  → 32 class subfolders (~147k images)
#   val/    → 32 class subfolders (~2.2k images)
#   test/   → 32 class subfolders (~50k images)
RSCD_PATH = "/home/talt_wireten_c/road-segmentation/datasets/rscd"

# ─── Train ────────────────────────────────────────────────────────────────────
model.train(
    data=RSCD_PATH,
    epochs=100,
    imgsz=224,
    batch=64,
    device=0,

    # Low LR for fine-tuning from ImageNet pretrained weights
    optimizer="SGD",
    lr0=0.0001,
    lrf=0.0001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    amp=True,
    pretrained=True,

    patience=20,

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    mosaic=0.0,   # CRITICAL: must be 0 for classification (lesson from Exp 2)
    mixup=0.2,

    project="runs/classify",
    name="exp8_rscd_ultralytics",
)

# ─── Validate ─────────────────────────────────────────────────────────────────
metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1}")
print(f"Top-5 Accuracy: {metrics.top5}")