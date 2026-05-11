from ultralytics import YOLO

# Load Step 1 best weights — Ultralytics → COCO-cls fine-tuned
model = YOLO("runs/classify/runs/classify/exp9_ultralytics_coco_cls2/weights/best.pt")

print("Model loaded OK")
print(model.info())

model.train(
    data="/home/talt_wireten_c/road-segmentation/datasets/rscd",
    epochs=100,
    imgsz=224,
    batch=64,
    device=0,

    # Identical to Exp 7 — only the init weights differ
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

    # Identical augmentation to Exp 7
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    mosaic=0.0,       # CRITICAL: off for classification
    mixup=0.2,
    erasing=0.4,
    auto_augment="randaugment",

    project="runs/classify",
    name="exp10_rscd_ultralytics_coco_cls",
)

metrics = model.val()
print(f"Val Top-1: {metrics.top1}")
print(f"Val Top-5: {metrics.top5}")

# Also evaluate on test split
metrics_test = model.val(
    data="/home/talt_wireten_c/road-segmentation/datasets/rscd",
    split="test",
)
print(f"Test Top-1: {metrics_test.top1}")
print(f"Test Top-5: {metrics_test.top5}")