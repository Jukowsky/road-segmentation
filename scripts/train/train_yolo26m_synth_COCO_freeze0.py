# exp20_coco_freeze0.py
from ultralytics import YOLO

model = YOLO(
    "/home/talt_wireten_c/road-segmentation/runs/segment/"
    "yolo26m-objv1-seg-finetune-coco=20260320/weights/best.pt"
)

print("Model loaded OK")
print(model.info())

model.train(
    task="segment",
    data="/home/talt_wireten_c/road-segmentation/config/masked_dataset_coco.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    name="exp20_coco_freeze0",

    # ── No freezing — entire network trainable ────────────────────────
    freeze=0,

    # ── Very low LR — full network updating on noisy synthetic data ───
    optimizer="SGD",
    lr0=0.00005,      # much lower than freeze=3 (0.0001) — full network active
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,
    amp=True,
    pretrained=True,
    patience=50,

    # ── Same augmentation as all other experiments ────────────────────
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
print(f"\nmAP50(B):    {metrics.box.map50:.4f}")
print(f"mAP50-95(B): {metrics.box.map:.4f}")
print(f"mAP50(M):    {metrics.seg.map50:.4f}")
print(f"mAP50-95(M): {metrics.seg.map:.4f}")