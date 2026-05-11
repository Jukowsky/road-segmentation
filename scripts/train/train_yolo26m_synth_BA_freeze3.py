# exp17_badapple_freeze3.py
from ultralytics import YOLO

model = YOLO(
    "/home/talt_wireten_c/road-segmentation/runs/segment/"
    "yolo26m-objv1-seg-finetune-coco=20260320/weights/best.pt"
)

model.train(
    task="segment",
    data="/home/talt_wireten_c/road-segmentation/config/masked_dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=2,
    name="exp17_badapple_freeze3",
    freeze=3,
    optimizer="SGD",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,
    amp=True,
    pretrained=True,
    patience=50,
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
print(f"mAP50(B): {metrics.box.map50:.4f}")
print(f"mAP50-95(B): {metrics.box.map:.4f}")
print(f"mAP50(M): {metrics.seg.map50:.4f}")
print(f"mAP50-95(M): {metrics.seg.map:.4f}")