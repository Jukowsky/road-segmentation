## Bunu default yolo26m cls modelini cropped ile egitip sonra rdsc ile train edecegiz.

from ultralytics import YOLO

model = YOLO("yolo26m-cls.pt")  # Ultralytics official — same starting point as Exp 8

print("Model loaded OK")
print(model.info())

model.train(
    data="/home/talt_wireten_c/road-segmentation/data/coco_cls",
    epochs=100,
    imgsz=224,
    batch=64,
    device=0,
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
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    mosaic=0.0,       # CRITICAL: off for classification
    mixup=0.2,
    project="runs/classify",
    name="exp9_ultralytics_coco_cls",
)

metrics = model.val()
print(f"Top-1: {metrics.top1}")
print(f"Top-5: {metrics.top5}")