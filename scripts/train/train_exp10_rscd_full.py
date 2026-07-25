# exp10-full — classification retrain on full 1M RSCD dataset
# Same hyperparameters as original exp10; starting weights: best COCO-cls pretrained model
from ultralytics import YOLO

COCO_CLS_WEIGHTS = (
    "runs/classify/runs/classify/exp9_ultralytics_coco_cls2/weights/best.pt"
)
RSCD_DATA = "/home/talt_wireten_c/road-segmentation/datasets/rscd"

model = YOLO(COCO_CLS_WEIGHTS)
print("Model loaded OK")
print(model.info())

model.train(
    data=RSCD_DATA,
    epochs=100,
    imgsz=224,
    batch=64,
    device=3,

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
    mosaic=0.0,
    mixup=0.2,
    erasing=0.4,
    auto_augment="randaugment",

    project="runs/classify",
    name="exp10_rscd_full",
)

metrics = model.val()
print(f"\nVal  Top-1: {metrics.top1:.4f}  Top-5: {metrics.top5:.4f}")

metrics_test = model.val(data=RSCD_DATA, split="test")
print(f"Test Top-1: {metrics_test.top1:.4f}  Top-5: {metrics_test.top5:.4f}")
