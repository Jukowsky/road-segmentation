# exp22 — full dataset, freeze=3 (comparison with exp19)
from ultralytics import YOLO

BASE_WEIGHTS = (
    "/home/talt_wireten_c/road-segmentation/runs/segment/"
    "exp22_full_freeze32/weights/last.pt"
)
DATA_CFG = "/home/talt_wireten_c/road-segmentation/config/masked_dataset_coco_full.yaml"

model = YOLO(BASE_WEIGHTS)
print("Model loaded OK")
print(model.info())

model.train(
    task="segment",
    data=DATA_CFG,
    epochs=100,
    imgsz=640,
    batch=64,
    workers=24,
    device=1,
    name="exp22_full_freeze3",

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
print(f"\nmAP50(B):    {metrics.box.map50:.4f}")
print(f"mAP50-95(B): {metrics.box.map:.4f}")
print(f"mAP50(M):    {metrics.seg.map50:.4f}")
print(f"mAP50-95(M): {metrics.seg.map:.4f}")
