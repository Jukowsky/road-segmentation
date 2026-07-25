# exp26 — semantic seg, yolo26m-sem, freeze=0 (mirrors exp23)
from ultralytics import YOLO

DATA_CFG = "/home/talt_wireten_c/road-segmentation/config/masked_dataset_sem_full.yaml"

model = YOLO("yolo26m-sem.pt")
print("Model loaded OK")
print(model.info())

model.train(
    data=DATA_CFG,
    epochs=100,
    imgsz=640,
    batch=64,
    workers=24,
    device=2,
    name="sem_exp26_freeze0",

    freeze=0,
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
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    scale=0.5,
    erasing=0.4,
)

metrics = model.val()
print(metrics)
