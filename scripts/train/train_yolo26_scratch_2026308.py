from ultralytics import YOLO
model = YOLO("yolo26m-seg.yaml")  # build a new model from YAML

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=1,
    name="yolo26m_coco_v2",
    optimizer="SGD",
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=5,
    amp=True,
    patience=50,
    cos_lr=True,
    pretrained=False,
    # augmentation
    mixup=0.1,        # non-default
    copy_paste=0.1,   # non-default
)

metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
