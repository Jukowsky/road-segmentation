from ultralytics import YOLO

model = YOLO("yolo26m-cls.yaml")  # from scratch

model.train(
    data="/home/talt_wireten_c/road-segmentation/data/coco_cls",
    epochs=100,
    imgsz=224,
    batch=64,
    device=0,
    pretrained=False,
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    warmup_epochs=5,
    amp=True,
    name="yolo26m_coco_cls_scratch-cropped",
)

metrics = model.val()
print(f"Top-1: {metrics.top1}")
print(f"Top-5: {metrics.top5}")