from ultralytics import YOLO

model = YOLO("runs/classify/yolo26m_imagenet100_scratch/weights/best.pt")

model.train(
    data="/home/talt_wireten_c/road-segmentation/data/coco_cls",
    epochs=50,
    imgsz=224,
    batch=64,
    device=2,
    lr0=0.0001,
    cos_lr=True,
    warmup_epochs=3,
    amp=True,
    name="yolo26m_coco_cls_finetuned",
)

metrics = model.val()
print(f"Top-1: {metrics.top1}")
print(f"Top-5: {metrics.top5}")