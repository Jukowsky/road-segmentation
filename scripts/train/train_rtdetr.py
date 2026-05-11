from ultralytics import RTDETR

# RT-DETR v2 models: rtdetr-l.pt (large), rtdetr-x.pt (extra large)
model = RTDETR("rtdetr-l.pt")

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=100,
    imgsz=640,
    batch=4,            # RT-DETR needs more memory, start with smaller batch
    device=0,
    name="rtdetr_l_coco",
    lr0=0.0001,         # Transformers need lower learning rate
    amp=True,           # RT-DETR works well with AMP
    warmup_epochs=5,
)

metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
