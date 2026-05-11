from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    name="rtdetr_l_coco_v2",
    lr0=0.0001,
    amp=True,
    warmup_epochs=5,
)

metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
