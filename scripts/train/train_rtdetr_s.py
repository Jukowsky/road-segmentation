from ultralytics import RTDETR

model = RTDETR("rtdetr-s.pt")

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    name="rtdetr_s_coco",
    lr0=0.0001,
    amp=True,
    warmup_epochs=5,
)
