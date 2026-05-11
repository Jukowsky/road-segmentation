from ultralytics import YOLO

model = YOLO("yolo26m.pt")  # s = good balance of speed/accuracy

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    name="yolo26m_coco",
    optimizer="SGD",        # YOLO26 uses MuSGD internally, SGD is the right base
    lr0=0.01,               # default works well for YOLO26
    warmup_epochs=3,
    amp=True,
    patience=50,
)

metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
