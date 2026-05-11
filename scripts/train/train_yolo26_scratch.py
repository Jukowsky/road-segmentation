from ultralytics import YOLO

model = YOLO("yolo26m-seg.yaml")  # build a new model from YAML

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    name="yolo26m_coco",
    optimizer="MuSGD",        
    lr0=0.01,               
    warmup_epochs=3,
    amp=True,
    patience=100,
    pretrained=False
)

metrics = model.val()
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
