from ultralytics import YOLO

model = YOLO("/home/talt_wireten_c/road-segmentation/runs/segment/yolo11n_seg_coco/weights/best.pt")

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=52,          # Only 52 more epochs
    imgsz=640,
    batch=8,
    optimizer="AdamW",
    device=0,
    name="yolo11n_seg_coco_resumed",
    lr0=0.001,
    amp=False,
    warmup_epochs=3,
)
