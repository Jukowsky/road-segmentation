from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/coco.yaml",
    epochs=100,
    imgsz=640,
    batch=-1,
    optimizer="auto",
    device=3,
    name="yolo11n_seg_coco",
    # Stability fixes
    lr0=0.001,          # Lower initial learning rate (default is 0.01)
    lrf=0.01,           # Final learning rate factor
    warmup_epochs=5,    # More warmup epochs
    amp=True,          # Disable mixed precision to prevent NaN
    patience=50,        # Early stopping patience
)

metrics = model.val()
print(f"Box mAP50-95: {metrics.box.map}")
print(f"Mask mAP50-95: {metrics.seg.map}")
