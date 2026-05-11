from ultralytics import YOLO

model = YOLO("runs/classify/runs/classify/exp8_rscd_ultralytics/weights/best.pt")
metrics = model.val(
    data="/home/talt_wireten_c/road-segmentation/datasets/rscd",
    split="test"  # explicitly use test split
)
print(f"Top-1: {metrics.top1}")
print(f"Top-5: {metrics.top5}")