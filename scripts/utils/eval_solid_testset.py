from ultralytics import YOLO

model = YOLO(
    "runs/segment/exp18_coco_freeze5/weights/best.pt"
)

metrics = model.val(
    data="/home/talt_wireten_c/road-segmentation/config/solid_label_testset.yaml",
    split="val",
    conf=0.25,
    iou=0.5,
)

print(f"\n{'='*50}")
print(f"mAP50(B)    on solid labels: {metrics.box.map50:.4f}")
print(f"mAP50-95(B) on solid labels: {metrics.box.map:.4f}")
print(f"mAP50(M)    on solid labels: {metrics.seg.map50:.4f}")
print(f"mAP50-95(M) on solid labels: {metrics.seg.map:.4f}")
print(f"{'='*50}")
print(f"\nFor reference — real test set mAP50(B): 0.804")
print(f"If solid label mAP is also high → model inflated by background")
print(f"If solid label mAP drops to near zero → model genuinely learned texture")