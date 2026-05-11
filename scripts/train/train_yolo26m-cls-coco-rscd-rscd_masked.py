from ultralytics import YOLO

model = YOLO(
    "/home/talt_wireten_c/road-segmentation/runs/segment/"
    "yolo26m-objv1-seg-finetune-coco=20260320/weights/best.pt"
)

model.train(
    data="/home/talt_wireten_c/road-segmentation/config/masked_dataset_coco.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=1,
    name="exp12_synthetic_frozen_backbone_coco_mask-2026-04-08",
    optimizer="SGD",
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.001,        # lower LR since backbone is frozen
    lrf=0.01,
    warmup_epochs=5,
    amp=True,
    patience=50,
    cos_lr=True,
    pretrained=True,
    freeze=10,        # freeze backbone
    mixup=0.1,
    copy_paste=0.1,
)

metrics = model.val()
print(f"mAP50(B):    {metrics.box.map50}")
print(f"mAP50-95(B): {metrics.box.map}")
print(f"mAP50(M):    {metrics.seg.map50}")
print(f"mAP50-95(M): {metrics.seg.map}")