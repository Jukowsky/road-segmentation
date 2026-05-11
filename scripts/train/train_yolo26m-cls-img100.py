from ultralytics import YOLO
model = YOLO("yolo26m-cls.yaml")  # build a new model from YAML

print("Model loaded OK")
print(model.info())

# Train the model
model.train(
    data="imagenet100",        # subset of ImageNet with 100 classes, ~13GB
    epochs=300,
    imgsz=224,              # standard for classification
    batch=64,
    device=0,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=5,
    cos_lr=True,
    amp=True,
    pretrained=False,
    name="yolo26m_imagenet100_scratch",
    # augmentation
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.4,
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    mosaic=0.0,
    mixup=0.2,              # very effective for classification
)

metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1}")
print(f"Top-5 Accuracy: {metrics.top5}")