"""
visualize_predictions.py
Runs Exp 18 (best seg model) on synthetic val images and saves
side-by-side: Original | Ground Truth | Prediction
"""
import cv2
import numpy as np
import os
import random
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = "runs/segment/exp18_coco_freeze5/weights/best.pt"
IMAGES_DIR   = "masked_rscd_dataset_coco/images"
LABELS_DIR   = "masked_rscd_dataset_coco/labels"
OUTPUT_DIR   = "pred_visualizations"
NUM_IMAGES   = 30       # how many to visualize
CONF_THRESH  = 0.25

# ── RSCD class names ──────────────────────────────────────────────────────────
CLASS_NAMES = [
    "dry_asphalt_severe","dry_asphalt_slight","dry_asphalt_smooth",
    "dry_concrete_severe","dry_concrete_slight","dry_concrete_smooth",
    "dry_gravel","dry_mud","fresh_snow","ice","melted_snow",
    "water_asphalt_severe","water_asphalt_slight","water_asphalt_smooth",
    "water_concrete_severe","water_concrete_slight","water_concrete_smooth",
    "water_gravel","water_mud",
    "wet_asphalt_severe","wet_asphalt_slight","wet_asphalt_smooth",
    "wet_concrete_severe","wet_concrete_slight","wet_concrete_smooth",
    "wet_gravel","wet_mud"
]

# ── Consistent colors per class ───────────────────────────────────────────────
random.seed(42)
COLORS = {i: (random.randint(50,255), random.randint(50,255), random.randint(50,255))
          for i in range(len(CLASS_NAMES))}

def draw_gt(img, label_path):
    """Draw ground truth polygons from YOLO label file."""
    h, w = img.shape[:2]
    vis = img.copy()
    overlay = img.copy()
    if not os.path.exists(label_path):
        return vis
    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 3: continue
            cls_id = int(parts[0])
            coords = np.array(parts[1:]).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            pts = coords.astype(np.int32).reshape(-1, 1, 2)
            color = COLORS[cls_id % len(COLORS)]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis, [pts], True, color, 2)
    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
    return vis

def draw_pred(img, result):
    """Draw model predictions from ultralytics result."""
    h, w = img.shape[:2]
    vis = img.copy()
    overlay = img.copy()
    if result.masks is None:
        cv2.putText(vis, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return vis
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    for i, mask in enumerate(masks):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        color  = COLORS[cls_id % len(COLORS)]
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        colored = np.zeros_like(img)
        colored[mask_resized > 0.5] = color
        cv2.addWeighted(colored, 0.45, overlay, 0.55, 0, overlay)
        # find contour for outline
        contours, _ = cv2.findContours(
            (mask_resized > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        # label on centroid
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                cv2.putText(vis, label, (cx-40, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.addWeighted(overlay, 0.45, vis, 0.55, 0, vis)
    return vis

def add_legend(img, title, color=(255,255,255)):
    """Add title banner to top of image."""
    banner = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, title, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return np.vstack([banner, img])

# ── Main ──────────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

img_files = sorted(Path(IMAGES_DIR).glob("*.jpg"))[:NUM_IMAGES]
print(f"Visualizing {len(img_files)} images...")

for img_path in img_files:
    img = cv2.imread(str(img_path))
    if img is None: continue

    label_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
    results = model.predict(str(img_path), conf=CONF_THRESH, verbose=False)

    panel_orig = add_legend(img.copy(),          "Original",       (200,200,200))
    panel_gt   = add_legend(draw_gt(img, str(label_path)), "Ground Truth",  (100,255,100))
    panel_pred = add_legend(draw_pred(img, results[0]),    "Prediction",    (100,200,255))

    combined = np.hstack([panel_orig, panel_gt, panel_pred])
    out_path = f"{OUTPUT_DIR}/{img_path.name}"
    cv2.imwrite(out_path, combined)
    print(f"Saved {img_path.name}")

print(f"\nDone. Results in: {OUTPUT_DIR}/")