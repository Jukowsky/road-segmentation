# visualize_synthetic.py
# Runs Exp18 on synthetic COCO mask val images
# Shows: Original | Ground Truth | Prediction side by side

from ultralytics import YOLO
import cv2
import numpy as np
import os
import random
from pathlib import Path

MODEL_PATH  = "runs/segment/exp18_coco_freeze5/weights/best.pt"
IMAGES_DIR  = "masked_rscd_dataset_coco/images"
LABELS_DIR  = "masked_rscd_dataset_coco/labels"
OUTPUT_DIR  = "synthetic_predictions_8"
NUM_IMAGES  = 10000
CONF_THRESH = 0.25   # lower threshold for synthetic data

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

random.seed(42)
COLORS = {i: (random.randint(50,255), random.randint(50,255), random.randint(50,255))
          for i in range(len(CLASS_NAMES))}

def draw_gt(img, label_path):
    h, w = img.shape[:2]
    vis     = img.copy()
    overlay = img.copy()
    if not os.path.exists(label_path):
        return vis, []
    gt_classes = []
    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 3: continue
            cls_id = int(parts[0])
            gt_classes.append(CLASS_NAMES[cls_id])
            coords = np.array(parts[1:]).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            pts = coords.astype(np.int32).reshape(-1, 1, 2)
            color = COLORS[cls_id]
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(vis, [pts], True, color, 2)
            # label
            M = cv2.moments(pts.reshape(-1, 2))
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis, CLASS_NAMES[cls_id],
                            (max(0, cx-50), cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
    return vis, list(set(gt_classes))

def draw_pred(img, result):
    h, w = img.shape[:2]
    vis     = img.copy()
    overlay = img.copy()
    pred_classes = []
    correct = []
    if result.masks is None:
        cv2.putText(vis, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return vis, pred_classes, correct

    masks = result.masks.data.cpu().numpy()
    for i, mask in enumerate(masks):
        cls_id   = int(result.boxes.cls[i].item())
        conf_val = float(result.boxes.conf[i].item())
        color    = COLORS[cls_id]
        name     = CLASS_NAMES[cls_id]
        pred_classes.append(name)

        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        colored = np.zeros_like(img)
        colored[mask_r > 0.5] = color
        cv2.addWeighted(colored, 0.4, overlay, 0.6, 0, overlay)
        contours, _ = cv2.findContours(
            (mask_r > 0.5).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(vis, f"{name} {conf_val:.2f}",
                            (max(0, cx-50), cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
    return vis, pred_classes, []

def add_banner(img, text, color=(200,200,200)):
    banner = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, text, (8, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return np.vstack([banner, img])

# ── Main ──────────────────────────────────────────────────────────────────
model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

img_files = sorted(Path(IMAGES_DIR).glob("*.jpg"))
# sample evenly across dataset
step = max(1, len(img_files) // NUM_IMAGES)
img_files = img_files[::step][:NUM_IMAGES]
print(f"Visualizing {len(img_files)} synthetic images...")

# Tracking
correct_count  = 0
wrong_count    = 0
missed_count   = 0
pred_freq      = {}
gt_freq        = {}
match_freq     = {}

for img_path in img_files:
    img = cv2.imread(str(img_path))
    if img is None: continue

    label_path = Path(LABELS_DIR) / (img_path.stem + ".txt")
    results    = model.predict(str(img_path), conf=CONF_THRESH, verbose=False)

    vis_gt,   gt_classes   = draw_gt(img, str(label_path))
    vis_pred, pred_classes, _ = draw_pred(img, results[0])

    # Track frequencies
    for c in gt_classes:
        gt_freq[c] = gt_freq.get(c, 0) + 1
    for c in pred_classes:
        pred_freq[c] = pred_freq.get(c, 0) + 1

    # Check class overlap (rough correctness)
    gt_set   = set(gt_classes)
    pred_set = set(pred_classes)
    matched  = gt_set & pred_set
    for c in matched:
        match_freq[c] = match_freq.get(c, 0) + 1
    if matched:
        correct_count += 1
    elif pred_classes:
        wrong_count += 1
    else:
        missed_count += 1

    # Build 3-panel output
    p_orig = add_banner(img.copy(),     "Original",      (180,180,180))
    p_gt   = add_banner(vis_gt,         f"GT: {', '.join(gt_classes)}",   (100,255,100))
    p_pred = add_banner(vis_pred,       f"Pred: {', '.join(pred_classes) or 'none'}", (100,200,255))

    combined = np.hstack([p_orig, p_gt, p_pred])
    cv2.imwrite(f"{OUTPUT_DIR}/{img_path.name}", combined)

print(f"\nDone. Saved to {OUTPUT_DIR}/")
print(f"\n{'='*55}")
print(f"SYNTHETIC DATA PREDICTION SUMMARY ({NUM_IMAGES} images)")
print(f"{'='*55}")
print(f"  Correct class (GT ∩ Pred ≠ ∅): {correct_count:3d} ({100*correct_count/NUM_IMAGES:.0f}%)")
print(f"  Wrong class   (pred but wrong): {wrong_count:3d} ({100*wrong_count/NUM_IMAGES:.0f}%)")
print(f"  Missed        (no detection):   {missed_count:3d} ({100*missed_count/NUM_IMAGES:.0f}%)")
print(f"\nTop GT classes:")
for name, cnt in sorted(gt_freq.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:<30} GT:{cnt:3d}  Matched:{match_freq.get(name,0):3d}")
print(f"\nTop predicted classes:")
for name, cnt in sorted(pred_freq.items(), key=lambda x: -x[1])[:10]:
    print(f"  {name:<30} {cnt:3d}")
print(f"\nDomain gap note: compare these results with vvis results")
print(f"{'='*55}")