from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import random

MODEL_PATH        = "runs/segment/exp18_coco_freeze5/weights/best.pt"
VVIS_DIR          = "/home/talt_wireten_c/road-segmentation/datasets/Spjute_Bjästa_SE_STA_CAMERA_Geni_4684_K1"
OUTPUT_DIR        = "vvis_predictions"
NUM_IMAGES        = 100
CONF_THRESH       = 0.30    
CROP_TOP_FRACTION = 0.10    # remove top 10% (header strip)
MAX_MASK_COVERAGE = 100    # ignore masks covering >100% of image

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

model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

extensions = ['.jpg', '.jpeg', '.png']
all_images = []
for ext in extensions:
    all_images.extend(Path(VVIS_DIR).rglob(f"*{ext}"))
all_images = sorted(all_images)[:NUM_IMAGES]
print(f"Found {len(all_images)} images in vvis")

detection_counts = {}

for img_path in all_images:
    img = cv2.imread(str(img_path))
    if img is None: continue
    h, w = img.shape[:2]

    # ── Fix 1: crop top 5% before inference ──────────────────────────
    crop_y      = int(h * CROP_TOP_FRACTION)
    img_cropped = img[crop_y:, :]
    ch, cw      = img_cropped.shape[:2]

    results = model.predict(img_cropped, conf=CONF_THRESH, verbose=False)
    result  = results[0]

    vis_cropped     = img_cropped.copy()
    overlay_cropped = img_cropped.copy()

    detected_classes = []

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            cls_id   = int(result.boxes.cls[i].item())
            conf_val = float(result.boxes.conf[i].item())
            color    = COLORS[cls_id]
            name     = CLASS_NAMES[cls_id]

            mask_r   = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

            # ── Fix 2: skip masks covering >80% of cropped area ───────
            coverage = (mask_r > 0.5).sum() / (ch * cw)
            if coverage > MAX_MASK_COVERAGE:
                continue

            detected_classes.append(name)
            detection_counts[name] = detection_counts.get(name, 0) + 1

            colored = np.zeros_like(img_cropped)
            colored[mask_r > 0.5] = color
            cv2.addWeighted(colored, 0.45, overlay_cropped, 0.55, 0, overlay_cropped)

            contours, _ = cv2.findContours(
                (mask_r > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_cropped, contours, -1, color, 2)

            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(vis_cropped, f"{name} {conf_val:.2f}",
                                (max(0, cx-60), cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.addWeighted(overlay_cropped, 0.45, vis_cropped, 0.55, 0, vis_cropped)

    if not detected_classes:
        cv2.putText(vis_cropped, "No detections", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # ── Paste cropped result back into full-size canvas ───────────────
    vis_full          = img.copy()
    vis_full[crop_y:] = vis_cropped

    # ── Draw cyan crop line on original panel ─────────────────────────
    panel_orig = img.copy()
    cv2.line(panel_orig, (0, crop_y), (w, crop_y), (0, 255, 255), 1)

    # ── Banner ────────────────────────────────────────────────────────
    banner = np.zeros((30, w * 2, 3), dtype=np.uint8)
    cv2.putText(banner, "Original", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(banner,
                f"Prediction (Exp18, COCO+freeze5) — {', '.join(detected_classes) or 'none'}",
                (w + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    combined = np.hstack([panel_orig, vis_full])
    combined = np.vstack([banner, combined])

    cv2.imwrite(f"{OUTPUT_DIR}/{img_path.name}", combined)
    print(f"Saved {img_path.name} → {detected_classes or ['no detections']}")

# ── Detection frequency report ────────────────────────────────────────────
print("\n" + "="*50)
print("DETECTION FREQUENCY ACROSS ALL vvis IMAGES")
print("="*50)
for name, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
    print(f"  {name:<30} {count:3d}  {'█' * count}")

never_detected = [c for c in CLASS_NAMES if c not in detection_counts]
print(f"\nNEVER DETECTED ({len(never_detected)} classes):")
for c in never_detected:
    print(f"  {c}")
print("="*50)
print(f"\nResults saved to: {OUTPUT_DIR}/")