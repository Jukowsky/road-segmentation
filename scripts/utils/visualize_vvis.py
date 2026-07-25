from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path

MODEL_PATH           = "runs/segment/exp21_full_freeze56/weights/best.pt"
VVIS_DIR             = "/home/talt_wireten_c/road-segmentation/datasets/rcm-samples"
OUTPUT_DIR           = "rcm_sample_predictions/exp21_freeze5_viz"
NUM_IMAGES           = 100
CONF_THRESH          = 0.20
IOU_THRESH           = 0.50        # NMS IoU — higher = fewer duplicate masks
CROP_TOP_FRACTION    = 0.50
CROP_BOTTOM_FRACTION = 0.25
MAX_MASK_COVERAGE    = 100
MASK_ALPHA           = 0.55        # mask opacity (higher = more visible)
LEGEND_WIDTH         = 220         # right-side legend panel width

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

# Semantically meaningful color palette (BGR)
COLORS = {
    "dry_asphalt_severe":    (60,  60,  60),
    "dry_asphalt_slight":    (90,  90,  90),
    "dry_asphalt_smooth":    (120, 120, 120),
    "dry_concrete_severe":   (80,  70,  60),
    "dry_concrete_slight":   (110, 100, 85),
    "dry_concrete_smooth":   (140, 130, 110),
    "dry_gravel":            (80,  110, 140),
    "dry_mud":               (40,  80,  120),
    "fresh_snow":            (240, 240, 255),
    "ice":                   (200, 230, 255),
    "melted_snow":           (170, 200, 220),
    "water_asphalt_severe":  (180, 50,  50),
    "water_asphalt_slight":  (200, 90,  90),
    "water_asphalt_smooth":  (220, 130, 130),
    "water_concrete_severe": (160, 60,  110),
    "water_concrete_slight": (180, 90,  140),
    "water_concrete_smooth": (200, 120, 170),
    "water_gravel":          (140, 80,  160),
    "water_mud":             (100, 50,  140),
    "wet_asphalt_severe":    (50,  130, 50),
    "wet_asphalt_slight":    (70,  160, 70),
    "wet_asphalt_smooth":    (90,  190, 90),
    "wet_concrete_severe":   (50,  150, 120),
    "wet_concrete_slight":   (70,  175, 145),
    "wet_concrete_smooth":   (90,  200, 170),
    "wet_gravel":            (60,  170, 160),
    "wet_mud":               (40,  130, 130),
}
COLOR_BY_ID = {i: COLORS[n] for i, n in enumerate(CLASS_NAMES)}


def draw_label(img, text, x, y, color, font_scale=0.42, thickness=1):
    """Draw text with a dark background box for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 3
    x = max(pad, min(x, img.shape[1] - tw - pad))
    y = max(th + pad, min(y, img.shape[0] - pad))
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_legend(h, detected):
    """Build a right-side legend panel listing detected classes with color swatches."""
    panel = np.zeros((h, LEGEND_WIDTH, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)
    cv2.putText(panel, "Detected", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(panel, (6, 28), (LEGEND_WIDTH - 6, 28), (80, 80, 80), 1)

    seen = {}
    for name, conf in detected:
        seen[name] = max(seen.get(name, 0), conf)

    y = 50
    for name, conf in sorted(seen.items(), key=lambda x: -x[1]):
        if y + 18 > h:
            break
        color = COLORS[name]
        cv2.rectangle(panel, (8, y - 10), (22, y + 2), color, -1)
        cv2.rectangle(panel, (8, y - 10), (22, y + 2), (180, 180, 180), 1)
        label = f"{name.replace('_', ' ')} {conf:.2f}"
        cv2.putText(panel, label, (28, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22

    if not seen:
        cv2.putText(panel, "none", (8, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (100, 100, 255), 1, cv2.LINE_AA)
    return panel


model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_images = sorted(
    p for ext in [".jpg", ".jpeg", ".png"]
    for p in Path(VVIS_DIR).rglob(f"*{ext}")
)[:NUM_IMAGES]
print(f"Found {len(all_images)} images")

detection_counts = {}

for img_path in all_images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    crop_y_top    = int(h * CROP_TOP_FRACTION)
    crop_y_bottom = h - int(h * CROP_BOTTOM_FRACTION)
    img_cropped   = img[crop_y_top:crop_y_bottom, :]
    ch, cw        = img_cropped.shape[:2]

    results = model.predict(
        img_cropped, conf=CONF_THRESH, iou=IOU_THRESH,
        verbose=False, device="cpu", agnostic_nms=True
    )
    result = results[0]

    vis_cropped     = img_cropped.copy()
    overlay_cropped = img_cropped.copy()
    detected = []   # (name, conf)

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            cls_id   = int(result.boxes.cls[i].item())
            conf_val = float(result.boxes.conf[i].item())
            color    = COLOR_BY_ID[cls_id]
            name     = CLASS_NAMES[cls_id]

            mask_r   = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
            coverage = (mask_r > 0.5).sum() / (ch * cw)
            if coverage > MAX_MASK_COVERAGE:
                continue

            detected.append((name, conf_val))
            detection_counts[name] = detection_counts.get(name, 0) + 1

            # Colored mask overlay
            colored = np.zeros_like(img_cropped)
            colored[mask_r > 0.5] = color
            cv2.addWeighted(colored, MASK_ALPHA, overlay_cropped,
                            1 - MASK_ALPHA, 0, overlay_cropped)

            # Contour
            contours, _ = cv2.findContours(
                (mask_r > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_cropped, contours, -1, color, 2)

            # Label with background box at mask centroid
            if contours:
                M = cv2.moments(contours[0])
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    draw_label(vis_cropped, f"{name} {conf_val:.2f}", cx - 60, cy, color)

    cv2.addWeighted(overlay_cropped, 0.5, vis_cropped, 0.5, 0, vis_cropped)

    if not detected:
        draw_label(vis_cropped, "No detections", 20, 40, (80, 80, 255), font_scale=0.6)

    # Paste prediction back into full canvas
    vis_full = img.copy()
    vis_full[crop_y_top:crop_y_bottom] = vis_cropped

    # Original panel with crop lines
    panel_orig = img.copy()
    cv2.line(panel_orig, (0, crop_y_top),    (w, crop_y_top),    (0, 255, 255), 1)
    cv2.line(panel_orig, (0, crop_y_bottom), (w, crop_y_bottom), (0, 255, 255), 1)

    # Legend panel
    legend = draw_legend(h, detected)

    # Top banner
    total_w = w * 2 + LEGEND_WIDTH
    banner = np.zeros((32, total_w, 3), dtype=np.uint8)
    banner[:] = (45, 45, 45)
    cv2.putText(banner, "Original", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(banner, f"Prediction — Exp21 freeze5 — {img_path.name}",
                (w + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1, cv2.LINE_AA)

    combined = np.hstack([panel_orig, vis_full, legend])
    combined = np.vstack([banner, combined])

    cv2.imwrite(f"{OUTPUT_DIR}/{img_path.name}", combined)
    names = list({n for n, _ in detected})
    print(f"  {img_path.name} → {names or ['no detections']}")

print("\n" + "=" * 50)
print("DETECTION FREQUENCY")
print("=" * 50)
for name, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
    print(f"  {name:<30} {count:3d}  {'█' * count}")
never = [c for c in CLASS_NAMES if c not in detection_counts]
print(f"\nNEVER DETECTED ({len(never)}): {', '.join(never)}")
print(f"\nResults saved to: {OUTPUT_DIR}/")
