from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = "runs/semantic/sem_exp26_freeze0/weights/best.pt"
IMG_DIR    = "/home/talt_wireten_c/road-segmentation/masked_rscd_dataset_coco_full/val_split/images"
OUTPUT_DIR = "sem_exp26_sample_predictions"
NUM_IMAGES = 30
DEVICE     = "cuda:2"
ALPHA      = 0.55

CLASS_NAMES = [
    "dry_asphalt_severe","dry_asphalt_slight","dry_asphalt_smooth",
    "dry_concrete_severe","dry_concrete_slight","dry_concrete_smooth",
    "dry_gravel","dry_mud","fresh_snow","ice","melted_snow",
    "water_asphalt_severe","water_asphalt_slight","water_asphalt_smooth",
    "water_concrete_severe","water_concrete_slight","water_concrete_smooth",
    "water_gravel","water_mud",
    "wet_asphalt_severe","wet_asphalt_slight","wet_asphalt_smooth",
    "wet_concrete_severe","wet_concrete_slight","wet_concrete_smooth",
    "wet_gravel","wet_mud",
]

COLORS = [
    (60,  60,  60),   # dry_asphalt_severe
    (90,  90,  90),   # dry_asphalt_slight
    (120, 120, 120),  # dry_asphalt_smooth
    (80,  70,  60),   # dry_concrete_severe
    (110, 100, 85),   # dry_concrete_slight
    (140, 130, 110),  # dry_concrete_smooth
    (80,  110, 140),  # dry_gravel
    (40,  80,  120),  # dry_mud
    (240, 240, 255),  # fresh_snow
    (200, 230, 255),  # ice
    (170, 200, 220),  # melted_snow
    (180, 50,  50),   # water_asphalt_severe
    (200, 90,  90),   # water_asphalt_slight
    (220, 130, 130),  # water_asphalt_smooth
    (160, 60,  110),  # water_concrete_severe
    (180, 90,  140),  # water_concrete_slight
    (200, 120, 170),  # water_concrete_smooth
    (140, 80,  160),  # water_gravel
    (100, 50,  140),  # water_mud
    (50,  130, 50),   # wet_asphalt_severe
    (70,  160, 70),   # wet_asphalt_slight
    (90,  190, 90),   # wet_asphalt_smooth
    (50,  150, 120),  # wet_concrete_severe
    (70,  175, 145),  # wet_concrete_slight
    (90,  200, 170),  # wet_concrete_smooth
    (60,  170, 160),  # wet_gravel
    (40,  130, 130),  # wet_mud
]

LEGEND_W = 230


def colorize_mask(mask_hw):
    """mask_hw: H×W uint8 with class ids (0-26) or 255 for ignore."""
    h, w = mask_hw.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(COLORS):
        rgb[mask_hw == cls_id] = color
    return rgb


def draw_legend(h, class_ids):
    panel = np.full((h, LEGEND_W, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "Predicted classes", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.line(panel, (4, 26), (LEGEND_W - 4, 26), (80, 80, 80), 1)
    y = 46
    for cid in sorted(class_ids):
        if y + 16 > h:
            break
        color = COLORS[cid]
        cv2.rectangle(panel, (6, y - 10), (20, y + 2), color, -1)
        cv2.rectangle(panel, (6, y - 10), (20, y + 2), (160, 160, 160), 1)
        cv2.putText(panel, CLASS_NAMES[cid].replace("_", " "), (26, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (210, 210, 210), 1, cv2.LINE_AA)
        y += 20
    return panel


model = YOLO(MODEL_PATH)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

images = sorted(Path(IMG_DIR).glob("*.jpg"))[:NUM_IMAGES]
print(f"Running inference on {len(images)} images → {OUTPUT_DIR}")

for img_path in images:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]

    results = model.predict(img, verbose=False, device=DEVICE)
    result  = results[0]

    # semantic_mask is H×W with class ids
    sem_mask = result.semantic_mask  # numpy array
    sem_mask = sem_mask.astype(np.uint8)

    colored = colorize_mask(sem_mask)
    colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(img, 1 - ALPHA, colored, ALPHA, 0)

    unique_ids = [int(c) for c in np.unique(sem_mask) if c < len(CLASS_NAMES)]
    legend = draw_legend(h, unique_ids)

    banner_w = w * 2 + LEGEND_W
    banner = np.full((28, banner_w, 3), 40, dtype=np.uint8)
    cv2.putText(banner, "Original", (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(banner, f"sem_exp26 best — {img_path.name}",
                (w + 6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA)

    combined = np.hstack([img, overlay, legend])
    combined = np.vstack([banner, combined])
    cv2.imwrite(f"{OUTPUT_DIR}/{img_path.name}", combined)

    names = [CLASS_NAMES[c] for c in unique_ids]
    print(f"  {img_path.name}: {names}")

print(f"\nDone. Results in {OUTPUT_DIR}/")
