"""
Side-by-side comparison: segmentation model vs sliding-window classifier.

Output per image:
  [  seg overlay  |  cls overlay  ]
  [          shared legend         ]

Edit the CONFIG block to change models or hyperparameters.
"""

import os, sys, importlib.util
import cv2
import numpy as np
import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEG_MODEL = "runs/segment/exp21_full_freeze56/weights/best.pt"
CLS_MODEL = "scripts/utils/best (1).pt"

# (image_path, wincls_crop_top, seg_crop_top_frac, seg_crop_bottom_frac)
IMAGES = [
    ("datasets/rcm-samples/RCMY196_20260302_0846_001.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_0846_003.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_0846_013.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_0846_014.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_0846_015.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_1301_010.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_1301_055.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_1551_009.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_1551_014.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260302_1551_015.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_008.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_009.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_010.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_014.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_015.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_018.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_019.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_021.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_022.jpg", 0, 0.50, 0.25),
    ("datasets/rcm-samples/RCMY196_20260305_0844_028.jpg", 0, 0.50, 0.25),
]

WIN_W        = 30
WIN_H        = 45
STRIDE       = 20
MODE         = "prob"

CONF_THRESH  = 0.20
IOU_THRESH   = 0.50
MASK_ALPHA   = 0.55

OUT_DIR      = "compare_seg_cls_2"
PANEL_W      = 960   # each panel resized to this width
# ─────────────────────────────────────────────────────────────────────────────

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── load yolo_swin ────────────────────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "wincls", os.path.join("scripts", "utils", "wincls(1).py"))
_wincls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wincls)
yolo_swin = _wincls.yolo_swin

from ultralytics import YOLO

# Colors and class names are taken from cls_model after loading (road_color_dict).
# Both seg and cls overlays use the same palette so the single legend is correct for both.


def resize_to_width(img, w):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def make_label_banner(text, w, h=32, bg=(40, 40, 40), fg=(200, 200, 200)):
    banner = np.full((h, w, 3), bg, dtype=np.uint8)
    cv2.putText(banner, text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 1, cv2.LINE_AA)
    return banner


def seg_overlay(img, seg_model, crop_top_frac, crop_bottom_frac, colors):
    """Run segmentation and return full-resolution overlay image."""
    h, w = img.shape[:2]
    y0 = int(h * crop_top_frac)
    y1 = h - int(h * crop_bottom_frac)
    crop = img[y0:y1].copy()
    ch, cw = crop.shape[:2]

    results = seg_model.predict(crop, conf=CONF_THRESH, iou=IOU_THRESH,
                                verbose=False, agnostic_nms=True)
    result = results[0]
    overlay = crop.copy()

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            cls_id = int(result.boxes.cls[i].item())
            color  = colors[cls_id]
            mask_r = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
            colored = np.zeros_like(crop)
            colored[mask_r > 0.5] = color
            cv2.addWeighted(colored, MASK_ALPHA, overlay, 1 - MASK_ALPHA, 0, overlay)

    vis = img.copy()
    vis[y0:y1] = overlay
    return vis


def cls_overlay(img, cls_model, crop_top):
    """Run wincls and return full-resolution overlay image."""
    mask = cls_model.swinpred(img, stride=STRIDE, mode=MODE, y_start=crop_top)
    h, w = img.shape[:2]
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(cls_model.num_classes):
        color_mask[mask == cls_id] = cls_model.colors[cls_id]
    return cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)


def make_legend(total_w, colors, rdict):
    """Single legend used by both panels (shared color palette)."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42
    thickness  = 1
    swatch_w   = 16
    item_h     = 22
    col_w      = 220
    cols       = max(1, total_w // col_w)
    rows       = -(-len(rdict) // cols)

    legend_h = rows * item_h + 30
    legend   = np.full((legend_h, total_w, 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(legend, "Classes", (8, 18),
                font, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    for idx, name in rdict.items():
        col_i = idx % cols
        row_i = idx // cols
        x = col_i * col_w + 8
        y = 30 + row_i * item_h + item_h // 2
        color = colors[idx]
        cv2.rectangle(legend, (x, y - 8), (x + swatch_w, y + 6), color, -1)
        cv2.putText(legend, name.replace("_", " "), (x + swatch_w + 4, y + 4),
                    font, font_scale, (210, 210, 210), thickness, cv2.LINE_AA)

    return legend


# ── load models ───────────────────────────────────────────────────────────────
print(f"Loading segmentation model: {SEG_MODEL}")
seg_model = YOLO(SEG_MODEL)

print(f"Loading classification model: {CLS_MODEL}")
cls_model = yolo_swin(CLS_MODEL, window_size=(WIN_W, WIN_H))

os.makedirs(OUT_DIR, exist_ok=True)
total_w = PANEL_W * 2

# ── per-image inference + comparison ─────────────────────────────────────────
for img_path, crop_top, seg_top_frac, seg_bot_frac in IMAGES:
    print(f"\n── {os.path.basename(img_path)}")
    img = cv2.imread(img_path)
    assert img is not None, f"Cannot read: {img_path}"

    print("  seg inference...")
    seg_vis = seg_overlay(img, seg_model, seg_top_frac, seg_bot_frac, cls_model.colors)

    print("  cls inference...")
    cls_vis = cls_overlay(img, cls_model, crop_top)

    # resize both panels to PANEL_W
    seg_panel = resize_to_width(seg_vis, PANEL_W)
    cls_panel = resize_to_width(cls_vis, PANEL_W)

    # match heights
    th = max(seg_panel.shape[0], cls_panel.shape[0])
    def pad_h(p, target_h):
        if p.shape[0] < target_h:
            pad = np.zeros((target_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            return np.vstack([p, pad])
        return p
    seg_panel = pad_h(seg_panel, th)
    cls_panel = pad_h(cls_panel, th)

    # label banners
    seg_banner = make_label_banner(
        f"SEG  {os.path.basename(SEG_MODEL)}  |  {os.path.basename(img_path)}", PANEL_W)
    cls_banner = make_label_banner(
        f"CLS  wincls  w{WIN_W}h{WIN_H} s{STRIDE} {MODE}  |  {os.path.basename(img_path)}", PANEL_W,
        fg=(100, 200, 255))

    left  = np.vstack([seg_banner, seg_panel])
    right = np.vstack([cls_banner, cls_panel])
    combined = np.hstack([left, right])

    legend  = make_legend(total_w, cls_model.colors, cls_model.rdict)
    result  = np.vstack([combined, legend])

    stem    = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(OUT_DIR, f"{stem}.jpg")
    cv2.imwrite(out_path, result)
    print(f"  saved → {out_path}")

print(f"\nDone. Results in {OUT_DIR}/")
