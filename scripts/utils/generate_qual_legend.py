"""
Generates qual_legend.png — standalone legend for qualitative result figures.
Uses road_color_dict (the shared palette used by both cls and seg overlays).
"""

import importlib.util, os
import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_spec = importlib.util.spec_from_file_location("wincls", "scripts/utils/wincls(1).py")
_wincls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wincls)

colors = _wincls.road_color_dict   # {int: (B, G, R)}
CLASS_NAMES = [
    "dry_asphalt_severe",   "dry_asphalt_slight",   "dry_asphalt_smooth",
    "dry_concrete_severe",  "dry_concrete_slight",  "dry_concrete_smooth",
    "dry_gravel",           "dry_mud",
    "fresh_snow",           "ice",                  "melted_snow",
    "water_asphalt_severe", "water_asphalt_slight", "water_asphalt_smooth",
    "water_concrete_severe","water_concrete_slight","water_concrete_smooth",
    "water_gravel",         "water_mud",
    "wet_asphalt_severe",   "wet_asphalt_slight",   "wet_asphalt_smooth",
    "wet_concrete_severe",  "wet_concrete_slight",  "wet_concrete_smooth",
    "wet_gravel",           "wet_mud",
]

COLS       = 3
SWATCH_W   = 18
SWATCH_H   = 14
TEXT_X_OFF = 24
ROW_H      = 26
COL_W      = 240
PAD        = 14
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.46
THICKNESS  = 1
BG         = (30, 30, 30)
FG         = (215, 215, 215)

rows   = -(-len(CLASS_NAMES) // COLS)
img_w  = COLS * COL_W + 2 * PAD
img_h  = rows * ROW_H + 2 * PAD
canvas = np.full((img_h, img_w, 3), BG, dtype=np.uint8)

for idx, name in enumerate(CLASS_NAMES):
    col_i = idx % COLS
    row_i = idx // COLS
    x = PAD + col_i * COL_W
    y = PAD + row_i * ROW_H + ROW_H // 2

    color = colors[idx]
    cv2.rectangle(canvas, (x, y - SWATCH_H // 2), (x + SWATCH_W, y + SWATCH_H // 2), color, -1)
    cv2.putText(canvas, name.replace("_", " "), (x + TEXT_X_OFF, y + 5),
                FONT, FONT_SCALE, FG, THICKNESS, cv2.LINE_AA)

out = "qual_legend.png"
cv2.imwrite(out, canvas)
print(f"Saved: {out}  ({img_w}×{img_h})")
