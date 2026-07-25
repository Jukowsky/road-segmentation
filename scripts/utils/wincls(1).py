import torch
import numpy as np
import cv2
import tqdm
from ultralytics import YOLO

import torch
import numpy as np
import cv2
from ultralytics import YOLO

road_color_dict = {
    # Asphalt (Dark Grays)
    0: (40, 40, 40),  # dry_asphalt_severe
    1: (60, 60, 60),  # dry_asphalt_slight
    2: (80, 80, 80),  # dry_asphalt_smooth

    # Concrete (Light Grays)
    3: (140, 140, 140),  # dry_concrete_severe
    4: (160, 160, 160),  # dry_concrete_slight
    5: (180, 180, 180),  # dry_concrete_smooth

    # Natural Dry (Browns/Tan)
    6: (139, 119, 101),  # dry_gravel
    7: (101, 67, 33),  # dry_mud

    # Winter Conditions (White/Cyan)
    8: (255, 250, 250),  # fresh_snow
    9: (173, 216, 230),  # ice
    10: (200, 225, 225),  # melted_snow

    # Water Covered (Deep Blues)
    11: (0, 0, 128),  # water_asphalt_severe
    12: (0, 0, 160),  # water_asphalt_slight
    13: (0, 0, 192),  # water_asphalt_smooth
    14: (30, 144, 255),  # water_concrete_severe
    15: (65, 105, 225),  # water_concrete_slight
    16: (100, 149, 237),  # water_concrete_smooth
    17: (0, 105, 148),  # water_gravel
    18: (47, 79, 79),  # water_mud

    # Wet/Damp (Slate Blues/Muted Grays)
    19: (70, 130, 180),  # wet_asphalt_severe
    20: (100, 149, 237),  # wet_asphalt_slight
    21: (119, 136, 153),  # wet_asphalt_smooth
    22: (176, 196, 222),  # wet_concrete_severe
    23: (188, 210, 238),  # wet_concrete_slight
    24: (211, 211, 211),  # wet_concrete_smooth
    25: (139, 137, 137),  # wet_gravel
    26: (84, 56, 30)  # wet_mud
}
import torch
import numpy as np
import cv2
import tqdm
import os
from ultralytics import YOLO



class yolo_swin:
    def __init__(this, model, window_size=(360, 240)):
        this.model = YOLO(model)
        this.rdict = this.model.names
        this.num_classes = len(this.rdict)
        imgsz = this.model.ckpt["train_args"]["imgsz"]
        this.yolo_input_size = imgsz[0] if isinstance(imgsz, list) else imgsz
        this.win_w, this.win_h = window_size
        this.colors = road_color_dict
    
    def swinpred(this, img, stride=112, mode='prob', patch_save_path=None, y_start=0):
        """
        patch_save_path: Base directory. Inside, subfolders will be created for each class label.
        y_start: first row to begin sliding window (rows above are left as no-prediction).
        """
        h, w = img.shape[:2]
        canvas = np.zeros((this.num_classes, h, w), dtype=np.float32)
        accumulate_fn = this.accumulate_prob if mode == 'prob' else this.accumulate_bin

        counter = 0
        for y in tqdm.tqdm(range(y_start, h, stride)):
            for x in range(0, w, stride):
                y_end = min(y + this.win_h, h)
                x_end = min(x + this.win_w, w)

                patch = img[y:y_end, x:x_end]

                # Prepare and Predict
                patch_for_model = cv2.resize(patch, (this.yolo_input_size, this.yolo_input_size))
                results = this.model.predict(patch_for_model, verbose=False)

                # Extract predicted class label for folder naming
                # results[0].probs.top1 gives the index of the highest probability class
                class_idx = results[0].probs.top1
                class_name = this.rdict[class_idx]

                # Save patch into its specific label folder
                if patch_save_path:
                    # Create path: patch_save_path/class_name/
                    label_dir = os.path.join(patch_save_path, str(class_name))
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir, exist_ok=True)

                    p_name = os.path.join(label_dir, f"patch_{counter}_{y}_{x}.jpg")
                    cv2.imwrite(p_name, patch)
                    counter += 1

                accumulate_fn(canvas, (y, x), (y_end, x_end), results)

        # Normalization and Mask creation
        sums = canvas.sum(axis=0, keepdims=True)
        sums[sums == 0] = 1.0
        norm_canvas = canvas / sums

        semantic_mask = np.argmax(norm_canvas, axis=0)
        semantic_mask[canvas.sum(axis=0) == 0] = -1

        return semantic_mask
    def accumulate_bin(this, canvas, win_tl, win_br, results):
        for r in results:
            top1_idx = r.probs.top1
            canvas[top1_idx, win_tl[0]:win_br[0], win_tl[1]:win_br[1]] += 1

    def accumulate_prob(this, canvas, win_tl, win_br, results):
        for r in results:
            p_vector = r.probs.data.cpu().numpy()
            canvas[:, win_tl[0]:win_br[0], win_tl[1]:win_br[1]] += p_vector[:, None, None]

    def visualize(this, img, semantic_mask, title="Classifier Swin-Segmentation", save_path=None):
        """
        save_path: Full file path (including .jpg/.png) to save the final montage.
        """
        h, w = img.shape[:2]
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_id in range(this.num_classes):
            color_mask[semantic_mask == cls_id] = this.colors[cls_id]

        overlay = cv2.addWeighted(img, 0.5, color_mask, 0.5, 0)
        montage = np.vstack([img, color_mask, overlay])

        target_w = 1080
        aspect_ratio = montage.shape[0] / montage.shape[1]
        target_h = int(target_w * aspect_ratio)
        montage_res = cv2.resize(montage, (target_w, target_h), interpolation=cv2.INTER_AREA)

        line_height = 36
        x_start, y_start = 20, 28
        x_ptr, y_ptr = x_start, y_start
        item_spacing = 220
        items_per_row = max(1, (target_w - x_start) // item_spacing)
        rows = -(-len(this.rdict) // items_per_row)  # ceiling division
        footer_h = y_start + rows * line_height + 10
        footer = np.zeros((footer_h, target_w, 3), dtype=np.uint8)

        for i, name in this.rdict.items():
            if x_ptr + item_spacing > target_w:
                x_ptr = x_start
                y_ptr += line_height
            c = [int(x) for x in this.colors[i]]
            cv2.rectangle(footer, (x_ptr, y_ptr - 18), (x_ptr + 22, y_ptr + 6), c, -1)
            cv2.putText(footer, str(name), (x_ptr + 30, y_ptr),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            x_ptr += item_spacing

        res = np.vstack([montage_res, footer])

        if save_path:
            # Ensure the directory for the save path exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, res)
            print(f"Result saved to {save_path}")

        cv2.namedWindow(title, 0)
        cv2.imshow(title, res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # ── CONFIG ────────────────────────────────────────────────────────────────
    MODEL  = "runs/classify/runs/classify/exp10_rscd_full2/weights/best.pt"
    OUT_DIR = "wincls_grillom_results"

    WIN_W        = 30     # window width  (px)
    WIN_H        = 45     # window height (px)
    STRIDE       = 20     # step size     (px)
    MODE         = "prob" # "prob" | "bin"
    SAVE_PATCHES = None   # set to a dir path to save patches per predicted class

    # (image_path, crop_top)  — crop_top rows are skipped by the sliding window
    IMAGES = [
        ("datasets/Grillom_SE_STA_CAMERA_Geni_4740_K1/2026-01-31_15_17_33_d6b88a6af1ceb2d2c7d0d122b28839ed46098ccb629ad14ddb0b5eb95253fdbb.jpg", 120),
        ("datasets/rcm-samples/RCMY196_20260305_0844_008.jpg", 400),
        ("datasets/rcm-samples/RCMY196_20260302_0846_014.jpg", 400),
        ("datasets/rcm-samples/RCMY196_20260305_0844_022.jpg", 400),
    ]
    # ─────────────────────────────────────────────────────────────────────────

    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    mod = yolo_swin(MODEL, window_size=(WIN_W, WIN_H))
    print(mod.rdict)

    for img_path, crop_top in IMAGES:
        im = cv2.imread(img_path)
        assert im is not None, f"Could not read: {img_path}"
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out = os.path.join(OUT_DIR, f"{stem}_w{WIN_W}h{WIN_H}_s{STRIDE}_{MODE}.jpg")
        mask = mod.swinpred(im, stride=STRIDE, mode=MODE, patch_save_path=SAVE_PATCHES, y_start=crop_top)
        mod.visualize(im, mask, save_path=out)