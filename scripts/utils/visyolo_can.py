import cv2
import numpy as np
import os
import random

def visyoloseg(imgroot, annoroot, id_color_dict=None, max_images=20):
    if id_color_dict is None:
        id_color_dict = {}
        random.seed(9)

    def get_color(class_id):
        if class_id not in id_color_dict:
            id_color_dict[class_id] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        return id_color_dict[class_id]

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(imgroot) if f.lower().endswith(img_extensions)][:max_images]

    os.makedirs("visyolo_output", exist_ok=True)

    for img_name in images:
        img_path = os.path.join(imgroot, img_name)
        anno_path = os.path.join(annoroot, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(anno_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        clean_img = img.copy()
        mask_only = np.zeros((h, w, 3), dtype=np.uint8)
        overlay = img.copy()
        result_img = img.copy()

        with open(anno_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 3:
                    continue
                class_id = int(parts[0])
                coords = np.array(parts[1:]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                pts = coords.astype(np.int32).reshape((-1, 1, 2))
                color = get_color(class_id)
                cv2.fillPoly(mask_only, [pts], color)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(result_img, [pts], True, color, 2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)
        combined_view = np.hstack((clean_img, mask_only, result_img))
        cv2.imwrite(f"visyolo_output/{img_name}", combined_view)
        print(f"Saved {img_name}")

visyoloseg("masked_rscd_dataset_coco/images", "masked_rscd_dataset_coco/labels")