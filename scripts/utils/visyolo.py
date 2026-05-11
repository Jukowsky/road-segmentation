import cv2
import numpy as np
import os
import random

def visyoloseg(imgroot, annoroot, id_color_dict=None):
    """
    Visualizes YOLO segmentation with three panels:
    Original, Black Mask, and Alpha-Blended Result.
    """
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
    images = [f for f in os.listdir(imgroot) if f.lower().endswith(img_extensions)]

    for img_name in images:
        img_path = os.path.join(imgroot, img_name)
        anno_path = os.path.join(annoroot, os.path.splitext(img_name)[0] + '.txt')

        if not os.path.exists(anno_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        # Panel 1: Original
        clean_img = img.copy()

        # Panel 2: Mask Only (Starts as black)
        mask_only = np.zeros((h, w, 3), dtype=np.uint8)

        # Panel 3: Blended Result
        overlay = img.copy()
        result_img = img.copy()

        # Parse YOLO segmentation annotations
        if os.path.exists(anno_path):
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

                    # Draw on Mask Only panel
                    cv2.fillPoly(mask_only, [pts], color)

                    # Draw on Result panel (Overlay + Contours)
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(result_img, [pts], True, color, 2)

        # Apply Alpha blending to the result panel
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0, result_img)

        # Concatenate: [Original | Mask | Result]
        combined_view = np.hstack((clean_img, mask_only, result_img))

        # Display
        import os
        os.makedirs("visyolo_output", exist_ok=True)
        cv2.imwrite(f"visyolo_output/{img_name}", combined_view)
        print(f"Saved {img_name}")

    cv2.destroyAllWindows()

# Use WINDOW_NORMAL to allow resizing since 3 panels can be very wide
cv2.namedWindow('YOLO: Original | Mask | Result', cv2.WINDOW_NORMAL)
visyoloseg("yolo_dataset/images", "yolo_dataset/labels")
