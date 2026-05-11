import os
import cv2
import random
import numpy as np
import hashlib
from pathlib import Path


def find_material_folders(root):
    root_path = Path(root)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    valid_folders = [
        str(d) for d in root_path.iterdir()
        if d.is_dir() and any(f.suffix.lower() in exts for f in d.iterdir())
    ]
    return sorted(valid_folders)


def get_material_file_paths(folder_paths):
    materials_paths = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for folder in folder_paths:
        paths = [str(f) for f in Path(folder).iterdir() if f.suffix.lower() in exts]
        materials_paths.append(paths)
    return materials_paths


def generate_material_color_map(material_folders):
    color_map = {-1: (50, 50, 50)}
    for i, folder_path in enumerate(material_folders):
        name = Path(folder_path).name
        hash_hex = hashlib.md5(name.encode()).hexdigest()
        color = (
            int(hash_hex[0:2], 16) % 195 + 60,
            int(hash_hex[2:4], 16) % 195 + 60,
            int(hash_hex[4:6], 16) % 195 + 60
        )
        color_map[i] = color
    return color_map


def draw_legend(height, material_folders, color_palette):
    """Creates a vertical legend panel showing folder names and colors."""
    panel_w = 250
    legend_img = np.zeros((height, panel_w, 3), dtype=np.uint8) + 30  # Dark gray background

    cv2.putText(legend_img, "MATERIAL LEGEND", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, folder in enumerate(material_folders):
        color = color_palette.get(i, (0, 0, 0))
        name = Path(folder).name
        y_pos = 70 + (i * 30)

        # Draw color swatch
        cv2.rectangle(legend_img, (10, y_pos - 15), (35, y_pos + 5), color, -1)
        cv2.rectangle(legend_img, (10, y_pos - 15), (35, y_pos + 5), (255, 255, 255), 1)

        # Draw text
        cv2.putText(legend_img, name, (45, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return legend_img


def binarize_and_make_gt_mask(imgpath, material_list, target_size=(512, 512)):
    mask = cv2.imread(str(imgpath), cv2.IMREAD_GRAYSCALE)
    if mask is None: return None, None
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    gt_mask = binary.astype(np.int32)
    for cls_id in [0, 1]:
        class_bits = (gt_mask == cls_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_bits, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 10:
                gt_mask[labels == i] = -1

    if len(material_list) < 2: return None, None

    selected_indices = random.sample(list(range(len(material_list))), 2)
    gt_mask_orig = gt_mask.copy()
    gt_mask[gt_mask_orig == 0] = selected_indices[0]
    gt_mask[gt_mask_orig == 1] = selected_indices[1]

    return gt_mask, (binary * 255).astype(np.uint8)


def mix_gt_mask_for_semseg(gt_mask, materials_paths):
    h, w = gt_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in np.unique(gt_mask):
        if cls_id == -1: continue
        class_mask = (gt_mask == cls_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(class_mask)
        for label_idx in range(1, num_labels):
            cc_mask = (labels == label_idx)
            if cls_id >= len(materials_paths) or not materials_paths[cls_id]: continue
            img_path = random.choice(materials_paths[cls_id])
            source_img = cv2.imread(img_path)
            if source_img is not None:
                source_res = cv2.resize(source_img, (w, h))
                canvas[cc_mask] = source_res[cc_mask]
    return canvas


def draw_bboxes_and_labels(image, gt_mask, material_folders, color_palette):
    for cls_id in np.unique(gt_mask):
        if cls_id == -1: continue
        box_color = color_palette.get(cls_id, (0, 255, 0))
        class_mask = (gt_mask == cls_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask)
        material_name = Path(material_folders[cls_id]).name
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < 100: continue
            cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
            label_size, _ = cv2.getTextSize(material_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(image, (x, y - 18), (x + label_size[0] + 5, y), box_color, -1)
            brightness = sum(box_color) / 3
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            cv2.putText(image, material_name, (x + 2, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)


def demo_src_folder(mask_folder, material_root, target_size=(512, 512)):
    material_folders = find_material_folders(material_root)
    if not material_folders: return
    materials_paths = get_material_file_paths(material_folders)
    color_palette = generate_material_color_map(material_folders)
    mask_files = sorted([f for f in Path(mask_folder).iterdir() if f.suffix.lower() in {'.jpg', '.png'}])

    legend_panel = draw_legend(target_size[1], material_folders, color_palette)

    paused, idx = False, 0
    while idx < len(mask_files):
        gt_mask, bin_img_gray = binarize_and_make_gt_mask(mask_files[idx], material_folders, target_size)
        if gt_mask is None:
            idx += 1
            continue

        bin_viz = cv2.cvtColor(bin_img_gray, cv2.COLOR_GRAY2BGR)

        if not paused:
            display_img = bin_viz
            cv2.putText(display_img, "RUNNING - BINARY MASK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 1. Generate the base synthesized image (Clean View)
            synth_img_clean = mix_gt_mask_for_semseg(gt_mask, materials_paths)

            # 2. Create a copy for the annotated version
            synth_img_annotated = synth_img_clean.copy()
            draw_bboxes_and_labels(synth_img_annotated, gt_mask, material_folders, color_palette)

            # 3. Generate the Ground Truth color map
            gt_colorized = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            for uid in np.unique(gt_mask):
                gt_colorized[gt_mask == uid] = color_palette.get(uid, (0, 0, 0))

            # 4. Stack four views: Clean -> Annotated -> GT Color -> Binary
            main_viz = np.hstack((synth_img_clean, synth_img_annotated, gt_colorized, bin_viz))

            # Append legend
            display_img = np.hstack((main_viz, legend_panel))
            cv2.putText(display_img, "PAUSED - FULL INSPECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255),
                        2)

        cv2.putText(display_img, "P: Pause/Play | Q: Quit | Any: Next", (20, display_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Processing Pipeline", display_img)
        key = cv2.waitKey(0 if paused else 100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        else:
            if not paused: idx += 1

    cv2.destroyAllWindows()
def export_to_yolo(gt_mask, output_path, img_width, img_height):
    """Converts gt_mask to YOLO segmentation polygons and saves to .txt."""
    lines = []
    unique_ids = np.unique(gt_mask)

    for cls_id in unique_ids:
        if cls_id == -1: continue # Skip ignored pixels

        # Create a binary mask for this specific class
        class_mask = (gt_mask == cls_id).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) < 3: continue # Need at least 3 points for a polygon

            # Flatten and normalize coordinates
            polygon = []
            for point in contour:
                x, y = point[0]
                polygon.append(str(x / img_width))
                polygon.append(str(y / img_height))

            line = f"{int(cls_id)} " + " ".join(polygon)
            lines.append(line)

    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
def save_dataset(mask_folder, material_root, output_dir, samples_per_mask=1, target_size=(512, 512)):
    """
    Main loop to process images and save them in YOLO format.

    Args:
        samples_per_mask (int): How many different texture variations to
                                generate for every single mask frame.
    """
    output_path = Path(output_dir)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "labels").mkdir(parents=True, exist_ok=True)

    material_folders = find_material_folders(material_root)
    materials_paths = get_material_file_paths(material_folders)
    mask_files = sorted([f for f in Path(mask_folder).iterdir() if f.suffix.lower() in {'.jpg', '.png'}])

    print(f"Starting export: {len(mask_files)} masks x {samples_per_mask} samples = {len(mask_files) * samples_per_mask} total images.")

    for mask_path in mask_files:
        # Process the same mask multiple times
        for s_idx in range(samples_per_mask):
            # 1. Binarize and assign random classes (this picks new random materials each call)
            gt_mask, _ = binarize_and_make_gt_mask(mask_path, material_folders, target_size)
            if gt_mask is None: continue

            # 2. Synthesize image with the randomly assigned materials
            synth_img = mix_gt_mask_for_semseg(gt_mask, materials_paths)

            # 3. Define unique filename for this sample
            base_name = mask_path.stem
            file_id = f"{base_name}_s{s_idx}"
            img_file = output_path / "images" / f"{file_id}.jpg"
            txt_file = output_path / "labels" / f"{file_id}.txt"

            # 4. Save Image and YOLO labels
            cv2.imwrite(str(img_file), synth_img)
            export_to_yolo(gt_mask, txt_file, target_size[0], target_size[1])

        print(f"Finished processing mask: {mask_path.name}")

if __name__ == "__main__":
    TEXTURE_ROOT = "datasets/rscd/train"
    MASK_FOLDER = "/home/talt_wireten_c/road-segmentation/datasets/coco_binary_masks"
    OUTPUT_DATASET = "/home/talt_wireten_c/road-segmentation/masked_rscd_dataset_coco"

    # Example: Generate 5 unique images for every 1 mask frame
    save_dataset(MASK_FOLDER, TEXTURE_ROOT, OUTPUT_DATASET, samples_per_mask=5)
# if __name__ == "__main__":
#     TEXTURE_ROOT = "/run/media/lasercat/writebuffer/tmp/wiertenc/extract/rscd/train/"
#     MASK_FOLDER = "/run/media/lasercat/writebuffer/tmp/wiertenc/apple/frames/"
#     cv2.namedWindow("Processing Pipeline", cv2.WINDOW_NORMAL)
#     demo_src_folder(MASK_FOLDER, TEXTURE_ROOT)
