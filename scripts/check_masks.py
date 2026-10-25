#!/usr/bin/env python3
"""
scripts/check_masks.py
Validates segmentation polygon annotations (minimum 3 points, normalized coords in [0, 1]),
summarizes background vs annotated images, and plots 16 visual sample overlays in figures/gt_samples/.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASETS_DIR = Path("datasets/package-seg")
TRAIN_NAMES_PATH = Path("splits/train400_names.txt")
FIGURES_DIR = Path("figures/gt_samples")


def validate_labels() -> bool:
    """Validates segmentation polygon labels for the selected 400 training images."""
    if not TRAIN_NAMES_PATH.exists():
        print(f"[!] Error: {TRAIN_NAMES_PATH} not found. Please run prepare_subset.py first.")
        return False

    with open(TRAIN_NAMES_PATH, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]

    labels_dir = DATASETS_DIR / "labels" / "train"

    empty_count = 0
    total_polygons = 0
    invalid_lines = []

    print("[*] Validating 400 training label files...")
    for img_name in selected_names:
        label_name = Path(img_name).stem + ".txt"
        label_path = labels_dir / label_name

        if not label_path.exists():
            empty_count += 1
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) == 0:
            empty_count += 1
            continue

        for idx, line in enumerate(lines):
            parts = line.split()
            coords = [float(x) for x in parts[1:]]

            # A valid polygon requires at least 3 points (6 normalized coordinates)
            if len(coords) < 6 or len(coords) % 2 != 0:
                invalid_lines.append((label_name, idx + 1, "Fewer than 3 vertices or odd number of coordinates"))
                continue

            # Coordinates must be normalized within [0, 1]
            if any(c < 0.0 or c > 1.0 for c in coords):
                invalid_lines.append((label_name, idx + 1, "Coordinates out of bounds [0, 1]"))
                continue

            total_polygons += 1

    print(f"[+] Total annotated package polygons: {total_polygons}")
    print(f"[+] Verified background images (no polygons): {empty_count}")

    if invalid_lines:
        print(f"[!] Detected {len(invalid_lines)} invalid label entries:")
        for inv in invalid_lines[:5]:
            print(f"    - {inv[0]} line {inv[1]}: {inv[2]}")
        return False

    print("[+] All label files passed polygon integrity validation.")
    return True


def draw_gt_samples(num_samples: int = 16):
    """Renders ground truth polygon overlays onto sample images and saves a grid image."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_NAMES_PATH, "r", encoding="utf-8") as f:
        selected_names = [line.strip() for line in f if line.strip()]

    labels_dir = DATASETS_DIR / "labels" / "train"
    images_dir = DATASETS_DIR / "images" / "train"

    sample_names = selected_names[:num_samples]
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i, img_name in enumerate(sample_names):
        img_path = images_dir / img_name
        label_path = labels_dir / (Path(img_name).stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()

        polygon_count = 0
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line in lines:
                parts = line.split()
                coords = [float(x) for x in parts[1:]]
                if len(coords) >= 6:
                    pts = np.array(
                        [[int(coords[j] * w), int(coords[j + 1] * h)] for j in range(0, len(coords), 2)],
                        np.int32,
                    )
                    pts = pts.reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                    cv2.polylines(img_rgb, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    polygon_count += 1

        # Semi-transparent overlay blending
        cv2.addWeighted(overlay, 0.4, img_rgb, 0.6, 0, img_rgb)

        axes[i].imshow(img_rgb)
        axes[i].set_title(f"{Path(img_name).stem[:16]}... ({polygon_count} pkgs)", fontsize=9)
        axes[i].axis("off")

    for j in range(num_samples, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    output_path = FIGURES_DIR / "gt_samples_grid.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[+] Saved GT visual overlay grid to {output_path}")


def main():
    print("=== Step 2: Validate Segmentation Annotations & Render Ground Truth ===")
    if validate_labels():
        draw_gt_samples()
    print("=== Annotation Validation Completed! ===")


if __name__ == "__main__":
    main()
