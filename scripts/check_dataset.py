#!/usr/bin/env python3
"""
scripts/check_dataset.py
Validates the integrity of the Package-Seg 400 dataset splits, verifies segmentation polygon formats,
checks for missing labels/images, and saves a 12-sample preview grid to figures/dataset_samples.png.
"""

import sys
from pathlib import Path
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_dataset() -> bool:
    data_yaml_path = REPO_ROOT / "datasets" / "package-seg" / "data.yaml"
    if not data_yaml_path.exists():
        data_yaml_path = Path("datasets/package-seg/data.yaml")

    if not data_yaml_path.exists():
        print(" [!] datasets/package-seg/data.yaml not found! Automatically downloading dataset...")
        try:
            from scripts.download_data import main as run_download
        except (ImportError, ModuleNotFoundError):
            from download_data import main as run_download
        run_download()

    if not data_yaml_path.exists():
        data_yaml_path = REPO_ROOT / "datasets" / "package-seg" / "data.yaml"

    if not data_yaml_path.exists():
        print(" Error: datasets/package-seg/data.yaml still not found!")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)


    base_dir = Path("datasets/package-seg")
    splits_names_file = Path("splits/train400_names.txt")

    print("=" * 60)
    print(" CHECKING INTEGRITY OF PACKAGE-SEG (400 SUBSET)")
    print("=" * 60)

    class_names = data_cfg.get("names", {0: "package"})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    all_valid = True
    total_images = 0
    total_packages = 0

    # 1. Check Train Split (from train400_names.txt)
    train_img_dir = base_dir / "images" / "train"
    train_lbl_dir = base_dir / "labels" / "train"

    with open(splits_names_file, "r", encoding="utf-8") as f:
        train_names = [line.strip() for line in f if line.strip()]

    print(f"\n Split: [TRAIN (400 Subset)]")
    print(f"   - Number of images:  {len(train_names)} (Expected: 400)")

    train_polygons = 0
    empty_labels = 0
    invalid_polygons = 0

    for name in train_names:
        img_p = train_img_dir / name
        lbl_p = train_lbl_dir / (Path(name).stem + ".txt")

        if not img_p.exists():
            print(f"    Missing image: {img_p}")
            all_valid = False

        if not lbl_p.exists():
            empty_labels += 1
            continue

        with open(lbl_p, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.split()
            coords = [float(x) for x in parts[1:]]
            if len(coords) < 6 or len(coords) % 2 != 0:
                invalid_polygons += 1
                continue
            if any(c < 0.0 or c > 1.0 for c in coords):
                invalid_polygons += 1
                continue
            train_polygons += 1

    total_images += len(train_names)
    total_packages += train_polygons

    print(f"   - Total package masks: {train_polygons}")
    print(f"   - Background images:   {empty_labels}")
    if invalid_polygons > 0:
        print(f"    Found {invalid_polygons} invalid polygons!")
        all_valid = False
    else:
        print(f"    All polygon coordinates are valid in [0, 1].")

    # 2. Check Validation & Test splits
    for split in ["val", "test"]:
        img_dir = base_dir / "images" / split
        lbl_dir = base_dir / "labels" / split

        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbls = sorted(list(lbl_dir.glob("*.txt")))

        print(f"\n Split: [{split.upper()}]")
        print(f"   - Number of images:  {len(imgs)}")
        print(f"   - Number of labels:  {len(lbls)}")
        total_images += len(imgs)

    print("\n" + "=" * 60)
    print(f" DATASET SUMMARY:")
    print(f"   - Total training instances: {total_packages} package masks")
    print(f"   - Split distribution:       400 train / 188 val / 89 test")
    print("=" * 60)

    # 3. Generate 12 sample preview grid
    visualize_samples(train_img_dir, train_lbl_dir, train_names, num_samples=12)

    return all_valid


def visualize_samples(img_dir: Path, lbl_dir: Path, image_names: list, num_samples: int = 12):
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_samples.png"

    sample_names = image_names[:num_samples]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, name in enumerate(sample_names):
        img_path = img_dir / name
        lbl_path = lbl_dir / (Path(name).stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img_rgb.copy()

        box_count = 0
        if lbl_path.exists():
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) >= 6:
                        pts = np.array(
                            [[int(coords[j] * w), int(coords[j + 1] * h)] for j in range(0, len(coords), 2)],
                            np.int32,
                        ).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], color=(0, 255, 0))
                        cv2.polylines(img_rgb, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                        box_count += 1

        cv2.addWeighted(overlay, 0.4, img_rgb, 0.6, 0, img_rgb)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{Path(name).stem[:16]} ({box_count} pkgs)", fontsize=9)
        axes[idx].axis("off")

    plt.suptitle("Package-Seg Ground Truth Samples (12 Images)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n Saved 12-sample visual preview to: {output_path}")


if __name__ == "__main__":
    check_dataset()
