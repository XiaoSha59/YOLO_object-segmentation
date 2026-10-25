#!/usr/bin/env python3
"""
scripts/demo_count.py
Package counting demo utilizing YOLO26 instance segmentation masks (conf=0.25, retina_masks=True).
Detects packages, counts individual instances, and saves annotated visualization to figures/demo_counting.png.
"""

import sys
import os
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Package Counting Demo with YOLO26-seg")
    parser.add_argument("--image", default=None, help="Path to input image (default: first image in test set)")
    parser.add_argument("--model", default=None, help="Path to checkpoint best.pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()

    # Search for available checkpoint
    model_path = args.model
    if not model_path:
        for cand in [
            "backup/yolo26s-seg/best.pt",
            "runs/segment/yolo26s-seg/weights/best.pt",
            "backup/yolo26m-seg/best.pt",
            "runs/segment/yolo26m-seg/weights/best.pt",
            "backup/yolo26n-seg/best.pt",
            "runs/segment/yolo26n-seg/weights/best.pt",
        ]:
            if Path(cand).exists():
                model_path = cand
                break

    if not model_path or not Path(model_path).exists():
        print(" Checkpoint best.pt not found! Train models first or specify --model.")
        return

    # Find test image
    image_path = args.image
    if not image_path:
        test_images = sorted(
            list(Path("datasets/package-seg/images/test").glob("*.jpg"))
            + list(Path("datasets/package-seg/images/test").glob("*.png"))
        )
        if not test_images:
            print(" Error: No test images found in datasets/package-seg/images/test!")
            return
        image_path = str(test_images[0])

    print(f" Running package counting demo with model: {model_path}")
    print(f" Input image: {image_path} (conf={args.conf})")

    model = YOLO(model_path)
    results = model.predict(image_path, conf=args.conf, retina_masks=True, save=False)
    res = results[0]

    count = len(res.masks.data) if (res.masks is not None and res.masks.data is not None) else 0

    print("\n" + "=" * 50)
    print(" PACKAGE COUNTING RESULTS:")
    print(f"    Total detected packages: {count}")
    print("=" * 50)

    # Plot annotated result image
    annotated_bgr = res.plot(labels=False, boxes=True)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "demo_counting.png"

    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_rgb)
    title_text = f"Package Instance Segmentation & Counting (YOLO26-seg)\nDetected Packages: {count}"
    plt.title(title_text, fontsize=13, fontweight="bold", pad=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"\n Saved annotated counting result to: {out_path}")


if __name__ == "__main__":
    main()
