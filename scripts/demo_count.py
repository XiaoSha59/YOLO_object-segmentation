#!/usr/bin/env python3
"""
scripts/demo_count.py
Package counting demo utilizing instance segmentation masks (conf=0.25, retina_masks=True).
Automatically selects the model with the highest validation Mask mAP50-95 or accepts a custom weight via CLI.
Renders annotated instance masks and counter overlays to figures/demo_counts/.
"""

import os
import argparse
from pathlib import Path
import csv
import cv2
import numpy as np
from ultralytics import YOLO

DEMO_DIR = Path("figures/demo_counts")
METRICS_CSV = Path("metrics/comparison.csv")
TEST_IMAGES_DIR = Path("datasets/package-seg/images/test")


def get_best_model_checkpoint() -> Path:
    """Finds the model with the highest Mask mAP50-95 on validation set from metrics/comparison.csv."""
    if METRICS_CSV.exists():
        best_map = -1.0
        best_model = None
        with open(METRICS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Split") == "val":
                    m_val = float(row.get("Mask_mAP50_95", 0))
                    if m_val > best_map:
                        best_map = m_val
                        best_model = row.get("Model")

        if best_model:
            for ckpt in [
                Path("runs/segment") / best_model / "weights" / "best.pt",
                Path("backup") / best_model / "best.pt",
            ]:
                if ckpt.exists():
                    print(f"[*] Auto-selected best model from validation metrics: {best_model} (mAP={best_map:.4f})")
                    return ckpt

    # Fallback search order
    for candidate in ["yolo26m-seg", "yolo26s-seg", "yolo26n-seg"]:
        p = Path("runs/segment") / candidate / "weights" / "best.pt"
        if p.exists():
            return p
        b = Path("backup") / candidate / "best.pt"
        if b.exists():
            return b

    return None


def run_demo(weights_path: str = None, num_images: int = 8):
    """Executes instance segmentation and package counting on test images."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    if weights_path:
        ckpt = Path(weights_path)
    else:
        ckpt = get_best_model_checkpoint()

    if not ckpt or not ckpt.exists():
        print("[!] Error: No model checkpoint found. Please train models first or specify --weights.")
        return

    print(f"[*] Loading model weights from: {ckpt}")
    model = YOLO(str(ckpt))

    if not TEST_IMAGES_DIR.exists():
        print(f"[!] Error: Test images directory not found at {TEST_IMAGES_DIR}")
        return

    test_imgs = sorted([f for f in TEST_IMAGES_DIR.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])[
        :num_images
    ]

    print(f"[*] Executing package counting demo on {len(test_imgs)} test images...")

    for img_path in test_imgs:
        results = model.predict(
            source=str(img_path),
            conf=0.25,
            retina_masks=True,
            save=False,
            verbose=False,
        )

        result = results[0]
        count = len(result.masks.data) if (result.masks is not None and result.masks.data is not None) else 0

        # Render instance masks and bounding boxes
        annotated_bgr = result.plot(labels=False, boxes=True)

        # Draw package counter banner
        banner_text = f"Count: {count} packages"
        cv2.rectangle(annotated_bgr, (10, 10), (330, 60), (0, 0, 0), -1)
        cv2.putText(
            annotated_bgr,
            banner_text,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        out_file = DEMO_DIR / f"demo_count_{img_path.name}"
        cv2.imwrite(str(out_file), annotated_bgr)
        print(f"[+] {img_path.name}: Detected {count} packages -> Saved to {out_file}")

    print(f"\n[+] Demo completed successfully! Results saved to {DEMO_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Package Counting Demo using YOLO26-seg Masks")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom model weights (.pt file)")
    parser.add_argument("--num", type=int, default=8, help="Number of test images to evaluate")
    args = parser.parse_args()

    run_demo(args.weights, args.num)


if __name__ == "__main__":
    main()
