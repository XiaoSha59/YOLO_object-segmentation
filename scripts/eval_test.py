#!/usr/bin/env python3
"""
scripts/eval_test.py
Evaluates best.pt checkpoints of YOLO26-seg models on both validation and test splits (conf=0.001, imgsz=512).
Computes segmentation and bounding box metrics and exports results to metrics/comparison.csv.
"""

import os
import csv
from pathlib import Path
import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/train.yaml")
METRICS_DIR = Path("metrics")


def load_config() -> dict:
    """Loads evaluation configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_checkpoint(model_name: str) -> Path:
    """Finds best.pt checkpoint in runs/ or backup/ directories."""
    candidates = [
        Path("runs/segment") / model_name / "weights" / "best.pt",
        Path("backup") / model_name / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def evaluate_all():
    """Runs evaluation across all models on val and test splits, exporting a comparison table."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    data_yaml = cfg["data"]
    models = cfg.get("models", ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt"])
    imgsz = cfg.get("imgsz", 512)
    conf = cfg.get("conf_eval", 0.001)

    results_data = []

    for model_weight in models:
        model_name = Path(model_weight).stem
        ckpt_path = find_checkpoint(model_name)
        if not ckpt_path:
            print(f"[!] Warning: No checkpoint found for {model_name}, skipping.")
            continue

        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        model = YOLO(str(ckpt_path))

        for split in ["val", "test"]:
            print(f"[*] Evaluating {model_name} on {split} split...")
            metrics = model.val(
                data=data_yaml,
                split=split,
                imgsz=imgsz,
                conf=conf,
                device=cfg.get("device", 0),
                verbose=False,
            )

            # Segmentation metrics
            mask_map50 = metrics.seg.map50 if hasattr(metrics, "seg") else 0.0
            mask_map = metrics.seg.map if hasattr(metrics, "seg") else 0.0
            mask_p = metrics.seg.mp if hasattr(metrics, "seg") else 0.0
            mask_r = metrics.seg.mr if hasattr(metrics, "seg") else 0.0

            # Box metrics
            box_map50 = metrics.box.map50 if hasattr(metrics, "box") else 0.0
            box_map = metrics.box.map if hasattr(metrics, "box") else 0.0

            row = {
                "Model": model_name,
                "Split": split,
                "Mask_mAP50": f"{mask_map50:.4f}",
                "Mask_mAP50_95": f"{mask_map:.4f}",
                "Mask_P": f"{mask_p:.4f}",
                "Mask_R": f"{mask_r:.4f}",
                "Box_mAP50": f"{box_map50:.4f}",
                "Box_mAP50_95": f"{box_map:.4f}",
                "Size_MB": f"{size_mb:.2f}",
            }
            results_data.append(row)

    if not results_data:
        print("[!] No evaluation metrics collected. Ensure trained checkpoints exist.")
        return

    csv_path = METRICS_DIR / "comparison.csv"
    headers = list(results_data[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\n[+] Saved evaluation summary to {csv_path}\n")

    # Display Markdown table in stdout
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in results_data:
        print("| " + " | ".join([r[k] for k in headers]) + " |")


if __name__ == "__main__":
    evaluate_all()
