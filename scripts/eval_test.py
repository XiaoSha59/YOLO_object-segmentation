#!/usr/bin/env python3
"""
scripts/eval_test.py
Evaluates trained YOLO26-seg models on the Package-Seg test split (conf=0.001).
Computes segmentation and bounding box benchmark metrics and exports to metrics/comparison_test.csv.
"""

import sys
import os
from pathlib import Path
import csv
import yaml
from ultralytics import YOLO

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluate_models():
    data_yaml = "datasets/package-seg/data.yaml"
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models = ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg"]
    rows = []

    print("=" * 65)
    print(" BENCHMARK EVALUATION ON TEST SET (Package-Seg, conf=0.001)")
    print("=" * 65)

    for model_tag in models:
        cand_paths = [
            Path(f"runs/segment/{model_tag}/weights/best.pt"),
            Path(f"backup/{model_tag}/best.pt"),
            Path(f"checkpoints/{model_tag}/best.pt"),
        ]
        best_pt = None
        for p in cand_paths:
            if p.exists():
                best_pt = p
                break

        if not best_pt:
            print(f" Checkpoint not found for {model_tag}. Skipping.")
            continue

        file_size_mb = best_pt.stat().st_size / (1024 * 1024)
        print(f"\n Evaluating {model_tag} ({best_pt})...")
        model = YOLO(str(best_pt))

        metrics = model.val(
            data=data_yaml,
            split="test",
            conf=0.001,
            device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
            project="runs/eval_test",
            name=model_tag,
            exist_ok=True,
            verbose=False,
        )

        # Segmentation metrics
        mask = metrics.seg if hasattr(metrics, "seg") else None
        mask_p = float(mask.mp) if mask else 0.0
        mask_r = float(mask.mr) if mask else 0.0
        mask_map50 = float(mask.map50) if mask else 0.0
        mask_map = float(mask.map) if mask else 0.0

        # Box metrics
        box = metrics.box if hasattr(metrics, "box") else None
        box_map50 = float(box.map50) if box else 0.0
        box_map = float(box.map) if box else 0.0

        rows.append({
            "Model": model_tag,
            "Size (MB)": round(file_size_mb, 2),
            "Mask_Precision": round(mask_p, 4),
            "Mask_Recall": round(mask_r, 4),
            "Mask_mAP50": round(mask_map50, 4),
            "Mask_mAP50-95": round(mask_map, 4),
            "Box_mAP50": round(box_map50, 4),
            "Box_mAP50-95": round(box_map, 4),
        })

    if rows:
        out_csv = metrics_dir / "comparison_test.csv"
        headers = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        print("\n" + "=" * 65)
        print(f" BENCHMARK RESULTS ON TEST SET (Saved to {out_csv}):")
        print("=" * 65)
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            print("| " + " | ".join([str(r[h]) for h in headers]) + " |")
    else:
        print(" No model checkpoints found for evaluation.")


if __name__ == "__main__":
    evaluate_models()
