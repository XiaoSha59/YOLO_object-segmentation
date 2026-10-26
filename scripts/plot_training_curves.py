#!/usr/bin/env python3
"""
scripts/plot_training_curves.py
Plots authentic training and validation trajectories across 30 epochs for YOLO26s-seg on Package-Seg 400.
"""

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
REPORT_DIR = REPO_ROOT / "report"

with open(REPO_ROOT / "kaggle_outputs/yolo26-object-segmentation part 1.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

s_out = "".join(nb["cells"][9]["outputs"][0]["text"])

# Parse train losses: (epoch, box_loss, seg_loss, cls_loss)
epochs_data = re.findall(r"\s+(\d+)/30\s+[\d\.]+G\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", s_out)
# Parse val metrics: (box_p, box_r, box_map50, box_map, mask_p, mask_r, mask_map50, mask_map)
val_data = re.findall(r"all\s+188\s+693\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)", s_out)

epochs = [int(x[0]) for x in epochs_data]
train_box_loss = [float(x[1]) for x in epochs_data]
train_seg_loss = [float(x[2]) for x in epochs_data]
train_cls_loss = [float(x[3]) for x in epochs_data]

# Take the first 30 val lines
val_lines = val_data[:len(epochs)]
val_box_map50 = [float(x[2]) for x in val_lines]
val_box_map = [float(x[3]) for x in val_lines]
val_mask_p = [float(x[4]) for x in val_lines]
val_mask_r = [float(x[5]) for x in val_lines]
val_mask_map50 = [float(x[6]) for x in val_lines]
val_mask_map = [float(x[7]) for x in val_lines]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), dpi=300)

plots_config = [
    (axes[0, 0], epochs, train_box_loss, "train/box_loss", "#2563eb", "Box Loss"),
    (axes[0, 1], epochs, train_seg_loss, "train/seg_loss", "#059669", "Segmentation Loss"),
    (axes[0, 2], epochs, train_cls_loss, "train/cls_loss", "#d97706", "Class Loss"),
    (axes[0, 3], epochs, val_mask_p, "metrics/mask_precision", "#7c3aed", "Mask Precision"),
    (axes[1, 0], epochs, val_mask_r, "metrics/mask_recall", "#db2777", "Mask Recall"),
    (axes[1, 1], epochs, val_mask_map50, "metrics/mask_mAP50", "#0284c7", "Mask mAP50"),
    (axes[1, 2], epochs, val_mask_map, "metrics/mask_mAP50-95", "#16a34a", "Mask mAP50-95"),
    (axes[1, 3], epochs, val_box_map, "metrics/box_mAP50-95", "#ea580c", "Box mAP50-95"),
]

for ax, x, y, title, color, ylabel in plots_config:
    ax.plot(x, y, color=color, linewidth=2.0, marker="o", markersize=3, label="YOLO26s-seg")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Epochs", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(1, 30)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=9)

plt.tight_layout()
out_fig = FIGURES_DIR / "training_curves.png"
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
fig.savefig(REPORT_DIR / "training_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[+] Successfully generated {out_fig}")
