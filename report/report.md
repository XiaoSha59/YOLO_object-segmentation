# Technical Report: Package Instance Segmentation with YOLO26-seg (Package-Seg 400)

## 1. Problem Formulation & Objectives
- **Task:** Instance Segmentation of individual packages and parcels in logistics and warehouse environments.
- **Objectives:**
  - Benchmark and compare three model scales: **YOLO26n-seg**, **YOLO26s-seg**, and **YOLO26m-seg** on a standardized 400-image training subset.
  - Evaluate performance primarily using **Mask mAP50–95** along with bounding box metrics.
  - Implement an automated package counting pipeline derived from instance mask detections.

## 2. Dataset & Subset Strategy (Package-Seg 400)
- **Source:** Ultralytics official Package-Seg release archive (`package-seg.zip`).
- **Archive SHA-256 Checksum:** `62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6`
- **Data Leakage Prevention & Filtering:**
  - Full dataset contains: 1,920 train images, 188 validation images, and 89 test images.
  - Many training samples contain synthetic augmentations generated with `.rf.` filename extensions sharing identical source images.
  - Filtered out all training image groups whose base names appear in validation or test splits.
  - Selected a deterministic subset of 400 unique images (`seed=42`, sorted filenames) recorded in `splits/train400_names.txt`.

## 3. Experimental Setup & Hyperparameters
- **Architectures:** YOLO26n-seg (~2.7M params), YOLO26s-seg (~10.4M params), YOLO26m-seg (~23.6M params).
- **Training Hyperparameters:**
  - Epochs: 30 (patience: 30)
  - Batch size: 4
  - Input resolution: 512x512
  - Optimizer: AdamW (initial lr0 = 0.001, weight decay = 0.0005)
  - Close mosaic: 5 epochs
  - Reproducibility: seed = 42, deterministic = True
- **Hardware Platform:** Kaggle GPU T4 (single-device training with `device=0`).

## 4. Benchmark Results (Mask mAP50–95)
*(Populated from `metrics/comparison.csv` post-training)*

| Model | Split | Mask mAP50 | Mask mAP50–95 | Mask Precision | Mask Recall | Box mAP50 | Box mAP50–95 | Training Time | Size (MB) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n-seg | Val | | | | | | | | |
| YOLO26s-seg | Val | | | | | | | | |
| YOLO26m-seg | Val | | | | | | | | |
| **Best Model** | Test | | | | | | | | |

## 5. Qualitative Analysis & Package Counting Application
- Qualitative inspection across dense clusters, severe occlusions, and background images without packages.
- Automated package counter using instance masks (`conf=0.25`, `retina_masks=True`).

## 6. Limitations & Future Directions
- Evaluated on a 400-image subset at 512x512 resolution.
- Single-seed evaluation.
- Does not infer 3D bounding volumes or spatial depth coordinates.
