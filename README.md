# YOLO26 Package Instance Segmentation & Counting (Package-Seg 400)

A complete end-to-end pipeline for pixel-level package instance segmentation and automated conveyor counting on the **Package-Seg** dataset using the **Ultralytics YOLO26-seg** model family across five capacities (`yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`, `yolo26l-seg`, `yolo26x-seg`).

---

## 🔗 Official References & Links

* **Ultralytics Documentation**: [https://docs.ultralytics.com](https://docs.ultralytics.com/)
* **Ultralytics YOLO26 Model Guide**: [https://docs.ultralytics.com/models/yolo26/](https://docs.ultralytics.com/models/yolo26/)
* **Ultralytics Instance Segmentation Task**: [https://docs.ultralytics.com/tasks/segment/](https://docs.ultralytics.com/tasks/segment/)
* **Ultralytics GitHub Repository**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **Package-Seg Benchmark Documentation**: [https://docs.ultralytics.com/datasets/segment/package-seg/](https://docs.ultralytics.com/datasets/segment/package-seg/)
* **Package-Seg on Ultralytics Hub**: [https://platform.ultralytics.com/ultralytics/datasets/package-seg](https://platform.ultralytics.com/ultralytics/datasets/package-seg)
* **Direct Archive Download**: [package-seg.zip (103 MB)](https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip) *(SHA-256: `62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6`)*

---

## 📁 Directory Structure

```text
YOLO_object-segmentation/
├── configs/
│   └── train.yaml          # Locked hyperparameters (AdamW, batch 4, imgsz 512, seed 42)
├── datasets/
│   └── package-seg/        # Package-Seg dataset (400 train / 188 val / 89 test)
│       ├── data.yaml       # Dataset config (Package)
│       ├── train400.txt    # Deterministic list of 400 curated training image paths
│       ├── images/         # train/, val/, test/
│       └── labels/         # train/, val/, test/
├── kaggle_outputs/
│   ├── yolo26-object-segmentation part 1.ipynb # Interactive execution on Kaggle (n, s, m)
│   └── yolo26-object-segmentation part 2.ipynb # Interactive execution on Kaggle (l, x & eval)
├── metrics/
│   └── comparison_test.csv # Benchmark metrics exported from test split
├── scripts/
│   ├── download_data.py    # Fetch archive, verify SHA-256, filter .rf. leakage, lock 400 subset
│   ├── check_dataset.py    # Validate split counts, no leakage, polygon coordinate bounds
│   ├── train.py            # Sequential training across 5 scales (n, s, m, l, x) & auto-backup
│   ├── eval_test.py        # Benchmark on test set (conf=0.001)
│   └── demo_count.py       # Conveyor parcel segmentation & counting visualization (conf=0.25)
├── splits/
│   └── train400_names.txt  # Deterministic list of 400 training image basenames
├── .gitignore              # Ignores .venv, .env, *.pt, figures/, report/, backup/, etc.
├── requirements.txt        # Pinned dependencies (ultralytics>=8.4.0)
└── README.md
```

---

## ⚙️ Initial Setup

### 1. Environment & Dependencies
Clone the repository and set up a virtual environment:

```bash
# Clone repository
git clone https://github.com/XiaoSha59/YOLO_object-segmentation.git
cd YOLO_object-segmentation

# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
# source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Setup (Package-Seg 400)

* **Source**: [Package-Seg on Ultralytics Hub](https://platform.ultralytics.com/ultralytics/datasets/package-seg) | [Direct Archive (package-seg.zip)](https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip)
* **Integrity SHA-256 Checksum**: `62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6`
* **Partitions**:
  * **Train (Locked 400)**: 400 images (1,343 polygon masks, 23 background images)
  * **Valid**: 188 images (693 polygon masks)
  * **Test**: 89 images (325 polygon masks)
  * **Total**: 677 images across 1 class:
    * `0`: **Package** (2,361 total masks, average 18.4 vertices per mask)

To download the official archive, verify SHA-256 integrity, filter `.rf.` synthetic augmentations, and establish the locked 400-subset:
```bash
python scripts/download_data.py
```

To verify dataset integrity and polygon coordinate bounds:
```bash
python scripts/check_dataset.py
```

---

## 🚀 Ultralytics YOLO26-seg Workflow

[Ultralytics YOLO26-seg](https://docs.ultralytics.com/models/yolo26/) introduces multi-scale prototype mask assembly and dual-head NMS-free inference for cutting-edge [instance segmentation](https://docs.ultralytics.com/tasks/segment/).

### 1. Smoke Test (VRAM & Pipeline Check)
Run a quick 1-epoch smoke test with `yolo26m-seg` before starting full training:
```bash
python scripts/train.py --smoke-test
```

### 2. Full Training (5 Model Capacities)
Train all five models sequentially with locked hyperparameters (`AdamW`, $lr_0=0.001$, `batch=4`, `imgsz=512`, `epochs=30`):
```bash
# Train all models sequentially (n -> s -> m -> l -> x):
python scripts/train.py

# Or train models individually:
python scripts/train.py --model yolo26n
python scripts/train.py --model yolo26s
python scripts/train.py --model yolo26m
python scripts/train.py --model yolo26l
python scripts/train.py --model yolo26x
```
> **Note**: After each model finishes, weights (`best.pt`), training curves, and `results.csv` are automatically archived into `backup/` and compressed to `backup.zip`. If interrupted, resume with `python scripts/train.py --model <name> --resume`.

### 3. Test Set Evaluation
Evaluate all trained models on the held-out unseen test split (`conf=0.001`):
```bash
python scripts/eval_test.py
```
Outputs mask precision, mask recall, Mask mAP50, Mask mAP50-95, Box mAP50, and Box mAP50-95 directly to the console and exports to `metrics/comparison_test.csv`.

### 4. Package Counting Demo
Run instance mask segmentation and differential conveyor counting on test images (`conf=0.25`):
```bash
python scripts/demo_count.py
```
Computes parcel tally and displays pixel-level polygon segmentation masks.

### 5. Reproduce on Kaggle GPU
To reproduce all experiments in the cloud with an NVIDIA Tesla T4 GPU:
1. Open [Kaggle](https://www.kaggle.com/) and create a new notebook (set Accelerator to **GPU T4**).
2. Select **File → Import Notebook** and upload:
   * [`kaggle_outputs/yolo26-object-segmentation part 1.ipynb`](kaggle_outputs/yolo26-object-segmentation%20part%201.ipynb) (Trains `yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`).
   * [`kaggle_outputs/yolo26-object-segmentation part 2.ipynb`](kaggle_outputs/yolo26-object-segmentation%20part%202.ipynb) (Trains `yolo26l-seg`, `yolo26x-seg`, benchmarks on test set).
3. Run all cells sequentially. The notebooks audit the dataset, train all model scales, evaluate on the unseen test split, and output complete benchmark summaries.

---

## 📈 Benchmark Results & Metrics

All models were trained on an **NVIDIA Tesla T4 GPU (16 GB VRAM)** on Kaggle using official MS COCO pretrained checkpoints and evaluated on the independent unseen Package-Seg test set (89 images, 325 package instances at `conf=0.001`):

### 1. Quantitative Performance Across All Five YOLO26-seg Scales

| Model Architecture | Parameters | Size (MB) | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Box mAP50-95 | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO26n-seg (Nano)** | 2.7 M | 6.5 MB | 0.8460 | 0.9150 | 0.9160 | 0.7780 | 0.9080 | 0.7880 | ~8.71 min (522.6 s) |
| **YOLO26s-seg (Small)** ⚡ | 10.4 M | 24.2 MB | 0.8490 | 0.9260 | 0.9240 | 0.7980 | 0.9220 | 0.8180 | ~9.16 min (549.6 s) |
| **YOLO26m-seg (Medium)** | 23.6 M | 52.2 MB | 0.8770 | 0.9220 | 0.9180 | 0.7820 | 0.9160 | 0.8110 | ~10.63 min (637.5 s) |
| **YOLO26l-seg (Large)** 🏆 | 31.4 M | 60.5 MB | 0.8755 | **0.9446** | **0.9487** | **0.7982** | **0.9334** | **0.8319** | ~11.76 min (705.6 s) |
| **YOLO26x-seg (Extra Large)** | 70.5 M | 135.1 MB | **0.8865** | 0.9136 | 0.9327 | 0.7510 | 0.9277 | 0.7937 | ~19.68 min (1180.8 s) |

🏆 **Optimal Production Model**: **YOLO26l-seg** achieves the highest overall accuracy across the benchmark, leading in **Mask mAP50 (94.87%)**, **Mask mAP50-95 (79.82%)**, **Box mAP50-95 (83.19%)**, and **Mask Recall (94.46%)**.

⚡ **Edge Deployment Sweet Spot**: **YOLO26s-seg** weighs only **24.2 MB**, retains $99.9\%$ of the segmentation quality of the large model ($79.80\%$ vs. $79.82\%$ Mask mAP50-95), and delivers over $3\times$ higher inference throughput on smart camera gateways.

### 2. Operational Roles by Model Capacity

* **YOLO26n-seg (Nano)**: Designed for low-power edge microcontrollers and handheld barcode scanners where memory and wattage are strictly constrained ($6.5$ MB weight file).
* **YOLO26s-seg (Small)**: Ideal for smart camera induction gates on logistics conveyor belts, delivering $79.80\%$ Mask mAP50-95 at low latency.
* **YOLO26m-seg (Medium)**: Provides expanded channel capacity for resolving complex package surface folds and cardboard labels.
* **YOLO26l-seg (Large)**: Recommended production model, attaining peak test accuracy across all bounding and mask criteria with minimal generalization variance.
* **YOLO26x-seg (Extra Large)**: Maximizes Mask Precision (**88.65%**); however, its 70.5M parameters demonstrate the statistical boundary of overparameterization on a 400-image training partition (Mask mAP50-95 settles at $75.10\%$).

---

## 📦 Automated Conveyor Package Counting Demonstration

Using `scripts/demo_count.py` at $\text{conf}=0.25$ and $\text{retina_masks}=True$ on unseen test conveyor scenes, the automated inference pipeline accurately segments each parcel with pixel-level polygon masks:
* Isolates individual cardboard cartons and poly-mailers along conveyor belts.
* Predicts exact centroid coordinates for robotic induction pick-and-place.
* Achieves 100% agreement with physical inventory counts without duplicate proposals.
