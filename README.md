# YOLO26 Package Instance Segmentation & Counting

A clean, reproducible pipeline for package instance segmentation and automated counting using **YOLO26-seg** (`yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`) on a standardized **Package-Seg (400-image subset)**.

---

## 1. Directory Structure

```text
YOLO_object-segmentation/
├── .gitignore
├── README.md
├── requirements.txt
├── configs/
│   └── train.yaml            # Locked training hyperparameters
├── scripts/
│   ├── prepare_subset.py     # Download, SHA-256 check, leakage-free 400 subset
│   ├── check_masks.py        # Validate polygon format and generate GT preview
│   ├── train.py              # Sequential model training with automated backup
│   ├── eval_test.py          # Dual split evaluation (val & test) and metric export
│   └── demo_count.py         # Instance mask package counting demo
├── splits/
│   └── train400_names.txt    # Tracked list of 400 selected training images
├── metrics/                  # Benchmark comparison tables (comparison.csv)
├── figures/                  # GT overlays and demo counting visualizations
└── report/
    └── report.md             # Experimental report outline
```

---

## 2. Initial Setup

### Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/XiaoSha59/YOLO_object-segmentation.git
cd YOLO_object-segmentation

# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies (Ultralytics >= 8.4)
pip install -r requirements.txt
```

---

## 3. Dataset Preparation

We use the official Ultralytics Package-Seg dataset archive with fixed SHA-256 verification (`62ff29db...`).

```bash
# 1. Download official archive, filter duplicate .rf. augmentations, lock 400 subset (seed 42)
python scripts/prepare_subset.py

# 2. Verify segmentation polygon integrity and generate 16 sample visual overlays
python scripts/check_masks.py
```
* **Leakage Prevention**: Base images sharing identical sources across synthetic augmentations (`.rf.`) in validation and test splits are strictly excluded from the training split.
* **Output**: Writes `splits/train400_names.txt` and generates `datasets/package-seg/package400.yaml`.

---

## 4. Ultralytics YOLO26 Workflow

All training hyperparameters are configured in `configs/train.yaml` (512x512 resolution, batch size 4, AdamW optimizer, lr0=0.001, seed 42).

### Step 1: Smoke Test
Verify GPU VRAM stability and confirm segmentation loss components with a 1-epoch test on `yolo26m-seg`:
```bash
python scripts/train.py --smoke
```

### Step 2: Sequential Training
Train `yolo26n-seg`, `yolo26s-seg`, and `yolo26m-seg` sequentially. Checkpoints and training curves are automatically backed up to `backup/<model_name>/` after each run:
```bash
python scripts/train.py
```

### Step 3: Dual-Split Evaluation
Evaluate all models on both `val` and `test` splits (`conf=0.001`, `imgsz=512`) and export metrics to `metrics/comparison.csv`:
```bash
python scripts/eval_test.py
```

### Step 4: Package Counting Demo
Run inference using instance masks (`conf=0.25`, `retina_masks=True`) to segment and count packages on test images:
```bash
python scripts/demo_count.py
```
Output visualizations are stored in `figures/demo_counts/`.
