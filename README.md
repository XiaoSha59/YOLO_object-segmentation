# YOLO26 Package Instance Segmentation & Counting (Package-Seg 400)

A complete pipeline for segmenting and counting packages on the **Package-Seg (400-image subset)** using the **Ultralytics YOLO26-seg** model family (`yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`).

* **Model Documentation**: [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/) | [Ultralytics Instance Segmentation](https://docs.ultralytics.com/tasks/segment/)
* **Dataset Sources**: [Package-Seg on Roboflow Universe](https://universe.roboflow.com/roboflow-100/package-seg) | [Ultralytics Official Archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip)

---

## 📍 Directory Structure

```text
YOLO_object-segmentation/
├── configs/
│   └── train.yaml          # Locked hyperparameters (AdamW, batch 4, imgsz 512)
├── datasets/
│   └── package-seg/        # Package-Seg dataset (400 train / 188 val / 89 test)
│       ├── data.yaml       # Dataset config
│       ├── train400.txt    # Relative image paths for locked 400 train split
│       ├── images/         # train/, val/, test/
│       └── labels/         # train/, val/, test/
├── figures/                # Visualizations (dataset_samples.png, demo_counting.png)
├── metrics/                # CSV outputs from evaluation runs (comparison_test.csv)
├── report/
│   └── report.md           # Experiment analysis and performance report
├── scripts/
│   ├── download_data.py    # Download archive, filter .rf. leakage, lock 400 subset
│   ├── check_dataset.py    # Validate split counts and polygon mask integrity
│   ├── train.py            # Sequential training (n -> s -> m) & auto-backup
│   ├── eval_test.py        # Benchmark on test set (conf=0.001)
│   ├── demo_count.py       # Package counting visualization (conf=0.25)
│   └── upload_kaggle.py    # Cloud backup to Kaggle Dataset
├── splits/
│   └── train400_names.txt  # Tracked 400 train image filenames
├── .gitignore              # Ignores .venv, .env, *.pt, *plan.md
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

The dataset originates from [Package-Seg on Roboflow Universe](https://universe.roboflow.com/roboflow-100/package-seg) and is officially distributed as an archive by Ultralytics:
* **Direct Archive**: [package-seg.zip](https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip)
* **Integrity SHA-256**: `62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6`

The dataset is partitioned into official fixed splits with an isolated 400-image training subset:
* **Train (Locked subset)**: 400 images (~1,343 package masks)
* **Valid**: 188 images
* **Test**: 89 images
* **Total**: 677 images

To download, verify SHA-256 integrity, filter `.rf.` synthetic augmentations, and lock the 400-subset:
```bash
python scripts/download_data.py
```

To verify dataset integrity and generate a 12-sample preview (`figures/dataset_samples.png`):
```bash
python scripts/check_dataset.py
```

---

## 🚀 Ultralytics YOLO26 Workflow

[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) introduces multi-scale proto masks and semantic loss guidance for cutting-edge [instance segmentation](https://docs.ultralytics.com/tasks/segment/).

### 1. Smoke Test (VRAM & Pipeline Check)
Run a quick 1-epoch smoke test with `yolo26m-seg` before starting full training:
```bash
python scripts/train.py --smoke-test
```

### 2. Full Training (Sequential $n \rightarrow s \rightarrow m$)
Train all three models sequentially with locked hyperparameters (`AdamW`, $lr_0=0.001$, `batch=4`, `imgsz=512`, `epochs=30`):
```bash
python scripts/train.py
```
> **Note**: After each model finishes, weights (`best.pt`), training curves, and `results.csv` are automatically archived into `backup/` and compressed to `backup.zip`. You can also train individual models with `python scripts/train.py --model yolo26n`.

### 3. Test Set Evaluation
Evaluate all trained models on the test set (`conf=0.001`):
```bash
python scripts/eval_test.py
```
Outputs precision, recall, Mask mAP50, Mask mAP50-95, and Box metrics to `metrics/comparison_test.csv`.

### 4. Package Counting Demo
Run instance mask detection and package counting on test images (`conf=0.25`):
```bash
python scripts/demo_count.py
```
Saves annotated package counts to `figures/demo_counting.png`.
