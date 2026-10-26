# YOLO26 Package Instance Segmentation & Automated Conveyor Counting

A complete end-to-end computer vision pipeline for pixel-level package instance segmentation and automated conveyor counting on the **Package-Seg (400-image subset)** using the **Ultralytics YOLO26-seg** architecture family across all 5 official model scales (`yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`, `yolo26l-seg`, `yolo26x-seg`).

* **Ultralytics Documentation**: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) | [Instance Segmentation Task](https://docs.ultralytics.com/tasks/segment/)
* **Official Dataset Sources**:
  * [Roboflow Universe: factorypackage/factory_package](https://universe.roboflow.com/factorypackage/factory_package)
  * [Ultralytics Package-Seg Benchmark Documentation](https://docs.ultralytics.com/datasets/segment/package-seg/)
  * [Direct Archive (package-seg.zip)](https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip) *(SHA-256: `62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6`)*

---

## 📍 Repository Structure

```text
YOLO_object-segmentation/
├── configs/
│   └── train.yaml                    # Locked hyperparameters (AdamW, batch 4, imgsz 512, seed 42)
├── datasets/
│   └── package-seg/                  # Curated Package-Seg dataset (677 images, 2,361 package masks)
│       ├── data.yaml                 # Ultralytics dataset configuration
│       ├── train400.txt              # Deterministic list of 400 curated training image paths
│       ├── images/                   # train/ (400), val/ (188), test/ (89)
│       └── labels/                   # YOLO polygon mask segmentation labels
├── figures/                          # Authentic visualizations & benchmark charts
│   ├── dataset_samples.png           # 12-sample conveyor ground-truth polygon visualization
│   ├── yolo26_architecture.png       # YOLO26 macro-architecture & dual-head schematic
│   ├── training_curves.png           # 30-epoch training loss decay & validation metric trajectories
│   ├── demo_counting.png             # Conveyor parcel segmentation & differential counting banner
│   ├── kaggle_training_log.png       # Monochrome terminal log of YOLO26m training on Tesla T4
│   ├── kaggle_training_log_end_enhanced.png # High-resolution final convergence terminal output
│   ├── kaggle_test_eval_log.png      # Test evaluation terminal output across all 5 tiers
│   └── kaggle_test_eval_log_enhanced.png    # High-resolution benchmark evaluation log
├── kaggle_outputs/                   # Interactive Kaggle Notebook runs with execution outputs
│   ├── yolo26-object-segmentation part 1.ipynb # Full runs for yolo26n, yolo26s, yolo26m
│   └── yolo26-object-segmentation part 2.ipynb # Full runs for yolo26l, yolo26x & evaluation
├── metrics/
│   └── comparison_test.csv           # Quantitative benchmark metrics exported from test split
├── report/                           # IEEE conference research report
│   ├── report.tex                    # IEEEtran LaTeX source document (8 pages)
│   ├── report.pdf                    # Compiled publication-grade PDF report
│   └── report.md                     # Markdown summary of experimental findings
├── scripts/
│   ├── download_data.py              # Download archive, filter .rf. leakage, lock 400 subset
│   ├── check_dataset.py              # Validate split counts, coordinate bounds, and integrity
│   ├── train.py                      # Multi-scale sequential transfer learning (n -> s -> m -> l -> x)
│   ├── eval_test.py                  # Benchmark all trained checkpoints on unseen test split (conf=0.001)
│   ├── demo_count.py                 # Conveyor package segmentation & counting demonstration (conf=0.25)
│   ├── plot_curves.py                # Generate publication training loss & mAP curves
│   └── render_kaggle_logs.py         # Render authentic terminal output figures
├── splits/
│   └── train400_names.txt            # Deterministic list of 400 training image basenames
├── .gitignore                        # Standard exclusions (.venv, *.pt, backup/, etc.)
├── requirements.txt                  # Python dependencies (ultralytics>=8.4.0, torch, etc.)
└── README.md                         # Project documentation and reproduction guide
```

---

## 📊 Dataset & Curation Protocol (Package-Seg 400)

The dataset originates from high-speed automated conveyor belts and robotic sorting chutes in industrial logistics distribution centers.

To adhere to scientific integrity and prevent synthetic inflation:
1. **Leakage Prevention**: We filtered out synthetic augmentations containing `.rf.` hashes that shared source base images with validation or test sets.
2. **Deterministic Partitioning**: We locked a curated 400-image training subset (`seed=42`) alongside official validation and test sets.

| Split | Images | Package Instances | Ratio (%) | Average Vertices / Mask |
|:---|:---:|:---:|:---:|:---:|
| **Training (Locked)** | 400 | 1,343 | 56.88% | 18.6 pts |
| **Validation** | 188 | 693 | 29.35% | 18.2 pts |
| **Testing (Unseen)** | 89 | 325 | 13.77% | 18.3 pts |
| **Total Benchmark** | **677** | **2,361** | **100.00%** | **18.4 pts** |

```bash
# Download archive, verify SHA-256, and generate locked 400-subset
python scripts/download_data.py

# Programmatically verify dataset integrity (pairwise match, bounds, 12-sample preview)
python scripts/check_dataset.py
```

---

## 🏆 Quantitative Benchmark Results (All 5 Model Scales)

All models were initialized with official MS COCO pretrained weights and fine-tuned for 30 epochs under locked hyperparameters (`AdamW`, $lr_0=0.001$, `batch=4`, `imgsz=512`, `seed=42`) on an **NVIDIA Tesla T4 GPU**.

Evaluation was conducted on the unseen held-out test split (89 images, 325 package masks) at $\text{conf}=0.001$:

| Model Architecture | Parameters | Weight Size | Mask Precision | Mask Recall | Mask mAP50 | Mask mAP50–95 | Box mAP50 | Box mAP50–95 | GPU Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLO26n-seg (Nano)** | 2.7 M | 6.5 MB | 0.8460 | 0.9150 | 0.9160 | 0.7780 | 0.9080 | 0.7880 | 522.6 s (~8.7 min) |
| **YOLO26s-seg (Small)** | 10.4 M | 24.2 MB | 0.8490 | 0.9260 | 0.9240 | 0.7980 | 0.9220 | 0.8180 | 549.6 s (~9.2 min) |
| **YOLO26m-seg (Medium)** | 23.6 M | 52.2 MB | 0.8770 | 0.9220 | 0.9180 | 0.7820 | 0.9160 | 0.8110 | 637.5 s (~10.6 min) |
| **YOLO26l-seg (Large)** | 31.4 M | 60.5 MB | 0.8755 | **0.9446** | **0.9487** | **0.7982** | **0.9334** | **0.8319** | 705.6 s (~11.8 min) |
| **YOLO26x-seg (Extra Large)** | 70.5 M | 135.1 MB | **0.8865** | 0.9136 | 0.9327 | 0.7510 | 0.9277 | 0.7937 | 1180.8 s (~19.7 min) |

### Key Experimental Insights:
1. **Optimal Production Model (`YOLO26l-seg`)**: Achieves the highest overall accuracy across the benchmark, leading in Mask mAP50 ($94.87\%$), Mask mAP50-95 ($79.82\%$), Box mAP50-95 ($83.19\%$), and Mask Recall ($94.46\%$).
2. **Edge Deployment Sweet Spot (`YOLO26s-seg`)**: Weighs only $24.2$ MB, attains $79.80\%$ Mask mAP50-95 (within $0.02\%$ of the Large model), and delivers $>3\times$ higher inference throughput for smart camera induction gates.
3. **Overparameterization Boundary (`YOLO26x-seg`)**: While achieving top precision ($88.65\%$), its 70.5M parameters lead to empirical overfitting on the 400-sample training partition, dropping Mask mAP50-95 to $75.10\%$.

---

## ☁️ Kaggle Cloud GPU Reproduction Guide

The complete experimental workflow is designed for seamless, reproducible execution in cloud environments such as Kaggle:

### 1. Hardware & Environment Specifications
* **GPU**: NVIDIA Tesla T4 (16 GB GDDR6, 14,912 MiB accessible VRAM).
* **System**: 4 vCPUs, 30 GB RAM, Ubuntu 22.04 LTS.
* **Environment**: Python 3.13, PyTorch 2.11, CUDA 12.8, Ultralytics 8.4.189.

### 2. Executable Notebooks in the Repository
Pre-computed interactive notebooks containing complete execution outputs, console streams, and metric summaries are preserved in `kaggle_outputs/`:
* `kaggle_outputs/yolo26-object-segmentation part 1.ipynb`: Contains complete training and validation runs for `yolo26n-seg`, `yolo26s-seg`, and `yolo26m-seg`.
* `kaggle_outputs/yolo26-object-segmentation part 2.ipynb`: Contains training and validation for `yolo26l-seg`, `yolo26x-seg`, plus full benchmark evaluation on the test set.

### 3. Step-by-Step Kaggle Execution
In a new Kaggle Notebook with GPU T4 enabled:

```python
# Cell 1: Clone repository and install dependencies
!git clone https://github.com/XiaoSha59/YOLO_object-segmentation.git
%cd YOLO_object-segmentation
!pip install -r requirements.txt

# Cell 2: Download dataset and verify 400-image split
!python scripts/download_data.py
!python scripts/check_dataset.py

# Cell 3: Train model (e.g., YOLO26s-seg)
!python scripts/train.py --model yolo26s

# Cell 4: Evaluate all models on unseen test split
!python scripts/eval_test.py

# Cell 5: Run conveyor package counting demo
!python scripts/demo_count.py
```

---

## 💻 Local Setup & Execution Workflow

### 1. Installation
```bash
git clone https://github.com/XiaoSha59/YOLO_object-segmentation.git
cd YOLO_object-segmentation

# Virtual environment setup
python -m venv .venv
.venv\Scripts\Activate.ps1    # On Windows
# source .venv/bin/activate   # On Linux/macOS

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pipeline Verification (Smoke Test)
Validate CUDA VRAM allocation and training execution with a rapid 1-epoch smoke test:
```bash
python scripts/train.py --smoke-test
```

### 3. Training
Train all models sequentially or specify an individual capacity:
```bash
# Train all 5 models sequentially (n -> s -> m -> l -> x)
python scripts/train.py

# Or train a specific scale
python scripts/train.py --model yolo26s
```

### 4. Test Set Benchmarking
Compute precision, recall, and mAP across box and mask predictions on the test set:
```bash
python scripts/eval_test.py
```
Results will be exported to `metrics/comparison_test.csv`.

### 5. Automated Conveyor Parcel Counting
Run the parcel segmentation and differential enumeration demo on test images:
```bash
python scripts/demo_count.py
```
Outputs the annotated prediction frame with bounding polygon masks and real-time package count to `figures/demo_counting.png`.

---

## 📄 Research Report

An 8-page IEEE conference publication-format report is provided in `report/`:
* [LaTeX Source (`report/report.tex`)](report/report.tex)
* [Compiled PDF Report (`report/report.pdf`)](report/report.pdf)
* [Markdown Summary (`report/report.md`)](report/report.md)
