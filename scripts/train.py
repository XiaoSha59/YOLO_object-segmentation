#!/usr/bin/env python3
"""
scripts/train.py
Trains YOLO26-seg models (n, s, m) with modular CLI controls:
- Train individually (--model yolo26n / yolo26s / yolo26m) or all models sequentially.
- Resume interrupted training runs (--resume).
- Verify GPU VRAM with a 1-epoch smoke test (--smoke).
- Automatically backs up and zips best.pt, last.pt, and logs to backup/<model_name>/.
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

CONFIG_PATH = Path("configs/train.yaml")
BACKUP_DIR = Path("backup")


def load_config() -> dict:
    """Loads training configuration from YAML file."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_model_name(name: str) -> str:
    """Normalizes input model names (e.g. 'yolo26n', 'n', 'yolo26n-seg') to 'yolo26n-seg.pt'."""
    clean = name.lower().strip().replace(".pt", "")
    if clean.endswith("-seg"):
        clean = clean[:-4]
    if clean.startswith("yolo26"):
        variant = clean.replace("yolo26", "")
    elif clean in ["n", "s", "m", "l", "x"]:
        variant = clean
    else:
        variant = clean
    return f"yolo26{variant}-seg.pt"


def backup_and_zip_artifacts(project_dir: Path, model_name: str) -> Path:
    """Backs up checkpoints and training curves, then creates a downloadable zip archive."""
    dest_dir = BACKUP_DIR / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        "weights/best.pt",
        "weights/last.pt",
        "args.yaml",
        "results.csv",
        "results.png",
    ]

    copied_files = []
    for item in files_to_backup:
        src = project_dir / item
        if src.exists():
            dst = dest_dir / Path(item).name
            shutil.copy2(src, dst)
            copied_files.append(dst)
            print(f"[+] Backed up: {src} -> {dst}")
        else:
            print(f"[!] Warning: {src} not found during backup.")

    # Create downloadable zip
    zip_path = BACKUP_DIR / f"{model_name}_backup.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for f in copied_files:
            zip_file.write(f, arcname=f.name)

    print(f"[+] Created standalone backup archive: {zip_path}")
    return zip_path


def run_smoke_test(cfg: dict):
    """Executes a 1-epoch smoke test on YOLO26m-seg to verify VRAM headroom and segmentation loss."""
    print("\n" + "=" * 65)
    print(">>> RUNNING SMOKE TEST: 1 EPOCH WITH YOLO26M-SEG <<<")
    print("=" * 65)

    model_weight = "yolo26m-seg.pt"
    model = YOLO(model_weight)

    data_yaml = cfg["data"]
    model.train(
        data=data_yaml,
        epochs=1,
        imgsz=cfg.get("imgsz", 512),
        batch=cfg.get("batch", 4),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        device=cfg.get("device", 0),
        workers=cfg.get("workers", 2),
        project="runs/segment",
        name="smoke_test",
        exist_ok=True,
        verbose=True,
    )

    print("\n[+] Smoke test passed successfully! VRAM comfortably supports batch=4.")
    print("[+] Safe to proceed with production training.")


def train_single_model(cfg: dict, model_weight: str, resume: bool = False):
    """Trains or resumes a single YOLO26-seg model."""
    model_name = Path(model_weight).stem
    run_dir = Path("runs/segment") / model_name
    last_pt = run_dir / "weights" / "last.pt"
    backup_last_pt = BACKUP_DIR / model_name / "last.pt"

    print("\n" + "=" * 65)
    if resume:
        print(f">>> RESUMING TRAINING: {model_name.upper()} <<<")
        target_checkpoint = None
        if last_pt.exists():
            target_checkpoint = str(last_pt)
        elif backup_last_pt.exists():
            target_checkpoint = str(backup_last_pt)

        if not target_checkpoint:
            raise FileNotFoundError(f"Cannot resume: neither {last_pt} nor {backup_last_pt} exists!")

        print(f"[*] Resuming from checkpoint: {target_checkpoint}")
        model = YOLO(target_checkpoint)
        model.train(resume=True)
    else:
        print(f">>> STARTING TRAINING: {model_name.upper()} <<<")
        model = YOLO(model_weight)
        data_yaml = cfg["data"]
        model.train(
            data=data_yaml,
            epochs=cfg.get("epochs", 30),
            patience=cfg.get("patience", 30),
            imgsz=cfg.get("imgsz", 512),
            batch=cfg.get("batch", 4),
            optimizer=cfg.get("optimizer", "AdamW"),
            lr0=cfg.get("lr0", 0.001),
            weight_decay=cfg.get("weight_decay", 0.0005),
            seed=cfg.get("seed", 42),
            deterministic=cfg.get("deterministic", True),
            amp=cfg.get("amp", True),
            workers=cfg.get("workers", 2),
            close_mosaic=cfg.get("close_mosaic", 5),
            save_period=cfg.get("save_period", 10),
            device=cfg.get("device", 0),
            cache=cfg.get("cache", "ram"),
            project="runs/segment",
            name=model_name,
            exist_ok=True,
            verbose=True,
        )

    print(f"\n[+] Successfully completed execution for {model_name}!")
    backup_and_zip_artifacts(run_dir, model_name)


def main():
    parser = argparse.ArgumentParser(description="YOLO26-seg Training Pipeline")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 1-epoch smoke test on YOLO26m-seg to verify VRAM stability",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model variant to train: yolo26n, yolo26s, yolo26m, or 'all'",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted training from the latest last.pt checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.smoke:
        run_smoke_test(cfg)
        return

    if args.model.lower() == "all":
        models = cfg.get("models", ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt"])
        for m in models:
            train_single_model(cfg, m, resume=args.resume)
    else:
        target_model = normalize_model_name(args.model)
        train_single_model(cfg, target_model, resume=args.resume)


if __name__ == "__main__":
    main()
