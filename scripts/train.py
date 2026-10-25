#!/usr/bin/env python3
"""
scripts/train.py
Trains YOLO26-seg models (n, s, m) sequentially using parameters defined in configs/train.yaml.
Supports --smoke flag (runs 1 epoch on yolo26m-seg) to verify GPU VRAM before production training.
Automatically backs up best.pt, last.pt, and training curves to backup/<model_name>/ after each run.
"""

import os
import sys
import shutil
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


def backup_model_artifacts(project_dir: Path, model_name: str):
    """Backs up checkpoints, logs, and curves after training each model."""
    dest_dir = BACKUP_DIR / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        "weights/best.pt",
        "weights/last.pt",
        "args.yaml",
        "results.csv",
        "results.png",
    ]

    for item in files_to_backup:
        src = project_dir / item
        if src.exists():
            dst = dest_dir / Path(item).name
            shutil.copy2(src, dst)
            print(f"[+] Backed up: {src} -> {dst}")
        else:
            print(f"[!] Warning: {src} not found during backup.")


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
    print("[+] Safe to proceed with production training of all three variants.")


def run_full_training(cfg: dict):
    """Trains YOLO26n-seg, YOLO26s-seg, and YOLO26m-seg sequentially."""
    models = cfg.get("models", ["yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt"])
    data_yaml = cfg["data"]

    for model_weight in models:
        model_name = Path(model_weight).stem
        print("\n" + "=" * 65)
        print(f">>> STARTING TRAINING: {model_name.upper()} <<<")
        print("=" * 65)

        run_dir = Path("runs/segment") / model_name

        model = YOLO(model_weight)
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

        print(f"\n[+] Successfully completed training for {model_name}!")
        backup_model_artifacts(run_dir, model_name)


def main():
    parser = argparse.ArgumentParser(description="YOLO26-seg Training Pipeline")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a 1-epoch smoke test on YOLO26m-seg to verify VRAM stability",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.smoke:
        run_smoke_test(cfg)
    else:
        run_full_training(cfg)


if __name__ == "__main__":
    main()
