#!/usr/bin/env python3
"""
scripts/train.py
Trains YOLO26-seg models (yolo26n-seg, yolo26s-seg, yolo26m-seg) on Package-Seg.
Provides modular CLI options: --smoke-test, --model, --resume, and automated backup.zip archiving.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import yaml
import zipfile
from ultralytics import YOLO

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def backup_model_run(run_dir: Path, model_name: str, backup_root: Path = Path("backup")):
    backup_root.mkdir(parents=True, exist_ok=True)
    dest_dir = backup_root / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = ["weights/best.pt", "weights/last.pt", "results.csv", "args.yaml", "results.png"]
    copied = []
    for f in files_to_copy:
        src = run_dir / f
        if src.exists():
            dst = dest_dir / Path(f).name
            shutil.copy2(src, dst)
            copied.append(Path(f).name)

    print(f" Backed up {copied} of {model_name} to: {dest_dir}")

    # Generate consolidated backup.zip
    zip_path = Path("backup.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(backup_root):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(backup_root.parent)
                zipf.write(file_path, arcname)
    print(f" Updated consolidated backup archive: {zip_path.resolve()} ({zip_path.stat().st_size / 1024:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26-seg on Package-Seg 400")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch smoke test on YOLO26m-seg to verify VRAM")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "yolo26n", "yolo26s", "yolo26m", "yolo26n-seg", "yolo26s-seg", "yolo26m-seg"],
        help="Select model variant to train (yolo26n, yolo26s, yolo26m, or all)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint last.pt if interrupted")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs count")
    parser.add_argument("--device", default=None, help="Device (0 or cpu)")
    args = parser.parse_args()

    config_path = Path("configs/train.yaml")
    if not config_path.exists():
        raise FileNotFoundError("configs/train.yaml not found!")

    cfg = load_yaml(config_path)

    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    # 1. Smoke test mode
    if args.smoke_test:
        print("\n" + "=" * 60)
        print(" RUNNING SMOKE TEST: YOLO26m-seg (1 epoch)")
        print("=" * 60)
        smoke_model = YOLO("yolo26m-seg.pt")
        smoke_model.train(
            data=cfg["data"],
            epochs=1,
            imgsz=cfg.get("imgsz", 512),
            batch=cfg.get("batch", 4),
            device=cfg.get("device", 0),
            project="runs/segment",
            name="smoke_test",
            exist_ok=True,
            amp=cfg.get("amp", True),
            verbose=True,
        )
        print("\n SMOKE TEST COMPLETED SUCCESSFULLY! Ready for production training.")
        return

    # 2. Model selection
    all_models = [
        ("yolo26n-seg", "yolo26n-seg.pt"),
        ("yolo26s-seg", "yolo26s-seg.pt"),
        ("yolo26m-seg", "yolo26m-seg.pt"),
    ]

    clean_model_arg = args.model.replace("-seg", "")
    if args.model != "all":
        models_to_train = [m for m in all_models if m[0].replace("-seg", "") == clean_model_arg]
    else:
        models_to_train = all_models

    print("\n" + "=" * 60)
    print(f" TARGET MODELS: {[m[0] for m in models_to_train]}")
    print(f" CONFIG: epochs={cfg['epochs']}, batch={cfg['batch']}, imgsz={cfg['imgsz']}, optimizer={cfg['optimizer']}")
    print("=" * 60)

    for tag, weight_name in models_to_train:
        print(f"\n Training model: {tag}...")
        run_project = "runs/segment"
        last_ckpt = Path(run_project) / tag / "weights" / "last.pt"

        if args.resume and last_ckpt.exists():
            print(f" Resuming interrupted training from: {last_ckpt}")
            model = YOLO(str(last_ckpt))
            model.train(resume=True)
        else:
            model = YOLO(weight_name)
            model.train(
                data=cfg["data"],
                epochs=cfg["epochs"],
                patience=cfg.get("patience", 30),
                imgsz=cfg["imgsz"],
                batch=cfg["batch"],
                optimizer=cfg.get("optimizer", "AdamW"),
                lr0=cfg.get("lr0", 0.001),
                weight_decay=cfg.get("weight_decay", 0.0005),
                seed=cfg.get("seed", 42),
                deterministic=cfg.get("deterministic", True),
                amp=cfg.get("amp", True),
                workers=cfg.get("workers", 2),
                close_mosaic=cfg.get("close_mosaic", 5),
                cache=cfg.get("cache", "ram"),
                device=cfg.get("device", 0),
                project=run_project,
                name=tag,
                exist_ok=True,
            )

        run_dir = Path(run_project) / tag
        backup_model_run(run_dir, tag)

    print("\n ALL MODELS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
