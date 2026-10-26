#!/usr/bin/env python3
"""
scripts/download_data.py
Downloads the official Package-Seg archive, verifies SHA-256 checksum,
filters duplicate .rf. synthetic augmentations to prevent data leakage,
selects the locked 400-image training subset (seed 42), and generates data.yaml.
"""

import os
import sys
import hashlib
import zipfile
import random
import shutil
from pathlib import Path
import urllib.request
import yaml

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip"
EXPECTED_SHA256 = "62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
TARGET_DIR = DATASETS_DIR / "package-seg"
ZIP_PATH = DATASETS_DIR / "package-seg.zip"
SPLITS_DIR = REPO_ROOT / "splits"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_dataset():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        print(f"[*] Found existing archive at {ZIP_PATH}. Verifying SHA-256...")
        actual_hash = compute_sha256(ZIP_PATH)
        if actual_hash == EXPECTED_SHA256:
            print("[+] SHA-256 verified successfully.")
            return
        else:
            print(f"[!] Hash mismatch: expected {EXPECTED_SHA256}, got {actual_hash}. Re-downloading...")
            ZIP_PATH.unlink()

    print(f"[1/3] Downloading Package-Seg dataset from {DATASET_URL}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            sys.stdout.write(
                f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)"
            )
            sys.stdout.flush()

    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, reporthook=reporthook)
    print("\n[+] Download completed.")

    actual_hash = compute_sha256(ZIP_PATH)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Downloaded file SHA-256 mismatch! Got: {actual_hash}")
    print("[+] SHA-256 checksum verified.")


def extract_dataset():
    if TARGET_DIR.exists() and (TARGET_DIR / "images").exists():
        print(f"[*] Dataset already extracted at {TARGET_DIR}.")
        return

    print(f"[2/3] Extracting archive to {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)

    # If the zip extracted into a nested TARGET_DIR / package-seg folder
    nested_dir = TARGET_DIR / "package-seg"
    if nested_dir.exists() and (nested_dir / "images").exists():
        for item in nested_dir.iterdir():
            dest = TARGET_DIR / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        try:
            nested_dir.rmdir()
        except Exception:
            pass

    # If the zip was extracted into DATASETS_DIR / images
    for sub in ["images", "labels"]:
        src_sub = DATASETS_DIR / sub
        dst_sub = TARGET_DIR / sub
        if src_sub.exists() and not dst_sub.exists():
            shutil.move(str(src_sub), str(dst_sub))

    for base in ["images", "labels"]:
        valid_dir = TARGET_DIR / base / "valid"
        val_dir = TARGET_DIR / base / "val"
        if valid_dir.exists() and not val_dir.exists():
            valid_dir.rename(val_dir)


def select_subset():
    print(f"[3/3] Partitioning leakage-free 400 training subset...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = TARGET_DIR / "images"
    train_dir = images_dir / "train"
    val_dir = images_dir / "val"
    test_dir = images_dir / "test"

    train_files = sorted([f.name for f in train_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    val_files = sorted([f.name for f in val_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    test_files = sorted([f.name for f in test_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    def get_base_name(filename: str) -> str:
        stem = Path(filename).stem
        return stem.split(".rf.")[0] if ".rf." in stem else stem

    val_test_bases = {get_base_name(f) for f in val_files} | {get_base_name(f) for f in test_files}

    group_to_file = {}
    for f in train_files:
        base = get_base_name(f)
        if base not in val_test_bases and base not in group_to_file:
            group_to_file[base] = f

    eligible_files = sorted(list(group_to_file.values()))
    if len(eligible_files) < 400:
        raise ValueError(f"Insufficient eligible candidates ({len(eligible_files)}) to sample 400 images!")

    rng = random.Random(42)
    selected_400 = sorted(rng.sample(eligible_files, 400))

    # Save splits/train400_names.txt
    names_txt = SPLITS_DIR / "train400_names.txt"
    with open(names_txt, "w", encoding="utf-8") as f:
        for name in selected_400:
            f.write(f"{name}\n")

    # Save datasets/package-seg/train400.txt
    dataset_txt = TARGET_DIR / "train400.txt"
    with open(dataset_txt, "w", encoding="utf-8") as f:
        for name in selected_400:
            f.write(f"./images/train/{name}\n")

    # Generate datasets/package-seg/data.yaml with resolved absolute path
    resolved_path = TARGET_DIR.resolve().as_posix()
    data_yaml = {
        "path": resolved_path,
        "train": "train400.txt",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "package"},
    }
    yaml_path = TARGET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Also save package400.yaml for backwards compatibility
    compat_yaml = TARGET_DIR / "package400.yaml"
    with open(compat_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"[+] Successfully generated {yaml_path} and {names_txt}")


def main():
    print("=" * 60)
    print("DOWNLOAD & PREPARE PACKAGE-SEG 400 DATASET")
    print("=" * 60)
    download_dataset()
    extract_dataset()
    select_subset()
    print("\n Download and configuration completed successfully!")


if __name__ == "__main__":
    main()
