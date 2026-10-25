#!/usr/bin/env python3
"""
scripts/prepare_subset.py
Downloads the official Package-Seg archive, verifies its SHA-256 checksum,
filters duplicate .rf. augmentation groups to prevent data leakage across val/test,
selects a locked 400-image training subset (seed 42), and generates package400.yaml.
"""

import os
import sys
import hashlib
import zipfile
import random
import shutil
from pathlib import Path
import urllib.request

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/package-seg.zip"
EXPECTED_SHA256 = "62ff29db4b04c2922ebee6879fc5e05de4bd8c8b35b61088337e0b64a68324f6"
DATASETS_DIR = Path("datasets")
TARGET_DIR = DATASETS_DIR / "package-seg"
ZIP_PATH = DATASETS_DIR / "package-seg.zip"
SPLITS_DIR = Path("splits")


def compute_sha256(filepath: Path) -> str:
    """Computes the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_dataset():
    """Downloads the official Package-Seg archive and verifies SHA-256 checksum."""
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

    print(f"[*] Downloading dataset from {DATASET_URL}...")

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

    print("[*] Verifying SHA-256 of downloaded file...")
    actual_hash = compute_sha256(ZIP_PATH)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"Downloaded file SHA-256 mismatch! Got: {actual_hash}")
    print("[+] SHA-256 matches expected checksum.")


def extract_dataset():
    """Extracts dataset and normalizes directory structure."""
    if TARGET_DIR.exists() and (TARGET_DIR / "images").exists():
        print(f"[*] Dataset already extracted at {TARGET_DIR}.")
        return

    print(f"[*] Extracting {ZIP_PATH} to {TARGET_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASETS_DIR)

    # Handle case where images directory is extracted to a different nested path
    if not (TARGET_DIR / "images").exists():
        for p in DATASETS_DIR.rglob("images"):
            if p.is_dir() and p != TARGET_DIR / "images":
                parent = p.parent
                if parent != TARGET_DIR:
                    TARGET_DIR.mkdir(parents=True, exist_ok=True)
                    for item in parent.iterdir():
                        shm_dest = TARGET_DIR / item.name
                        if not shm_dest.exists():
                            shutil.move(str(item), str(shm_dest))
                break

    # Normalize 'valid' split folder to 'val' if present
    for base in ["images", "labels"]:
        valid_dir = TARGET_DIR / base / "valid"
        val_dir = TARGET_DIR / base / "val"
        if valid_dir.exists() and not val_dir.exists():
            print(f"[*] Normalizing {valid_dir} -> {val_dir}")
            valid_dir.rename(val_dir)

    print("[+] Extraction & split normalization completed.")


def get_base_name(filename: str) -> str:
    """Extracts the base image name by stripping Roboflow '.rf.' augment suffixes."""
    stem = Path(filename).stem
    if ".rf." in stem:
        return stem.split(".rf.")[0]
    return stem


def select_subset():
    """Filters duplicate groups, selects 400 unique train images, and writes config files."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = TARGET_DIR / "images"
    train_dir = images_dir / "train"
    val_dir = images_dir / "val"
    test_dir = images_dir / "test"

    train_files = sorted([f.name for f in train_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    val_files = sorted([f.name for f in val_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    test_files = sorted([f.name for f in test_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    print(f"[*] Raw image counts: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Collect base names present in validation and test splits
    val_test_bases = set()
    for f in val_files:
        val_test_bases.add(get_base_name(f))
    for f in test_files:
        val_test_bases.add(get_base_name(f))

    print(f"[*] Identified {len(val_test_bases)} unique base groups in Val & Test sets.")

    # Filter training images: keep 1 image per base group that does not appear in val/test
    group_to_file = {}
    for f in train_files:
        base = get_base_name(f)
        if base in val_test_bases:
            continue
        if base not in group_to_file:
            group_to_file[base] = f

    eligible_files = sorted(list(group_to_file.values()))
    print(f"[*] Eligible leakage-free train candidates: {len(eligible_files)}")

    if len(eligible_files) < 400:
        raise ValueError(f"Insufficient eligible candidates ({len(eligible_files)}) to sample 400 images!")

    # Deterministic sampling with seed 42
    rng = random.Random(42)
    selected_400 = rng.sample(eligible_files, 400)
    selected_400.sort()

    # Save splits/train400_names.txt (tracked in Git)
    names_txt = SPLITS_DIR / "train400_names.txt"
    with open(names_txt, "w", encoding="utf-8") as f:
        for name in selected_400:
            f.write(f"{name}\n")
    print(f"[+] Saved 400 subset filenames to {names_txt}")

    # Save datasets/package-seg/train400.txt with relative paths
    dataset_txt = TARGET_DIR / "train400.txt"
    with open(dataset_txt, "w", encoding="utf-8") as f:
        for name in selected_400:
            f.write(f"./images/train/{name}\n")
    print(f"[+] Saved relative image path list to {dataset_txt}")

    # Save package400.yaml
    yaml_content = f"""# Package-Seg 400 Subset Configuration
path: datasets/package-seg
train: train400.txt
val: images/val
test: images/test

names:
  0: package
"""
    yaml_path = TARGET_DIR / "package400.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"[+] Generated dataset configuration file at {yaml_path}")


def main():
    print("=== Step 1: Download & Prepare Package-Seg 400 Subset ===")
    download_dataset()
    extract_dataset()
    select_subset()
    print("=== Dataset Preparation Completed Successfully! ===")


if __name__ == "__main__":
    main()
