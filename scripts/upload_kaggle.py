#!/usr/bin/env python3
"""
scripts/upload_kaggle.py
Automates permanent backup of trained checkpoints to a private Kaggle Dataset via the Kaggle API.
"""

import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BACKUP_DIR = Path("backup")
METADATA_FILE = BACKUP_DIR / "dataset-metadata.json"


def setup_kaggle_credentials():
    """Sets up Kaggle credentials from .env or environment variables."""
    token = os.getenv("KAGGLE_API_TOKEN")
    key = os.getenv("KAGGLE_KEY")
    username = os.getenv("KAGGLE_USERNAME", "salala1706")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"

    # If user provided a Bearer / API token or username + key
    if token and not key:
        os.environ["KAGGLE_API_TOKEN"] = token
        # Write json for official kaggle client compatibility
        cred = {"username": username, "key": token}
    else:
        cred = {"username": username, "key": key or token or ""}

    with open(kaggle_json, "w", encoding="utf-8") as f:
        json.dump(cred, f)
    try:
        kaggle_json.chmod(0o600)
    except Exception:
        pass


def upload_to_kaggle_dataset(dataset_slug: str = "yolo26-package-seg-checkpoints"):
    """Creates or updates a Kaggle dataset containing the backup directory artifacts."""
    if not BACKUP_DIR.exists():
        print(f"[!] Backup directory {BACKUP_DIR} does not exist.")
        return

    setup_kaggle_credentials()
    username = os.getenv("KAGGLE_USERNAME", "salala1706")

    # Generate metadata file if missing
    if not METADATA_FILE.exists():
        metadata = {
            "title": "YOLO26 Package Segmentation Checkpoints",
            "id": f"{username}/{dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[+] Created {METADATA_FILE}")

    print(f"[*] Uploading checkpoints from {BACKUP_DIR} to Kaggle Dataset ({username}/{dataset_slug})...")

    # Try creating dataset first, if exists push new version
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(BACKUP_DIR), "-u"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("[+] Successfully created new Kaggle dataset!")
            print(result.stdout)
            return
        elif "already exists" in result.stderr.lower():
            print("[*] Dataset already exists, pushing new version...")
            res_ver = subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(BACKUP_DIR), "-m", "Updated checkpoints"],
                capture_output=True,
                text=True,
            )
            print(res_ver.stdout)
            if res_ver.stderr:
                print(res_ver.stderr)
        else:
            print(f"[!] Kaggle CLI output: {result.stderr or result.stdout}")
    except FileNotFoundError:
        print("[!] 'kaggle' command-line tool not found. Install it with: pip install kaggle")


if __name__ == "__main__":
    upload_to_kaggle_dataset()
