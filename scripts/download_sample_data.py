#!/usr/bin/env python3
"""Manifest-driven downloader for public sample point cloud datasets.

Usage:
    python scripts/download_sample_data.py --dataset 3dscannerapp_samples --output data/samples
    python scripts/download_sample_data.py --list

Add new sources to DATASETS below. Downloads are idempotent (skipped if the
target directory already exists and is non-empty).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATASETS = {
    "3dscannerapp_samples": {
        "description": "Official 3D Scanner App sample exports + parsing code (Laan Labs).",
        "kind": "git",
        "url": "https://github.com/laanlabs/3dScannerApp_samples.git",
    },
    "arkitscenes": {
        "description": (
            "Apple ARKitScenes -- real iPad/iPhone LiDAR RGB-D + pose scans. "
            "Requires accepting Apple's dataset license; this only clones the "
            "repo, which contains its own download_data.py for the actual data."
        ),
        "kind": "git",
        "url": "https://github.com/apple/ARKitScenes.git",
    },
}


def list_datasets() -> None:
    for name, info in DATASETS.items():
        print(f"{name}: {info['description']}\n  {info['url']}")


def download_git(url: str, dest: Path) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch this dataset but was not found on PATH")
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def download_zip(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "download.zip"
    urlretrieve(url, zip_path)  # noqa: S310 - trusted, user-specified dataset URL
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    zip_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="Dataset name to fetch.")
    parser.add_argument("--output", default="data/samples", help="Directory to download into.")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit.")
    args = parser.parse_args()

    if args.list or not args.dataset:
        list_datasets()
        return 0

    info = DATASETS[args.dataset]
    dest = Path(args.output) / args.dataset

    if dest.exists() and any(dest.iterdir()):
        print(f"{dest} already exists and is non-empty, skipping download.")
        return 0

    print(f"Fetching {args.dataset} -> {dest}")
    if info["kind"] == "git":
        download_git(info["url"], dest)
    elif info["kind"] == "zip":
        download_zip(info["url"], dest)
    else:
        raise ValueError(f"unknown dataset kind: {info['kind']}")

    print(f"Done. See {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
