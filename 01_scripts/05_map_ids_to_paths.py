#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_map_ids_to_paths.py

Given:
- A text file with image IDs (one per line, e.g. 'dog_12s.jpg')
- The root directory containing the THINGS images
- An output path for a .txt file listing absolute image paths

Output:
- A .txt file with full paths, order-matched to the IDs

Usage:
python 05_map_ids_to_paths.py \
    --ids_file train_image_ids.txt \
    --images_root /path/to/THINGS-images \
    --out_file train_image_paths.txt
"""

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Map THINGS image IDs to full file paths.")
    ap.add_argument("--ids_file", type=Path, required=True, help="Text file with one image filename per line")
    ap.add_argument("--images_root", type=Path, required=True, help="Root folder containing THINGS images")
    ap.add_argument("--out_file", type=Path, required=True, help="Output .txt file for absolute image paths")
    args = ap.parse_args()

    if not args.images_root.exists():
        raise FileNotFoundError(f"Images root not found: {args.images_root}")

    # Read IDs
    with open(args.ids_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]

    # Map to full paths
    paths = []
    for img_id in ids:
        img_path = args.images_root / img_id
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        paths.append(str(img_path.resolve()))

    # Save
    with open(args.out_file, "w") as f:
        for p in paths:
            f.write(f"{p}\n")

    print(f"[DONE] Saved {len(paths)} image paths to {args.out_file}")

if __name__ == "__main__":
    main()
