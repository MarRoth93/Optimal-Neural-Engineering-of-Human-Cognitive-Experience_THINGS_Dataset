#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_map_ids_to_paths.py

Map THINGS image IDs to full file paths (supports subfolders).

Given:
- A text file with image IDs (one per line, e.g. 'dog_12s.jpg')
- The root directory containing the THINGS images (may contain subfolders)
- An output path for a .txt file listing absolute image paths

Output:
- A .txt file with full paths, order-matched to the IDs

Usage:
python 05_map_ids_to_paths.py \
    --ids_file train_image_ids.txt \
    --images_root /home/rothermm/THINGS/02_data/stimuli/images \
    --out_file train_image_paths.txt
"""

import argparse
from pathlib import Path
from collections import defaultdict

def build_image_index(images_root: Path, exts):
    """Recursively index all images under images_root by lowercase basename."""
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    norm_exts = set(e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts)
    index = {}
    dups = defaultdict(list)

    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in norm_exts:
            continue
        key = p.name.lower()
        abs_str = str(p.resolve())
        if key in index:
            if not dups[key]:
                dups[key].append(index[key])
            dups[key].append(abs_str)
        else:
            index[key] = abs_str

    return index, dups

def choose_deterministically(paths):
    """Pick one path deterministically for duplicates."""
    return sorted(paths, key=lambda s: (len(s), s))[0]

def main():
    ap = argparse.ArgumentParser(description="Map THINGS image IDs to full file paths (supports subfolders).")
    ap.add_argument("--ids_file", type=Path, required=True, help="Text file with one image filename per line")
    ap.add_argument("--images_root", type=Path, required=True, help="Root folder containing THINGS images (with subfolders)")
    ap.add_argument("--out_file", type=Path, required=True, help="Output .txt file for absolute image paths")
    ap.add_argument("--allow-duplicates", action="store_true",
                    help="If multiple files have the same basename, pick one deterministically instead of failing")
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"],
                    help="File extensions to consider (default: .jpg .jpeg .png)")
    ap.add_argument("--debug", action="store_true", help="Print summary of indexing and matching before writing output")
    args = ap.parse_args()

    # Build index
    index, dups = build_image_index(args.images_root, args.exts)

    if args.debug:
        print(f"[DEBUG] Indexed {len(index)} unique basenames from {args.images_root}")
        if dups:
            print(f"[DEBUG] Found {len(dups)} duplicate basenames")
        else:
            print("[DEBUG] No duplicates found")

    if dups and not args.allow_duplicates:
        msg_lines = [
            "Duplicate basenames found. Re-run with --allow-duplicates to auto-pick one,",
            "or remove/rename duplicates. Examples:"
        ]
        shown = 0
        for name, paths in dups.items():
            msg_lines.append(f"  {name} ->")
            for p in paths[:5]:
                msg_lines.append(f"    {p}")
            if len(paths) > 5:
                msg_lines.append(f"    ... (+{len(paths)-5} more)")
            shown += 1
            if shown >= 10:
                msg_lines.append("  (showing first 10 duplicate groups)")
                break
        raise RuntimeError("\n".join(msg_lines))

    # Read IDs
    with open(args.ids_file, "r") as f:
        raw_ids = [line.strip() for line in f if line.strip()]

    if args.debug:
        print(f"[DEBUG] Loaded {len(raw_ids)} IDs from {args.ids_file}")

    # Map to full paths
    paths = []
    missing = []
    chosen_from_dups = 0

    for img_id in raw_ids:
        key = Path(img_id).name.lower()
        if key in index:
            paths.append(index[key])
        elif key in dups and args.allow_duplicates:
            chosen = choose_deterministically(dups[key])
            paths.append(chosen)
            chosen_from_dups += 1
        else:
            missing.append(img_id)

    if missing:
        sample = "\n  ".join(missing[:20])
        more = f"\n  ... (+{len(missing)-20} more)" if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"{len(missing)} image(s) from IDs were not found under {args.images_root}.\n"
            f"First missing entries:\n  {sample}{more}"
        )

    if args.debug:
        print(f"[DEBUG] Matched {len(paths)} paths")
        if chosen_from_dups:
            print(f"[DEBUG] Chose deterministically from {chosen_from_dups} duplicate cases")

    # Save
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        for p in paths:
            f.write(f"{p}\n")

    print(f"[DONE] Saved {len(paths)} image paths to {args.out_file}")

if __name__ == "__main__":
    main()
