#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot first N original test images with their BLIP-2 captions, using robust path matching.
Reads captions from JSONL with entries: {"image_path": "...", "caption": "..."}.

Output: /home/rothermm/THINGS/03_results/plots/caption_test.png
"""

import argparse, json, textwrap, re, os
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_lines(p: Path) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def load_image_safe(p: Path) -> Image.Image:
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return Image.new("RGB", (384, 384), color=(240, 240, 240))

def first_nonempty_line(s: str) -> str:
    if not s: return "(no caption)"
    parts = [t.strip() for t in re.split(r"\n+", s) if t.strip()]
    return parts[0] if parts else "(no caption)"

def rel_after_stimuli(path: Path) -> Optional[str]:
    # return posix suffix after ".../stimuli/images/"
    try:
        s = path.as_posix()
        key = "/stimuli/images/"
        i = s.lower().find(key)
        if i >= 0:
            return s[i+len(key):]  # e.g., "alligator/alligator_14n.jpg"
    except Exception:
        pass
    return None

def parent_plus_name(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"

def build_caption_indices(jsonl_path: Path):
    """
    Build multiple indices so we can match even if the absolute paths differ:
      idx_full[realpath]          -> caption
      idx_full_lower[lower str]   -> caption
      idx_rel[relative after stimuli/images] -> caption
      idx_pair["parent/name"]     -> caption
      idx_name[filename]          -> caption (only if unique)
    """
    idx_full: Dict[Path, str] = {}
    idx_full_lower: Dict[str, str] = {}
    idx_rel: Dict[str, str] = {}
    idx_pair: Dict[str, str] = {}
    name_to_caps: Dict[str, List[str]] = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            ip = Path(obj["image_path"])
            cap = first_nonempty_line(str(obj.get("caption", "")).strip())

            # realpath (don’t explode on missing)
            try:
                real = ip.resolve()
            except Exception:
                real = ip

            idx_full[real] = cap
            idx_full_lower[real.as_posix().lower()] = cap

            rel = rel_after_stimuli(real)
            if rel: idx_rel[rel] = cap

            pair = parent_plus_name(real)
            idx_pair[pair] = cap

            name = real.name
            name_to_caps.setdefault(name, []).append(cap)

    # build unique filename index
    idx_name: Dict[str, str] = {
        k: v[0] for k, v in name_to_caps.items() if len(v) == 1
    }

    return idx_full, idx_full_lower, idx_rel, idx_pair, idx_name

def lookup_caption(path: Path, indices) -> str:
    idx_full, idx_full_lower, idx_rel, idx_pair, idx_name = indices

    # 1) exact resolved
    try:
        real = path.resolve()
    except Exception:
        real = path
    cap = idx_full.get(real)
    if cap: return cap

    # 2) lowercased full
    cap = idx_full_lower.get(real.as_posix().lower())
    if cap: return cap

    # 3) suffix after stimuli/images
    rel = rel_after_stimuli(real)
    if rel:
        cap = idx_rel.get(rel)
        if cap: return cap

    # 4) parent/name
    cap = idx_pair.get(parent_plus_name(real))
    if cap: return cap

    # 5) filename only (unique only)
    cap = idx_name.get(real.name)
    if cap: return cap

    return "(no caption)"

def wrap(text: str, width: int = 42) -> str:
    return textwrap.fill(text, width=width)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=1)
    ap.add_argument("--test_paths", type=Path, default=None)
    ap.add_argument("--captions_jsonl", type=Path, default=None)
    ap.add_argument("--out_png", type=Path, default=None)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    if args.test_paths is None:
        args.test_paths = Path(f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sp}/test_image_paths.txt")
    if args.captions_jsonl is None:
        args.captions_jsonl = Path(f"/home/rothermm/THINGS/02_data/captions/subj{sp}/test_captions.jsonl")
    if args.out_png is None:
        args.out_png = Path("/home/rothermm/THINGS/03_results/plots/caption_test.png")

    if not args.test_paths.exists(): raise FileNotFoundError(args.test_paths)
    if not args.captions_jsonl.exists(): raise FileNotFoundError(args.captions_jsonl)
    ensure_dir(args.out_png.parent)

    # load
    img_paths = [Path(p).expanduser() for p in read_lines(args.test_paths)]
    indices = build_caption_indices(args.captions_jsonl)

    m = max(0, min(args.n, len(img_paths)))
    img_paths = img_paths[:m]

    # figure
    cols = m if m > 0 else 1
    fig_w, fig_h = 4.0 * cols, 6.0
    fig, axes = plt.subplots(1, cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0] if m > 0 else [axes]

    for i in range(cols):
        p = img_paths[i]
        img = load_image_safe(p)
        ax = axes[i]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
        ax.set_title(f"#{i} — {p.name}", fontsize=10, pad=6)

        cap = lookup_caption(p, indices)
        ax.text(0.5, -0.12, wrap(cap), transform=ax.transAxes, ha="center", va="top", fontsize=10)

    fig.suptitle(f"BLIP-2 captions — first {m} test images (sub-{sp})",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.92])
    fig.savefig(args.out_png, dpi=200); plt.close(fig)
    print(f"[DONE] Saved -> {args.out_png.resolve()}")

if __name__ == "__main__":
    main()
