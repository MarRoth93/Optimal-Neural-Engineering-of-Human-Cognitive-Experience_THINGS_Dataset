#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_caption_images.py

Batch-caption THINGS images using BLIP or BLIP-2.
Aligns exactly with train/test_image_paths.txt ordering.

Outputs (in --out_dir):
- train_captions.tsv / test_captions.tsv (image_path, caption)
- train_captions.jsonl / test_captions.jsonl
- train_captions.txt / test_captions.txt  (one caption per line, same order as *_image_paths.txt)

Usage (defaults for subj01):
python 12_caption_images.py --sub 1
"""

import os, sys, json, argparse
from pathlib import Path
from typing import List, Optional
import torch
from PIL import Image
from tqdm import tqdm

def read_lines(p: Path) -> List[str]:
    with open(p, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_sidecars(paths: List[str], caps: List[str], prefix: str, out_dir: Path):
    # TSV
    tsv = out_dir / f"{prefix}_captions.tsv"
    with open(tsv, "w") as f:
        f.write("image_path\tcaption\n")
        for p, c in zip(paths, caps):
            f.write(f"{p}\t{c}\n")
    # JSONL
    jsonl = out_dir / f"{prefix}_captions.jsonl"
    with open(jsonl, "w") as f:
        for p, c in zip(paths, caps):
            f.write(json.dumps({"image_path": p, "caption": c}, ensure_ascii=False) + "\n")
    # TXT (captions only, aligned with *_image_paths.txt)
    txt = out_dir / f"{prefix}_captions.txt"
    with open(txt, "w") as f:
        for c in caps:
            f.write(c + "\n")
    print(f"[SAVED] {tsv}\n[SAVED] {jsonl}\n[SAVED] {txt}")

def load_image_safe(p: str, size: Optional[int] = None) -> Image.Image:
    img = Image.open(p).convert("RGB")
    if size:
        # conservative resize to keep detail but bound memory
        img = img.resize((size, size))
    return img

def main():
    ap = argparse.ArgumentParser(description="Caption THINGS images (train/test) with BLIP/BLIP-2.")
    ap.add_argument("--sub", type=int, default=1, help="Subject number for default paths (e.g., 1)")
    ap.add_argument("--train_paths", type=Path, default=None, help="train_image_paths.txt")
    ap.add_argument("--test_paths",  type=Path, default=None, help="test_image_paths.txt")
    ap.add_argument("--out_dir",     type=Path, default=None, help="Output dir for captions")
    ap.add_argument("--model", type=str, default="Salesforce/blip2-opt-2.7b",
                  help="HF model: 'Salesforce/blip-image-captioning-large' or 'Salesforce/blip2-opt-2.7b'")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=30)
    ap.add_argument("--beam_search", action="store_true", help="Use beam search (slower, often better)")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--prompt", type=str, default=None,
                    help="Optional prefix prompt for BLIP/BLIP-2 (e.g., 'a detailed photo of')")
    ap.add_argument("--image_size", type=int, default=384,
                    help="Square resize for captioning input (BLIP typical 384). Set 0 to keep native.")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if None)")
    args = ap.parse_args()

    sp = f"{args.sub:02d}"
    if args.train_paths is None:
        args.train_paths = Path(f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sp}/train_image_paths.txt")
    if args.test_paths is None:
        args.test_paths  = Path(f"/home/rothermm/THINGS/02_data/preprocessed_data/subj{sp}/test_image_paths.txt")
    if args.out_dir is None:
        args.out_dir = Path(f"/home/rothermm/THINGS/02_data/captions/subj{sp}")
    ensure_dir(args.out_dir)

    for p in [args.train_paths, args.test_paths]:
        if not p.exists():
            raise FileNotFoundError(f"Missing paths file: {p}")

    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Device:", device)

    # ---- load model ----
    # BLIP variants
    if "blip2" in args.model.lower():
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        is_blip2 = True
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(args.model)
        model = BlipForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        is_blip2 = False

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams if args.beam_search else 1,
        "do_sample": False,
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
    }

    def caption_batch(img_paths: List[str]) -> List[str]:
        images = [load_image_safe(p, size=(args.image_size or None)) for p in img_paths]
        if is_blip2:
            # Only pass `text` if user provided one that contains the <image> token.
            # Otherwise, omit `text` entirely (BLIP-2 handles image-only prompts internally).
            if args.prompt and "<image>" in args.prompt:
                inputs = processor(images=images, text=[args.prompt]*len(images), return_tensors="pt")
            else:
                inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device, dtype=model.dtype) for k, v in inputs.items()}
        else:
            if args.prompt is not None:
                inputs = processor(images=images, text=[args.prompt]*len(images), return_tensors="pt").to(device, dtype=model.dtype)
            else:
                inputs = processor(images=images, return_tensors="pt").to(device, dtype=model.dtype)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
            out = model.generate(**inputs, **gen_kwargs)
        if is_blip2:
            captions = processor.batch_decode(out, skip_special_tokens=True)
        else:
            captions = [c.strip() for c in processor.decode(out[i], skip_special_tokens=True) for i in range(len(images))]
            # The above line isn't correct for batch; use this instead:
        return captions

    # fix BLIP batch decode for non-blip2:
    def caption_batch_blip(img_paths: List[str]) -> List[str]:
        images = [load_image_safe(p, size=(args.image_size or None)) for p in img_paths]
        if args.prompt is not None:
            inputs = processor(images=images, text=[args.prompt]*len(images), return_tensors="pt").to(device, dtype=model.dtype)
        else:
            inputs = processor(images=images, return_tensors="pt").to(device, dtype=model.dtype)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type=="cuda")):
            out = model.generate(**inputs, **gen_kwargs)
        return processor.batch_decode(out, skip_special_tokens=True)

    # choose captioner
    captioner = caption_batch if is_blip2 else caption_batch_blip

    def run_split(prefix: str, paths_file: Path):
        img_paths = read_lines(paths_file)
        n = len(img_paths)
        print(f"[INFO] {prefix}: {n} images")

        # resume support
        out_txt = args.out_dir / f"{prefix}_captions.txt"
        existing = []
        if out_txt.exists():
            existing = [ln.rstrip("\n") for ln in open(out_txt, "r", encoding="utf-8")]
            if len(existing) > n:
                print(f"[WARN] {out_txt} has more lines than images; truncating to {n}")
                existing = existing[:n]
        captions = list(existing) + [None] * (n - len(existing))

        bs = max(1, args.batch_size)
        pbar = tqdm(total=n, desc=f"Captioning {prefix}", unit="img")
        pbar.update(len(existing))
        i = len(existing)

        while i < n:
            j = min(i + bs, n)
            batch_paths = img_paths[i:j]
            batch_caps = captioner(batch_paths)
            # clean
            batch_caps = [c.strip() if isinstance(c, str) else "" for c in batch_caps]
            # commit in memory
            captions[i:j] = batch_caps
            # append to TXT incrementally (resume-friendly)
            with open(out_txt, "a", encoding="utf-8") as f:
                for c in batch_caps:
                    f.write(c + "\n")
            i = j
            pbar.update(len(batch_caps))
        pbar.close()

        # sidecars
        save_sidecars(img_paths, captions, prefix, args.out_dir)

    # run both splits
    run_split("train", args.train_paths)
    run_split("test",  args.test_paths)

if __name__ == "__main__":
    main()
