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
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- I/O helpers ----------------
def read_lines(p: Path) -> List[str]:
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_sidecars(paths: List[str], caps: List[str], prefix: str, out_dir: Path):
    # TSV
    tsv = out_dir / f"{prefix}_captions.tsv"
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("image_path\tcaption\n")
        for p, c in zip(paths, caps):
            f.write(f"{p}\t{c}\n")
    # JSONL
    jsonl = out_dir / f"{prefix}_captions.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for p, c in zip(paths, caps):
            f.write(json.dumps({"image_path": p, "caption": c}, ensure_ascii=False) + "\n")
    # TXT
    txt = out_dir / f"{prefix}_captions.txt"
    with open(txt, "w", encoding="utf-8") as f:
        for c in caps:
            f.write(c + "\n")
    print(f"[SAVED] {tsv}\n[SAVED] {jsonl}\n[SAVED] {txt}")

def load_image_safe(p: str, size: Optional[int] = None) -> Image.Image:
    try:
        img = Image.open(p).convert("RGB")
    except Exception:
        # Return a tiny blank image if file is problematic; will still produce a caption
        img = Image.new("RGB", (size or 384, size or 384), color=(0, 0, 0))
    if size and size > 0:
        img = img.resize((size, size), resample=Image.BICUBIC)
    return img

def _to_device_cast_floats(d: dict, device: torch.device, float_dtype: torch.dtype) -> dict:
    """Move tensors to device; cast only floating tensors to float_dtype (keep int/long as-is)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(float_dtype)
        out[k] = v
    return out

# ---------------- main ----------------
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
    ap.add_argument("--beam_search", action="store_true", help="Use beam search (BLIP-2: only safe for batch_size=1)")
    ap.add_argument("--num_beams", type=int, default=5)

    ap.add_argument("--prompt", type=str, default=None,
                    help="Optional prompt (do NOT include '<image>' for BLIP-2).")
    ap.add_argument("--image_size", type=int, default=384,
                    help="Square resize for captioning input (384 common). Set 0 for native.")
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
    is_blip2 = "blip2" in args.model.lower()
    if is_blip2:
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
    else:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        processor = BlipProcessor.from_pretrained(args.model)
        model = BlipForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)

    # base generate kwargs
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams if args.beam_search else 1,
        "do_sample": False,
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
    }

    # ---- captioning helpers ----
    def caption_batch_blip2(img_paths: List[str]) -> List[str]:
        images = [load_image_safe(p, size=args.image_size) for p in img_paths]

        # BLIP-2: do NOT include "<image>" in prompt; processor handles image tokens.
        user_prompt = args.prompt if args.prompt else "Question: describe this image in one sentence. Answer:"
        user_prompt = user_prompt.replace("<image>", "").strip()
        if not user_prompt:
            user_prompt = "Question: describe this image in one sentence. Answer:"

        inputs = processor(images=images, text=[user_prompt] * len(images), return_tensors="pt")
        inputs = _to_device_cast_floats(inputs, device, model.dtype)

        # Use tokenizer eos/pad if available
        gk = dict(gen_kwargs)
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            if getattr(tok, "eos_token_id", None) is not None:
                gk["eos_token_id"] = tok.eos_token_id
            if getattr(tok, "pad_token_id", None) is not None:
                gk["pad_token_id"] = tok.pad_token_id

        # Safety: beam search can break for batch_size>1 — force greedy unless user uses bs=1
        if gk["num_beams"] > 1 and len(images) > 1:
            gk["num_beams"] = 1

        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model.generate(**inputs, **gk)
        caps = [c.strip() for c in processor.batch_decode(out, skip_special_tokens=True)]

        # Fallback: retry empties with sampling
        need_retry = [i for i, c in enumerate(caps) if len(c) == 0]
        if need_retry:
            retry_inputs = {k: (v[need_retry] if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            gk_retry = dict(gk); gk_retry.update(dict(do_sample=True, top_p=0.9, temperature=0.7, num_beams=1))
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                out2 = model.generate(**retry_inputs, **gk_retry)
            caps_retry = [c.strip() for c in processor.batch_decode(out2, skip_special_tokens=True)]
            for idx, c in zip(need_retry, caps_retry):
                caps[idx] = c
        return caps

    def caption_batch_blip(img_paths: List[str]) -> List[str]:
        images = [load_image_safe(p, size=args.image_size) for p in img_paths]
        if args.prompt:
            inputs = processor(images=images, text=[args.prompt] * len(images), return_tensors="pt")
        else:
            inputs = processor(images=images, return_tensors="pt")
        inputs = _to_device_cast_floats(inputs, device, model.dtype)
        gk = dict(gen_kwargs)
        # BLIP beam search is stable; keep user choice
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            out = model.generate(**inputs, **gk)
        return [c.strip() for c in processor.batch_decode(out, skip_special_tokens=True)]

    captioner = caption_batch_blip2 if is_blip2 else caption_batch_blip

    # ---- runner (resume-friendly) ----
    def run_split(prefix: str, paths_file: Path):
        img_paths = read_lines(paths_file)
        n = len(img_paths)
        print(f"[INFO] {prefix}: {n} images")

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
            batch_caps = [c if isinstance(c, str) and len(c) > 0 else "a photo" for c in batch_caps]  # final guard
            captions[i:j] = batch_caps
            with open(out_txt, "a", encoding="utf-8") as f:
                for c in batch_caps:
                    f.write(c + "\n")
            i = j
            pbar.update(len(batch_caps))
        pbar.close()

        save_sidecars(img_paths, captions, prefix, args.out_dir)

    run_split("train", args.train_paths)
    run_split("test",  args.test_paths)

if __name__ == "__main__":
    main()
