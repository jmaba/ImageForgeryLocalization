"""Precompute perturbations that need a model (cannot run inside the feature DataLoader).

    sr4   generative super-resolution: downscale by 4, then LDM 4x super-resolution
          (CompVis/ldm-super-resolution-4x-openimages) at <= 512 px, then bicubic back to the original size.
          Every pixel is re-synthesised, as when an edited image
          is "enhanced" before sharing (TGIF2 reports this weakens forensic evidence).

Output: data/perturbed/<mode>/<img_path>.png, read by extract_features.py --harmonize <mode>.

    python scripts/make_perturbed.py --mode sr4 --datasets tgif2_orig tgif2_sd2_fr tgif2_sd2_sp \
        tgif2_sdxl_fr tgif2_flux1dev_fr tgif2_flux1dev_sp
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA  # noqa: E402
from ifl.data.dataset import load_manifest, perturbed_path  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="sr4", choices=["sr4"])
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--max-side", type=int, default=512, help="cap on the synthesised size (speed); resized back")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    from diffusers import LDMSuperResolutionPipeline
    pipe = LDMSuperResolutionPipeline.from_pretrained(
        "CompVis/ldm-super-resolution-4x-openimages", torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    df = load_manifest()
    df = df[df.dataset.isin(args.datasets)].reset_index(drop=True)
    t0 = time.time()
    for n, r in enumerate(df.itertuples()):
        out = perturbed_path(r.img_path, args.mode)
        if out.exists():
            continue
        im = Image.open(DATA / r.img_path).convert("RGB")
        s = min(1.0, args.max_side / max(im.size))
        w, h = max(16, round(im.width * s / 4)), max(16, round(im.height * s / 4))
        lo = im.resize((w, h), Image.BICUBIC)
        with torch.no_grad():
            hi = pipe(image=lo, num_inference_steps=args.steps, eta=1.0,
                      generator=torch.Generator(DEV).manual_seed(n)).images[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        hi.convert("RGB").resize(im.size, Image.BICUBIC).save(out)
        if n % 50 == 0:
            print(f"{n}/{len(df)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(df)} images in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
