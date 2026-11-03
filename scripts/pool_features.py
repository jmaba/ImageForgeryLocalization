"""Cell features of the REAL pool (data/ssl_real) for real-only predictive training (scripts/train_predictor.py).

Same preprocessing as the splice shards (make_splices.py), so the predictor sees the feature distribution the
comparator was trained on. Each variant is a global laundering of the whole image (no region differs):
    clean      as loaded (max side --max-side)
    jpeg       re-saved as JPEG, quality uniform in [60, 95]
    resize     downscaled by a factor uniform in [0.5, 0.8]
Output: data/ssl_real/feats/<variant>/{dinov2_l6,srm,pixels}.npy  [N, 24, 24, D] fp16, rows = pool.csv rows.

    python scripts/pool_features.py --variants clean jpeg resize
"""
import argparse
import zlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifl import DATA  # noqa: E402
from ifl.features.backbones import build  # noqa: E402
from make_splices import jpeg, load_pool_image  # noqa: E402

SPECS = {"dinov2_l6": ("dinov2:l6", 336), "srm": ("srm", None), "pixels": ("pixels", 216)}


def launder(a, variant, rng):
    if variant == "jpeg":
        return jpeg(a, rng.integers(60, 96))
    if variant == "resize":
        s = rng.uniform(0.5, 0.8)
        h, w = a.shape[:2]
        return np.asarray(Image.fromarray(a).resize((max(8, round(w * s)), max(8, round(h * s))), Image.BICUBIC))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["clean", "jpeg", "resize"])
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--limit", type=int, default=None, help="first N pool images only")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    pool = pd.read_csv(DATA / "ssl_real" / "pool.csv")
    if args.limit:
        pool = pool.iloc[:args.limit]
    ext = {k: build(spec) for k, (spec, _) in SPECS.items()}
    for variant in args.variants:
        rng = np.random.default_rng(args.seed + zlib.crc32(variant.encode()))
        out = DATA / "ssl_real" / "feats" / variant
        out.mkdir(parents=True, exist_ok=True)
        mm, t0 = {}, time.time()
        for i in range(len(pool)):
            im = Image.fromarray(launder(load_pool_image(pool.path[i], args.max_side, 8), variant, rng))
            for k, e in ext.items():
                size = SPECS[k][1]
                x = im.resize((size, size), Image.BICUBIC) if size else im
                _, p = e(torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1)[None])
                if k not in mm:
                    mm[k] = np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.float16,
                                                      (len(pool),) + p.shape[1:])
                mm[k][i] = p[0].astype(np.float16)
            if i % 200 == 0:
                print(f"[{variant}] {i}/{len(pool)}  {time.time() - t0:.0f}s", flush=True)
        for m in mm.values():
            m.flush()


if __name__ == "__main__":
    main()
