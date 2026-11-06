"""Check the B-Free wrapper against the official demo results (code/demo_images/results.csv) and time the map mode.

    IFL_BASELINES=/workspace/baselines python scripts/remote/bfree_check.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ifl.features.baselines import ROOT, BFreeMap  # noqa: E402


def main():
    demo = ROOT / "B-Free" / "code" / "demo_images"
    ref = pd.read_csv(demo / "results.csv").set_index("filename")["BFREE_dino2reg4"]
    bm = BFreeMap(local=True)
    for f, r in ref.items():
        x = torch.from_numpy(np.asarray(Image.open(demo / f).convert("RGB")).copy()).permute(2, 0, 1)[None]
        t0 = time.time()
        g, cells = bm(x)
        p = float(cells[0, 0, 0, 2])
        logit = np.log(p / (1 - p + 1e-12) + 1e-12)
        print(f"{f}: official logit {r:+.3f}  wrapper {logit:+.3f}  | map mean {cells[0, ..., 0].mean():.3f} "
              f"min {cells[0, ..., 0].min():.3f} max {cells[0, ..., 0].max():.3f}  | {time.time() - t0:.2f}s "
              f"({x.shape[-1]}x{x.shape[-2]})", flush=True)
    im = Image.open(Path("data/samples/bench26/fluxfill_sp") / sorted(p.name for p in
                    Path("data/samples/bench26/fluxfill_sp").glob("*.png"))[0])
    x = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1)[None]
    t0 = time.time()
    bm(x)
    print(f"benchmark image {im.size}: {time.time() - t0:.2f}s with map", flush=True)


if __name__ == "__main__":
    main()
