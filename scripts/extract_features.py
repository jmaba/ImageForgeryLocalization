"""Extract frozen features for every manifest row and cache them.

    python scripts/extract_features.py --backbones dinov2 dinov2:l6 siglip2 clip srm vaerec

Cache layout (fp16):  data/features/<spec>/<dataset>/{global.npy, patch.npy, ids.txt}
Mask grids (shared): data/features/_masks/<dataset>.npy   [N, G, G] manipulated-area fraction
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA  # noqa: E402
from ifl.data.dataset import ManifestImages, load_manifest, mask_grid  # noqa: E402
from ifl.features.backbones import GRID, build, input_size  # noqa: E402

FEAT = DATA / "features"


def spec_dir(spec: str) -> str:
    return spec.replace(":", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["dinov2", "siglip2", "clip", "srm", "vaerec"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--harmonize", default=None, help="e.g. jpeg95 to re-encode every image first")
    ap.add_argument("--native-max-side", type=int, default=1024)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=2, help="DataLoader workers (each holds its own copy of torch)")
    args = ap.parse_args()

    df = load_manifest()
    datasets = args.datasets or sorted(df.dataset.unique())
    tag = f"__{args.harmonize}" if args.harmonize else ""

    (FEAT / "_masks").mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        p = FEAT / "_masks" / f"{ds}.npy"
        sub = df[df.dataset == ds]
        if args.force or not p.exists() or len(np.load(p, mmap_mode="r")) != len(sub):
            np.save(p, np.stack([mask_grid(r, GRID) for r in sub.itertuples()]).astype(np.float16))

    for spec in args.backbones:
        try:
            ext = build(spec)
        except Exception as e:
            print(f"[{spec}] cannot build: {type(e).__name__}: {str(e)[:200]}")
            continue
        size = input_size(ext)
        for ds in datasets:
            sub = df[df.dataset == ds].reset_index(drop=True)
            out = FEAT / (spec_dir(spec) + tag) / ds
            if not args.force and (out / "ids.txt").exists() and \
                    len((out / "ids.txt").read_text().split()) == len(sub):
                continue
            out.mkdir(parents=True, exist_ok=True)
            loader = torch.utils.data.DataLoader(
                ManifestImages(sub, size, args.harmonize, None if size else args.native_max_side),
                batch_size=args.batch if size else 1, num_workers=args.workers, shuffle=False)
            G, P = [], []
            t0 = time.time()
            for x, _ in tqdm(loader, desc=f"{spec}/{ds}", leave=False):
                g, p = ext(x)
                G.append(g.astype(np.float16))
                P.append(p.astype(np.float16))
            np.save(out / "global.npy", np.concatenate(G))
            np.save(out / "patch.npy", np.concatenate(P))
            (out / "ids.txt").write_text("\n".join(sub.sample_id.astype(str)))
            print(f"[{spec}] {ds}: {len(sub)} imgs  D={P[0].shape[-1]}  {time.time() - t0:.0f}s", flush=True)
        del ext
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
