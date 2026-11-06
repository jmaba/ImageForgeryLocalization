"""Check the FLAME wrapper (ifl/features/baselines.py FlameMap) against FLAME's own single-image path on its demo
images, and on one benchmark image: same preprocessing should give the same probability map.

    IFL_BASELINES=/workspace/baselines python scripts/remote/flame_check.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ifl.features.baselines import ROOT, FlameMap  # noqa: E402


def main():
    fm = FlameMap()
    code = ROOT / "FLAME" / "FLAME"
    sys.path.insert(0, str(code))
    from torchvision.transforms import functional as TF
    demo = sorted((code / "demo_test_set" / "magicbrush" / "target").glob("*.png"))
    for p in demo:
        im = Image.open(p).convert("RGB")
        # official path: PIL bilinear resize to 512, ToTensor, model(orig, [], output_extras=True)
        t = TF.to_tensor(im.resize((fm.size, fm.size), Image.BILINEAR))[None].cuda()
        with torch.no_grad(), torch.autocast("cuda"):
            lo = fm.model(t, [], output_extras=True)
        lo = lo[0] if isinstance(lo, tuple) else lo
        p_off = torch.sigmoid(lo.float())
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None]
        g, cells = fm(x)
        off_cells = F.adaptive_avg_pool2d(p_off, 24)[0, 0].cpu().numpy()
        gt = np.asarray(Image.open(code / "demo_test_set" / "magicbrush" / "mask" / p.name).convert("L")) > 127
        gt_cells = F.adaptive_avg_pool2d(torch.from_numpy(gt.astype(np.float32))[None, None], 24)[0, 0].numpy()
        r = np.corrcoef(off_cells.ravel(), cells[0, ..., 0].ravel())[0, 1]
        print(f"{p.name}: wrapper vs official cell map r={r:.4f}  map mean {cells[0, ..., 0].mean():.3f} "
              f"max {cells[0, ..., 1].max():.3f}  det {cells[0, 0, 0, 2]:.3f}  "
              f"corr with GT cells {np.corrcoef(gt_cells.ravel(), cells[0, ..., 0].ravel())[0, 1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
