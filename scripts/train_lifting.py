"""Train LiftNet on the real-only pool (data/ssl_real). No manipulated image is ever seen.

    python scripts/train_lifting.py --steps 4000
-> data/checkpoints/liftnet.pt, outputs/logs/liftnet_train.csv
Crops are taken at native resolution (no resizing: it would erase the acquisition statistics).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.ssl.lifting import LiftNet  # noqa: E402


class Crops(torch.utils.data.Dataset):
    def __init__(self, paths, crop, n):
        self.paths, self.crop, self.n = paths, crop, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        im = Image.open(DATA / self.paths[rng.integers(len(self.paths))]).convert("RGB")
        a = np.asarray(im)
        H, W = a.shape[:2]
        c = min(self.crop, H, W) // 8 * 8
        y, x = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
        a = a[y:y + c, x:x + c]
        if c < self.crop:  # small image: tile-pad by reflection to keep batch shape
            a = np.pad(a, ((0, self.crop - c), (0, self.crop - c), (0, 0)), mode="reflect")
        if rng.random() < 0.5:
            a = a[:, ::-1]
        return torch.from_numpy(a.copy()).permute(2, 0, 1).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    pool = pd.read_csv(DATA / "ssl_real" / "pool.csv")
    val_paths = pool.path.values[::10]                 # 10% held out to watch over-fitting
    tr_paths = np.setdiff1d(pool.path.values, val_paths)
    dl = torch.utils.data.DataLoader(Crops(tr_paths, args.crop, args.steps * args.batch), batch_size=args.batch,
                                     num_workers=4, shuffle=False, drop_last=True)
    val = torch.stack([Crops(val_paths, args.crop, 64)[i] for i in range(64)]).cuda()
    net = LiftNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    log, t0 = [], time.time()
    for step, x in enumerate(dl):
        loss = net.loss(x.cuda())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            with torch.no_grad():
                vl = float(np.mean([net.loss(val[k:k + 16]).item() for k in range(0, 64, 16)]))
            log.append(dict(step=step, train=loss.item(), val=vl, sec=time.time() - t0))
            print(f"step {step:5d}  train {loss.item():.4f}  val {vl:.4f}  {time.time() - t0:.0f}s", flush=True)
    ck = DATA / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), ck / "liftnet.pt")
    (OUTPUTS / "logs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(OUTPUTS / "logs" / "liftnet_train.csv", index=False)


if __name__ == "__main__":
    main()
