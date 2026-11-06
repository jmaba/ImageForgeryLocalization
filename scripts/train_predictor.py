"""Predictive self-consistency (FoJEPA-lite): learn, on REAL images only, to predict a cell's features from
its context; at test time a cell whose features are badly predicted is flagged.

Tokens are the 24 x 24 frozen cell features already used by the comparator (DINOv2-L6 + SRM, PCA-256 per
stream + standardisation, fitted on the real pool), plus each cell's pixel statistics (colour mean/std and a
3x3 thumbnail) as CONTENT conditioning. A block of cells is hidden (mask token instead of its features) and a
small transformer predicts the hidden features from the visible ones; the hidden cells keep their content
token unless --no-own-content, so the error measures what the cell's own colour/texture plus its context do
not explain. Training data: data/ssl_real/feats/<variant> (scripts/pool_features.py); global launderings
(jpeg, resize) are separate training images, so the predictor is fitted to what real photos look like after
re-sharing, not to one pipeline. No edited image, generator output or mask is ever seen.

Test: every cell is hidden by a sliding 4x4 block (stride 2); its score is the mean prediction error over the
blocks that hid it, overall (ctx_all) and per stream (ctx_dino, ctx_srm). Evaluated with the comparator's
harness (scripts/train_comparator.py test()): edited AUC, interior-only AUC, matched transplant null, pixel
control, image-level TPR, perturbed test copies.

    python scripts/train_predictor.py --tag ctx_small
    python scripts/train_predictor.py --tag ctx_big --d 384 --layers 8 --steps 40000 --batch 64   # bigger GPU
    python scripts/train_predictor.py --tag ctx_f25 --pool-frac 0.25          # data scaling (same validation set)
    python scripts/train_predictor.py --tag fuse --fuse outputs/comparator/model_base.pt   # + comparator fusion
    python scripts/train_predictor.py --tag trace --targets srm --fuse outputs/comparator/model_base.pt
--targets: the streams that are hidden and predicted (default: all). The others (e.g. DINOv2) are never hidden and
join the content token, so a hidden cell's noise residual is predicted from its own semantics + colour and from
its neighbours: the error then measures the trace, not how predictable the content is.
ctx_norm: the log error minus the log error expected for REAL cells with the same content token (ridge regression
fitted on held-out pool images), so hard-to-predict texture alone is not suspicious.
--fuse adds the saved comparator's pair_ref/point, their own ensemble (cmp_z2, the control for any fusion gain)
and per-image fusions with the predictor: fuse_rank (mean of within-image ranks of ctx_all and pair_ref),
fuse_z (sum of within-image z-scores), fuse_z_point (+ point), fuse_norm (ctx_norm + pair_ref + point).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifl import DATA, OUTPUTS  # noqa: E402
from train_comparator import G2, Projector, cell_score, load_models, test  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
G = 24


class CtxPredictor(nn.Module):
    def __init__(self, d_x, d_c, d=192, layers=4, heads=6, own_content=True):
        super().__init__()
        self.own_content = own_content
        self.inx, self.inc = nn.Linear(d_x, d), nn.Linear(d_c, d)
        self.mask = nn.Parameter(torch.zeros(d))
        self.pos = nn.Parameter(torch.randn(G2, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_x))

    def forward(self, x, c, m):
        """x [B,576,Dx] features, c [B,576,Dc] content, m [B,576] bool (True = hidden) -> predicted x."""
        h = torch.where(m[..., None], self.mask.expand(*m.shape, -1), self.inx(x))
        cc = self.inc(c)
        h = h + (cc if self.own_content else torch.where(m[..., None], torch.zeros_like(cc), cc))
        return self.out(self.body(h + self.pos))


def block_masks(B, rng, lo=0.15, hi=0.4):
    """1-4 random rectangles of 3-8 cells per image, union in [lo, hi] of the grid."""
    M = np.zeros((B, G, G), bool)
    for b in range(B):
        target = rng.uniform(lo, hi)
        while M[b].mean() < target:
            h, w = rng.integers(3, 9, 2)
            y, x = rng.integers(0, G - h + 1), rng.integers(0, G - w + 1)
            M[b, y:y + h, x:x + w] = True
    return M.reshape(B, G2)


def sliding_masks(k=4, stride=2):
    out = []
    for y in range(0, G - k + 1, stride):
        for x in range(0, G - k + 1, stride):
            m = np.zeros((G, G), bool)
            m[y:y + k, x:x + k] = True
            out.append(m.reshape(-1))
    return np.stack(out)


def flip(t):
    """horizontal flip of a [B,576,D] cell grid"""
    return t.reshape(t.shape[0], G, G, -1).flip(2).reshape(t.shape)


def load_predictor(path):
    """-> score(xr [576, D raw features in ck['feats'] order], C [576,33] pixel stats) -> {ctx_all, ctx_norm}"""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = CtxPredictor(ck["d_x"], ck["d_c"], ck["d"], ck["layers"], ck["heads"], ck["own"]).to(DEV).eval()
    model.load_state_dict(ck["state"])
    proj = Projector(ck["proj"]["dims"])
    vars(proj).update(ck["proj"])
    ti, ci = ck["ti"].to(DEV), ck["ci"].to(DEV)
    SM = torch.from_numpy(sliding_masks()).to(DEV)
    cover = SM.float().sum(0)

    @torch.no_grad()
    def score(xr, C):
        x = torch.from_numpy(proj(xr)).to(DEV)
        c = torch.from_numpy(((C - ck["cmu"]) / ck["csd"]).astype(np.float32)).to(DEV)
        xt, cc = x[..., ti], torch.cat([c, x[..., ci]], -1)
        n = len(SM)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            pred = model(xt[None].expand(n, -1, -1), cc[None].expand(n, -1, -1), SM).float()
        e = (((pred - xt[None]) ** 2 * SM[..., None]).sum(0) / cover[:, None]).mean(-1).cpu().numpy()
        return {"ctx_all": e, "ctx_norm": np.log(e + 1e-6) - ck["norm"].predict(cc.cpu().numpy())}
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["dinov2_l6", "srm"])
    ap.add_argument("--variants", nargs="+", default=["clean", "jpeg", "resize"])
    ap.add_argument("--tag", default="ctx")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-own-content", action="store_true")
    ap.add_argument("--max-per-dataset", type=int, default=150)
    ap.add_argument("--test-suffixes", nargs="+", default=[""])
    ap.add_argument("--pool-frac", type=float, default=1.0, help="fraction of the training pool images used")
    ap.add_argument("--fuse", default=None, help="comparator checkpoint (train_comparator.py --save)")
    ap.add_argument("--targets", nargs="+", default=None, help="streams to predict (default: all of --feats)")
    ap.add_argument("--norm-images", type=int, default=300, help="held-out pool images for the ctx_norm fit")
    ap.add_argument("--save", action="store_true", help="write outputs/comparator/predictor_<tag>.pt (ensemble expert)")
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---------------- real pool, all launderings; validation = held-out pool images (all their variants)
    root = DATA / "ssl_real" / "feats"
    raw = {v: [np.load(root / v / f"{f}.npy", mmap_mode="r") for f in args.feats] for v in args.variants}
    pixr = {v: np.load(root / v / "pixels.npy") for v in args.variants}
    N = len(pixr[args.variants[0]])
    dims = [a.shape[-1] for a in raw[args.variants[0]]]
    cat = lambda v, idx: np.concatenate([np.asarray(a[idx], np.float32).reshape(len(idx), G2, -1)  # noqa: E731
                                         for a in raw[v]], -1)
    va_img = rng.choice(N, int(0.15 * N), replace=False)
    tr_img = np.setdiff1d(np.arange(N), va_img)
    if args.pool_frac < 1:                                   # data scaling: nested subsets, same validation set
        tr_img = np.sort(np.random.default_rng(2).permutation(tr_img)[:int(round(args.pool_frac * len(tr_img)))])
    proj = Projector(dims).fit(cat("clean", rng.choice(tr_img, min(150, len(tr_img)), replace=False))
                               .reshape(-1, sum(dims)))
    X = np.empty((len(args.variants) * N, G2, len(proj.mu)), np.float16)          # [V*N,576,D], fp16 in RAM
    for k, v in enumerate(args.variants):
        for i in range(N):
            X[k * N + i] = proj(cat(v, [i])[0])
    Cr = np.concatenate([pixr[v].reshape(N, G2, -1).astype(np.float32) for v in args.variants])
    cmu, csd = Cr[np.tile(np.isin(np.arange(N), tr_img), len(args.variants))].reshape(-1, Cr.shape[-1]).mean(0), \
        Cr.reshape(-1, Cr.shape[-1]).std(0) + 1e-4
    Cz = ((Cr - cmu) / csd).astype(np.float32)
    rows_tr = np.concatenate([tr_img + k * N for k in range(len(args.variants))])
    rows_va = np.concatenate([va_img + k * N for k in range(len(args.variants))])
    # training tensors on the GPU when they take < 1/3 of its memory, else in (pinned) CPU RAM, batch by batch
    on_gpu = DEV == "cuda" and X.nbytes + Cz.nbytes < torch.cuda.get_device_properties(0).total_memory / 3
    Xt, Ct = torch.from_numpy(X), torch.from_numpy(Cz)
    Xt, Ct = (Xt.to(DEV), Ct.to(DEV)) if on_gpu else (Xt, Ct)
    del X
    split = np.cumsum([0] + [min(d, 256) for d in dims])        # stream boundaries after projection
    tgt = args.targets or args.feats
    is_t = [f in tgt for f in args.feats]
    span = [np.arange(split[k], split[k + 1]) for k in range(len(dims))]
    ti = torch.from_numpy(np.concatenate([a for a, t in zip(span, is_t) if t])).to(DEV)
    ci = torch.from_numpy(np.concatenate([a for a, t in zip(span, is_t) if not t] + [np.zeros(0, int)])).to(DEV)
    tsplit = np.cumsum([0] + [len(a) for a, t in zip(span, is_t) if t])

    def parts(x, c):
        """projected features + normalised pixel stats -> (targets, content = pixels + non-target streams)"""
        return x[..., ti], torch.cat([c, x[..., ci]], -1)
    print(f"[{args.tag}] pool {N} x {len(args.variants)} variants  train {len(rows_tr)}  val {len(rows_va)}  "
          f"D={Xt.shape[-1]} streams {dict(zip(args.feats, np.diff(split)))}  targets {tgt}  "
          f"data on {'GPU' if on_gpu else 'CPU'}", flush=True)

    model = CtxPredictor(len(ti), Cz.shape[-1] + len(ci), args.d, args.layers, args.heads,
                         not args.no_own_content).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)

    def loss_on(rows, masks, train):
        idx = torch.from_numpy(np.asarray(rows))
        x, c = parts(Xt[idx].to(DEV).float(), Ct[idx].to(DEV))
        if train and rng.random() < 0.5:
            x, c = flip(x), flip(c)
        m = torch.from_numpy(masks).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            pred = model(x, c, m)
        err = F.mse_loss(pred.float(), x, reduction="none").mean(-1)
        return (err * m).sum() / m.sum()

    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    t0 = time.time()
    vm = block_masks(len(rows_va), np.random.default_rng(1))
    for step in range(args.steps):
        model.train()
        rows = rng.choice(rows_tr, args.batch)
        loss = loss_on(rows, block_masks(args.batch, rng), True)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 1000 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vl = np.mean([loss_on(rows_va[k:k + 64], vm[k:k + 64], False).item()
                              for k in range(0, len(rows_va), 64)])
            print(f"  step {step}  train {loss.item():.4f}  val {vl:.4f}  {time.time() - t0:.0f}s", flush=True)
    val_tab = pd.DataFrame({"val masked MSE": [vl], "train images": [len(rows_tr)], "targets": [" ".join(tgt)]})

    # ---------------- per-cell error under the sliding masks
    SM = torch.from_numpy(sliding_masks()).to(DEV)
    cover = SM.float().sum(0)
    model.eval()

    @torch.no_grad()
    def cell_err(x, c):
        """x [576,D] projected, c [576,Dc] normalised pixel stats (GPU) -> ([576,Dt] error, [576,Dc'] content)"""
        x, c = parts(x, c)
        n = len(SM)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            pred = model(x[None].expand(n, -1, -1), c[None].expand(n, -1, -1), SM).float()
        return ((pred - x[None]) ** 2 * SM[..., None]).sum(0) / cover[:, None], c

    # ---------------- content normalisation: expected log error of a REAL cell given its content token
    from sklearn.linear_model import Ridge
    E, Q = [], []
    for r in np.random.default_rng(3).permutation(rows_va)[:args.norm_images]:
        e, c = cell_err(Xt[r].to(DEV).float(), Ct[r].to(DEV))
        E.append(torch.log(e.mean(-1) + 1e-6).cpu().numpy())
        Q.append(c.cpu().numpy())
    E, Q = np.concatenate(E), np.concatenate(Q)
    norm = Ridge(alpha=10.0).fit(Q, E)
    val_tab["content R2 of log error"] = r2 = norm.score(Q, E)
    print(f"  content explains {r2:.2f} of the real-cell log-error variance", flush=True)
    del Xt, Ct, E, Q
    if args.save:
        torch.save(dict(state=model.state_dict(), d_x=len(ti), d_c=Cz.shape[-1] + len(ci), d=args.d,
                        layers=args.layers, heads=args.heads, own=not args.no_own_content, ti=ti.cpu(), ci=ci.cpu(),
                        proj=vars(proj), cmu=cmu, csd=csd, norm=norm, feats=args.feats),
                   OUTPUTS / "comparator" / f"predictor_{args.tag}.pt")

    cmp = load_models(args.fuse) if args.fuse else None

    def rank(s):
        return np.argsort(np.argsort(s)) / (len(s) - 1)

    def zs(s):
        return (s - s.mean()) / (s.std() + 1e-8)

    @torch.no_grad()
    def score(xr, C):
        """xr: raw (unprojected) cell features, so the comparator can apply its own projector"""
        out = ctx_score(proj(xr), C)
        if cmp:
            models, cproj, cg = cmp
            xc = cproj(xr[:, cg])
            sc = {**cell_score(models["pair"], "pair", xc), **cell_score(models["point"], "point", xc)}
            cz = zs(sc["pair_ref"]) + zs(sc["point"])
            out |= {"cmp_pair_ref": sc["pair_ref"], "cmp_point": sc["point"], "cmp_z2": cz,
                    "fuse_rank": (rank(out["ctx_all"]) + rank(sc["pair_ref"])) / 2,
                    "fuse_z": zs(out["ctx_all"]) + zs(sc["pair_ref"]),
                    "fuse_z_point": zs(out["ctx_all"]) + cz,
                    "fuse_norm": zs(out["ctx_norm"]) + cz}
        return out

    @torch.no_grad()
    def ctx_score(x, C):
        e, c = cell_err(torch.from_numpy(x).to(DEV), torch.from_numpy(((C - cmu) / csd).astype(np.float32)).to(DEV))
        out = {"ctx_all": e.mean(-1).cpu().numpy()}
        tn = [f for f, t in zip(args.feats, is_t) if t]
        if len(tn) > 1:
            out |= {f"ctx_{f.split('_')[0]}": e[:, tsplit[k]:tsplit[k + 1]].mean(-1).cpu().numpy()
                    for k, f in enumerate(tn)}
        out["ctx_norm"] = np.log(out["ctx_all"] + 1e-6) - norm.predict(c.cpu().numpy())
        return out

    args.cols, args.shards, args.exclude_types = [], ["ssl_real/" + v for v in args.variants], []
    args.gcols = np.arange(sum(dims))
    args.title = "Predictive self-consistency (real-only context predictor)"
    val_tab["pool frac"] = args.pool_frac
    for suffix in [s for s in args.test_suffixes if s != "none"]:   # identity projector: score() projects itself
        test(score, lambda r: np.asarray(r, np.float32), args, args.tag + suffix, suffix, val_tab)


if __name__ == "__main__":
    main()
