"""Bigger test: a LEARNED region-consistency relation, trained real-only on self-made splices.

Two models share the same cell encoder (frozen-feature cell token -> 64-D), trained on data/splices only:
    pair   f(u_i, u_j) = P(cells i and j come from the same source), symmetric head on [|h_i-h_j|, h_i*h_j]
    point  g(u_i)      = P(cell i was replaced)  (same data, absolute instead of relative supervision)
At test time the pairwise model is turned into a cell score by the majority assumption:
    pair_mean   s_i = 1 - mean_j f(i, j)
    pair_ref    s_i = 1 - mean_{j in R} f(i, j),  R = cells whose pair_mean is below the image median
Evaluation uses the protocol that exposed the training-free scorers (scripts/region_null.py):
    edited per-image AUC (12 datasets), transplant null on authentic images (must be 0.5), increment over the
    pixel-colour score, and image-level TPR at a conformal 5% false-alarm threshold.
No test editor, no COCO image and no test mask is ever seen in training.

    python scripts/train_comparator.py --feats dinov2_l6 srm
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load, load_multi  # noqa: E402
from region_null import DOMAIN, cell_scores, resid  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
G2 = 24 * 24


class Enc(nn.Module):
    def __init__(self, d_in, d=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, d), nn.LayerNorm(d))

    def forward(self, x):
        return self.net(x)


class Pair(nn.Module):
    def __init__(self, d_in, d=64):
        super().__init__()
        self.enc = Enc(d_in, d)
        self.head = nn.Sequential(nn.Linear(2 * d, 64), nn.GELU(), nn.Linear(64, 1))

    def logits(self, hi, hj):
        return self.head(torch.cat([(hi - hj).abs(), hi * hj], -1)).squeeze(-1)


class Point(nn.Module):
    def __init__(self, d_in, d=64):
        super().__init__()
        self.enc = Enc(d_in, d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(-1)


def labels(M):
    lab = np.full(M.shape, -1, np.int8)
    lab[M >= 0.5] = 1
    lab[M == 0] = 0
    return lab


def train(kind, X, lab, tr, va, d_in, steps, seed=0, B=32, P=256):
    """X [N,576,D] fp16 numpy (standardised), lab [N,576]. Pairs/cells are sampled within images."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = (Pair if kind == "pair" else Point)(d_in).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    bce = nn.BCEWithLogitsLoss()
    for step in range(steps):
        model.train()
        b = rng.choice(tr, min(B, len(tr)), replace=False)
        x = torch.from_numpy(X[b].astype(np.float32)).to(DEV)          # B,576,D
        l = torch.from_numpy(lab[b].astype(np.int64)).to(DEV)          # B,576
        if kind == "point":
            v = l >= 0
            loss = bce(model(x)[v], l[v].float())
        else:
            h = model.enc(x)
            # half the pairs are forced across the edit boundary when the image has one
            i = torch.randint(0, G2, (len(b), P), device=DEV)
            j = torch.randint(0, G2, (len(b), P), device=DEV)
            for k in range(len(b)):
                pos, neg = torch.where(l[k] == 1)[0], torch.where(l[k] == 0)[0]
                if len(pos) and len(neg):
                    h2 = P // 2
                    i[k, :h2] = pos[torch.randint(0, len(pos), (h2,), device=DEV)]
                    j[k, :h2] = neg[torch.randint(0, len(neg), (h2,), device=DEV)]
            li, lj = torch.gather(l, 1, i), torch.gather(l, 1, j)
            v = (li >= 0) & (lj >= 0)
            hi = torch.gather(h, 1, i[..., None].expand(-1, -1, h.shape[-1]))
            hj = torch.gather(h, 1, j[..., None].expand(-1, -1, h.shape[-1]))
            loss = bce(model.logits(hi, hj)[v], (li == lj)[v].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"  [{kind}] step {step} loss {loss.item():.4f}  val per-image AUC "
                  f"{np.nanmean(evaluate(model, kind, X[va], lab[va])):.3f}", flush=True)
    return model.eval()


@torch.no_grad()
def cell_score(model, kind, x):
    """x [576,D] float32 -> dict of cell scores"""
    x = torch.from_numpy(x).to(DEV)
    if kind == "point":
        return {"point": torch.sigmoid(model(x[None]))[0].cpu().numpy()}
    h = model.enc(x[None])[0]
    A = torch.sigmoid(model.logits(h[:, None].expand(-1, G2, -1), h[None].expand(G2, -1, -1)))
    A = (A + A.T) / 2
    s = 1 - A.mean(1)
    ref = s <= s.median()
    return {"pair_mean": s.cpu().numpy(), "pair_ref": (1 - A[:, ref].mean(1)).cpu().numpy()}


def evaluate(model, kind, X, lab):
    model.eval()
    out = []
    for x, l in zip(X, lab):
        v = l >= 0
        if (l[v] == 1).any() and (l[v] == 0).any():
            s = cell_score(model, kind, x.astype(np.float32))
            out.append(roc_auc_score(l[v], list(s.values())[-1][v]))
    return out


def shard_dir(name):
    return DATA / "splices" if name == "v1" else DATA / "splices_v2" / name


class Projector:
    """Per-feature PCA (only for D > n_pca) followed by standardisation, fitted on training splices and
    applied unchanged to the test features. Keeps ~6.5k images x 576 cells in RAM (DINOv2 is 768-D)."""

    def __init__(self, dims, n_pca=256):
        self.dims, self.n_pca, self.P, self.mu, self.sd = dims, n_pca, {}, None, None

    def _split(self, x):
        return np.split(x, np.cumsum(self.dims)[:-1], axis=-1)

    def fit(self, sample):
        parts = []
        for k, p in enumerate(self._split(sample)):
            if p.shape[-1] > self.n_pca:
                m = p.mean(0)
                _, _, vt = np.linalg.svd(p - m, full_matrices=False)
                self.P[k] = (m, vt[:self.n_pca].T)
                p = (p - m) @ self.P[k][1]
            parts.append(p)
        z = np.concatenate(parts, -1)
        self.mu, self.sd = z.mean(0), z.std(0) + 1e-4
        return self

    def __call__(self, x):
        parts = [((p - self.P[k][0]) @ self.P[k][1]) if k in self.P else p for k, p in enumerate(self._split(x))]
        return ((np.concatenate(parts, -1) - self.mu) / self.sd).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["dinov2_l6", "srm"])
    ap.add_argument("--shards", nargs="+", default=["vae", "inpaint"], help="data/splices_v2/<shard>, or v1")
    ap.add_argument("--exclude-types", nargs="*", default=[], help="splice types left out of TRAINING")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--max-per-dataset", type=int, default=150)
    args = ap.parse_args()
    tag = args.tag or "+".join(args.feats)
    out = OUTPUTS / "comparator"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- training data: real-only splices (memmapped shards, projected chunk-wise)
    metas, mms = [], []
    for s in args.shards:
        m = pd.read_csv(shard_dir(s) / "meta.csv")
        m["shard"] = s
        metas.append(m)
        mms.append([np.load(shard_dir(s) / f"{f}.npy", mmap_mode="r") for f in args.feats] +
                   [np.load(shard_dir(s) / "masks.npy", mmap_mode="r")])
    meta = pd.concat(metas, ignore_index=True)
    if "mask_kind" not in meta:
        meta["mask_kind"] = np.where(meta.type == "none", "none", "random")
    dims = [mm.shape[-1] for mm in mms[0][:-1]]
    rng = np.random.default_rng(0)
    pool_ids = meta.pool_idx.unique()
    va_pool = set(rng.choice(pool_ids, int(0.15 * len(pool_ids)), replace=False))
    is_va = meta.pool_idx.isin(va_pool).values
    va = np.where(is_va)[0]
    tr = np.where(~is_va & ~meta.type.isin(args.exclude_types).values)[0]
    rows_of = []                                   # global row -> (shard, local row)
    for si, m in enumerate(metas):
        rows_of += [(si, r) for r in range(len(m))]

    def raw(idx):
        return np.stack([np.concatenate([np.asarray(mms[rows_of[g][0]][k][rows_of[g][1]], np.float32)
                                         .reshape(G2, -1) for k in range(len(dims))], -1) for g in idx])

    proj = Projector(dims).fit(raw(rng.choice(tr, min(150, len(tr)), replace=False)).reshape(-1, sum(dims)))
    X = np.empty((len(meta), G2, len(proj.mu)), np.float16)
    for k in range(0, len(meta), 100):
        idx = np.arange(k, min(k + 100, len(meta)))
        X[idx] = proj(raw(idx)).astype(np.float16)
    lab = labels(np.stack([np.asarray(mms[s][-1][r], np.float32).reshape(G2) for s, r in rows_of]))
    print(f"[{tag}] shards {args.shards}  train {len(tr)}  val {len(va)}  D={X.shape[-1]}  "
          f"excluded {args.exclude_types}", flush=True)
    models = {k: train(k, X, lab, tr, va, X.shape[-1], args.steps) for k in ("pair", "point")}
    val_rows = []
    for k, m in models.items():
        for (t, mk), sel in meta.iloc[va].groupby(["type", "mask_kind"]).groups.items():
            if t != "none":
                sel = np.asarray(sel)
                val_rows.append(dict(model=k, type=t + (" (held out)" if t in args.exclude_types else ""),
                                     mask=mk, auc=np.nanmean(evaluate(m, k, X[sel], lab[sel]))))
    val_tab = pd.DataFrame(val_rows).pivot_table(index=["type", "mask"], columns="model", values="auc").round(3)
    print(val_tab, flush=True)
    del X

    # ---------------- test: all edited datasets + authentic images with transplanted masks
    fs = load_multi(args.feats) if len(args.feats) > 1 else load(args.feats[0])
    pix = load("pixels")
    pkey = {k: i for i, k in enumerate(pix.meta.dataset + "/" + pix.meta.sample_id.astype(str))}
    tm = fs.meta.reset_index(drop=True)
    keys = (tm.dataset + "/" + tm.sample_id.astype(str)).values
    eidx, tlab = R._edited(fs)
    ds = tm.dataset.values
    jobs = []
    for d in np.unique(ds[eidx]):
        dom = "tgif2" if d.startswith("tgif2") else d
        for i in rng.permutation(eidx[ds[eidx] == d])[:args.max_per_dataset]:
            jobs.append((i, tlab[i].reshape(-1), dict(dataset=d, domain=dom, kind="edited")))
    src = {}
    for i in eidx[np.char.startswith(ds[eidx].astype(str), "tgif2_")]:
        src.setdefault(str(tm.pair_id[i]).split("_")[0], i)
    for d, dom in DOMAIN.items():
        donors = eidx[ds[eidx] == d]
        for i in np.where((ds == d) & (tm.label.values == 0))[0][:args.max_per_dataset]:
            j = src.get(str(tm.pair_id[i])) if dom == "tgif2" else rng.choice(donors)
            if j is not None:
                jobs.append((i, tlab[j].reshape(-1), dict(dataset=d, domain=dom, kind="authentic")))
    rows = []
    for n, (i, l, info) in enumerate(jobs):
        x = proj(np.asarray(fs.P[i], np.float32).reshape(G2, -1))
        C = np.asarray(pix.P[pkey[keys[i]]], np.float32).reshape(G2, -1)
        ps = cell_scores(C, C)["loc_cf"]
        v = l >= 0
        sc = {**cell_score(models["pair"], "pair", x), **cell_score(models["point"], "point", x)}
        for name, s in list(sc.items()) + [("PIX_loc_cf", ps)]:
            for nm, ss in ((name, s), (f"{name}|pix", resid(s, ps))) if not name.startswith("PIX") else ((name, s),):
                rows.append(dict(info, scorer=nm, auc=R._auc(l[v], ss[v]) if (l[v] == 1).any() else np.nan,
                                 T=float(np.sort(s)[-10:].mean())))
        if n % 300 == 0:
            print(f"  test {n}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"rows_{tag}.csv", index=False)

    loc = df[df.kind == "edited"].pivot_table(index="scorer", columns="dataset", values="auc")
    loc["ALL"] = df[df.kind == "edited"].groupby("scorer").auc.mean()
    null = df[df.kind == "authentic"].pivot_table(index="scorer", columns="domain", values="auc")
    img = []
    for (sc, dom), g in df.groupby(["scorer", "domain"]):
        a = np.sort(g[g.kind == "authentic"]["T"].values)
        if len(a) < 20:
            continue
        thr = a[min(len(a) - 1, int(np.ceil(0.95 * (len(a) + 1))) - 1)]
        for d, e in g[g.kind == "edited"].groupby("dataset"):
            y = np.r_[np.zeros(len(a)), np.ones(len(e))]
            img.append(dict(scorer=sc, dataset=d, tpr5=(e["T"] > thr).mean(),
                            img_auc=roc_auc_score(y, np.r_[a, e["T"].values])))
    img = pd.DataFrame(img)
    with open(out / f"results_{tag}.md", "w") as f:
        f.write(f"# Learned real-only comparator - {tag}\n\nfeatures {args.feats}, shards {args.shards}, "
                f"types excluded from training {args.exclude_types}\n\n## Validation (held-out pool images)\n\n")
        f.write(val_tab.to_markdown())
        f.write("\n\n## Edited: per-image AUC\n\n" + loc.round(3).to_markdown())
        f.write("\n\n## Authentic + transplanted masks: per-image AUC (must be 0.5)\n\n" + null.round(3).to_markdown())
        f.write("\n\n## Image-level AUC (edited vs authentic of the same domain)\n\n" +
                img.pivot_table(index="scorer", columns="dataset", values="img_auc").round(3).to_markdown())
        f.write("\n\n## Image-level TPR at conformal 5% FPR\n\n" +
                img.pivot_table(index="scorer", columns="dataset", values="tpr5").round(3).to_markdown() + "\n")
    pd.set_option("display.width", 320)
    print(loc.round(3).to_string())
    print(null.round(3).to_string())
    print(img.pivot_table(index="scorer", columns="dataset", values="img_auc").round(3).to_string())


if __name__ == "__main__":
    main()
