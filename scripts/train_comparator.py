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
Labels are per-cell SOURCE ids (0 camera, k >= 1 region k, -1 ignored), so multi-source splices (shard `multi`)
train "same source?" across regions too; `point` sees them as binary (edited or not).
--test-suffixes evaluates the same trained models on perturbed copies of the test features
(extract_features.py --harmonize jpeg75 -> data/features/<spec>__jpeg75), i.e. robustness to re-sharing.

    python scripts/train_comparator.py --feats dinov2_l6 srm
    python scripts/train_comparator.py --shards vae inpaint multi --test-suffixes "" __jpeg75 __webp80 __resize50
"""
import argparse
import zlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import ndimage
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load, load_multi  # noqa: E402
from region_null import DOMAIN, MATCHED, cell_scores, pair_source, resid  # noqa: E402

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


def binary(src):
    return np.where(src > 0, 1, src).astype(np.int8)


def train(kind, X, src, tr, va, d_in, steps, seed=0, B=32, P=256):
    """X [N,576,D] fp16 numpy (standardised), src [N,576] source ids. Pairs/cells are sampled within images."""
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
        l = torch.from_numpy(src[b].astype(np.int64)).to(DEV)          # B,576
        if kind == "point":
            v = l >= 0
            loss = bce(model(x)[v], (l[v] > 0).float())
        else:
            h = model.enc(x)
            # half the pairs are forced across the edit boundary when the image has one, and with several
            # regions a further quarter is drawn among edited cells (about half of those cross two regions)
            i = torch.randint(0, G2, (len(b), P), device=DEV)
            j = torch.randint(0, G2, (len(b), P), device=DEV)
            h2, q = P // 2, P // 4
            for k in range(len(b)):
                pos, neg = torch.where(l[k] > 0)[0], torch.where(l[k] == 0)[0]
                if len(pos) and len(neg):
                    i[k, :h2] = pos[torch.randint(0, len(pos), (h2,), device=DEV)]
                    j[k, :h2] = neg[torch.randint(0, len(neg), (h2,), device=DEV)]
                if len(pos) and l[k].max() >= 2:
                    i[k, h2:h2 + q] = pos[torch.randint(0, len(pos), (q,), device=DEV)]
                    j[k, h2:h2 + q] = pos[torch.randint(0, len(pos), (q,), device=DEV)]
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
                  f"{np.nanmean(evaluate(model, kind, X[va], binary(src[va]))):.3f}", flush=True)
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


@torch.no_grad()
def pair_auc(model, X, src, n_pairs=2000, seed=0):
    """Per-image AUC of the pairwise affinity for "same source" over random valid cell pairs (multi-source)."""
    rng = np.random.default_rng(seed)
    out = []
    for x, l in zip(X, src):
        ok = np.where(l >= 0)[0]
        if len(np.unique(l[ok])) < 3:
            continue
        i, j = rng.choice(ok, n_pairs), rng.choice(ok, n_pairs)
        h = model.enc(torch.from_numpy(x.astype(np.float32)).to(DEV)[None])[0]
        a = torch.sigmoid(model.logits(h[i], h[j])).cpu().numpy()
        out.append(roc_auc_score(l[i] == l[j], a))
    return out


def load_models(path):
    """-> (models dict, Projector, gcols) saved by --save"""
    ck = torch.load(path, map_location=DEV, weights_only=False)
    models = {}
    for k, cls in (("pair", Pair), ("point", Point)):
        models[k] = cls(ck["d_in"]).to(DEV)
        models[k].load_state_dict(ck[k])
        models[k].eval()
    proj = Projector(ck["proj"]["dims"])
    vars(proj).update(ck["proj"])
    return models, proj, ck["gcols"]


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
    ap.add_argument("--test-suffixes", nargs="+", default=[""],
                    help='feature-dir suffixes of perturbed test copies, e.g. "" __jpeg75 __sr4')
    ap.add_argument("--cols", nargs="*", default=[],
                    help="keep only some columns of a stream (ablations), e.g. dtcwt_phase=1,3,5,6,8,11,12,14,15")
    ap.add_argument("--save", action="store_true", help="also write outputs/comparator/model_<tag>.pt (fusion)")
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
        sp = shard_dir(s) / "sources.npy"           # older shards: binary sources derived from the masks
        mms.append([np.load(shard_dir(s) / f"{f}.npy", mmap_mode="r") for f in args.feats] +
                   [np.load(sp, mmap_mode="r") if sp.exists() else labels(np.load(shard_dir(s) / "masks.npy"))])
    meta = pd.concat(metas, ignore_index=True)
    if "mask_kind" not in meta:
        meta["mask_kind"] = np.where(meta.type == "none", "none", "random")
    full = [mm.shape[-1] for mm in mms[0][:-1]]
    keep = {k: [int(c) for c in v.split(",")] for k, v in (c.split("=") for c in args.cols)}
    sel = [np.array(keep.get(f, range(d))) for f, d in zip(args.feats, full)]
    args.gcols = np.concatenate([o + c for o, c in zip(np.cumsum([0] + full[:-1]), sel)])
    dims = [len(c) for c in sel]
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
                                         .reshape(G2, -1) for k in range(len(dims))], -1)[:, args.gcols]
                         for g in idx])

    proj = Projector(dims).fit(raw(rng.choice(tr, min(150, len(tr)), replace=False)).reshape(-1, sum(dims)))
    X = np.empty((len(meta), G2, len(proj.mu)), np.float16)
    for k in range(0, len(meta), 100):
        idx = np.arange(k, min(k + 100, len(meta)))
        X[idx] = proj(raw(idx)).astype(np.float16)
    src = np.stack([np.asarray(mms[s][-1][r]).reshape(G2) for s, r in rows_of]).astype(np.int8)
    lab = binary(src)
    print(f"[{tag}] shards {args.shards}  train {len(tr)}  val {len(va)}  D={X.shape[-1]}  "
          f"excluded {args.exclude_types}", flush=True)
    models = {k: train(k, X, src, tr, va, X.shape[-1], args.steps) for k in ("pair", "point")}
    if args.save:
        torch.save(dict(feats=args.feats, gcols=args.gcols, d_in=X.shape[-1], proj=vars(proj),
                        **{k: m.state_dict() for k, m in models.items()}), out / f"model_{tag}.pt")
    val_rows = []
    for k, m in models.items():
        for (t, mk), sel in meta.iloc[va].groupby(["type", "mask_kind"]).groups.items():
            if t != "none":
                sel = np.asarray(sel)
                val_rows.append(dict(model=k, type=t + (" (held out)" if t in args.exclude_types else ""),
                                     mask=mk, auc=np.nanmean(evaluate(m, k, X[sel], lab[sel]))))
        if k == "pair":
            for t, sel in meta.iloc[va].groupby("type").groups.items():
                pa = pair_auc(m, X[np.asarray(sel)], src[np.asarray(sel)])
                if pa:
                    held = " (held out)" if t in args.exclude_types else ""
                    val_rows.append(dict(model="same_source", type=t + held, mask="all", auc=np.mean(pa)))
    val_tab = pd.DataFrame(val_rows).pivot_table(index=["type", "mask"], columns="model", values="auc").round(3)
    print(val_tab, flush=True)
    del X
    for suffix in args.test_suffixes:
        test(models, proj, args, tag + suffix, suffix, val_tab)


def test(models, proj, args, tag, suffix, val_tab):
    """Same images for every suffix: the subsample and the donor masks are seeded per dataset.
    `models` is the trained pair/point dict, or any callable (x [576,D], pixel stats C [576,33]) -> {name: scores}."""
    out = OUTPUTS / "comparator"

    # ---------------- test: all edited datasets + authentic images with transplanted masks
    names = [f + suffix for f in args.feats]
    fs = load_multi(names) if len(names) > 1 else load(names[0])
    pix = load("pixels" + suffix)
    pkey = {k: i for i, k in enumerate(pix.meta.dataset + "/" + pix.meta.sample_id.astype(str))}
    tm = fs.meta.reset_index(drop=True)
    keys = (tm.dataset + "/" + tm.sample_id.astype(str)).values
    eidx, tlab = R._edited(fs)
    ds = tm.dataset.values
    jobs = []
    for d in np.unique(ds[eidx]):
        dom = "tgif2" if d.startswith("tgif2") else d
        perm = np.random.default_rng(zlib.crc32(d.encode())).permutation(eidx[ds[eidx] == d])
        for i in perm[:args.max_per_dataset]:
            jobs.append((i, tlab[i].reshape(-1), dict(dataset=d, domain=dom, kind="edited")))
    src = pair_source(tm, eidx)
    for d, dom in DOMAIN.items():
        donors, r = eidx[ds[eidx] == d], np.random.default_rng(zlib.crc32(d.encode()) + 1)
        for i in np.where((ds == d) & (tm.label.values == 0))[0][:args.max_per_dataset]:
            j = src.get(str(tm.pair_id[i])) if dom in MATCHED else r.choice(donors)
            if j is not None:
                jobs.append((i, tlab[j].reshape(-1), dict(dataset=d, domain=dom, kind="authentic")))
    rows = []
    for n, (i, l, info) in enumerate(jobs):
        x = proj(np.asarray(fs.P[i], np.float32).reshape(G2, -1)[:, args.gcols])
        C = np.asarray(pix.P[pkey[keys[i]]], np.float32).reshape(G2, -1)
        ps = cell_scores(C, C)["loc_cf"]
        v = l >= 0
        # interior-only: drop every cell within one cell (~43 px) of the edit boundary, so a seam detector
        # cannot score (paste-back and exchange splices leave a sharp transition there)
        e = (l == 1).reshape(24, 24)
        band = (ndimage.binary_dilation(e) & ~ndimage.binary_erosion(e)).reshape(-1)
        vi = v & ~band
        sc = models(x, C) if callable(models) else \
            {**cell_score(models["pair"], "pair", x), **cell_score(models["point"], "point", x)}
        for name, s in list(sc.items()) + [("PIX_loc_cf", ps)]:
            for nm, ss in ((name, s), (f"{name}|pix", resid(s, ps))) if not name.startswith("PIX") else ((name, s),):
                rows.append(dict(info, scorer=nm, auc=R._auc(l[v], ss[v]) if (l[v] == 1).any() else np.nan,
                                 auc_int=R._auc(l[vi], ss[vi]) if (l[vi] == 1).any() and (l[vi] == 0).any()
                                 else np.nan, T=float(np.sort(s)[-10:].mean())))
        if n % 300 == 0:
            print(f"  test {n}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"rows_{tag}.csv", index=False)

    loc = df[df.kind == "edited"].pivot_table(index="scorer", columns="dataset", values="auc")
    loc["ALL"] = df[df.kind == "edited"].groupby("scorer").auc.mean()
    loc_int = df[df.kind == "edited"].pivot_table(index="scorer", columns="dataset", values="auc_int")
    loc_int["ALL"] = df[df.kind == "edited"].groupby("scorer").auc_int.mean()
    null =df[df.kind == "authentic"].pivot_table(index="scorer", columns="domain", values="auc")
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
        f.write(f"# {getattr(args, 'title', 'Learned real-only comparator')} - {tag}\n\nfeatures {args.feats} cols {args.cols}, shards {args.shards}, "
                f"types excluded from training {args.exclude_types}, test perturbation '{suffix or 'none'}'\n\n"
                "## Validation (held-out pool images)\n\n")
        f.write(val_tab.to_markdown())
        f.write("\n\n## Edited: per-image AUC\n\n" + loc.round(3).to_markdown())
        f.write("\n\n## Edited: interior-only per-image AUC (1-cell boundary band excluded)\n\n" +
                loc_int.round(3).to_markdown())
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
