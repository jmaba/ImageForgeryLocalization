"""Region self-consistency - Tests 3, 5 and 6 of reports/Region self consistency localization.md.

Everything is training-free and computed from ONE image at a time (cached 24x24 cell descriptors D and
content covariates C from the `pixels` stream). Evaluation: per-image AUC against the GT mask on partial
edits (cells with mask fraction in (0, 0.5) are ignored).

Cross-fitting: the 24x24 grid is split into 3x3 macro-blocks of 8x8 cells. A cell's reference set and
its content regression are fitted ONLY on cells of the other 8 macro-blocks, so an edit cannot serve as
its own reference unless it spans most of the image (the majority-authentic breakdown, Test 7).

Scorers (all oriented so that "higher = more inconsistent"; signed ones are also reported negated):
  loc_raw      |D - median(D)| / MAD, all cells as reference              (region-vs-rest, location)
  loc_reg      same on residuals of a ridge fit D ~ C on ALL cells          (content-regressed, naive)
  loc_cf       same on residuals of a cross-fitted Huber fit D ~ C         (content-regressed, cross-fit)
  loc_match    distance to median of the k content-nearest cells, cross-fitted   (like-with-like)
  scale_cf     signed log(local 3x3 MAD / reference median local MAD) on cross-fitted residuals
  energy_cf    energy distance between the 3x3 neighbourhood and a reference sample (distributional)
Negative controls (Test 3): descriptor = the `pixels` stream itself, and descriptor = random projection
of C plus independent noise. After content conditioning both must sit at AUC 0.50 +- 0.02.

    python scripts/region_tests.py --specs srm dtcwt shear lift dinov2_l6 --controls
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load  # noqa: E402

G, B = 24, 8                                   # grid, macro-block side (cells)
YY, XX = np.divmod(np.arange(G * G), G)
BLOCK = (YY // B) * (G // B) + XX // B         # macro-block id of every cell (0..8)
NB = np.stack([(np.clip(YY[:, None] + dy, 0, G - 1)) * G + np.clip(XX[:, None] + dx, 0, G - 1)
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)], -1)[:, 0, :]   # 3x3 neighbourhood ids [576, 9]
EPS = 1e-6


def rz(D, ref=None):
    ref = D if ref is None else ref
    med = np.median(ref, 0)
    mad = 1.4826 * np.median(np.abs(ref - med), 0) + EPS
    return (D - med) / mad, med, mad


def ridge(X, Y, lam=1.0):
    X1 = np.c_[np.ones(len(X)), X]
    return np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ Y)


def huber(X, Y, lam=1.0, it=5, c=1.345):
    """IRLS Huber regression, one weight per row (robust to edited cells inside the training blocks)."""
    X1 = np.c_[np.ones(len(X)), X]
    beta = ridge(X, Y, lam)
    for _ in range(it):
        r = Y - X1 @ beta
        s = 1.4826 * np.median(np.abs(r), 0) + EPS
        u = np.sqrt(((r / s) ** 2).mean(1))
        w = np.where(u <= c, 1.0, c / u)
        Xw = X1 * w[:, None]
        beta = np.linalg.solve(X1.T @ Xw + lam * np.eye(X1.shape[1]), Xw.T @ Y)
    return beta


def cf_residuals(D, Cz):
    """Cross-fitted content residuals: for each macro-block, the D~C map is fitted on the other blocks."""
    R_ = np.empty_like(D)
    for b in range(9):
        tr, te = BLOCK != b, BLOCK == b
        beta = huber(Cz[tr], D[tr])
        R_[te] = D[te] - np.c_[np.ones(te.sum()), Cz[te]] @ beta
    return R_


def energy_dist(A, Bm):
    """Sample energy distance between point sets A [n,d] and B [m,d] (Szekely & Rizzo)."""
    dab = np.sqrt(((A[:, None] - Bm[None]) ** 2).sum(-1)).mean()
    daa = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1)).mean()
    dbb = np.sqrt(((Bm[:, None] - Bm[None]) ** 2).sum(-1)).mean()
    return 2 * dab - daa - dbb


def score_image(D, C, rng, k=24, n_ref=96):
    out = {}
    Cz, _, _ = rz(C)
    out["loc_raw"] = np.sqrt((rz(D)[0] ** 2).mean(1))
    beta = ridge(Cz, D)
    out["loc_reg"] = np.sqrt((rz(D - np.c_[np.ones(len(D)), Cz] @ beta)[0] ** 2).mean(1))
    Rc = cf_residuals(D, Cz)
    Rz = np.empty_like(Rc)
    for b in range(9):                                     # robust z with cross-fitted location/scale
        Rz[BLOCK == b] = rz(Rc[BLOCK == b], Rc[BLOCK != b])[0]
    out["loc_cf"] = np.sqrt((Rz ** 2).mean(1))
    # like-with-like: k content-nearest cells from OTHER macro-blocks
    Dz = rz(D)[0]
    d2 = ((Cz[:, None] - Cz[None]) ** 2).sum(-1)
    d2[BLOCK[:, None] == BLOCK[None]] = np.inf
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    out["loc_match"] = np.sqrt(((Dz - np.median(Dz[nn], 1)) ** 2).mean(1))
    # signed dispersion of residuals in the 3x3 neighbourhood vs the cross-fitted reference
    loc_mad = np.median(np.abs(Rz[NB] - np.median(Rz[NB], 1, keepdims=True)), 1).mean(1) + EPS
    sc = np.empty(len(D))
    for b in range(9):
        sc[BLOCK == b] = np.log(loc_mad[BLOCK == b] / np.median(loc_mad[BLOCK != b]))
    out["scale_cf+"] = sc
    out["scale_cf-"] = -sc
    # distributional: energy distance of the 3x3 neighbourhood to a sample of other-block cells
    en = np.empty(len(D))
    for b in range(9):
        pool = np.where(BLOCK != b)[0]
        ref = Rz[rng.choice(pool, min(n_ref, len(pool)), replace=False)]
        for i in np.where(BLOCK == b)[0]:
            en[i] = energy_dist(Rz[NB[i]], ref)
    out["energy_cf"] = en
    return out


def reduce_dim(P, n=24):
    if P.shape[1] <= n:
        return P
    Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    return Pc @ Vt[:n].T


def run_one(spec, i, D, C, l, meta_row, seed):
    rng = np.random.default_rng(seed)
    v = l >= 0
    res = []
    for name, s in score_image(D, C, rng).items():
        res.append(dict(spec=spec, scorer=name, dataset=meta_row["dataset"], sample_id=meta_row["sample_id"],
                        pair_id=meta_row.get("pair_id", ""), mask_frac=meta_row["mask_frac"],
                        auc=R._auc(l[v], s[v])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["srm", "dtcwt", "shear", "lift", "dinov2_l6"])
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--max-per-dataset", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    out = OUTPUTS / "region"
    out.mkdir(parents=True, exist_ok=True)
    content = load("pixels")
    ckey = (content.meta.dataset + "/" + content.meta.sample_id.astype(str)).values
    cidx = {k: i for i, k in enumerate(ckey)}
    specs = list(args.specs) + (["CTRL_pixels", "CTRL_randproj"] if args.controls else [])
    rows = []
    for spec in specs:
        fs = content if spec.startswith("CTRL") else load(spec)
        idx, lab = R._edited(fs)
        rng = np.random.default_rng(0)
        # balanced, seeded subsample per dataset (keeps the run to minutes)
        ds = fs.meta.dataset.values
        idx = np.concatenate([rng.permutation(idx[ds[idx] == d])[:args.max_per_dataset] for d in np.unique(ds[idx])])
        keys = (fs.meta.dataset + "/" + fs.meta.sample_id.astype(str)).values
        proj = rng.normal(size=(content.P.shape[-1], 8))
        jobs = []
        for n, i in enumerate(idx):
            C = np.asarray(content.P[cidx[keys[i]]], np.float32).reshape(G * G, -1)
            if spec == "CTRL_pixels":
                D = C.copy()
            elif spec == "CTRL_randproj":
                D = rz(C)[0] @ proj + np.random.default_rng(n).normal(size=(G * G, 8))
            else:
                D = reduce_dim(np.asarray(fs.P[i], np.float32).reshape(G * G, -1))
            m = fs.meta.iloc[i].to_dict()
            m["mask_frac"] = float(fs.M[i].mean())
            jobs.append((spec, i, D, C, lab[i].reshape(-1), m, n))
        for r in Parallel(n_jobs=args.jobs, batch_size=8)(delayed(run_one)(*j) for j in jobs):
            rows += r
        print(f"[{spec}] {len(idx)} images", flush=True)
        pd.DataFrame(rows).to_csv(out / "region_tests_rows.csv", index=False)   # checkpoint
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["spec", "scorer"], columns="dataset", values="auc")
    piv["ALL"] = df.groupby(["spec", "scorer"]).auc.mean()
    piv["frac<0.5"] = df.groupby(["spec", "scorer"]).auc.apply(lambda a: float((a < 0.5).mean()))
    piv.round(3).to_csv(out / "region_tests.csv")
    pd.set_option("display.width", 320)
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
