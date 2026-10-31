"""Training-free region self-consistency: is each cell consistent with the rest of ITS OWN image?

No labels, no fitting across images - every score is computed from one image alone, then evaluated with
the per-image AUC against the GT mask (partial edits only).

Scorers (D = descriptor of each 24x24 cell, C = content covariates of the cell):
  raw        robust z-distance of D to the image median (per-channel MAD scaling)  -> region-vs-rest
  regress    residual of D after a per-image ridge regression on C, then robust z  -> content-normalised
  matched    distance of D to the median D of the k cells with the most similar C (excluding the 3x3
             neighbourhood)                                                         -> like-with-like
The content covariates come from the 'pixels' stream (per-cell RGB mean/std + 3x3 thumbnail), i.e. what
a texture/colour segmentation would group on.

    python scripts/region_consistency.py --specs srm dtcwt shear lift dinov2_l6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load  # noqa: E402

G = 24
YY, XX = np.divmod(np.arange(G * G), G)
NEAR = (np.abs(YY[:, None] - YY[None]) <= 1) & (np.abs(XX[:, None] - XX[None]) <= 1)


def robust_z(D):
    med = np.median(D, 0)
    mad = 1.4826 * np.median(np.abs(D - med), 0) + 1e-6
    return (D - med) / mad


def score_raw(D, C):
    return np.sqrt((robust_z(D) ** 2).mean(1))


def score_regress(D, C, lam=1.0):
    Cz = robust_z(C)
    X = np.c_[np.ones(len(Cz)), Cz]
    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ D)
    return score_raw(D - X @ beta, None)


def score_matched(D, C, k=24):
    Cz = robust_z(C)
    Dz = robust_z(D)
    d2 = ((Cz[:, None] - Cz[None]) ** 2).sum(-1)
    d2[NEAR] = np.inf
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    ref = np.median(Dz[nn], axis=1)
    return np.sqrt(((Dz - ref) ** 2).mean(1))


SCORERS = {"raw": score_raw, "regress": score_regress, "matched": score_matched}


def reduce_dim(P, n=32):
    """PCA per image would leak nothing across images; for high-D tokens use a fixed per-image projection
    onto the top-n principal axes of that image's own cells."""
    if P.shape[1] <= n:
        return P
    Pc = P - P.mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    return Pc @ Vt[:n].T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["srm", "dtcwt", "shear", "lift", "dinov2_l6"])
    ap.add_argument("--content", default="pixels")
    args = ap.parse_args()
    out = OUTPUTS / "region"
    out.mkdir(parents=True, exist_ok=True)
    content = load(args.content)
    ckey = content.meta.dataset + "/" + content.meta.sample_id.astype(str)
    cidx = {k: i for i, k in enumerate(ckey)}
    rows = []
    for spec in args.specs:
        fs = load(spec)
        idx, lab = R._edited(fs)
        keys = (fs.meta.dataset + "/" + fs.meta.sample_id.astype(str)).values
        for i in idx:
            if keys[i] not in cidx:
                continue
            D = reduce_dim(np.asarray(fs.P[i], np.float32).reshape(G * G, -1))
            C = np.asarray(content.P[cidx[keys[i]]], np.float32).reshape(G * G, -1)
            l = lab[i].reshape(-1)
            v = l >= 0
            for name, f in SCORERS.items():
                s = f(D, C)
                rows.append(dict(spec=spec, scorer=name, dataset=fs.meta.dataset.iloc[i],
                                 auc=R._auc(l[v], s[v]), mask_frac=float(fs.M[i].mean())))
        print(f"[{spec}] done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "region_consistency_rows.csv", index=False)
    piv = df.pivot_table(index=["spec", "scorer"], columns="dataset", values="auc")
    piv["ALL"] = df.groupby(["spec", "scorer"]).auc.mean()
    # how often is the score inverted (edit MORE consistent than the rest)?
    piv["frac_auc<0.5"] = df.groupby(["spec", "scorer"]).auc.apply(lambda a: float((a < 0.5).mean()))
    piv.round(3).to_csv(out / "region_consistency.csv")
    pd.set_option("display.width", 300)
    print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
