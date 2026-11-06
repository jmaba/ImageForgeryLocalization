"""Region self-consistency - Test 1 (false alarms on authentic images) and the pixel-increment check.

Why: in region_tests.py the CTRL_pixels "descriptor" (cell colour statistics) already reached AUC ~0.61, i.e.
edited regions differ in CONTENT, so a descriptor only carries a forensic trace if it beats pixels.

  1. Mask-transplant null. Score AUTHENTIC images against a mask borrowed from an edited image of the same
     source (TGIF2: the exact mask of the same pair_id; OpenSDI / SID-Set: a random edited image's mask).
     Any scorer that is independent of the edit has expected AUC exactly 0.5 here, so the deviation measures
     scorer bias (centre, borders, object priors) rather than tampering.
  2. Pixel increment. Per image, regress the descriptor's cell score on the CTRL_pixels cell score and
     evaluate the residual ("<scorer>|pix"): what the descriptor sees that colour/texture statistics do not.
  3. Image-level false alarms. T = mean of the top-10 cell scores. The threshold is the conformal 95% quantile
     over the authentic images of the same domain; we report TPR of edited images at that threshold.

    python scripts/region_null.py --specs srm dinov2_l6 dtcwt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifl import OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load  # noqa: E402
from region_tests import BLOCK, G, NB, EPS, cf_residuals, reduce_dim, rz  # noqa: E402

DOMAIN = {"tgif2_orig": "tgif2", "opensdi": "opensdi", "sid_set": "sid_set",
          "korus": "korus", "imd2020_real": "imd2020_real", "nc2016": "nc2016", "cmfd_velo": "cmfd_velo",
          "bench_orig": "bench26", "benchx_orig": "benchx",
          "coco_orig_gpt_image_2": "gpt_image_2", "coco_orig_nanobanana2": "nanobanana2"}
MATCHED = {"tgif2", "korus", "imd2020_real", "nc2016", "cmfd_velo", "bench26", "benchx", "gpt_image_2", "nanobanana2"}   # authentic image = the edit's own original


def pair_source(meta, eidx):
    """pair key -> an edited row, so an original can borrow the mask of its own edit. TGIF2 edits are
    '<coco id>_<variant>' and share the random mask of the COCO id; local benchmarks use the full pair_id."""
    src = {}
    for i in eidx:
        p, d = meta.pair_id[i], str(meta.dataset[i])
        if isinstance(p, str):
            src.setdefault(p.split("_")[0] if d.startswith("tgif2_") else p, i)
    return src


def cell_scores(D, C):
    """Only the scorers that survived region_tests.py (location and dispersion, cross-fitted)."""
    Cz = rz(C)[0]
    Rc = cf_residuals(D, Cz)
    Rz = np.empty_like(Rc)
    for b in range(9):
        Rz[BLOCK == b] = rz(Rc[BLOCK == b], Rc[BLOCK != b])[0]
    loc = np.sqrt((Rz ** 2).mean(1))
    loc_mad = np.median(np.abs(Rz[NB] - np.median(Rz[NB], 1, keepdims=True)), 1).mean(1) + EPS
    sc = np.empty(len(D))
    for b in range(9):
        sc[BLOCK == b] = np.log(loc_mad[BLOCK == b] / np.median(loc_mad[BLOCK != b]))
    return {"loc_cf": loc, "scale_cf+": sc}


def resid(s, p):
    X = np.c_[np.ones(len(p)), p]
    return s - X @ np.linalg.lstsq(X, s, rcond=None)[0]


def run_one(D, C, l, info):
    s = cell_scores(D, C)
    pix = cell_scores(C, C)
    v = l >= 0
    out = []
    for k in s:
        for name, sc in ((k, s[k]), (f"{k}|pix", resid(s[k], pix[k])), (f"PIX_{k}", pix[k])):
            out.append(dict(info, scorer=name, auc=R._auc(l[v], sc[v]) if (l[v] == 1).any() else np.nan,
                            T=float(np.sort(sc)[-10:].mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["srm", "dinov2_l6", "dtcwt"])
    ap.add_argument("--max-per-dataset", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    out = OUTPUTS / "region"
    content = load("pixels")
    ckey = (content.meta.dataset + "/" + content.meta.sample_id.astype(str)).values
    cidx = {k: i for i, k in enumerate(ckey)}
    rows = []
    for spec in args.specs:
        fs = load(spec)
        meta = fs.meta.reset_index(drop=True)
        keys = (meta.dataset + "/" + meta.sample_id.astype(str)).values
        eidx, lab = R._edited(fs)
        rng = np.random.default_rng(0)
        ds = meta.dataset.values
        jobs = []
        # edited images (same subsample as region_tests.py, restricted to domains that have authentic images)
        for d in np.unique(ds[eidx]):
            dom = "tgif2" if d.startswith("tgif2") else d
            if dom not in DOMAIN.values():
                continue
            for i in rng.permutation(eidx[ds[eidx] == d])[:args.max_per_dataset]:
                jobs.append((i, lab[i].reshape(-1), dict(spec=spec, dataset=d, domain=dom, kind="edited",
                                                          pair_id=meta.pair_id[i])))
        # authentic images with transplanted masks
        src = pair_source(meta, eidx)
        for d, dom in DOMAIN.items():
            ridx = np.where((ds == d) & (meta.label.values == 0))[0][:args.max_per_dataset]
            donors = eidx[ds[eidx] == d] if dom not in MATCHED else None
            for i in ridx:
                j = src.get(str(meta.pair_id[i])) if dom in MATCHED else rng.choice(donors)
                if j is None:
                    continue
                jobs.append((i, lab[j].reshape(-1), dict(spec=spec, dataset=d, domain=dom, kind="authentic",
                                                          pair_id=meta.pair_id[i])))
        work = []
        for i, l, info in jobs:
            C = np.asarray(content.P[cidx[keys[i]]], np.float32).reshape(G * G, -1)
            D = reduce_dim(np.asarray(fs.P[i], np.float32).reshape(G * G, -1))
            work.append((D, C, l, info))
        for r in Parallel(n_jobs=args.jobs, batch_size=8)(delayed(run_one)(*w) for w in work):
            rows += r
        print(f"[{spec}] {len(work)} images", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "region_null_rows.csv", index=False)

    # 1+2: localisation AUC on edited vs transplant-null AUC on authentic
    piv = df.pivot_table(index=["spec", "scorer"], columns=["kind", "domain"], values="auc").round(3)
    # 3: image-level TPR at the conformal 95% authentic threshold, per edited dataset
    tpr = []
    for (spec, sc, dom), g in df.groupby(["spec", "scorer", "domain"]):
        a = np.sort(g[g.kind == "authentic"]["T"].values)
        if len(a) < 20:
            continue
        thr = a[min(len(a) - 1, int(np.ceil(0.95 * (len(a) + 1))) - 1)]
        for d, e in g[g.kind == "edited"].groupby("dataset"):
            tpr.append(dict(spec=spec, scorer=sc, dataset=d, n_auth=len(a), tpr_at_fpr5=(e["T"] > thr).mean()))
    tpr = pd.DataFrame(tpr).pivot_table(index=["spec", "scorer"], columns="dataset", values="tpr_at_fpr5").round(3)
    with open(out / "region_null.md", "w") as f:
        f.write("# Region self-consistency: transplant null, pixel increment, image-level false alarms\n\n")
        f.write("## Per-image AUC (edited = real localisation; authentic = transplanted mask, must be 0.5)\n\n")
        f.write(piv.to_markdown() + "\n\n## Image-level TPR at 5% FPR (conformal threshold on authentic)\n\n")
        f.write(tpr.to_markdown() + "\n")
    pd.set_option("display.width", 320)
    print(piv.to_string())
    print(tpr.to_string())


if __name__ == "__main__":
    main()
