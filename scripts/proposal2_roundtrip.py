"""Proposal 2 prototype - round-trip vs. de-novo localization.

    python scripts/proposal2_roundtrip.py --rt rt2-sd rt2-qwen --sem dinov2_l6 --train tgif2_sd2_fr

E1  training-free: per-image AUC of every single round-trip statistic (sign-free |AUC-.5| ranking)
E2  matched TGIF2 pairs: cell classes generated (FR, in mask) / re-decoded (FR, outside mask) /
    real (original image, same cell) -> pairwise AUCs per statistic
E3  leave-one-editor-out: linear patch probe trained ONLY on --train (SD2 random-mask inpainting), tested
    on every other dataset; feature sets = round-trip, semantic, both
E4  INP-X control: the same probe on matched SP (background restored) vs FR images - does the in-mask
    score survive, and does the outside-mask score drop (it should if it reads 're-decoded')?
Writes outputs/proposal2/{results.md, *.png, metrics.json}.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load, load_multi  # noqa: E402
from ifl.features.roundtrip import FEATURE_NAMES  # noqa: E402

OUT = OUTPUTS / "proposal2"
SEED = 0


def per_image_auc(S, lab):
    a = [R._auc(l[l >= 0], s[l >= 0]) for s, l in zip(S, lab)]
    return float(np.nanmean(a))


def e1_univariate(fs, rt_name, datasets):
    idx, lab = R._edited(fs)
    ds = fs.meta.dataset.values
    rows = []
    for d in datasets:
        ii = idx[ds[idx] == d]
        if len(ii) < 5:
            continue
        for k, f in enumerate(FEATURE_NAMES):
            rows.append(dict(vae=rt_name, dataset=d, feature=f,
                             auc=per_image_auc(np.asarray(fs.P[ii][..., k], np.float32), lab[ii])))
    return pd.DataFrame(rows)


def e2_three_class(fs, rt_name):
    """FR in-mask (generated) vs FR out-mask (re-decoded) vs original (real), matched by pair_id."""
    meta = fs.meta
    if "pair_id" not in meta or "tgif2_orig" not in set(meta.dataset):
        return pd.DataFrame()
    orig = {str(p): i for i, p in zip(meta.index, meta.pair_id) if meta.dataset[i] == "tgif2_orig"}
    rows = []
    # only FLUX edits share the exact crop of `orig-flux`; SD2/SDXL used other crops/sizes
    for d in sorted(x for x in meta.dataset.unique() if x.startswith("tgif2_flux") and x.endswith("_fr")):
        gen, rec, real = [], [], []
        for i in np.where(meta.dataset == d)[0]:
            cid = str(meta.pair_id[i]).split("_")[0]
            if cid not in orig:
                continue
            m = fs.M[i]
            gen.append(fs.P[i][m >= 0.5]); rec.append(fs.P[i][m == 0]); real.append(fs.P[orig[cid]][m == 0])
        if len(gen) < 5:
            continue
        G_, Rc, Rl = (np.concatenate(x).astype(np.float32) for x in (gen, rec, real))
        for k, f in enumerate(FEATURE_NAMES):
            def auc(a, b):
                return R._auc(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a[:, k], b[:, k]])
            rows.append(dict(vae=rt_name, dataset=d, n_pairs=len(gen), feature=f,
                             real_vs_redecoded=auc(Rl, Rc), redecoded_vs_generated=auc(Rc, G_),
                             real_vs_generated=auc(Rl, G_)))
    return pd.DataFrame(rows)


def e3_e4_transfer(fs, train_ds, rng):
    idx, lab = R._edited(fs)
    ds = fs.meta.dataset.values
    grp = R._groups(fs.meta).values
    tr_pool = idx[ds[idx] == train_ds]
    ug = np.unique(grp[tr_pool])
    hold = set(rng.choice(ug, len(ug) // 3, replace=False))
    tr = np.array([i for i in tr_pool if grp[i] not in hold])
    X, y = R._train_patches(fs, tr, lab)
    model = R._linear(X.shape[1]).fit(X, y)
    # TGIF2 subsets reuse COCO photos: never test on a source photo the probe was trained on
    cid = fs.meta.get("pair_id", pd.Series([""] * len(fs.meta))).astype(str).str.split("_").str[0].values
    seen = set(cid[tr]) - {"nan", ""}
    res = {}
    for d in sorted(np.unique(ds[idx])):
        te = idx[ds[idx] == d]
        if d == train_ds:
            te = np.array([i for i in te if grp[i] in hold])
        elif d.startswith("tgif2_"):
            te = np.array([i for i in te if cid[i] not in seen])
        if len(te) < 5:
            continue
        r, _ = R._loc_metrics(R._score_images(model, fs, te), lab[te])
        res[d] = r["img_mean_auc"]
    # E4: matched FR vs SP pairs - mean score inside / outside the mask
    e4 = []
    for fr in [d for d in np.unique(ds) if d.startswith("tgif2_") and d.endswith("_fr")]:
        sp = fr[:-3] + "_sp"
        if sp not in set(ds):
            continue
        pid = fs.meta.pair_id.astype(str).values
        a = {pid[i]: i for i in np.where(ds == fr)[0]}
        b = {pid[i]: i for i in np.where(ds == sp)[0]}
        common = sorted(set(a) & set(b))
        if len(common) < 5:
            continue
        Sfr = R._score_images(model, fs, np.array([a[c] for c in common]))
        Ssp = R._score_images(model, fs, np.array([b[c] for c in common]))
        Mfr = fs.M[[a[c] for c in common]]
        Msp = fs.M[[b[c] for c in common]]
        e4.append(dict(pair=fr[6:-3], n=len(common),
                       fr_in=float(Sfr[Mfr >= .5].mean()), fr_out=float(Sfr[Mfr == 0].mean()),
                       sp_in=float(Ssp[Msp >= .5].mean()), sp_out=float(Ssp[Msp == 0].mean()),
                       fr_img_auc=per_image_auc(Sfr, R._patch_labels(Mfr)),
                       sp_img_auc=per_image_auc(Ssp, R._patch_labels(Msp))))
    return res, pd.DataFrame(e4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", nargs="+", default=["rt2-sd", "rt2-qwen"])
    ap.add_argument("--sem", default="dinov2_l6")
    ap.add_argument("--train", default="tgif2_sd2_fr")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    md = ["# Proposal 2 prototype - round-trip vs. de-novo\n",
          f"Train set for probes: `{args.train}` only (SD2 inpainting, random masks). "
          "Every other column is an unseen editor / dataset. Numbers are per-image patch AUC (0.5 = chance).\n"]
    metrics = {}

    # ---------------- E1 / E2 per VAE
    e1s, e2s = [], []
    for spec in args.rt:
        fs = load(spec)
        e1s.append(e1_univariate(fs, spec, sorted(fs.meta.dataset.unique())))
        e2s.append(e2_three_class(fs, spec))
    e1 = pd.concat(e1s)
    piv = e1.pivot_table(index=["vae", "feature"], columns="dataset", values="auc")
    piv.to_csv(OUT / "e1_univariate.csv")
    md += ["## E1 - training-free single statistics (per-image AUC; <0.5 means the statistic is LOWER in the "
           "edit)\n", piv.round(3).to_markdown(), ""]
    e2 = pd.concat(e2s) if any(len(x) for x in e2s) else pd.DataFrame()
    if len(e2):
        e2.to_csv(OUT / "e2_three_class.csv", index=False)
        md += ["## E2 - matched TGIF2 cells: real / re-decoded / generated (pooled cell AUC)\n",
               e2.round(3).to_markdown(index=False), ""]

    # ---------------- E3 / E4 feature sets
    sets = {f"rt[{'+'.join(args.rt)}]": args.rt, args.sem: [args.sem], f"{args.sem}+rt": [args.sem] + args.rt}
    for s in args.rt:
        sets[s] = [s]
    e3, e4s = {}, []
    for name, specs in sets.items():
        fs = load_multi(specs)
        res, e4 = e3_e4_transfer(fs, args.train, rng)
        e3[name] = res
        if len(e4):
            e4.insert(0, "features", name)
            e4s.append(e4)
        print(f"[{name}] " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
    e3 = pd.DataFrame(e3).T
    e3.to_csv(OUT / "e3_transfer.csv")
    md += ["## E3 - leave-one-editor-out: probe trained on SD2 random-mask inpainting only\n",
           e3.round(3).to_markdown(), ""]
    if e4s:
        e4 = pd.concat(e4s)
        e4.to_csv(OUT / "e4_inpx.csv", index=False)
        md += ["## E4 - INP-X control on matched pairs (mean probe logit inside / outside the mask)\n",
               "If the probe reads *re-decoding*, `sp_out` (restored camera pixels) should drop well below "
               "`fr_out`; if it reads *generation*, `fr_in` and `sp_in` should stay similar.\n",
               e4.round(3).to_markdown(index=False), ""]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * e3.shape[1]), 4))
    e3.T.plot.bar(ax=ax, width=0.85)
    ax.axhline(0.5, color="k", lw=0.8, ls="--"); ax.set_ylim(0.3, 1.0)
    ax.set_title(f"Per-image patch AUC, probe trained on {args.train} only", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e3_transfer.png", dpi=110); plt.close(fig)
    md.append("![E3](e3_transfer.png)")
    metrics["e3"] = e3.to_dict()
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (OUT / "results.md").write_text("\n".join(md), encoding="utf-8")
    print((OUT / "results.md").read_text(encoding="utf-8")[:200])


if __name__ == "__main__":
    main()
