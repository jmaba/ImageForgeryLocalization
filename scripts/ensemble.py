"""Mixture of forensic experts -> one localization map, evaluated under our protocol (docs/07).

Input: per-cell scores of every expert for every test image, dumped by the shared harness
(scripts/eval_physics.py writes outputs/ensemble/dump_phys<suffix>.pkl). Experts:
    mesorch, trufor          supervised, published weights (trace + learned edit appearance)
    comparator               real-only same-source consistency, z(pair_ref) + z(point)
    predictor                real-only context-prediction error (ctx_all)
    light, illum, blur       physics: shading vs global SH lighting fit, illuminant colour, defocus vs depth
Aggregators (all maps are z-scored within the image first, so scales never matter):
    zsum      equal-weight sum. Training-free.
    rel       reliability-weighted sum: TruFor weighted by its confidence map, blur by its validity, light by
              how well the cell's normals determine a direction. Training-free.
    stack     logistic regression on the cell's expert z-scores and their products with image-level gate
              descriptors (each expert's peakiness, mean inter-expert agreement, the TruFor image score):
              a per-image gated linear mixture. Fitted LEAVE-ONE-DOMAIN-OUT on edited images (every TGIF2
              variant + tgif2_orig form one domain), so a domain is never scored by a model that saw it.
    --train-suffix fits the stacker on one condition (e.g. clean) and applies it to another (e.g. __jpeg75).
Reported: mean per-image cell AUC (all / excl. IMD2020 / interior), matched null, AUC - null, per edit family,
and drop-one-expert ablations of zsum and stack.

    python scripts/ensemble.py --suffixes "" __jpeg75
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402

EXPERTS = {"mesorch": "x_mesorch", "trufor": "x_trufor", "comparator": "cmp_z2", "predictor": "x_ctx_all",
           "light": "phys_light", "illum": "phys_illum", "blur": "phys_blur"}
FAMILY = {"regenerated / LLM edit": ["gpt_image_2", "nanobanana2", "tgif2_sd2_fr", "tgif2_sdxl_fr", "tgif2_flux1dev_fr"],
          "diffusion splice / inpaint": ["tgif2_sd2_sp", "tgif2_flux1dev_sp", "cocoglide", "diverse_inpaint",
                                         "defacto_inpaint", "imd2020_inpaint", "opensdi", "sid_set", "diffseg30k",
                                         "autosplice"],
          "manual / classical": ["nc2016", "korus", "imd2020_real", "cmfd_velo"]}
IMD = {"imd2020_inpaint", "imd2020_real"}


def zs(s):
    s = np.nan_to_num(np.asarray(s, np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return (s - s.mean()) / (s.std() + 1e-8)


def load(suffix):
    with open(OUTPUTS / "ensemble" / f"dump_phys{suffix}.pkl", "rb") as fh:
        return pickle.load(fh)


def expert_z(job, names):
    return np.stack([zs(job["scores"][EXPERTS[k]]) for k in names], -1)          # [576, K]


def gate(Z, job):
    """image-level descriptors: per-expert peakiness (mean of the top 10 z), mean inter-expert correlation,
    the TruFor image score (mean of its map), plus a constant"""
    peak = np.sort(Z, 0)[-10:].mean(0)
    C = np.corrcoef(Z.T) if Z.shape[1] > 1 else np.ones((1, 1))
    agree = np.nanmean(C[np.triu_indices(len(C), 1)]) if len(C) > 1 else 1.0
    return np.r_[1.0, peak, np.nan_to_num(agree), float(np.mean(job["scores"]["x_trufor"]))]


def stack_feats(Z, g):
    return np.concatenate([Z, (Z[:, :, None] * g[None, None, 1:]).reshape(len(Z), -1)], 1)


def rel_score(job, names):
    sc, out = job["scores"], 0.0
    for k in names:
        z = zs(sc[EXPERTS[k]])
        if k == "trufor":
            z = z * np.clip(sc["x_trufor_conf"], 0, 1) * 2
        elif k == "blur":
            z = z * sc["x_blur_valid"]
        elif k == "light":
            c = sc["x_light_cond"]
            z = z * np.clip(c / (np.median(c) + 1e-8), 0, 2)
        out = out + z
    return out


def metrics(jobs, maps):
    rows = []
    for j, s in zip(jobs, maps):
        l = j["l"].astype(int)
        v = l >= 0
        vi = v & ~j["band"]
        rows.append(dict(dataset=j["dataset"], domain=j["domain"], kind=j["kind"],
                         auc=R._auc(l[v], s[v]) if (l[v] == 1).any() else np.nan,
                         auc_int=R._auc(l[vi], s[vi]) if (l[vi] == 1).any() and (l[vi] == 0).any() else np.nan))
    return pd.DataFrame(rows)


def summarize(df):
    e, a = df[df.kind == "edited"], df[df.kind == "authentic"]
    out = {"ALL": e.auc.mean(), "excl. IMD2020": e[~e.dataset.isin(IMD)].auc.mean(), "interior": e.auc_int.mean(),
           "null": a.auc.mean()}
    out["ALL - null"] = out["ALL"] - out["null"]
    for f, ds in FAMILY.items():
        out[f] = e[e.dataset.isin(ds)].auc.mean()
    return out


def stack_maps(train_jobs, test_jobs, names, seed=0, cells=60):
    """leave-one-domain-out: each test job is scored by a model fitted without its domain.
    Cell features are built once per job; folds only re-select which jobs train."""
    rng = np.random.default_rng(seed)
    tr = []                                                                   # (domain, X, y) per edited job
    for j in train_jobs:
        if j["kind"] != "edited":
            continue
        l = j["l"].astype(int)
        pos, neg = np.where(l == 1)[0], np.where(l == 0)[0]
        if not len(pos) or not len(neg):
            continue
        Z = expert_z(j, names)
        k = min(cells, len(pos), len(neg))
        idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)]
        tr.append((j["domain"], stack_feats(Z, gate(Z, j))[idx], l[idx]))
    te = [stack_feats(Z, gate(Z, j)) for j in test_jobs for Z in [expert_z(j, names)]]
    maps = [None] * len(test_jobs)
    for dom in sorted({j["domain"] for j in test_jobs}):
        X = np.concatenate([x for d, x, _ in tr if d != dom])
        y = np.concatenate([t for d, _, t in tr if d != dom])
        # standardised inputs: the gate products have very different scales (lbfgs converges far faster)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X, y)
        for i, j in enumerate(test_jobs):
            if j["domain"] == dom:
                maps[i] = m.decision_function(te[i])
    return maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffixes", nargs="+", default=[""])
    ap.add_argument("--train-suffix", default=None, help="fit the stacker on this condition (default: same)")
    ap.add_argument("--experts", nargs="+", default=list(EXPERTS))
    args = ap.parse_args()
    out = OUTPUTS / "ensemble"
    names = [k for k in args.experts if k in EXPERTS]
    train_jobs = load(args.train_suffix) if args.train_suffix is not None else None
    for suffix in args.suffixes:
        jobs = load(suffix)
        names_ok = [k for k in names if EXPERTS[k] in jobs[0]["scores"]]
        tj = train_jobs or jobs
        res = {}
        for k in names_ok:                                                     # single experts
            res[k] = summarize(metrics(jobs, [zs(j["scores"][EXPERTS[k]]) for j in jobs]))
        res["pixel control"] = summarize(metrics(jobs, [j["pix"] for j in jobs]))
        res["zsum"] = summarize(metrics(jobs, [expert_z(j, names_ok).sum(1) for j in jobs]))
        res["rel"] = summarize(metrics(jobs, [rel_score(j, names_ok) for j in jobs]))
        res["stack"] = summarize(metrics(jobs, stack_maps(tj, jobs, names_ok)))
        sub = [k for k in names_ok if k in ("mesorch", "trufor", "comparator", "predictor")]
        res["zsum w/o physics"] = summarize(metrics(jobs, [expert_z(j, sub).sum(1) for j in jobs]))
        res["stack w/o physics"] = summarize(metrics(jobs, stack_maps(tj, jobs, sub)))
        abl = {}
        for k in names_ok:                                                     # drop-one-expert ablation
            rest = [n for n in names_ok if n != k]
            abl[f"zsum - {k}"] = summarize(metrics(jobs, [expert_z(j, rest).sum(1) for j in jobs]))
            abl[f"stack - {k}"] = summarize(metrics(jobs, stack_maps(tj, jobs, rest)))
        main_t, abl_t = pd.DataFrame(res).round(3), pd.DataFrame(abl).round(3)
        hdr = f"# Ensemble of forensic experts - condition '{suffix or 'clean'}'" + \
              (f", stacker fitted on '{args.train_suffix or 'clean'}'" if args.train_suffix is not None else "")
        tag = (suffix or "_clean") + (f"_from{args.train_suffix or '_clean'}" if args.train_suffix is not None else "")
        (out / f"results{tag}.md").write_text(
            f"{hdr}\n\nexperts {names_ok}\n\n## Experts and aggregators\n\n{main_t.to_markdown()}\n\n"
            f"## Drop-one-expert ablation\n\n{abl_t.to_markdown()}\n")
        print(hdr)
        print(main_t.to_string())
        print(abl_t.to_string())


if __name__ == "__main__":
    main()
