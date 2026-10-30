"""Are the frozen latents representative of 'fake' / 'manipulated'?  Quantitative tools.

Image level (global vector)
  probe_image          linear-probe CV (grouped) real vs fake AUC, kNN acc, silhouette
  lodo_image           leave-one-dataset-out: generalisation to an unseen generator/editor
  confound_image       how well the SAME features predict nuisance factors (format, dataset) and the
                       real-vs-fake AUC *within* a single file format (JPEG-vs-PNG shortcut check)
Patch level (dense vector, localization)
  patch_probe          supervised linear patch probe, GroupKFold by image -> patch AUC, per-image
                       AUC / F1 / IoU, image-level AUC from max patch score, Fisher d'
  patch_transfer       train the probe on dataset A, test on B (within-image patch AUC)
  intra_image_zero_shot  training-free self-consistency: distance of each patch to the rest of
                       its own image (prototype + kNN) -> per-image AUC
  one_class_real       Gaussian (PCA + Ledoit-Wolf) fitted on REAL-image patches only; Mahalanobis
                       score of edited-image patches (Forensic-Twins / GMM style, no fake labels)

Patch labels: manipulated fraction >= POS_T -> 1, == 0 -> 0, in between (boundary) -> ignored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ifl import LOCAL_EDIT, REAL

POS_T = 0.5
SEED = 0


def _auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def _linear(D, n_comp=128, C=0.5):
    steps = [StandardScaler()]
    if D > n_comp:
        steps += [PCA(n_comp, random_state=SEED), StandardScaler()]
    return make_pipeline(*steps, LogisticRegression(C=C, max_iter=3000))


def _groups(meta):
    # images derived from the same source photo must not straddle folds
    # (AutoSplice: <src>_<variant>, GPT-Image-2 mapping: <src>_edit<k>)
    return meta.dataset + "/" + meta.sample_id.astype(str).str.replace(r"(_edit\d+|_\d)$", "", regex=True)


# =========================================================================== image level
def probe_image(fs, folds=5) -> dict:
    X, meta = fs.G, fs.meta
    y = (meta.label != REAL).astype(int).values
    out = dict(n=len(y), n_real=int((y == 0).sum()), n_fake=int(y.sum()))
    if out["n_real"] < folds or out["n_fake"] < folds:
        return out
    oof = np.zeros(len(y))
    knn_pred = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(X, y, _groups(meta)):
        m = _linear(X.shape[1]).fit(X[tr], y[tr])
        oof[te] = m.decision_function(X[te])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        knn_pred[te] = KNeighborsClassifier(10, metric="cosine").fit(Xn[tr], y[tr]).predict(Xn[te])
    out.update(probe_auc=_auc(y, oof), probe_bacc=float(balanced_accuracy_score(y, oof > 0)),
               knn_bacc=float(balanced_accuracy_score(y, knn_pred)))
    Xs = StandardScaler().fit_transform(X)
    out["silhouette_cos"] = float(silhouette_score(Xs, y, metric="cosine", sample_size=min(3000, len(y)),
                                                   random_state=SEED))
    for lab, g in meta.groupby("label"):  # AUC per fake type against all reals
        if lab != REAL:
            sel = (meta.label == REAL) | (meta.label == lab)
            out[f"probe_auc_label{lab}"] = _auc(y[sel], oof[sel])
    out["_oof"] = oof
    return out


def lodo_image(fs) -> pd.DataFrame:
    """Hold out every fake sample of one dataset; reals split into a train half / test half."""
    meta, X = fs.meta, fs.G
    y = (meta.label != REAL).astype(int).values
    real_idx = np.where(y == 0)[0]
    if len(real_idx) < 10:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    grp = _groups(meta).values
    ug = np.unique(grp[real_idx])
    test_groups = set(rng.choice(ug, len(ug) // 2, replace=False))
    real_te = np.array([i for i in real_idx if grp[i] in test_groups])
    real_tr = np.setdiff1d(real_idx, real_te)
    rows = []
    for ds in sorted(meta.dataset[y == 1].unique()):
        fake_te = np.where((y == 1) & (meta.dataset == ds).values)[0]
        fake_tr = np.where((y == 1) & (meta.dataset != ds).values)[0]
        if len(fake_tr) < 10:
            continue
        tr = np.concatenate([real_tr, fake_tr])
        m = _linear(X.shape[1]).fit(X[tr], y[tr])
        te = np.concatenate([real_te, fake_te])
        s = m.decision_function(X[te])
        rows.append(dict(held_out=ds, n_fake=len(fake_te), auc=_auc(y[te], s),
                         tpr_at_real_median=float((s[len(real_te):] > np.median(s[:len(real_te)])).mean())))
    return pd.DataFrame(rows)


def confound_image(fs) -> dict:
    meta, X = fs.meta, fs.G
    out = {}
    fmt = (meta.format == "jpg").astype(int).values
    grp = _groups(meta)
    if 5 <= fmt.sum() <= len(fmt) - 5:
        oof = np.zeros(len(fmt))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, fmt, grp):
            oof[te] = _linear(X.shape[1]).fit(X[tr], fmt[tr]).decision_function(X[te])
        out["format_probe_auc"] = _auc(fmt, oof)
    # dataset identity (how 'dataset-specific' is the embedding)
    ds = meta.dataset.astype("category").cat.codes.values
    if len(np.unique(ds)) > 1:
        pred = np.zeros(len(ds))
        for tr, te in GroupKFold(5).split(X, ds, grp):
            pred[te] = _linear(X.shape[1]).fit(X[tr], ds[tr]).predict(X[te])
        out["dataset_probe_bacc"] = float(balanced_accuracy_score(ds, pred))
        out["dataset_chance"] = 1.0 / len(np.unique(ds))
    # real-vs-fake inside a single format (removes the JPEG/PNG shortcut)
    y = (meta.label != REAL).astype(int).values
    for f in ["jpg", "png"]:
        sel = (meta.format == f).values
        if sel.sum() > 20 and 10 <= y[sel].sum() <= sel.sum() - 10:
            oof = np.zeros(sel.sum())
            Xs, ys, gs = X[sel], y[sel], grp[sel]
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Xs, ys, gs):
                oof[te] = _linear(X.shape[1]).fit(Xs[tr], ys[tr]).decision_function(Xs[te])
            out[f"probe_auc_within_{f}"] = _auc(ys, oof)
    return out


# =========================================================================== patch level
def _patch_labels(M):
    lab = np.full(M.shape, -1, np.int8)
    lab[M >= POS_T] = 1
    lab[M == 0] = 0
    return lab


def _edited(fs):
    """indices of locally edited images that have both authentic and manipulated cells"""
    lab = _patch_labels(fs.M)
    ok = (fs.meta.label == LOCAL_EDIT).values & ((lab == 1).any((1, 2))) & ((lab == 0).any((1, 2)))
    return np.where(ok)[0], lab


def _train_patches(fs, idx, lab, per_class=24, rng=None, extra_real=None):
    rng = rng or np.random.default_rng(SEED)
    Xs, ys = [], []
    D = fs.P.shape[-1]
    for i in idx:
        P = fs.P[i].reshape(-1, D)
        l = lab[i].reshape(-1)
        for c in (0, 1):
            w = np.where(l == c)[0]
            if len(w):
                w = rng.choice(w, min(per_class, len(w)), replace=False)
                Xs.append(P[w]); ys.append(np.full(len(w), c))
    for i in (extra_real if extra_real is not None else []):
        w = rng.choice(fs.P.shape[1] * fs.P.shape[2], per_class, replace=False)
        Xs.append(fs.P[i].reshape(-1, D)[w]); ys.append(np.zeros(per_class, int))
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def _score_images(model, fs, idx):
    D = fs.P.shape[-1]
    g = fs.grid
    out = np.zeros((len(idx), g, g), np.float32)
    for k in range(0, len(idx), 64):
        chunk = idx[k:k + 64]
        X = fs.P[chunk].reshape(-1, D).astype(np.float32)
        out[k:k + len(chunk)] = model.decision_function(X).reshape(len(chunk), g, g)
    return out


def _loc_metrics(S, lab, prefix=""):
    """S: [n,g,g] scores (higher = manipulated), lab: [n,g,g] in {-1,0,1}"""
    v = lab >= 0
    per_auc, per_f1, per_iou = [], [], []
    for s, l, vv in zip(S, lab, v):
        yy, ss = l[vv], s[vv]
        per_auc.append(_auc(yy, ss))
        pred = ss > 0
        tp, fp, fn = (pred & (yy == 1)).sum(), (pred & (yy == 0)).sum(), (~pred & (yy == 1)).sum()
        per_f1.append(2 * tp / max(2 * tp + fp + fn, 1))
        per_iou.append(tp / max(tp + fp + fn, 1))
    yv, sv = lab[v], S[v]
    pos, neg = sv[yv == 1], sv[yv == 0]
    dprime = (pos.mean() - neg.mean()) / np.sqrt(0.5 * (pos.var() + neg.var()) + 1e-12)
    return {f"{prefix}patch_auc": _auc(yv, sv), f"{prefix}img_mean_auc": float(np.nanmean(per_auc)),
            f"{prefix}img_mean_f1": float(np.mean(per_f1)), f"{prefix}img_mean_iou": float(np.mean(per_iou)),
            f"{prefix}dprime": float(dprime)}, np.array(per_auc)


def patch_probe(fs, folds=5) -> tuple[dict, np.ndarray, np.ndarray]:
    """Supervised linear patch probe. Reals (if present) join training as all-authentic images and
    are scored for image-level detection-from-localization (max patch score)."""
    idx, lab = _edited(fs)
    reals = np.where((fs.meta.label == REAL).values)[0]
    if len(idx) < folds:
        return {}, idx, None
    grp = _groups(fs.meta).values
    S = np.zeros((len(fs.meta), fs.grid, fs.grid), np.float32)
    all_idx = np.concatenate([idx, reals])
    for tr, te in GroupKFold(folds).split(all_idx, groups=grp[all_idx]):
        tri, tei = all_idx[tr], all_idx[te]
        tr_edit = tri[np.isin(tri, idx)]
        tr_real = tri[np.isin(tri, reals)]
        X, y = _train_patches(fs, tr_edit, lab, extra_real=tr_real)
        m = _linear(X.shape[1]).fit(X, y)
        S[tei] = _score_images(m, fs, tei)
    res, per_auc = _loc_metrics(S[idx], lab[idx])
    res["n_edited_imgs"] = len(idx)
    if len(reals) >= 5:
        smax = np.concatenate([S[reals].reshape(len(reals), -1).max(1), S[idx].reshape(len(idx), -1).max(1)])
        res["det_auc_from_maxpatch"] = _auc(np.r_[np.zeros(len(reals)), np.ones(len(idx))], smax)
        res["real_fp_patch_rate"] = float((S[reals] > 0).mean())
    # per-dataset breakdown
    for ds in fs.meta.dataset.iloc[idx].unique():
        sel = (fs.meta.dataset.iloc[idx] == ds).values
        r, _ = _loc_metrics(S[idx[sel]], lab[idx[sel]])
        res[f"ds:{ds}:img_mean_auc"] = r["img_mean_auc"]
        res[f"ds:{ds}:img_mean_f1"] = r["img_mean_f1"]
    return res, idx, S


def patch_transfer(fs, min_imgs=30) -> pd.DataFrame:
    idx, lab = _edited(fs)
    ds_of = fs.meta.dataset.values
    names = [d for d in sorted(np.unique(ds_of[idx])) if (ds_of[idx] == d).sum() >= min_imgs]
    rng = np.random.default_rng(SEED)
    grp = _groups(fs.meta).values
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        ia = idx[ds_of[idx] == a]
        ga = np.unique(grp[ia])
        tr_groups = set(rng.choice(ga, len(ga) // 2, replace=False))  # same-dataset cell uses a held-out half
        tr = np.array([i for i in ia if grp[i] in tr_groups])
        X, y = _train_patches(fs, tr, lab)
        m = _linear(X.shape[1]).fit(X, y)
        for b in names:
            ib = idx[ds_of[idx] == b]
            if a == b:
                ib = np.array([i for i in ib if grp[i] not in tr_groups])
            r, _ = _loc_metrics(_score_images(m, fs, ib), lab[ib])
            mat.loc[a, b] = r["img_mean_auc"]
    return mat


def intra_image_zero_shot(fs, k=10) -> tuple[dict, dict]:
    """Training-free: a manipulated region should be *inconsistent* with the rest of its image.
    proto : 1 - cos(patch, mean of image patches)
    knn   : mean cos-distance to the k nearest other patches of the same image (excluding the 3x3
            spatial neighbourhood, which would trivially match)."""
    idx, lab = _edited(fs)
    g, D = fs.grid, fs.P.shape[-1]
    yy, xx = np.divmod(np.arange(g * g), g)
    near = (np.abs(yy[:, None] - yy[None]) <= 1) & (np.abs(xx[:, None] - xx[None]) <= 1)
    S_proto = np.zeros((len(idx), g, g), np.float32)
    S_knn = np.zeros_like(S_proto)
    for n, i in enumerate(idx):
        Z = fs.P[i].reshape(-1, D).astype(np.float32)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)  # per-channel standardisation within the image
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        mu = Zn.mean(0)
        S_proto[n] = (1 - Zn @ mu / (np.linalg.norm(mu) + 1e-8)).reshape(g, g)
        sim = Zn @ Zn.T
        sim[near] = -np.inf
        S_knn[n] = (1 - np.sort(sim, axis=1)[:, -k:].mean(1)).reshape(g, g)
    out, maps = {}, {}
    for name, S in [("proto", S_proto), ("knn", S_knn)]:
        # AUC is threshold-free; F1/IoU use a per-image z-score > 1 rule
        Sz = (S - S.mean((1, 2), keepdims=True)) / (S.std((1, 2), keepdims=True) + 1e-8) - 1
        r, per = _loc_metrics(Sz, lab[idx], prefix=f"zs_{name}_")
        out.update(r)
        maps[name] = S
    return out, maps


def one_class_real(fs, n_comp=64, per_img=64) -> tuple[dict, np.ndarray | None]:
    """Fit N(mu, Sigma) on patches of half of the REAL images; score edited images + other half."""
    reals = np.where((fs.meta.label == REAL).values)[0]
    idx, lab = _edited(fs)
    if len(reals) < 20 or len(idx) < 5:
        return {}, None
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(reals)
    fit_r, test_r = perm[: len(perm) // 2], perm[len(perm) // 2:]
    D = fs.P.shape[-1]
    X = np.concatenate([fs.P[i].reshape(-1, D)[rng.choice(fs.grid ** 2, per_img, replace=False)]
                        for i in fit_r]).astype(np.float32)
    sc = StandardScaler().fit(X)
    pca = PCA(min(n_comp, D), random_state=SEED).fit(sc.transform(X)) if D > n_comp else None
    tf = (lambda Z: pca.transform(sc.transform(Z))) if pca else sc.transform
    lw = LedoitWolf().fit(tf(X))

    def score(ids):
        Z = fs.P[ids].reshape(-1, D).astype(np.float32)
        return np.sqrt(lw.mahalanobis(tf(Z))).reshape(len(ids), fs.grid, fs.grid)

    S_edit = score(idx)
    # within-image AUC uses a per-image z-score so the rule is comparable across images
    Sz = (S_edit - S_edit.mean((1, 2), keepdims=True)) / (S_edit.std((1, 2), keepdims=True) + 1e-8) - 1
    res, _ = _loc_metrics(Sz, lab[idx], prefix="oc_")
    S_real = score(test_r)
    res["oc_det_auc_mean"] = _auc(np.r_[np.zeros(len(test_r)), np.ones(len(idx))],
                                  np.r_[S_real.mean((1, 2)), S_edit.mean((1, 2))])
    res["oc_det_auc_p95"] = _auc(np.r_[np.zeros(len(test_r)), np.ones(len(idx))],
                                 np.r_[np.percentile(S_real.reshape(len(test_r), -1), 95, 1),
                                       np.percentile(S_edit.reshape(len(idx), -1), 95, 1)])
    # raw (un-normalised) patch AUC: are manipulated patches globally 'less real' than authentic ones?
    v = lab[idx] >= 0
    res["oc_patch_auc_raw"] = _auc(lab[idx][v], S_edit[v])
    return res, S_edit
