"""Image-level triage: authentic / fully synthetic / regenerated edit / local splice (docs/08 follow-up).

Classes
  0 authentic        camera photos (benchmark originals, TGIF2 originals, COCO originals, OpenSDI / SID-Set real)
  1 fully synthetic  text-to-image (SDXL, FLUX; OpenSDI / SID-Set full generations)
  2 regenerated      a local edit whose output is the editor's full decode (benchmark "fr", TGIF2 "fr",
                     GPT-Image-2, Nano-Banana-2)
  3 spliced          the edited region pasted back into the photo (benchmark "sp", TGIF2 "sp")
Features per image (no pixels, no training of the maps themselves)
  autoencoder round trip of SD2 / SDXL / SD3.5 / FLUX (scripts/remote/gen_consistency.py): per model the image mean,
  10th / 90th percentile and spread of the LPIPS map over the 24 x 24 cells, and the mean squared error (log scale)
  TruFor and Mesorch maps: mean, max, 90th percentile, spread; TruFor confidence
Splits
  train   benchmark extension (benchx, 200 photos x 3 editors) + 2/3 of the synthetic set (split by caption)
  test    the pre-registered benchmark (bench, 120 photos) + the other 1/3 of the synthetic set
  out of distribution   TGIF2, GPT-Image-2, Nano-Banana-2 (+ their COCO originals), CocoGlide, OpenSDI, SID-Set
Models: gradient boosting (main) and logistic regression, each on autoencoder features, detector features, both.

    python scripts/triage.py
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA, OUTPUTS  # noqa: E402

FEAT = DATA / "features"
VAES = ("sd2", "sdxl", "sd35", "flux")
CLASSES = ["authentic", "fully synthetic", "regenerated", "spliced"]
GC_DIRS = ["gencons", "gencons_bench", "gencons_benchx", "gencons_synth", "gencons_ood"]


def label_of(ds, manifest_label=None):
    if ds.endswith("_orig") or ds.startswith("coco_orig"):
        return 0
    if ds.startswith("synth_"):
        return 1
    if ds.endswith("_fr") or ds in ("gpt_image_2", "nanobanana2"):
        return 2
    if ds.endswith("_sp"):
        return 3
    if ds in ("opensdi", "sid_set"):
        return {0: 0, 1: 1}.get(manifest_label, -2)        # their local edits: regenerated or spliced, unknown
    return -1                                               # CocoGlide: pixel-space GLIDE edit, reported apart


def key(k):
    k = str(k)
    return k.zfill(12) if k.isdigit() and len(k) > 6 else k


def ae_features():
    rows = []
    for d in GC_DIRS:
        for f in sorted((FEAT / d).glob("*.npz")) if (FEAT / d).exists() else []:
            if "__" in f.stem:                              # re-decoded controls of the pilot
                continue
            z = np.load(f, allow_pickle=True)
            names = list(z["names"])
            P = z["patch"].astype(np.float32).reshape(len(z["ids"]), -1, len(names))
            for k, p in zip(z["ids"], P):
                r = {"dataset": f.stem, "sample_id": key(k)}
                for m in VAES:
                    lp, l2 = p[:, names.index(f"ae_lp_{m}")], p[:, names.index(f"ae_l2_{m}")]
                    r |= {f"lp_mean_{m}": lp.mean(), f"lp_p10_{m}": np.percentile(lp, 10),
                          f"lp_p90_{m}": np.percentile(lp, 90), f"lp_sd_{m}": lp.std(),
                          f"l2_log_{m}": np.log10(l2.mean() + 1e-8)}
                means = np.array([r[f"lp_mean_{m}"] for m in VAES])
                r |= {"lp_min": means.min(), "lp_contrast": np.median(means) - means.min()}
                rows.append(r)
    return pd.DataFrame(rows).drop_duplicates(["dataset", "sample_id"])


def detector_features(datasets):
    rows = []
    for ds in datasets:
        t, m = FEAT / "trufor" / ds, FEAT / "mesorch" / ds
        if not (t / "ids.txt").exists() or not (m / "ids.txt").exists():
            continue
        ids = [key(s) for s in (t / "ids.txt").read_text().split()]
        T = np.load(t / "patch.npy").astype(np.float32).reshape(len(ids), -1, 3)
        mids = [key(s) for s in (m / "ids.txt").read_text().split()]
        M = dict(zip(mids, np.load(m / "patch.npy").astype(np.float32).reshape(len(mids), -1, 2)))
        for k, tp in zip(ids, T):
            mp = M.get(k)
            if mp is None:
                continue
            rows.append(dict(dataset=ds, sample_id=k, tru_mean=tp[:, 0].mean(), tru_max=tp[:, 1].max(),
                             tru_p90=np.percentile(tp[:, 0], 90), tru_sd=tp[:, 0].std(), tru_conf=tp[:, 2].mean(),
                             mes_mean=mp[:, 0].mean(), mes_max=mp[:, 1].max(), mes_p90=np.percentile(mp[:, 0], 90),
                             mes_sd=mp[:, 0].std()))
    return pd.DataFrame(rows)


def split_of(ds, sid):
    if ds.startswith("benchx_"):
        return "train"
    if ds.startswith("bench_"):
        return "test"
    if ds.startswith("synth_"):
        return "train" if zlib.crc32(sid.encode()) % 3 else "test"     # same caption -> same split for both
    return "ood"


def source_of(ds):
    for pre, src in (("benchx_", "extension"), ("bench_", "benchmark"), ("synth_", "synthetic"), ("tgif2", "TGIF2"),
                     ("coco_orig_gpt", "GPT-Image-2"), ("gpt_image_2", "GPT-Image-2"),
                     ("coco_orig_nano", "Nano-Banana-2"), ("nanobanana2", "Nano-Banana-2")):
        if ds.startswith(pre):
            return src
    return ds


def loso(df, cols):
    """leave one source out: boosting (missing detector features allowed) trained on every other labelled source,
    the pre-registered benchmark never in training; then all sources -> the benchmark"""
    lab = df[(df.y >= 0) & (df.source != "benchmark")]
    rows = []
    for src in sorted(set(df.source) - {"benchmark"}) + ["benchmark"]:
        tr = lab[lab.source != src]
        te = df[df.source == src]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                             random_state=0).fit(tr[cols].to_numpy(float), tr.y.values)
        p = clf.predict_proba(te[cols].to_numpy(float))
        pred = p.argmax(1)
        y = te.y.values
        r = dict(held_out=src, n=len(te), classes=", ".join(CLASSES[v] for v in sorted(set(y[y >= 0]))))
        for i, c in enumerate(CLASSES):
            if (y == i).any():
                r[f"recall {c}"] = (pred[y == i] == i).mean()
        if (y == -2).any():
            r["edited (fr/sp) called edited"] = np.isin(pred[y == -2], [2, 3]).mean()
        gen = (y != 0) & (y != -1)
        if (y == 0).any() and gen.any():
            r["auroc authentic vs rest"] = roc_auc_score(gen[(y >= 0) | (y == -2)], 1 - p[(y >= 0) | (y == -2), 0])
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    man = pd.read_csv(DATA / "samples" / "manifest.csv", dtype={"sample_id": str})
    mlab = {(r.dataset, key(r.sample_id)): r.label for r in man.itertuples()}
    A = ae_features()
    D = detector_features(sorted(A.dataset.unique()))
    df = A.merge(D, on=["dataset", "sample_id"], how="left")
    df["y"] = [label_of(d, mlab.get((d, s))) for d, s in zip(df.dataset, df.sample_id)]
    df["split"] = [split_of(d, s) for d, s in zip(df.dataset, df.sample_id)]
    df["source"] = [source_of(d) for d in df.dataset]
    ae_cols = [c for c in A.columns if c not in ("dataset", "sample_id")]
    det_cols = [c for c in D.columns if c not in ("dataset", "sample_id")]
    print(df.groupby(["split", "y"]).size().unstack(fill_value=0))

    tr, te, ood = df[df.split == "train"], df[df.split == "test"], df[df.split == "ood"]
    md = ["# Image-level triage: authentic / fully synthetic / regenerated / spliced\n",
          f"train {len(tr)} images, test {len(te)}, out of distribution {len(ood)}\n",
          "Train: benchmark extension (200 photos x 3 editors) + 2/3 of the synthetic set. "
          "Test: the pre-registered benchmark (120 photos x 3 editors, never used for training) + 1/3 synthetic.\n"]
    results, models = [], {}
    for fname, cols in (("autoencoder", ae_cols), ("detectors", det_cols), ("both", ae_cols + det_cols)):
        for mname in ("boosting", "logistic"):
            Xtr = tr[cols].to_numpy(float)
            ok = np.isfinite(Xtr).all(1)
            if mname == "boosting":
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                                     random_state=0)
            else:
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000,
                                                                         class_weight="balanced"))
            clf.fit(Xtr[ok], tr.y.values[ok])
            Xte = te[cols].to_numpy(float)
            okt = np.isfinite(Xte).all(1)
            pred = clf.predict(Xte[okt])
            yt = te.y.values[okt]
            proba = clf.predict_proba(Xte[okt])
            results.append(dict(features=fname, model=mname, n_test=int(okt.sum()),
                                balanced_acc=balanced_accuracy_score(yt, pred),
                                macro_f1=f1_score(yt, pred, average="macro"),
                                **{f"recall {c}": (pred[yt == i] == i).mean() for i, c in enumerate(CLASSES)},
                                auroc_authentic_vs_rest=roc_auc_score(yt != 0, 1 - proba[:, 0])))
            models[(fname, mname)] = (clf, cols)
    res = pd.DataFrame(results)
    md.append("## Held-out test (pre-registered benchmark + held-out synthetic)\n\n" + res.round(3).to_markdown(index=False))

    clf, cols = models[("both", "boosting")]
    okt = np.isfinite(te[cols].to_numpy(float)).all(1)
    pred = clf.predict(te[cols].to_numpy(float)[okt])
    cm = pd.DataFrame(confusion_matrix(te.y.values[okt], pred, labels=range(4)),
                      index=[f"true {c}" for c in CLASSES], columns=[f"pred {c}" for c in CLASSES])
    md.append("\n## Confusion matrix (both feature sets, boosting), held-out test\n\n" + cm.to_markdown())
    pi = permutation_importance(clf, te[cols].to_numpy(float)[okt], te.y.values[okt], n_repeats=10, random_state=0,
                                scoring="balanced_accuracy")
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False).head(12)
    md.append("\n## Most important features (permutation, held-out test)\n\n" + imp.round(3).to_frame("drop in "
              "balanced accuracy").to_markdown())

    # out of distribution: per dataset, how the images are classified (autoencoder-only model where detectors absent)
    rows = []
    for ds, g in ood.groupby("dataset"):
        for fname in ("both", "autoencoder"):
            c, cs = models[(fname, "boosting")]
            X = g[cs].to_numpy(float)
            ok = np.isfinite(X).all(1)
            if ok.sum() == 0:
                continue
            p = c.predict(X[ok])
            y = g.y.values[ok]
            known = y >= 0
            rows.append(dict(dataset=ds, features=fname, n=int(ok.sum()),
                             true_class=", ".join(sorted({CLASSES[v] if v >= 0 else ("edited (fr or sp)" if v == -2
                                                                                     else "other") for v in y})),
                             accuracy=(p[known] == y[known]).mean() if known.any() else np.nan,
                             **{f"-> {c}": (p == i).mean() for i, c in enumerate(CLASSES)}))
            break                                           # the richest feature set available for this dataset
    md.append("\n## Out of distribution (generators, sizes and sources not seen in training)\n\n"
              + pd.DataFrame(rows).round(3).to_markdown(index=False))
    md.append("\n## Leave one source out (boosting on autoencoder + detector features; train on all other sources, "
              "the pre-registered benchmark never in training; last row: all sources -> benchmark)\n\n"
              + loso(df, ae_cols + det_cols).round(3).to_markdown(index=False))
    out = OUTPUTS / "triage"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text("\n".join(md) + "\n")
    df.to_csv(out / "features.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
