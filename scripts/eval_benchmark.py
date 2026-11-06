"""Pre-registered analysis of the controlled diffusion-edit benchmark (docs/08_benchmark_prereg.md). Written before the
benchmark existed; the tests, directions and thresholds below are the ones registered there.

Steps
  merge     append data/samples/bench26/manifest.csv to data/samples/manifest.csv (paths relative to data/)
  harness   Mesorch, TruFor and the comparator on the benchmark through the shared test harness (needs the features:
            extract_features.py --backbones pixels dinov2:l6 srm mesorch trufor --datasets <bench datasets>);
            writes outputs/ensemble/dump_bench.pkl with every expert's per-cell scores
  analyse   H1 / H1b / H1c / H2 / H3 + baselines from the dump and the autoencoder maps in
            data/features/gencons_bench/<dataset>.npz (scripts/remote/gen_consistency.py --ae-only); writes
            outputs/benchmark/results.md

    python scripts/eval_benchmark.py merge harness analyse
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.data.dataset import area_resize  # noqa: E402

BENCH_NAME = os.environ.get("IFL_BENCH", "bench26")     # another folder only for dry runs (e.g. bench_smoke)
BENCH = DATA / "samples" / BENCH_NAME
EDITORS = ("sdxlinp", "fluxfill", "kontext")
VARIANTS = [f"bench_{e}_{v}" for e in EDITORS for v in ("fr", "sp")]
MATCH = {"sdxlinp": "sdxl", "fluxfill": "flux", "kontext": "flux"}     # the editor's own autoencoder
VAES = ("sd2", "sdxl", "sd35", "flux")
GCB = DATA / "features" / ("gencons_bench" if BENCH_NAME == "bench26" else f"gencons_{BENCH_NAME}")
B = 10_000
rng = np.random.default_rng(0)


def merge():
    man = DATA / "samples" / "manifest.csv"
    m = pd.read_csv(man, dtype={"sample_id": str, "pair_id": str})
    b = pd.read_csv(BENCH / "manifest.csv", dtype={"sample_id": str, "pair_id": str})
    b["pair_id"] = "bench" + b.sample_id                  # never numeric, so pair matching keeps working
    m = pd.concat([m[~m.dataset.str.startswith("bench_")], b], ignore_index=True)
    m.to_csv(man, index=False)
    print(f"manifest: {len(b)} benchmark rows merged ({b.dataset.value_counts().to_dict()})")


def harness():
    import argparse
    from eval_physics import zs
    from train_comparator import cell_score, load_models, test
    models, cproj, cg = load_models("outputs/comparator/model_base.pt")
    nc = len(cg)
    a = argparse.Namespace(feats=["dinov2_l6", "srm", "mesorch", "trufor"], cols=[], shards=["benchmark"],
                           exclude_types=[], gcols=np.arange(nc + 5), max_per_dataset=10_000,
                           title="Controlled diffusion-edit benchmark", dump=True,
                           only_datasets=VARIANTS + ["bench_orig"])

    def score(x, C):
        xc = cproj(x[:, cg])
        sc = {**cell_score(models["pair"], "pair", xc), **cell_score(models["point"], "point", xc)}
        return {"x_mesorch": x[:, nc], "x_trufor": x[:, nc + 2], "x_trufor_conf": x[:, nc + 4],
                "x_cmp_pair_ref": sc["pair_ref"], "x_cmp_point": sc["point"],
                "cmp_z2": zs(sc["pair_ref"]) + zs(sc["point"])}

    try:
        test(score, lambda r: np.asarray(r, np.float32), a, "bench", "", pd.DataFrame({"comparator": ["model_base"]}))
    except KeyError as e:          # the image-level table is empty for tiny sets; the dump is written before it
        if not (OUTPUTS / "ensemble" / "dump_bench.pkl").exists():
            raise
        print(f"harness: per-cell dump written; image-level table skipped ({e})")


# ---------------------------------------------------------------- analysis helpers
def auc_l(l, s):
    v = l >= 0
    return R._auc(l[v], s[v]) if (l[v] == 1).any() and (l[v] == 0).any() else np.nan


def boot_mean(x):
    """mean and one-sided 95 % lower / upper bounds of the mean over source photos"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    bs = rng.choice(x, (B, len(x))).mean(1)
    return x.mean(), np.percentile(bs, 5), np.percentile(bs, 95), len(x)


def load_ae():
    """dataset -> sample id -> [576, F] raw autoencoder errors, and the feature names"""
    out, names = {}, None
    for f in GCB.glob("*.npz"):
        z = np.load(f, allow_pickle=True)
        names = list(z["names"])
        # COCO ids are zero-padded to 12 digits (maps written before the extractor kept them as text lost the zeros)
        out[f.stem] = {str(k).zfill(12) if str(k).isdigit() else str(k): p.astype(np.float32).reshape(-1, len(names))
                       for k, p in zip(z["ids"], z["patch"])}
    return out, names


def diffmask_labels(sid):
    m = np.asarray(Image.open(BENCH / "kontext_diffmask" / f"{sid}.png").convert("L"), np.float32) / 255
    return R._patch_labels(area_resize(m, 24)).reshape(-1)


def analyse():
    jobs = pickle.load(open(OUTPUTS / "ensemble" / "dump_bench.pkl", "rb"))
    ae, names = load_ae()
    col = {n: i for i, n in enumerate(names)}
    for j in jobs:
        j["sid"] = j["sample"].split("/", 1)[1]
        P = ae.get(j["dataset"], {}).get(j["sid"])
        if P is not None:
            for m in VAES:
                j["scores"][f"ae_lp_{m}"] = P[:, col[f"ae_lp_{m}"]]          # higher error = edited (H1 direction)
                j["scores"][f"ae_l2_{m}"] = P[:, col[f"ae_l2_{m}"]]
    by = {}
    for j in jobs:
        by.setdefault(j["dataset"], {})[j["sid"]] = j
    null = by.get("bench_orig", {})
    md = ["# Controlled diffusion-edit benchmark - pre-registered analysis (docs/08)\n",
          f"jobs: { {d: len(v) for d, v in sorted(by.items())} }\n"]

    # H1 / H1b / H1c: matched autoencoder LPIPS map, higher error = edited, paired with the original + same mask
    def paired(ds, key, labels=None):
        rows = []
        for sid, j in by.get(ds, {}).items():
            if key not in j["scores"] or sid not in null or key not in null[sid]["scores"]:
                continue
            l = j["l"].astype(int) if labels is None else labels(sid)
            ln = null[sid]["l"].astype(int) if labels is None else labels(sid)
            rows.append(dict(sid=sid, auc=auc_l(l, j["scores"][key]), null=auc_l(ln, null[sid]["scores"][key])))
        return pd.DataFrame(rows)

    h = []
    pooled = []
    for e in ("sdxlinp", "fluxfill"):
        d = paired(f"bench_{e}_fr", f"ae_lp_{MATCH[e]}")
        pooled.append(d.assign(e=e))
        mu, lo, hi, n = boot_mean(d.auc - d.null)
        h.append(dict(test="H1", set=f"{e} fr", map=f"ae_lp_{MATCH[e]}", n=n, auc=d.auc.mean(), null=d.null.mean(),
                      diff=mu, diff_lo95=lo, verdict=bool(lo > 0 and d.auc.mean() > 0.55)))
    pp = pd.concat(pooled)
    pp = pp.groupby("sid")[["auc", "null"]].mean()
    mu, lo, hi, n = boot_mean(pp.auc - pp.null)
    h.insert(0, dict(test="H1 (primary, pooled)", set="sdxlinp + fluxfill fr", map="matched ae_lp", n=n,
                     auc=pp.auc.mean(), null=pp.null.mean(), diff=mu, diff_lo95=lo,
                     verdict=bool(lo > 0 and pp.auc.mean() > 0.55)))
    for lab, fn in (("object mask", None), ("difference mask", diffmask_labels)):
        d = paired("bench_kontext_fr", "ae_lp_flux", fn)
        mu, lo, hi, n = boot_mean(d.auc - d.null)
        h.append(dict(test="H1b", set=f"kontext fr ({lab})", map="ae_lp_flux", n=n, auc=d.auc.mean(),
                      null=d.null.mean(), diff=mu, diff_lo95=lo, verdict=bool(lo > 0 and d.auc.mean() > 0.55)))
    for e in EDITORS:                                  # H1c: pasted decoder output reconstructs better (AUC < 0.5)
        d = paired(f"bench_{e}_sp", f"ae_lp_{MATCH[e]}")
        mu, lo, hi, n = boot_mean(d.auc)
        h.append(dict(test="H1c", set=f"{e} sp", map=f"ae_lp_{MATCH[e]}", n=n, auc=mu, null=d.null.mean(),
                      diff=mu - d.null.mean(), diff_lo95=np.nan, auc_hi95=hi, verdict=bool(hi < 0.5)))
    md.append("## H1 / H1b / H1c - matched autoencoder LPIPS map, higher error = edited\n\n"
              + pd.DataFrame(h).round(3).to_markdown(index=False) + "\n")

    # H2: image level, mean matched-autoencoder error (lower = decoded by that model); LPIPS primary, L2 shown too
    h2 = []
    for e in EDITORS:
        for v in ("fr", "sp"):
            for kind in ("ae_lp", "ae_l2"):
                k = f"{kind}_{MATCH[e]}"
                o = [P[:, col[k]].mean() for P in ae.get("bench_orig", {}).values()]
                x = [P[:, col[k]].mean() for P in ae.get(f"bench_{e}_{v}", {}).values()]
                if not o or not x:
                    continue
                a = R._auc(np.r_[np.zeros(len(o)), np.ones(len(x))], -np.r_[o, x])
                ok = a > 0.9 if v == "fr" else a < 0.75
                h2.append(dict(test="H2", originals_vs=f"{e} {v}", feature=k, auroc=a,
                               verdict=bool(ok) if kind == "ae_lp" else "(secondary)"))
    md.append("## H2 - image level: originals vs edited, image-mean matched autoencoder error\n\n"
              + pd.DataFrame(h2).round(3).to_markdown(index=False) + "\n")

    # baselines and every autoencoder map: AUC, null, IoU at 0.5 and false-positive area on originals
    rows = []
    keys = ["x_mesorch", "x_trufor", "cmp_z2"] + [f"ae_lp_{m}" for m in VAES]
    for ds in VARIANTS:
        for k in keys:
            d = paired(ds, k)
            if not len(d):
                continue
            r = dict(set=ds.replace("bench_", ""), scorer=k, auc=d.auc.mean(), null=d.null.mean(),
                     auc_minus_null=(d.auc - d.null).mean())
            if k in ("x_mesorch", "x_trufor"):
                iou = []
                for sid, j in by[ds].items():
                    l = j["l"].astype(int)
                    v = l >= 0
                    p, t = j["scores"][k][v] > 0.5, l[v] == 1
                    iou.append((p & t).sum() / max(1, (p | t).sum()))
                fp = [np.mean(null[s]["scores"][k] > 0.5) for s in by[ds] if s in null]
                r |= dict(iou_at_0p5=np.mean(iou), fp_area_on_originals=np.mean(fp))
            rows.append(r)
    md.append("## Baselines and autoencoder maps per variant (AUC paired with the original + same mask)\n\n"
              + pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n")

    # H3 (exploratory): stacker trained on the pilot domains, applied to the benchmark as a new domain
    try:
        md.append(h3(jobs))
    except FileNotFoundError as e:
        md.append(f"## H3\n\nskipped: {e}\n")
    out = OUTPUTS / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text("\n".join(md))
    print("\n".join(md))


def h3(bench_jobs):
    import ensemble as E
    from eval_gencons import DOMAINS, load_gc
    pilot = [j for j in pickle.load(open(OUTPUTS / "ensemble" / "dump_phys.pkl", "rb")) if j["domain"] in DOMAINS]
    gc = {f.stem: load_gc(f.stem)[0] for f in (DATA / "features" / "gencons").glob("*.npz") if "__" not in f.stem}
    names = load_gc("tgif2_orig")[1]
    col = {n: i for i, n in enumerate(names)}
    train = []
    for j in pilot:
        P = gc.get(j["dataset"], {}).get(j["sample"].split("/", 1)[1])
        if P is None:
            continue
        train.append(dict(j, scores={**j["scores"], **{f"ae_lp_{m}": P[:, col[f"ae_lp_{m}"]] for m in VAES}}))
    test = [j for j in bench_jobs if all(f"ae_lp_{m}" in j["scores"] for m in VAES)]
    base = {"mesorch": "x_mesorch", "trufor": "x_trufor", "comparator": "cmp_z2"}
    aes = {f"ae_{m}": f"ae_lp_{m}" for m in VAES}
    E.EXPERTS.update(base | aes)
    res = {}
    for tag, nm in (("stack: mesorch+trufor+comparator", list(base)),
                    ("stack: + autoencoder maps", list(base) + list(aes))):
        maps = E.stack_maps(train, test, nm)
        m = E.metrics(test, maps)
        e = m[m.kind == "edited"]
        res[tag] = {"null": m[m.kind == "authentic"].auc.mean(),
                    **{ds.replace("bench_", ""): e[e.dataset == ds].auc.mean() for ds in VARIANTS},
                    "fr mean": e[e.dataset.str.endswith("_fr")].auc.mean(),
                    "sp mean": e[e.dataset.str.endswith("_sp")].auc.mean()}
    return ("## H3 (exploratory) - stacker trained on the pilot domains, applied to the benchmark\n\n"
            + pd.DataFrame(res).round(3).to_markdown() + "\n")


if __name__ == "__main__":
    for step in sys.argv[1:] or ["analyse"]:
        {"merge": merge, "harness": harness, "analyse": analyse}[step]()
