"""Decoder-trace detector (ifl/features/dtrace.py, docs/09) against Mesorch, TruFor and the real-only comparator, on
every dataset, through the shared harness (matched nulls, colour control), plus image-level detection.

  localization  test() over all edited datasets (150 per dataset, same subsets as every earlier table) with
                dtraceA (trained on 5 autoencoders, 5 held out) and dtraceB (all 10), cell means and maxima
  detection     image score = mean of the top 5 % of the trace map (dtrace), the TruFor / Mesorch map maximum;
                AUROC of each dataset against the authentic images of its own source, and against all authentic
                images pooled

    python scripts/eval_dtrace.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifl import DATA, OUTPUTS  # noqa: E402
from train_comparator import cell_score, load_models, test  # noqa: E402

FEAT = DATA / "features"


def zs(s):
    s = np.nan_to_num(np.asarray(s, np.float64))
    return (s - s.mean()) / (s.std() + 1e-8)


def localization(max_per_dataset, suffix="", with_flame=False, with_bfree=False, dump=True):
    models, cproj, cg = load_models("outputs/comparator/model_base.pt")
    nc = len(cg)  # comparator streams, then mesorch (2), trufor (3), dtraceA (3), dtraceB (3) [, flame (3)]
    feats = (["dinov2_l6", "srm", "mesorch", "trufor", "dtraceA", "dtraceB"] + (["flame"] if with_flame else [])
             + (["bfree"] if with_bfree else []))
    a = argparse.Namespace(feats=feats, cols=[],
                           shards=["autoencoder round trips of real photos (dtrace) / real-only (comparator)"],
                           exclude_types=[], gcols=np.arange(nc + 11 + 3 * with_flame + 3 * with_bfree),
                           max_per_dataset=max_per_dataset,
                           title="Decoder-trace detector vs published localizers", dump=dump)

    def score(x, C):
        xc = cproj(x[:, cg])
        sc = {**cell_score(models["pair"], "pair", xc), **cell_score(models["point"], "point", xc)}
        cmp = zs(sc["pair_ref"]) + zs(sc["point"])
        mes, tru, dA, dB = x[:, nc], x[:, nc + 2], x[:, nc + 5], x[:, nc + 8]
        out = {"x_mesorch": mes, "x_trufor": tru, "x_trufor_conf": x[:, nc + 4], "cmp_z2": cmp,
               "dtA": dA, "dtA_max": x[:, nc + 6], "dtB": dB, "dtB_max": x[:, nc + 9],
               "sota+cmp": zs(mes) + zs(tru) + cmp, "sota+cmp+dtB": zs(mes) + zs(tru) + cmp + zs(dB),
               "dtB+cmp": zs(dB) + cmp}
        if with_flame:
            fl = x[:, nc + 11]
            out |= {"x_flame": fl, "x_flame_max": x[:, nc + 12], "dtB+flame": zs(dB) + zs(fl),
                    "sota+cmp+flame": zs(mes) + zs(tru) + cmp + zs(fl),
                    "sota+cmp+dtB+flame": zs(mes) + zs(tru) + cmp + zs(dB) + zs(fl)}
        if with_bfree:
            bf = x[:, nc + 11 + 3 * with_flame]
            out |= {"x_bfree": bf, "x_bfree_max": x[:, nc + 12 + 3 * with_flame], "dtB+bfree": zs(dB) + zs(bf),
                    "sota+cmp+dtB+bfree": zs(mes) + zs(tru) + cmp + zs(dB) + zs(bf)}
            if with_flame:
                out["sota+cmp+dtB+flame+bfree"] = out["sota+cmp+dtB+flame"] + zs(bf)
        return out

    tag = ("dtrace" if not (with_flame or with_bfree) else "dt" + "flame" * with_flame + "bfree" * with_bfree) + suffix
    try:
        test(score, lambda r: np.asarray(r, np.float32), a, tag, suffix,
             pd.DataFrame({"model": ["dtraceA / dtraceB / comparator model_base" + (" / FLAME" * with_flame) + (" / B-Free" * with_bfree)]}))
    except KeyError as e:                                    # image-level table of test() on tiny sets
        print(f"localization: image-level table skipped ({e})")


def image_level(suffix="", with_flame=False, with_bfree=False):
    man = pd.read_csv(DATA / "samples" / "manifest.csv", dtype={"sample_id": str})
    rows = []
    for ds, g in man.groupby("dataset"):
        stores = {}
        specs = (("dtraceA", "dtraceB", "trufor", "mesorch") + (("flame",) if with_flame else ())
                 + (("bfreeimg",) if with_bfree else ()))
        for spec in specs:
            d = FEAT / (spec + suffix) / ds
            if (d / "ids.txt").exists():
                ids = (d / "ids.txt").read_text().split()
                P = np.load(d / "patch.npy").astype(np.float32)
                stores[spec] = dict(zip(ids, P))
        if len(stores) < len(specs):
            continue
        lab = dict(zip(g.sample_id.astype(str), g.label))
        fam = "tgif2" if ds.startswith("tgif2") else ds.split("_")[0] if ds.startswith(("bench", "synth")) else ds
        for sid in stores["dtraceB"]:
            if sid not in lab or any(sid not in stores[s] for s in stores):
                continue
            rows.append(dict(dataset=ds, family=fam, sample_id=sid, label=int(lab[sid]),
                             dtA=stores["dtraceA"][sid][0, 0, 2], dtB=stores["dtraceB"][sid][0, 0, 2],
                             trufor=stores["trufor"][sid][..., 1].max(), mesorch=stores["mesorch"][sid][..., 1].max(),
                             **({"flame": stores["flame"][sid][0, 0, 2], "flame_map": stores["flame"][sid][..., 1].max()}
                                if with_flame else {}),
                             **({"bfree": stores["bfreeimg"][sid][0, 0, 2]} if with_bfree else {})))
    df = pd.DataFrame(rows)
    auth = df[df.label == 0]
    fam_auth = {"bench": "bench_orig", "benchx": "benchx_orig", "tgif2": "tgif2_orig", "synth": "benchx_orig",
                "gpt_image_2": "coco_orig_gpt_image_2", "nanobanana2": "coco_orig_nanobanana2"}
    out = []
    for ds, g in df[df.label != 0].groupby("dataset"):
        fam = g.family.iloc[0]
        own = auth[auth.dataset == fam_auth[fam]] if fam in fam_auth else auth[auth.dataset == ds]
        r = dict(dataset=ds, n=len(g), kind={1: "fully synthetic", 2: "edited"}.get(int(g.label.iloc[0]), "?"),
                 own_authentic=(fam_auth.get(fam, ds) if len(own) else "-"))
        for s in (("dtA", "dtB", "trufor", "mesorch") + (("flame", "flame_map") if with_flame else ())
                  + (("bfree",) if with_bfree else ())):
            y = np.r_[np.zeros(len(auth)), np.ones(len(g))]
            r[f"{s} vs all authentic"] = roc_auc_score(y, np.r_[auth[s], g[s]])
            if len(own):
                yo = np.r_[np.zeros(len(own)), np.ones(len(g))]
                r[f"{s} vs own"] = roc_auc_score(yo, np.r_[own[s], g[s]])
        out.append(r)
    t = pd.DataFrame(out).round(3)
    fp = auth.groupby("dataset")[["dtA", "dtB"]].mean().round(3)
    o = OUTPUTS / "dtrace"
    o.mkdir(parents=True, exist_ok=True)
    md = (f"# Decoder-trace detector - image-level detection (condition '{suffix or 'clean'}')\n\n"
          "Image score: mean of the top 5 % of the trace probability map (dtrace), maximum of the TruFor / Mesorch "
          f"map. {len(auth)} authentic images in the pool.\n\n" + t.to_markdown(index=False)
          + "\n\n## Mean image score on authentic images (lower is better)\n\n" + fp.to_markdown() + "\n")
    name = "image_level" + ("_flame" if with_flame else "") + ("_bfree" if with_bfree else "") + suffix
    (o / f"{name}.md").write_text(md)
    t.to_csv(o / f"{name}.csv", index=False)
    print(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("steps", nargs="*", default=["localization", "image"])
    ap.add_argument("--max-per-dataset", type=int, default=150)
    ap.add_argument("--suffix", default="", help="perturbation condition, e.g. __jpeg75")
    ap.add_argument("--with-flame", action="store_true", help="add FLAME (arXiv:2606.02178) as a scorer")
    ap.add_argument("--with-bfree", action="store_true", help="add B-Free (CVPR 2025) as a scorer")
    ap.add_argument("--no-dump", action="store_true", help="skip the per-cell score dump (it grows to ~15 GB)")
    args = ap.parse_args()
    if "localization" in args.steps:
        localization(args.max_per_dataset, args.suffix, args.with_flame, args.with_bfree, not args.no_dump)
    if "image" in args.steps:
        image_level(args.suffix, args.with_flame, args.with_bfree)
