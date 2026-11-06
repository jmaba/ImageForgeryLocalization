"""Physical-consistency scores (ifl/features/physics.py) under our protocol, alone and fused with the comparator
and the published localizers. Nothing is trained on edits; fusions are equal-weight sums of within-image z-scores.

Scorers
    phys_light    |shading - global SH lighting fit|          phys_ang     local vs global light direction
    phys_illum    illuminant-colour deviation                 phys_blur    |blur - thin-lens fit on disparity|
    phys_cf       cross-fitted region consistency (region_null.cell_scores loc_cf) on the whole physics vector,
                  with the cell's colour statistics as content control
    phys_any      z(light) + z(illum) + z(blur)
    cmp+phys, sota+cmp, sota+cmp+phys   (sota = Mesorch + TruFor)

    python scripts/eval_physics.py --test-suffixes "" __jpeg75
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from region_null import cell_scores  # noqa: E402
from train_comparator import cell_score, load_models, test  # noqa: E402
from train_predictor import load_predictor  # noqa: E402

N_PHYS = 15


def zs(s):
    s = np.nan_to_num(s, nan=np.nanmedian(s) if np.isfinite(s).any() else 0.0)
    return (s - s.mean()) / (s.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparator", default="outputs/comparator/model_base.pt")
    ap.add_argument("--max-per-dataset", type=int, default=150)
    ap.add_argument("--test-suffixes", nargs="+", default=[""])
    ap.add_argument("--tag", default="phys")
    ap.add_argument("--predictor", default="outputs/comparator/predictor_fuse2.pt",
                    help="context predictor (train_predictor.py --save); skipped when missing")
    args = ap.parse_args()
    models, cproj, cg = load_models(args.comparator)
    pred = load_predictor(args.predictor) if Path(args.predictor).exists() else None
    nc = len(cg)                                 # comparator streams, then mesorch (2), trufor (3), physics (15)
    a = argparse.Namespace(feats=["dinov2_l6", "srm", "mesorch", "trufor", "physics"], cols=[],
                           shards=["none (physics) / real-only (comparator)"], exclude_types=[],
                           gcols=np.arange(nc + 5 + N_PHYS), max_per_dataset=args.max_per_dataset,
                           title="Physical consistency: lighting / illuminant / defocus-depth", dump=True)

    def score(x, C):
        p = x[:, nc + 5:]
        xc = cproj(x[:, cg])
        sc = {**cell_score(models["pair"], "pair", xc), **cell_score(models["point"], "point", xc)}
        cmp = zs(sc["pair_ref"]) + zs(sc["point"])
        sota = zs(x[:, nc]) + zs(x[:, nc + 2])
        light, ang = p[:, 1], p[:, 2]
        illum = np.hypot(p[:, 4], p[:, 5])
        blur = np.where(p[:, 14] > 0, p[:, 13], np.nan)
        pz = (p - p.mean(0)) / (p.std(0) + 1e-6)
        cf = cell_scores(pz, C)["loc_cf"]
        anyp = zs(light) + zs(illum) + zs(blur)
        out = {"phys_light": light, "phys_ang": ang, "phys_illum": illum, "phys_blur": np.nan_to_num(blur),
               "phys_cf": cf, "phys_any": anyp, "cmp_z2": cmp, "cmp+phys": cmp + zs(anyp),
               "sota+cmp": sota + cmp, "sota+cmp+phys": sota + cmp + zs(anyp),
               # raw expert maps and reliabilities, kept for the offline ensemble (scripts/ensemble.py)
               "x_mesorch": x[:, nc], "x_trufor": x[:, nc + 2], "x_trufor_conf": x[:, nc + 4],
               "x_cmp_pair_ref": sc["pair_ref"], "x_cmp_point": sc["point"],
               "x_light_cond": p[:, 3], "x_blur_valid": p[:, 14], "x_light_res": p[:, 0]}
        if pred:
            out |= {f"x_{k}": v for k, v in pred(x[:, cg], C).items()}
        return out

    for suffix in args.test_suffixes:
        test(score, lambda r: np.asarray(r, np.float32), a, args.tag + suffix, suffix,
             pd.DataFrame({"comparator": [args.comparator]}))


if __name__ == "__main__":
    main()
