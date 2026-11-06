"""Are the published supervised localizers and our real-only comparator complementary? Per-image z-fusion of
their cell maps, under our protocol (docs/06). Needs the extracted cell maps of the baselines (eval_baselines.py)
and a saved comparator (train_comparator.py --save).

Scorers: the parts alone (mesorch, trufor, cmp_z2 = z(pair_ref) + z(point)) and the sums of within-image z-scores
    mes+cmp, tru+cmp, mes+tru, mes+tru+cmp   (equal weights: nothing is tuned on test data)

    python scripts/eval_fusion.py --test-suffixes "" __jpeg75 __webp80 __resize50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_comparator import cell_score, load_models, test  # noqa: E402


def zs(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparator", default="outputs/comparator/model_base.pt")
    ap.add_argument("--max-per-dataset", type=int, default=150)
    ap.add_argument("--test-suffixes", nargs="+", default=[""])
    ap.add_argument("--tag", default="blfuse")
    args = ap.parse_args()
    models, cproj, cg = load_models(args.comparator)
    n_cmp = len(cg)                                   # comparator streams first, then mesorch (2), trufor (3)
    a = argparse.Namespace(feats=["dinov2_l6", "srm", "mesorch", "trufor"], cols=[], shards=["pretrained+real-only"],
                           exclude_types=[], gcols=np.arange(n_cmp + 5), max_per_dataset=args.max_per_dataset,
                           title="Fusion: published localizers + real-only comparator")

    def score(x, C):
        sc = {**cell_score(models["pair"], "pair", cproj(x[:, cg])), **cell_score(models["point"], "point",
                                                                                   cproj(x[:, cg]))}
        mes, tru = x[:, n_cmp], x[:, n_cmp + 2]
        cmp = zs(sc["pair_ref"]) + zs(sc["point"])
        return {"mesorch": mes, "trufor": tru, "cmp_z2": cmp,
                "mes+cmp": zs(mes) + zs(cmp), "tru+cmp": zs(tru) + zs(cmp), "mes+tru": zs(mes) + zs(tru),
                "mes+tru+cmp": zs(mes) + zs(tru) + zs(cmp)}

    for suffix in args.test_suffixes:
        test(score, lambda r: np.asarray(r, np.float32), a, args.tag + suffix, suffix,
             pd.DataFrame({"comparator": [args.comparator]}))


if __name__ == "__main__":
    main()
