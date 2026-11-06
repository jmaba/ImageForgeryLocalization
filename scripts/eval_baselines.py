"""Published localizers (TruFor, Mesorch; ifl/features/baselines.py) under our protocol: the same images, masks,
interior AUC, matched transplant null, pixel control, image-level TPR and perturbed copies as
train_comparator.py / train_predictor.py. Cell maps must be extracted first:

    python scripts/extract_features.py --backbones trufor mesorch --workers 0
    python scripts/extract_features.py --backbones trufor mesorch --harmonize jpeg75 --workers 0
    python scripts/eval_baselines.py --specs trufor mesorch --test-suffixes "" __jpeg75 __webp80 __resize50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_comparator import test  # noqa: E402

COLS = {"trufor": ["mean", "max", "conf"], "mesorch": ["mean", "max"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", default=["trufor", "mesorch"])
    ap.add_argument("--max-per-dataset", type=int, default=150)
    ap.add_argument("--test-suffixes", nargs="+", default=[""])
    args = ap.parse_args()
    for spec in args.specs:
        a = argparse.Namespace(feats=[spec], cols=[], shards=["pretrained"], exclude_types=[],
                               gcols=np.arange(len(COLS[spec])), max_per_dataset=args.max_per_dataset,
                               title=f"Published baseline: {spec} (supervised, official weights)")
        names = [f"{spec}_{c}" for c in COLS[spec] if c != "conf"]

        def score(x, C, names=names):
            return {n: x[:, k] for k, n in enumerate(names)}
        for suffix in args.test_suffixes:
            test(score, lambda r: np.asarray(r, np.float32), a, f"bl_{spec}{suffix}", suffix,
                 pd.DataFrame({"model": [spec]}))


if __name__ == "__main__":
    main()
