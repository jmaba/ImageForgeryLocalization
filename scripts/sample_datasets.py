"""Download a small random subset of each configured dataset and build data/samples/manifest.csv.

    python scripts/sample_datasets.py                     # all sources
    python scripts/sample_datasets.py --only sid_set opensdi
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA, ROOT  # noqa: E402
from ifl.data.sources import SAMPLERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = ROOT / cfg["out_dir"]
    manifest = out / "manifest.csv"
    old = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame()

    for name, scfg in cfg["sources"].items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}] sampling ({scfg['kind']})", flush=True)
        rng = random.Random(f"{cfg['seed']}-{name}")
        try:
            rows = SAMPLERS[scfg["kind"]](name, scfg, out, rng)
        except Exception as e:  # keep going - one broken source must not kill the run
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        new = pd.DataFrame(rows)
        if len(old):  # a source may write several datasets (e.g. tgif2 -> tgif2_sd2_fr, tgif2_orig ...)
            old = old[(old.dataset != name) & ~old.dataset.str.startswith(name + "_")]
        old = pd.concat([old, new], ignore_index=True)
        old.to_csv(manifest, index=False)  # checkpoint after every source
        print(f"[{name}] {len(new)} samples  labels={new.label.value_counts().to_dict()}", flush=True)

    print(f"\nmanifest: {manifest.relative_to(ROOT)}  ({len(old)} rows)")
    if len(old):
        print(old.groupby(["dataset", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
