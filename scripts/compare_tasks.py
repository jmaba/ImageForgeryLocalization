"""Two-task comparison of cached feature streams.

  Task A (whole image): real vs fully synthetic, image-level, local edits excluded.
  Task B (partial):     localization inside locally edited images, patch level - the main task.
                        Run with raw / image-relative / both patch descriptors.

    python scripts/compare_tasks.py --specs pixels srm dinov2_l6 clip dtcwt
Writes outputs/tasks/{task_a.csv, task_b.csv, task_b_per_dataset.csv, summary.md}
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load, load_multi  # noqa: E402

OUT = OUTPUTS / "tasks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="+", required=True, help="spec or a+b for channel concat")
    ap.add_argument("--modes", nargs="+", default=["raw", "relative", "both"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    A, B, Bds = {}, {}, {}
    for spec in args.specs:
        fs = load_multi(spec.split("+"), args.datasets) if "+" in spec else load(spec, args.datasets)
        A[spec] = R.task_whole_image(fs)
        for mode in args.modes:
            R.FEATURE_MODE = mode
            r, _, _ = R.patch_probe(fs)
            mat = R.patch_transfer(fs)
            off = mat.values[~np.eye(len(mat), dtype=bool)] if len(mat) > 1 else np.array([np.nan])
            key = f"{spec} [{mode}]"
            B[key] = dict(img_mean_auc=r["img_mean_auc"], img_mean_f1=r["img_mean_f1"], patch_auc=r["patch_auc"],
                          transfer_offdiag=float(np.nanmean(off)), det_from_maxpatch=r.get("det_auc_from_maxpatch"))
            Bds[key] = {k.split(":")[1]: v for k, v in r.items() if k.endswith(":img_mean_auc")}
            print(f"{key:40s} " + "  ".join(f"{k}={v:.3f}" for k, v in B[key].items() if v is not None), flush=True)
        R.FEATURE_MODE = "raw"
    tag = f"_{args.tag}" if args.tag else ""
    a, b, bd = pd.DataFrame(A).T, pd.DataFrame(B).T, pd.DataFrame(Bds).T
    a.to_csv(OUT / f"task_a{tag}.csv"); b.to_csv(OUT / f"task_b{tag}.csv"); bd.to_csv(OUT / f"task_b_per_dataset{tag}.csv")
    md = ["# Task A - whole image (real vs fully synthetic), image-level AUC\n", a.round(3).to_markdown(), "",
          "# Task B - partial edits, patch localization (per-image AUC; transfer = mean off-diagonal)\n",
          b.astype(float).round(3).to_markdown(), "", "## Task B per dataset (per-image AUC)\n",
          bd.round(3).to_markdown()]
    (OUT / f"summary{tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
