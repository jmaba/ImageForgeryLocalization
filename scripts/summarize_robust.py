"""Robustness of the decoder-trace detector and the published localizers (docs/09): the key numbers of
summarize_dtrace.py / eval_dtrace.py image level side by side for clean, JPEG-75, WebP-80 and resize-50 %.

    python scripts/summarize_robust.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import OUTPUTS  # noqa: E402

CONDS = {"": "clean", "__jpeg75": "JPEG-75", "__webp80": "WebP-80", "__resize50": "resize 50%"}
SCORERS = ["x_trufor", "x_mesorch", "cmp_z2", "dtA", "dtB", "dtB_max", "sota+cmp", "sota+cmp+dtB", "x_flame",
           "dtB+flame", "sota+cmp+dtB+flame"]
LOC = ["ALL - null", "benchmark regenerated (fr) - null", "extension regenerated (fr) - null", "GPT-Image-2 - null",
       "Nano-Banana-2 - null", "TGIF2 regenerated (fr) - null", "benchmark spliced (sp)",
       "diffusion splice / inpaint (published)"]
IMG = {"benchmark fr": ["bench_sdxlinp_fr", "bench_fluxfill_fr", "bench_kontext_fr"],
       "extension fr": ["benchx_sdxlinp_fr", "benchx_fluxfill_fr", "benchx_kontext_fr"],
       "TGIF2 fr": ["tgif2_sd2_fr", "tgif2_sdxl_fr", "tgif2_flux1dev_fr"],
       "GPT-Image-2": ["gpt_image_2"], "Nano-Banana-2": ["nanobanana2"],
       "fully synthetic (SDXL, FLUX)": ["synth_sdxl", "synth_fluxfill"],
       "benchmark sp": ["bench_sdxlinp_sp", "bench_fluxfill_sp", "bench_kontext_sp"]}


def main(flame=False):
    out = OUTPUTS / "dtrace"
    loc, img = ("localization_flame", "image_level_flame") if flame else ("localization", "image_level")
    md = ["# Robustness to re-sharing: clean vs JPEG-75 vs WebP-80 vs resize 50 %\n",
          "Every image of every dataset re-encoded / resized before any method sees it; same images, masks and "
          "matched nulls in every condition.\n"]
    for metric in LOC:
        t = {}
        for suf, name in CONDS.items():
            f = out / f"{loc}_summary{suf}.csv"
            if f.exists():
                s = pd.read_csv(f, index_col=0)
                if metric in s.index:
                    t[name] = s.loc[metric, [c for c in SCORERS if c in s.columns]]
        if t:
            md.append(f"## Localization: {metric}\n\n" + pd.DataFrame(t).round(3).to_markdown() + "\n")
    for det in ("dtA", "dtB", "trufor", "mesorch") + (("flame", "flame_map", "bfree") if flame else ()):
        t = {}
        for suf, name in CONDS.items():
            f = out / f"{img}{suf}.csv"
            if not f.exists():
                continue
            s = pd.read_csv(f).set_index("dataset")
            col = f"{det} vs own"
            t[name] = {g: s.loc[[d for d in ds if d in s.index], col].mean() for g, ds in IMG.items()
                       if col in s.columns}
        if t:
            md.append(f"## Image-level AUROC vs the images' own originals: {det}\n\n"
                      + pd.DataFrame(t).round(3).to_markdown() + "\n")
    (out / ("robustness_flame.md" if flame else "robustness.md")).write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main("--flame" in sys.argv)
