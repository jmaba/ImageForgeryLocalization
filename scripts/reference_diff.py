"""Edited image vs its own original, for the datasets that ship (or name) the source photo:
TGIF2 (tgif2_orig, same COCO id), GPT-Image-2 and Nano-Banana-2 (COCO-2017 train id in the bench's mapping.json;
originals fetched once from images.cocodataset.org into data/_cache/coco_orig).

1. Diagnostic: how much of the image did the editor change? Per image, inside vs outside the mask:
     mad_in / mad_out     mean |dL*| (CIELAB lightness) between the edit and its original
     frac_out_changed     fraction of OUTSIDE pixels with |dL*| > 3
     trace_corr_out       correlation of the high-pass residuals (3x3 Laplacian) of edit and original OUTSIDE
                          the mask: ~1 = the camera/compression traces survive outside, ~0 = re-synthesised
   A trace-consistency localizer needs trace_corr_out high (the unedited part keeps its traces) and a contrast
   inside; full regeneration should drive it to ~0 everywhere.
2. Reference-based localization (upper bound, needs the original): cell score = mean |dLab| per 24x24 cell after
   a small blur, per-image cell AUC under the same labels as every other scorer; also with the edit re-shared
   (JPEG-75, WebP-80, resize 50 %), the original untouched.

    python scripts/reference_diff.py            -> outputs/reference/{pairs.csv, summary.md}
"""
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from scipy import ndimage
from skimage.color import rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.analysis import representativeness as R  # noqa: E402
from ifl.analysis.store import load  # noqa: E402
from ifl.data.dataset import harmonize  # noqa: E402

BENCH = {"gpt_image_2": ("GPT-Image-2/newapi_gpt_image_2_bench.tar.gz", "newapi_gpt_image_2_bench"),
         "nanobanana2": ("NanoBanana2/NanoBanana2_gemini-3.1-flash-image_bench.tar.gz", "gemini-3.1-flash-image_bench")}
TGIF2 = ["tgif2_sd2_sp", "tgif2_flux1dev_sp", "tgif2_sd2_fr", "tgif2_sdxl_fr", "tgif2_flux1dev_fr"]
LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
PERTURB = ["", "jpeg75", "webp80", "resize50"]


def coco_ids(name):
    tar, root = BENCH[name]
    with tarfile.open(DATA / "raw" / "AIGC-Loc-Testsets" / tar) as tf:
        mp = json.load(tf.extractfile(f"{root}/mapping.json"))
    out = {}
    for k, v in mp.items():
        src = v.get("original") or v.get("source") or v.get("sample_id")
        out[k] = Path(src).stem.split("_")[0]
    return out


def coco_image(cid):
    p = DATA / "_cache" / "coco_orig" / f"{int(cid):012d}.jpg"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"http://images.cocodataset.org/train2017/{int(cid):012d}.jpg", timeout=60) as r:
            p.write_bytes(r.read())
    return Image.open(p).convert("RGB")


def lab(im):
    return rgb2lab(np.asarray(im, np.float32) / 255).astype(np.float32)


def stats(ed, orig, mask):
    """ed, orig PIL RGB of equal size; mask bool HxW"""
    a, b = lab(ed), lab(orig)
    dl = np.abs(a[..., 0] - b[..., 0])
    ga, gb = (ndimage.convolve(np.asarray(x.convert("L"), np.float32), LAP) for x in (ed, orig))
    out = ~ndimage.binary_dilation(mask, iterations=8)                 # keep a margin away from the edit
    c = np.corrcoef(ga[out], gb[out])[0, 1] if out.sum() > 100 else np.nan
    ci = np.corrcoef(ga[mask], gb[mask])[0, 1] if mask.sum() > 100 else np.nan
    return dict(mad_in=dl[mask].mean(), mad_out=dl[out].mean(), frac_out_changed=(dl[out] > 3).mean(),
                trace_corr_out=c, trace_corr_in=ci)


def cell_diff(ed, orig, G=24):
    a = lab(ed.filter(ImageFilter.GaussianBlur(1.5)))
    b = lab(orig.filter(ImageFilter.GaussianBlur(1.5)))
    d = np.sqrt(((a - b) ** 2).sum(-1))
    t = np.asarray(Image.fromarray(d).resize((G, G), Image.BOX))
    return t.reshape(-1)


def main():
    out = OUTPUTS / "reference"
    out.mkdir(parents=True, exist_ok=True)
    fs = load("pixels", TGIF2 + ["tgif2_orig", "gpt_image_2", "nanobanana2"])
    meta = fs.meta.reset_index(drop=True)
    eidx, lab_grid = R._edited(fs)
    orig_of = {str(r.pair_id).split("_")[0]: DATA / r.img_path for r in meta[meta.dataset == "tgif2_orig"].itertuples()}
    cmap = {d: coco_ids(d) for d in BENCH}
    rows = []
    for n, i in enumerate(eidx):
        r = meta.iloc[i]
        if r.dataset in BENCH:
            cid = cmap[r.dataset].get(str(r.sample_id))
            try:
                orig = coco_image(cid) if cid else None
            except Exception as e:                 # a missing COCO file must not stop the run
                print(f"  {r.sample_id}: no original ({e})", flush=True)
                orig = None
        else:
            p = orig_of.get(str(r.pair_id).split("_")[0])
            orig = Image.open(p).convert("RGB") if p else None
        if orig is None:
            continue
        ed = Image.open(DATA / r.img_path).convert("RGB")
        orig = orig.resize(ed.size, Image.BICUBIC)
        mask = np.asarray(Image.open(DATA / r.mask_path).convert("L").resize(ed.size, Image.NEAREST)) > 127
        l = lab_grid[i].reshape(-1)
        v = l >= 0
        row = dict(dataset=r.dataset, sample_id=r.sample_id, mask_frac=mask.mean(), **stats(ed, orig, mask))
        for pm in PERTURB:
            e2 = harmonize(ed, pm).resize(ed.size, Image.BICUBIC) if pm else ed
            row[f"ref_auc{('_' + pm) if pm else ''}"] = R._auc(l[v], cell_diff(e2, orig)[v])
        rows.append(row)
        if n % 100 == 0:
            print(f"  {n}/{len(eidx)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairs.csv", index=False)
    cols = ["mad_in", "mad_out", "frac_out_changed", "trace_corr_in", "trace_corr_out"] + \
        [c for c in df if c.startswith("ref_auc")]
    tab = df.groupby("dataset")[cols].mean().round(3)
    tab.insert(0, "n", df.groupby("dataset").size())
    (out / "summary.md").write_text("# Edit vs own original (scripts/reference_diff.py)\n\n" + tab.to_markdown() + "\n")
    print(tab.to_string())


if __name__ == "__main__":
    main()
