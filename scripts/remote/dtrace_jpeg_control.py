"""Format control for the decoder-trace image-level result on GPT-Image-2 / Nano-Banana-2 (docs/09): their outputs are
PNG while the COCO originals are JPEG. Re-encode the edited images as JPEG (95 / 90 / 75) and score them again against
the same originals; if the AUROC holds, the detector is not just separating "no JPEG artefacts".

    python scripts/remote/dtrace_jpeg_control.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.features.dtrace import DecoderTraceMap  # noqa: E402


def score(det, im):
    x = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1)[None]
    return float(det(x)[1][0, 0, 0, 2])


def jpeg(im, q):
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue()))


def main():
    man = pd.read_csv(DATA / "samples" / "manifest.csv", dtype={"sample_id": str})
    rows = []
    for split in ("A", "B"):
        det = DecoderTraceMap(split)
        for ds in ("gpt_image_2", "nanobanana2"):
            e = man[man.dataset == ds].head(150)
            o = man[man.dataset == "coco_orig_" + ds]
            so = [score(det, Image.open(DATA / p)) for p in o.img_path]
            r = {"model": f"dtrace{split}", "dataset": ds, "n_edited": len(e), "n_originals": len(so)}
            for q in (None, 95, 90, 75):
                se = [score(det, Image.open(DATA / p) if q is None else jpeg(Image.open(DATA / p), q))
                      for p in e.img_path]
                y = np.r_[np.zeros(len(so)), np.ones(len(se))]
                r["edited as PNG" if q is None else f"edited as JPEG {q}"] = roc_auc_score(y, np.r_[so, se])
            rows.append(r)
            print(r, flush=True)
    t = pd.DataFrame(rows).round(3)
    (OUTPUTS / "dtrace" / "jpeg_control.md").write_text(
        "# GPT-Image-2 / Nano-Banana-2 vs their COCO originals (JPEG): image AUROC with the edited images re-encoded\n\n"
        + t.to_markdown(index=False) + "\n")
    print(t.to_markdown(index=False))


if __name__ == "__main__":
    main()
