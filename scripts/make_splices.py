"""Real-only self-made splices for the pairwise "same source?" comparator (bigger test after region_tests).

Every training image starts from the real pool (data/ssl_real: DIV2K + ImageNet-val, no COCO). A random mask
is drawn and the masked region is replaced by one of:
    vae_sd    the SAME image after one SD-VAE encode/decode      (content preserved, decoder trace only)
    vae_sdxl  the SAME image after one SDXL-VAE encode/decode    (content preserved, decoder trace only)
    proc      the SAME image after JPEG / blur / resampling / noise (content preserved, processing trace)
    real      a region of ANOTHER pool image                      (classic splice, content changes)
    none      nothing (authentic, all cells same source)
The content-preserving types dominate so that a model cannot solve the task by reading content: that is
exactly the confound region_tests.py exposed. Qwen-Image / FLUX VAEs and every test editor stay unseen.

Output (features only, images are not kept):
    data/splices/{dinov2_l6,srm,pixels}.npy   [N, 24, 24, D] fp16
    data/splices/masks.npy                     [N, 24, 24]    fp16 edited-area fraction per cell
    data/splices/meta.csv                      pool index, type, size
    outputs/splices/examples/*.png             a few composites + masks for eyeballing

    python scripts/make_splices.py --versions 2
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.data.dataset import area_resize  # noqa: E402
from ifl.features.backbones import DEV, GRID, build  # noqa: E402
from ifl.features.roundtrip import RoundTrip  # noqa: E402

TYPES = ["vae_sd", "vae_sdxl", "proc", "real", "none"]
PROBS = [0.30, 0.20, 0.25, 0.15, 0.10]
MAX_SIDE = 768


def load_pool_image(path):
    im = Image.open(DATA / path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        s = MAX_SIDE / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    return np.asarray(im)


def random_mask(h, w, rng):
    """Soft mask in [0,1] covering 5-40% of the image: smooth random blob, ellipse or rectangle."""
    area = rng.uniform(0.05, 0.40)
    kind = rng.choice(["blob", "ellipse", "rect"], p=[0.5, 0.3, 0.2])
    yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
    if kind == "blob":
        g = rng.normal(size=(rng.integers(3, 8), rng.integers(3, 8))).astype(np.float32)
        f = F.interpolate(torch.from_numpy(g)[None, None], size=(h, w), mode="bicubic", align_corners=False)[0, 0].numpy()
        m = (f >= np.quantile(f, 1 - area)).astype(np.float32)
    else:
        cy, cx = rng.uniform(0.2, 0.8) * h / max(h, w), rng.uniform(0.2, 0.8) * w / max(h, w)
        ar = rng.uniform(0.5, 2.0)
        r = np.sqrt(area * h * w / max(h, w) ** 2 / (np.pi if kind == "ellipse" else 4))
        ry, rx = r * np.sqrt(ar), r / np.sqrt(ar)
        if kind == "ellipse":
            m = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1).astype(np.float32)
        else:
            m = ((np.abs(yy - cy) <= ry) & (np.abs(xx - cx) <= rx)).astype(np.float32)
    if m.sum() < 0.02 * h * w:   # degenerate: fall back to a centred square
        m[h // 3:2 * h // 3, w // 3:2 * w // 3] = 1
    if rng.random() < 0.5:       # feathered edge, as in paste-back inpainting
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(rng.uniform(1.5, 6))), np.float32) / 255
    return m


def jpeg(a, q):
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=int(q))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def processed(a, rng):
    op = rng.choice(["jpeg", "blur", "resample", "noise"])
    if op == "jpeg":
        return jpeg(a, rng.integers(50, 91))
    im = Image.fromarray(a)
    if op == "blur":
        return np.asarray(im.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.0))))
    if op == "resample":
        s = rng.uniform(0.5, 0.85)
        small = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BICUBIC)
        return np.asarray(small.resize(im.size, Image.BICUBIC))
    return np.clip(a + rng.normal(0, rng.uniform(1, 4), a.shape), 0, 255).astype(np.uint8)


@torch.no_grad()
def vae_once(rt, a):
    h, w = a.shape[:2]
    x = torch.from_numpy(a).permute(2, 0, 1)[None].float().to(DEV) / 127.5 - 1
    x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="reflect")
    x1 = rt._dec(rt._enc(x.half())).float().clamp(-1, 1)[..., :h, :w]
    return ((x1[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def donor_region(pool, i, h, w, rng):
    j = rng.integers(len(pool))
    while j == i:
        j = rng.integers(len(pool))
    d = Image.fromarray(load_pool_image(pool.path[j]))
    d = d.resize((w, h), Image.BICUBIC)
    a = np.asarray(d)
    return jpeg(a, rng.integers(60, 96)) if rng.random() < 0.5 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=int, default=2, help="splices per pool image")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pool = pd.read_csv(DATA / "ssl_real" / "pool.csv")
    if args.limit:
        pool = pool.sample(args.limit, random_state=0).reset_index(drop=True)
    out = DATA / "splices"
    out.mkdir(parents=True, exist_ok=True)
    ex = OUTPUTS / "splices" / "examples"
    ex.mkdir(parents=True, exist_ok=True)

    vaes = {"vae_sd": RoundTrip("sd"), "vae_sdxl": RoundTrip("sdxl")}
    ext = {"dinov2_l6": build("dinov2:l6"), "srm": build("srm"), "pixels": build("pixels")}
    sizes = {"dinov2_l6": 336, "srm": None, "pixels": 216}
    feats = {k: [] for k in ext}
    masks, meta = [], []
    t0 = time.time()
    for i in range(len(pool)):
        a = load_pool_image(pool.path[i])
        h, w = a.shape[:2]
        for v in range(args.versions):
            typ = rng.choice(TYPES, p=PROBS)
            if typ == "none":
                comp, m = a, np.zeros((h, w), np.float32)
            else:
                m = random_mask(h, w, rng)
                src = (vae_once(vaes[typ], a) if typ.startswith("vae") else
                       processed(a, rng) if typ == "proc" else donor_region(pool, i, h, w, rng))
                comp = (m[..., None] * src + (1 - m[..., None]) * a).round().clip(0, 255).astype(np.uint8)
            if rng.random() < 0.3:   # whole-image re-save, as when an edit is shared
                comp = jpeg(comp, rng.integers(85, 99))
            im = Image.fromarray(comp)
            for k, e in ext.items():
                x = im.resize((sizes[k], sizes[k]), Image.BICUBIC) if sizes[k] else im
                _, p = e(torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1)[None])
                feats[k].append(p[0].astype(np.float16))
            masks.append(area_resize((m >= 0.5).astype(np.float32), GRID).astype(np.float16))
            meta.append(dict(pool_idx=i, version=v, type=typ, source=pool.source[i], h=h, w=w,
                             mask_frac=float((m >= 0.5).mean())))
            if i < 12 and v == 0:
                im.save(ex / f"{i:03d}_{typ}.png")
                Image.fromarray((m * 255).astype(np.uint8)).save(ex / f"{i:03d}_{typ}_mask.png")
        if i % 50 == 0:
            print(f"{i}/{len(pool)}  {time.time() - t0:.0f}s", flush=True)
    for k in feats:
        np.save(out / f"{k}.npy", np.stack(feats[k]))
    np.save(out / "masks.npy", np.stack(masks))
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    print(pd.DataFrame(meta).type.value_counts(), flush=True)


if __name__ == "__main__":
    main()
