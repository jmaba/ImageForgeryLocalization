"""Real-only self-made splices for the pairwise "same source?" comparator.

Every training image starts from the real pool (data/ssl_real: DIV2K + ImageNet-val, no COCO). A mask is drawn
(random blob/ellipse/rectangle, or an OBJECT-shaped segment of the image's own DINOv2 token clusters, so that
"distinct object" is decoupled from "edited") and the masked region is replaced by one of:
    vae:<name>   the SAME image after one encode/decode by decoder <name>   (content preserved, decoder trace)
                 sd, sdxl, qwen (16-ch), taesd (tiny AE), dcae (SANA DC-AE f32), movq (Kandinsky VQ)
    proc         the SAME image after JPEG / blur / resampling / noise      (content preserved, processing trace)
    real         a region of ANOTHER pool image                             (classic splice, content changes)
    inpaint_sp   SD1.5-inpainting output pasted back inside the mask        (generated content, paste-back)
    inpaint_fr   the whole SD1.5-inpainting output                          (generated inside, re-decoded outside)
    none         nothing (authentic)
    multi        2-3 disjoint regions, each from a DIFFERENT one of --multi-types (multi-source / multi-step edits,
                 as in DiffSeg30k or RITA-style edit chains); labels are per-cell source ids, not a binary mask
FLUX's VAE is gated and every test editor is unseen; SD1.5-inpainting shares the SD VAE with TGIF2-SD2, so
TGIF2-SD2 is NOT an unseen test set for models trained with vae:sd / inpaint_*.

Output shard (features only, images are not kept), written incrementally as memmaps:
    data/splices_v2/<shard>/{dinov2_l6,srm,pixels}.npy  [N, 24, 24, D] fp16
    data/splices_v2/<shard>/masks.npy                     [N, 24, 24]    edited-area fraction per cell
    data/splices_v2/<shard>/sources.npy                   [N, 24, 24]    int8 source id per cell: 0 camera,
                                                          k >= 1 region k, -1 mixed cell (ignored)
    data/splices_v2/<shard>/meta.csv                      pool index, type, mask kind, size
    outputs/splices_v2/<shard>/examples/*.png

    python scripts/make_splices.py --shard vae --versions 4 \
        --types vae:sd vae:sdxl vae:qwen vae:taesd vae:dcae vae:movq proc real none
    python scripts/make_splices.py --shard inpaint --types inpaint_sp inpaint_fr --versions 1 --max-side 512
    python scripts/make_splices.py --shard multi --types multi --versions 2 --max-side 640
Adding feature streams to an existing shard: rerun its exact command plus --add-feats (same seed => same splices;
each row's type/mask kind and the stored SRM features are checked, and the run stops on a mismatch):
    python scripts/make_splices.py --shard multi --types multi --versions 2 --max-side 640 --add-feats dtcwt_phase wph
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
from scipy import ndimage
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA, OUTPUTS  # noqa: E402
from ifl.data.dataset import area_resize  # noqa: E402
from ifl.features.backbones import DEV, GRID, build  # noqa: E402
from ifl.features.roundtrip import RoundTrip  # noqa: E402

PROMPTS = ["", "a photo", "an object", "a high quality photo", "a natural scene", "something", "a texture",
           "a person", "an animal", "a building", "a plant", "food on a table", "a vehicle", "a sign"]


def load_pool_image(path, max_side, mult=1):
    im = Image.open(DATA / path).convert("RGB")
    s = min(1.0, max_side / max(im.size))
    w, h = max(mult, round(im.width * s) // mult * mult), max(mult, round(im.height * s) // mult * mult)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.BICUBIC)
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


@torch.no_grad()
def object_mask(a, dino, rng):
    """Mask = one connected segment of a k-means clustering of the image's own DINOv2 tokens (object-like,
    not taken from COCO annotations). None when no segment covers 4-45% of the image."""
    x = np.asarray(Image.fromarray(a).resize((336, 336), Image.BICUBIC)).copy()
    _, p = dino(torch.from_numpy(x).permute(2, 0, 1)[None])
    t = p[0].reshape(-1, p.shape[-1])
    t = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-6)
    lab = KMeans(int(rng.integers(3, 7)), n_init=2, random_state=int(rng.integers(1 << 30))).fit_predict(t)
    cands = []
    for c in np.unique(lab):
        cc, n = ndimage.label((lab == c).reshape(GRID, GRID))
        cands += [cc == k for k in range(1, n + 1) if 0.04 <= (cc == k).mean() <= 0.45]
    if not cands:
        return None
    seg = cands[rng.integers(len(cands))].astype(np.float32)
    h, w = a.shape[:2]
    m = F.interpolate(torch.from_numpy(seg)[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    m = (m.numpy() >= 0.5).astype(np.float32)
    if rng.random() < 0.5:
        m = ndimage.binary_dilation(m, iterations=int(rng.integers(2, 12))).astype(np.float32)
    if rng.random() < 0.5:
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


class Codec:
    """One encode/decode pass through an image autoencoder; uint8 HWC in and out."""

    def __init__(self, name):
        self.name, self.mult = name, 8
        if name in ("sd", "sdxl", "qwen"):
            rt = RoundTrip(name)
            self.enc, self.dec = rt._enc, rt._dec
            return
        import diffusers
        if name == "taesd":
            m = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
            self.enc, self.dec = (lambda x: m.encode(x).latents), (lambda z: m.decode(z).sample)
        elif name == "dcae":
            m = diffusers.AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers",
                                                        torch_dtype=torch.float16)
            self.enc, self.dec, self.mult = (lambda x: m.encode(x).latent), (lambda z: m.decode(z).sample), 32
        elif name == "movq":
            m = diffusers.VQModel.from_pretrained("kandinsky-community/kandinsky-2-2-decoder-inpaint",
                                                  subfolder="movq", torch_dtype=torch.float16)
            self.enc, self.dec = (lambda x: m.encode(x).latents), (lambda z: m.decode(z).sample)
        else:
            raise KeyError(name)
        self.m = m.to(DEV).eval()

    @torch.no_grad()
    def __call__(self, a):
        h, w = a.shape[:2]
        x = torch.from_numpy(a).permute(2, 0, 1)[None].float().to(DEV) / 127.5 - 1
        x = F.pad(x, (0, (-w) % self.mult, 0, (-h) % self.mult), mode="reflect")
        x1 = self.dec(self.enc(x.half())).float()[..., :h, :w]
        if not torch.isfinite(x1).all():
            raise FloatingPointError(f"{self.name}: non-finite decode")
        x1 = x1.clamp(-1, 1)
        return ((x1[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


class Inpainter:
    def __init__(self):
        from diffusers import StableDiffusionInpaintPipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting", variant="fp16", torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False).to(DEV)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.enable_attention_slicing()

    @torch.no_grad()
    def __call__(self, a, m, rng):
        h, w = a.shape[:2]
        out = self.pipe(prompt=str(rng.choice(PROMPTS)), image=Image.fromarray(a),
                        mask_image=Image.fromarray(((m >= 0.5) * 255).astype(np.uint8)), height=h, width=w,
                        num_inference_steps=int(rng.integers(15, 31)), guidance_scale=float(rng.uniform(4, 9)),
                        generator=torch.Generator(DEV).manual_seed(int(rng.integers(1 << 30)))).images[0]
        return np.asarray(out.convert("RGB").resize((w, h), Image.BICUBIC))


def source_grid(S, k):
    """Per-cell source id from a pixel source map S (0 camera, 1..k regions): a cell is camera only if it has
    no edited pixel, region r if r covers at least half of it, otherwise mixed (-1)."""
    frac = np.stack([area_resize((S == r).astype(np.float32), GRID) for r in range(k + 1)])
    g = np.full((GRID, GRID), -1, np.int8)
    g[frac[0] >= 1 - 1e-6] = 0
    for r in range(1, k + 1):
        g[frac[r] >= 0.5] = r
    return g


def donor_region(pool, i, h, w, rng):
    j = rng.integers(len(pool))
    while j == i:
        j = rng.integers(len(pool))
    d = Image.fromarray(load_pool_image(pool.path[j], 2048))
    a = np.asarray(d.resize((w, h), Image.BICUBIC))
    return jpeg(a, rng.integers(60, 96)) if rng.random() < 0.5 else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--types", nargs="+", required=True)
    ap.add_argument("--probs", nargs="*", type=float, help="one per type; default uniform")
    ap.add_argument("--versions", type=int, default=4, help="splices per pool image")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--object-p", type=float, default=0.5, help="probability of an object-shaped mask")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--multi-types", nargs="+",
                    default=["vae:sd", "vae:sdxl", "vae:qwen", "vae:taesd", "vae:movq", "proc", "real", "inpaint_sp"])
    ap.add_argument("--add-feats", nargs="+", default=None,
                    help="regenerate an existing shard and only write these extra feature streams")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pool = pd.read_csv(DATA / "ssl_real" / "pool.csv")
    if args.limit:
        pool = pool.sample(args.limit, random_state=0).reset_index(drop=True)
    probs = np.array(args.probs or [1.0] * len(args.types))
    probs = probs / probs.sum()
    out = DATA / "splices_v2" / args.shard
    out.mkdir(parents=True, exist_ok=True)
    ex = OUTPUTS / "splices_v2" / args.shard / "examples"
    ex.mkdir(parents=True, exist_ok=True)

    used = set(args.types) | (set(args.multi_types) if "multi" in args.types else set())
    codecs = {t: Codec(t[4:]) for t in sorted(used) if t.startswith("vae:")}
    inpaint = Inpainter() if any(t.startswith("inpaint") for t in used) else None
    mult = max([8] + [c.mult for c in codecs.values()])
    ext = {"dinov2_l6": build("dinov2:l6"), "srm": build("srm"), "pixels": build("pixels")}
    sizes = {"dinov2_l6": 336, "srm": None, "pixels": 216}
    N = len(pool) * args.versions
    mm = {}
    add = args.add_feats
    if add:
        old_meta = pd.read_csv(out / "meta.csv")
        old_srm = np.load(out / "srm.npy", mmap_mode="r")
        N = min(N, len(old_meta))
        del ext["pixels"]
        ext.update({k: build(k) for k in add})
        sizes.update({k: None for k in add})
        store, srm_err = set(add), []
    else:
        store = set(ext)
        masks = np.lib.format.open_memmap(out / "masks.npy", "w+", np.float16, (N, GRID, GRID))
        sources = np.lib.format.open_memmap(out / "sources.npy", "w+", np.int8, (N, GRID, GRID))
    meta = []
    t0 = time.time()
    n = 0
    def draw_mask(a):
        m = object_mask(a, ext["dinov2_l6"], rng) if rng.random() < args.object_p else None
        return (m, "object") if m is not None else (random_mask(h, w, rng), "random")

    def region_source(typ, base, m):
        if typ.startswith("vae:"):
            return codecs[typ](a)
        if typ == "proc":
            return processed(a, rng)
        if typ == "real":
            return donor_region(pool, i, h, w, rng)
        return inpaint(base, m, rng)

    def paste(base, src, m):
        return (m[..., None] * src + (1 - m[..., None]) * base).round().clip(0, 255).astype(np.uint8)

    for i in range(len(pool)):
        if n >= N:
            break
        a = load_pool_image(pool.path[i], args.max_side, mult)
        h, w = a.shape[:2]
        for v in range(args.versions):
            typ = str(rng.choice(args.types, p=probs))
            mkind, parts = "none", ""
            S, nk = np.zeros((h, w), np.int8), 1
            if typ == "none":
                comp, m = a, np.zeros((h, w), np.float32)
            elif typ == "multi":
                # disjoint regions, each from a different source; later regions avoid earlier ones and the
                # union stays below half the image (the majority-authentic assumption of pair_ref)
                kinds = list(rng.choice(args.multi_types, int(rng.integers(2, 4)), replace=False))
                comp, mkind, done = a, "multi", []
                for t in kinds:
                    for _ in range(6):
                        mr, _ = draw_mask(a)
                        mr = mr * (S == 0)
                        frac = (mr >= 0.5).mean()
                        if frac >= 0.03 and (S > 0).mean() + frac <= 0.5:
                            break
                    else:
                        continue
                    comp = paste(comp, region_source(t, comp, mr), mr)
                    done.append(t)
                    S[(mr >= 0.5) & (S == 0)] = len(done)
                m = (S > 0).astype(np.float32)
                nk, parts = max(1, len(done)), "+".join(done)
            else:
                m, mkind = draw_mask(a)
                src = region_source(typ, a, m)
                comp = src if typ == "inpaint_fr" else paste(a, src, m)
                S[m >= 0.5] = 1
            if rng.random() < 0.3:   # whole-image re-save, as when an edit is shared
                comp = jpeg(comp, rng.integers(85, 99))
            im = Image.fromarray(comp)
            if add and (typ, mkind) != (old_meta.type[n], old_meta.mask_kind[n]):
                raise SystemExit(f"row {n}: regenerated {typ}/{mkind} != stored {old_meta.type[n]}/"
                                 f"{old_meta.mask_kind[n]} (different --types order, seed or code)")
            for k, e in ext.items():
                if k not in store and k != "srm":
                    continue
                x = im.resize((sizes[k], sizes[k]), Image.BICUBIC) if sizes[k] else im
                _, p = e(torch.from_numpy(np.asarray(x).copy()).permute(2, 0, 1)[None])
                if add and k == "srm":       # same splice? relative error of the stored SRM features
                    ref = np.asarray(old_srm[n], np.float32)
                    srm_err.append(float(np.abs(p[0] - ref).mean() / (np.abs(ref).mean() + 1e-6)))
                    if srm_err[-1] > 0.05:
                        raise SystemExit(f"row {n}: SRM features differ from the stored shard ({srm_err[-1]:.3f})")
                if k not in store:
                    continue
                if k not in mm:
                    mm[k] = np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.float16, (N,) + p.shape[1:])
                mm[k][n] = p[0].astype(np.float16)
            if add:
                n += 1
                continue
            masks[n] = area_resize((m >= 0.5).astype(np.float32), GRID).astype(np.float16)
            sources[n] = source_grid(S, nk)
            meta.append(dict(pool_idx=i, version=v, type=typ, mask_kind=mkind, source=pool.source[i], h=h, w=w,
                             mask_frac=float((m >= 0.5).mean()), parts=parts))
            if i < 16 and v == 0:
                tag = typ.replace(":", "-")
                im.save(ex / f"{i:03d}_{tag}_{mkind}.png")
                vis = (S.astype(np.float32) / max(1, nk) * 255) if typ == "multi" else m * 255
                Image.fromarray(vis.astype(np.uint8)).save(ex / f"{i:03d}_{tag}_{mkind}_mask.png")
            n += 1
        if i % 50 == 0:
            print(f"{i}/{len(pool)}  {time.time() - t0:.0f}s" +
                  (f"  SRM rel. err. max {max(srm_err):.4f}" if add else ""), flush=True)
            if add:
                continue
            pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)   # checkpoint: rows < len(meta) are valid
    for k in mm:
        mm[k].flush()
    if add:
        print(f"added {sorted(store)} to {n} rows; SRM rel. err. mean {np.mean(srm_err):.4f} max {max(srm_err):.4f}")
        return
    masks.flush()
    sources.flush()
    df = pd.DataFrame(meta)
    df.to_csv(out / "meta.csv", index=False)
    print(df.groupby(["type", "mask_kind"]).size(), flush=True)


if __name__ == "__main__":
    main()
