"""Controlled diffusion-edit benchmark (docs/08_benchmark_prereg.md): COCO val2017 photos, one object each, replaced
by three open editors; every edit kept as the full output (fr) and pasted back inside the mask (sp).

    python scripts/remote/make_benchmark.py --n 120 --editors sdxlinp fluxfill kontext
    python scripts/remote/make_benchmark.py --n 2 --out data/samples/bench_smoke        # smoke test

Output (under --out): orig/, masks/, <editor>_fr/, <editor>_sp/, kontext_diffmask/, meta.csv, and manifest.csv in the
repository's manifest format (dataset = bench_<variant>, paths relative to the data folder).
Resumable: an image that already exists is not generated again.
"""
import argparse
import io
import json
import random
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

SEED = 2026
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
CACHE = Path("data/_cache/coco_val")


def coco_annotations(kind="instances"):
    p = CACHE / f"{kind}_val2017.json"
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(ANN_URL, timeout=600) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        for k in ("instances", "captions"):
            (CACHE / f"{k}_val2017.json").write_bytes(z.read(f"annotations/{k}_val2017.json"))
    return json.loads(p.read_text())


def coco_image(cid):
    p = CACHE / f"{cid:012d}.jpg"
    if not p.exists():
        with urllib.request.urlopen(IMG_URL.format(cid), timeout=120) as r:
            p.write_bytes(r.read())
    return Image.open(p).convert("RGB")


def select(n):
    """seeded choice of photos, one eligible object each, and a replacement category from the same supercategory"""
    from pycocotools import mask as mutils
    js = coco_annotations()
    cats = {c["id"]: c for c in js["categories"]}
    by_super = {}
    for c in js["categories"]:
        by_super.setdefault(c["supercategory"], []).append(c["name"])
    imgs = {i["id"]: i for i in js["images"]}
    elig = {}
    for a in js["annotations"]:
        im, c = imgs[a["image_id"]], cats[a["category_id"]]
        frac = a["area"] / (im["width"] * im["height"])
        if (not a["iscrowd"] and c["name"] != "person" and len(by_super[c["supercategory"]]) >= 2
                and 0.03 <= frac <= 0.25 and min(im["width"], im["height"]) >= 400):
            elig.setdefault(a["image_id"], []).append(a)
    rng = random.Random(SEED)
    ids = sorted(elig)
    rng.shuffle(ids)
    out = []
    for cid in ids[:n]:
        a = rng.choice(sorted(elig[cid], key=lambda x: x["id"]))
        c = cats[a["category_id"]]
        new = rng.choice(sorted(set(by_super[c["supercategory"]]) - {c["name"]}))
        im = imgs[cid]
        rle = mutils.frPyObjects(a["segmentation"], im["height"], im["width"]) if isinstance(a["segmentation"], list) \
            else a["segmentation"]
        m = mutils.decode(mutils.merge(rle) if isinstance(rle, list) else rle).astype(np.uint8)
        out.append(dict(coco_id=cid, ann_id=a["id"], category=c["name"], new_category=new, mask=m))
    return out


def cached(repo):
    """local snapshot folder of a (partially) downloaded repo in the HF cache"""
    import os
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    d = hub / ("models--" + repo.replace("/", "--"))
    return str(d / "snapshots" / (d / "refs" / "main").read_text().strip())


def target_size(w, h, long=1024):
    s = long / max(w, h)
    return max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)


def prep(item):
    im = coco_image(item["coco_id"])
    W, H = target_size(*im.size)
    im = im.resize((W, H), Image.BICUBIC)
    m = Image.fromarray(item["mask"] * 255).resize((W, H), Image.NEAREST)
    k = max(3, round(0.015 * max(W, H)) * 2 + 1)
    m = m.filter(ImageFilter.MaxFilter(k)).point(lambda v: 255 if v > 127 else 0)
    return im, m


class Editors:
    """loads one editor at a time (they do not fit together); text conditioning is encoded up front and the text
    encoders freed, so the FLUX transformer (fp8 storage, bf16 compute) and its VAE stay on the GPU"""

    def __init__(self, name, items):
        self.name, self.dev = name, "cuda"
        dt = torch.bfloat16
        if name == "sdxlinp":
            from diffusers import (AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLInpaintPipeline,
                                   UNet2DConditionModel)
            from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
            base, inp = "stabilityai/stable-diffusion-xl-base-1.0", "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            f16 = dict(torch_dtype=torch.float16, variant="fp16")
            # assembled from cached parts (the inpainting repo ships only its UNet here; text encoders are SDXL base's)
            self.pipe = StableDiffusionXLInpaintPipeline(
                vae=AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32),
                text_encoder=CLIPTextModel.from_pretrained(base, subfolder="text_encoder", **f16),
                text_encoder_2=CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", **f16),
                tokenizer=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer"),
                tokenizer_2=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2"),
                unet=UNet2DConditionModel.from_pretrained(inp, subfolder="unet", **f16),
                scheduler=EulerDiscreteScheduler.from_pretrained(inp, subfolder="scheduler"),
                requires_aesthetics_score=False, force_zeros_for_empty_prompt=True).to(self.dev)
            self.pipe.set_progress_bar_config(disable=True)
        else:
            from diffusers import FluxFillPipeline, FluxKontextPipeline
            # loaded from the cached snapshot folders: only the diffusers components were downloaded, and both
            # diffusers' hub check and huggingface_hub (incomplete snapshot) refuse that offline
            fill = cached("black-forest-labs/FLUX.1-Fill-dev")
            if name == "fluxfill":
                pipe = FluxFillPipeline.from_pretrained(fill, torch_dtype=dt)
            else:                                    # Kontext shares the text encoders of the FLUX.1 family
                from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
                pipe = FluxKontextPipeline.from_pretrained(
                    cached("black-forest-labs/FLUX.1-Kontext-dev"), torch_dtype=dt,
                    text_encoder=CLIPTextModel.from_pretrained(fill, subfolder="text_encoder", torch_dtype=dt),
                    tokenizer=CLIPTokenizer.from_pretrained(fill, subfolder="tokenizer"),
                    text_encoder_2=T5EncoderModel.from_pretrained(fill, subfolder="text_encoder_2", torch_dtype=dt),
                    tokenizer_2=T5TokenizerFast.from_pretrained(fill, subfolder="tokenizer_2"))
            pipe.text_encoder.to(self.dev)
            pipe.text_encoder_2.to(self.dev)
            self.emb = {}
            with torch.no_grad():
                for it in items:
                    p = self.prompt(it)
                    if p not in self.emb:
                        pe, pooled, _ = pipe.encode_prompt(prompt=p, prompt_2=None, device=self.dev,
                                                           max_sequence_length=512)
                        self.emb[p] = (pe.cpu(), pooled.cpu())
            pipe.text_encoder, pipe.text_encoder_2 = None, None
            torch.cuda.empty_cache()
            pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dt)
            pipe.transformer.to(self.dev)
            pipe.vae.to(self.dev)
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe

    def prompt(self, it):
        if self.name == "kontext":
            return f"Replace the {it['category']} with a {it['new_category']}. Keep everything else unchanged."
        return f"a photo of a {it['new_category']}"

    @torch.no_grad()
    def __call__(self, it, im, m, seed):
        g = torch.Generator(self.dev).manual_seed(seed)
        W, H = im.size
        if self.name == "sdxlinp":
            return self.pipe(prompt=self.prompt(it), image=im, mask_image=m, width=W, height=H, strength=0.99,
                             guidance_scale=7.5, num_inference_steps=30, generator=g).images[0]
        pe, pooled = (t.to(self.dev) for t in self.emb[self.prompt(it)])
        if self.name == "fluxfill":
            return self.pipe(prompt_embeds=pe, pooled_prompt_embeds=pooled, image=im, mask_image=m, width=W,
                             height=H, guidance_scale=30.0, num_inference_steps=28, generator=g).images[0]
        return self.pipe(prompt_embeds=pe, pooled_prompt_embeds=pooled, image=im, width=W, height=H,
                         guidance_scale=2.5, num_inference_steps=28, generator=g, _auto_resize=False,
                         max_area=W * H).images[0]   # otherwise Kontext rescales to ~1 MP and the output size changes


def diff_mask(a, b):
    d = np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).mean(-1) / 255
    d = np.asarray(Image.fromarray((d * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), np.float32) / 255
    return Image.fromarray(((d > 0.1) * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(
        ImageFilter.MaxFilter(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--editors", nargs="+", default=["sdxlinp", "fluxfill", "kontext"])
    ap.add_argument("--out", default="data/samples/bench26")
    ap.add_argument("--skip", type=int, default=0, help="skip the first k photos of the seeded order (extensions)")
    ap.add_argument("--prefix", default="bench", help="dataset name prefix in the manifest")
    args = ap.parse_args()
    out = Path(args.out)
    items = select(args.skip + args.n)[args.skip:]     # same seeded order: an extension never overlaps the first k
    for sub in ["orig", "masks"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    meta = []
    for i, it in enumerate(items):
        sid = f"{it['coco_id']:012d}"
        im, m = prep(it)
        if not (out / "orig" / f"{sid}.png").exists():
            im.save(out / "orig" / f"{sid}.png")
            m.save(out / "masks" / f"{sid}.png")
        meta.append(dict(sample_id=sid, coco_id=it["coco_id"], ann_id=it["ann_id"], category=it["category"],
                         new_category=it["new_category"], width=im.width, height=im.height,
                         mask_frac=float((np.asarray(m) > 127).mean()), seed=SEED + args.skip + i))
    pd.DataFrame(meta).to_csv(out / "meta.csv", index=False)
    print(f"{len(items)} sources ready", flush=True)

    for ed in args.editors:
        todo = [(it, mt) for it, mt in zip(items, meta) if not (out / f"{ed}_sp" / f"{mt['sample_id']}.png").exists()]
        if not todo:
            continue
        for sub in [f"{ed}_fr", f"{ed}_sp"] + (["kontext_diffmask"] if ed == "kontext" else []):
            (out / sub).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        E = Editors(ed, [it for it, _ in todo])
        print(f"[{ed}] loaded in {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB",
              flush=True)
        t0, bad = time.time(), 0
        for k, (it, mt) in enumerate(todo):
            sid = mt["sample_id"]
            im = Image.open(out / "orig" / f"{sid}.png").convert("RGB")
            m = Image.open(out / "masks" / f"{sid}.png").convert("L")
            fr = E(it, im, m, mt["seed"])
            if fr.size != im.size:
                print(f"  {ed} {sid}: output {fr.size} != {im.size}, resized", flush=True)
                fr = fr.resize(im.size, Image.BICUBIC)
            a = np.asarray(fr, np.float32)
            if not np.isfinite(a).all() or a.std() < 1:
                bad += 1
            mb = (np.asarray(m) > 127)[..., None]
            sp = np.where(mb, np.asarray(fr), np.asarray(im)).astype(np.uint8)
            fr.save(out / f"{ed}_fr" / f"{sid}.png")
            if ed == "kontext":
                diff_mask(fr, im).save(out / "kontext_diffmask" / f"{sid}.png")
            Image.fromarray(sp).save(out / f"{ed}_sp" / f"{sid}.png")
            if k % 10 == 0:
                print(f"  {ed} {k}/{len(todo)}  {(time.time() - t0) / (k + 1):.1f} s/img", flush=True)
        print(f"[{ed}] {len(todo)} edits in {time.time() - t0:.0f}s, {bad} degenerate", flush=True)
        del E                                    # the fp8 casting hooks form reference cycles: collect them
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    root = out.relative_to("data") if out.parts[0] == "data" else out
    for mt in meta:
        sid = mt["sample_id"]
        base = dict(subset="all", sample_id=sid, width=mt["width"], height=mt["height"], format="png", mode="RGB",
                    pair_id=f"{args.prefix}{sid}", category=mt["category"])
        rows.append(dict(base, dataset=f"{args.prefix}_orig", label=0, img_path=f"{root}/orig/{sid}.png",
                         mask_path="", mask_frac=0.0))
        for ed in args.editors:
            for v in ("fr", "sp"):
                p = out / f"{ed}_{v}" / f"{sid}.png"
                if p.exists():
                    rows.append(dict(base, dataset=f"{args.prefix}_{ed}_{v}", label=2,
                                     img_path=f"{root}/{ed}_{v}/{sid}.png",
                                     mask_path=f"{root}/masks/{sid}.png", mask_frac=mt["mask_frac"]))
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"manifest: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
