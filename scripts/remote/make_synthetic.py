"""Fully synthetic images for the image-level triage (authentic / fully synthetic / regenerated edit / local splice):
COCO val2017 captions rendered from scratch by SDXL base and by FLUX.1-Fill-dev with an all-masked canvas (pure
text-to-image), at the size the photo of that caption would have in the benchmark (long side 1024). The photos used
by the benchmark and its extension (the first 420 of its seeded order) are excluded.

    python scripts/remote/make_synthetic.py --n 300 --out data/samples/synth26

Output: <out>/<generator>/<id>.png, meta.csv, manifest.csv (dataset synth_<generator>, label 1, no mask). Resumable.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_benchmark import SEED, cached, coco_annotations, select, target_size  # noqa: E402


def pick(n):
    caps = coco_annotations("captions")
    imgs = {i["id"]: i for i in caps["images"]}
    first = {}
    for a in sorted(caps["annotations"], key=lambda a: a["id"]):
        first.setdefault(a["image_id"], a["caption"].strip().rstrip("."))
    used = {it["coco_id"] for it in select(420)}
    ids = sorted(set(first) - used)
    random.Random(SEED + 1).shuffle(ids)
    out = []
    for cid in ids[:n]:
        W, H = target_size(imgs[cid]["width"], imgs[cid]["height"])
        out.append(dict(sample_id=f"{cid:012d}", coco_id=cid, caption=first[cid], width=W, height=H))
    return out


class SDXL:
    def __init__(self):
        from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        base, f16 = "stabilityai/stable-diffusion-xl-base-1.0", dict(torch_dtype=torch.float16, variant="fp16")
        self.pipe = StableDiffusionXLPipeline(
            # fp16 VAE: its config has force_upcast, so the pipeline decodes in fp32 itself (an fp32 VAE with fp16
            # latents fails in the text-to-image pipeline)
            vae=AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=torch.float16),
            text_encoder=CLIPTextModel.from_pretrained(base, subfolder="text_encoder", **f16),
            text_encoder_2=CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2", **f16),
            tokenizer=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer"),
            tokenizer_2=CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2"),
            unet=UNet2DConditionModel.from_pretrained(base, subfolder="unet", **f16),
            scheduler=EulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler")).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def __call__(self, it, seed):
        g = torch.Generator("cuda").manual_seed(seed)
        return self.pipe(prompt=it["caption"], width=it["width"], height=it["height"], guidance_scale=7.0,
                         num_inference_steps=30, generator=g).images[0]


class FluxFillT2I:
    """FLUX.1-Fill-dev with the whole canvas masked: nothing of an input image survives, so it is text-to-image"""

    def __init__(self, items):
        from diffusers import FluxFillPipeline
        dt = torch.bfloat16
        pipe = FluxFillPipeline.from_pretrained(cached("black-forest-labs/FLUX.1-Fill-dev"), torch_dtype=dt)
        pipe.text_encoder.to("cuda")
        pipe.text_encoder_2.to("cuda")
        self.emb = {}
        with torch.no_grad():
            for it in items:
                pe, pooled, _ = pipe.encode_prompt(prompt=it["caption"], prompt_2=None, device="cuda",
                                                   max_sequence_length=512)
                self.emb[it["sample_id"]] = (pe.cpu(), pooled.cpu())
        pipe.text_encoder, pipe.text_encoder_2 = None, None
        torch.cuda.empty_cache()
        pipe.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dt)
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    @torch.no_grad()
    def __call__(self, it, seed):
        g = torch.Generator("cuda").manual_seed(seed)
        W, H = it["width"], it["height"]
        pe, pooled = (t.to("cuda") for t in self.emb[it["sample_id"]])
        return self.pipe(prompt_embeds=pe, pooled_prompt_embeds=pooled, image=Image.new("RGB", (W, H), (128,) * 3),
                         mask_image=Image.new("L", (W, H), 255), width=W, height=H, guidance_scale=30.0,
                         num_inference_steps=28, generator=g).images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--generators", nargs="+", default=["sdxl", "fluxfill"])
    ap.add_argument("--out", default="data/samples/synth26")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = pick(args.n)
    pd.DataFrame(items).to_csv(out / "meta.csv", index=False)
    print(f"{len(items)} captions", flush=True)
    for gname in args.generators:
        (out / gname).mkdir(exist_ok=True)
        todo = [(i, it) for i, it in enumerate(items) if not (out / gname / f"{it['sample_id']}.png").exists()]
        if not todo:
            continue
        t0, gen = time.time(), None
        try:                                     # one generator failing must not lose the other's images
            gen = SDXL() if gname == "sdxl" else FluxFillT2I([it for _, it in todo])
            print(f"[{gname}] loaded in {time.time() - t0:.0f}s", flush=True)
            t0 = time.time()
            for k, (i, it) in enumerate(todo):
                gen(it, SEED + 10_000 + i).save(out / gname / f"{it['sample_id']}.png")
                if k % 25 == 0:
                    print(f"  {gname} {k}/{len(todo)}  {(time.time() - t0) / (k + 1):.1f} s/img", flush=True)
            print(f"[{gname}] {len(todo)} images in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print(f"FAILED {gname}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        del gen
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    root = out.relative_to("data") if out.parts[0] == "data" else out
    rows = [dict(dataset=f"synth_{g}", subset="all", sample_id=it["sample_id"], label=1,
                 img_path=f"{root}/{g}/{it['sample_id']}.png", mask_path="", width=it["width"], height=it["height"],
                 format="png", mode="RGB", mask_frac=1.0, pair_id=f"synth{it['sample_id']}", category="")
            for g in args.generators for it in items if (out / g / f"{it['sample_id']}.png").exists()]
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    print(f"manifest: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
