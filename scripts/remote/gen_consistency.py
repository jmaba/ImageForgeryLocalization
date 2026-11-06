"""Generator-consistency maps (docs/07 pilot): per-cell evidence that a region was produced by a latent-diffusion
decoder or sampled by a diffusion model, measured with open models used as instruments. Nothing is trained.

Per image, at native resolution (reflect-padded to a multiple of 64, maps cropped back, then averaged into the
24 x 24 grid of the mask cells):
  ae_l2_<m>, ae_lp_<m>   autoencoder round trip x -> D(E(x)) of model m: squared error and LPIPS-vgg distance.
                         Decoder outputs are near fixed points of the autoencoder (AEROBLADE, Ricker et al.,
                         CVPR 2024, arXiv:2401.17879) - a baseline here, not the novelty.
  den_<m>_t<k>           denoising error of model m at timestep k: z0 = E(x) (posterior mode, scaled),
                         z_t = sqrt(a_k) z0 + sqrt(1 - a_k) eps, error = |unet(z_t, k, "") - target|^2 per latent
                         cell, mean of DRAWS noise draws. Hypothesis: latents that were freely sampled by a diffusion
                         model are more typical for it than encodings of real pixels, also inside a fully
                         re-decoded ("regenerated") image, where the round trip above is uniform.
Models: sd2 (sd2-community/stable-diffusion-2-1-base) and sdxl (SDXL base 1.0) with their UNets; sd35 (SD3.5-large)
and flux (FLUX.1-dev) autoencoders only.
Controls: rows with control=vae_sd2 / vae_flux are first re-decoded by that autoencoder (benign processing of an
authentic image; detectors of this kind flag such images, see BG-REAL, arXiv:2607.26232).

Input: CSV with columns group, key, path (relative to --root), control (none | vae_sd2 | vae_flux).
Output: <out>/<group>[__<control>].npz with ids, patch [N, 24, 24, F] (float16), names.

    python scripts/remote/gen_consistency.py --list pilot_list.csv --out data/features/gencons
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

G = 24
TS = (100, 300, 500)
DRAWS = 2
DEV = "cuda"
REPO = {"sd2": "sd2-community/stable-diffusion-2-1-base", "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd35": "stabilityai/stable-diffusion-3.5-large", "flux": "black-forest-labs/FLUX.1-dev"}
DENOISERS = ("sd2", "sdxl")


def names():
    out = []
    for m in REPO:
        out += [f"ae_l2_{m}", f"ae_lp_{m}"]
        if m in DENOISERS:
            out += [f"den_{m}_t{t}" for t in TS]
    return out


def pix_cells(mp, H, W):
    return F.adaptive_avg_pool2d(mp[:H, :W].float()[None, None], G)[0, 0]


def lat_cells(mp, H, W):
    return F.adaptive_avg_pool2d(mp[:math.ceil(H / 8), :math.ceil(W / 8)].float()[None, None], G)[0, 0]


class Instruments:
    def __init__(self):
        import lpips
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        self.lp = lpips.LPIPS(net="vgg", spatial=True, verbose=False).to(DEV).eval()
        self.vae = {m: AutoencoderKL.from_pretrained(r, subfolder="vae", torch_dtype=torch.float32).to(DEV).eval()
                    for m, r in REPO.items()}
        self.unet, self.cond, self.ac, self.ptype = {}, {}, {}, {}
        for m in DENOISERS:
            r = REPO[m]
            self.unet[m] = UNet2DConditionModel.from_pretrained(r, subfolder="unet", torch_dtype=torch.float16,
                                                                variant="fp16").to(DEV).eval()
            sch = DDPMScheduler.from_pretrained(r, subfolder="scheduler")
            self.ac[m] = sch.alphas_cumprod.to(DEV)
            self.ptype[m] = sch.config.prediction_type
        if not DENOISERS:
            return
        with torch.no_grad():                    # empty-prompt conditioning, then the text encoders are dropped
            r = REPO["sd2"]
            tok = CLIPTokenizer.from_pretrained(r, subfolder="tokenizer")
            te = CLIPTextModel.from_pretrained(r, subfolder="text_encoder", torch_dtype=torch.float16).to(DEV)
            ids = tok([""], padding="max_length", max_length=tok.model_max_length, return_tensors="pt").input_ids
            self.cond["sd2"] = dict(encoder_hidden_states=te(ids.to(DEV))[0])
            del te
            r = REPO["sdxl"]
            t1 = CLIPTokenizer.from_pretrained(r, subfolder="tokenizer")
            t2 = CLIPTokenizer.from_pretrained(r, subfolder="tokenizer_2")
            e1 = CLIPTextModel.from_pretrained(r, subfolder="text_encoder", torch_dtype=torch.float16,
                                               variant="fp16").to(DEV)
            e2 = CLIPTextModelWithProjection.from_pretrained(r, subfolder="text_encoder_2", torch_dtype=torch.float16,
                                                             variant="fp16").to(DEV)
            i1 = t1([""], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEV)
            i2 = t2([""], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(DEV)
            h1 = e1(i1, output_hidden_states=True).hidden_states[-2]
            o2 = e2(i2, output_hidden_states=True)
            self.cond["sdxl"] = dict(encoder_hidden_states=torch.cat([h1, o2.hidden_states[-2]], -1),
                                     pooled=o2.text_embeds)
            del e1, e2
        torch.cuda.empty_cache()

    @torch.no_grad()
    def roundtrip(self, m, x):
        """x [1,3,H,W] in [-1,1] -> D(E(x)) of model m, same size"""
        xp, H, W = pad64(x)
        return self.vae[m].decode(self.vae[m].encode(xp).latent_dist.mode()).sample.clamp(-1, 1)[..., :H, :W]

    @torch.no_grad()
    def denoise(self, m, z0, Hp, Wp, H, W):
        B = len(TS)
        t = torch.tensor(TS, device=DEV)
        a = self.ac[m][t].view(B, 1, 1, 1).float()
        z0b = z0.float().expand(B, -1, -1, -1)
        c = self.cond[m]
        kw = dict(encoder_hidden_states=c["encoder_hidden_states"].expand(B, -1, -1))
        if m == "sdxl":
            tid = torch.tensor([[Hp, Wp, 0, 0, Hp, Wp]], device=DEV, dtype=torch.float16).expand(B, -1)
            kw["added_cond_kwargs"] = dict(text_embeds=c["pooled"].expand(B, -1), time_ids=tid)
        g = torch.Generator(DEV).manual_seed(0)
        err = 0.0
        for _ in range(DRAWS):
            eps = torch.randn(z0b.shape, generator=g, device=DEV)
            zt = a.sqrt() * z0b + (1 - a).sqrt() * eps
            pred = self.unet[m](zt.half(), t, **kw).sample.float()
            target = eps if self.ptype[m] == "epsilon" else a.sqrt() * eps - (1 - a).sqrt() * z0b
            err = err + ((pred - target) ** 2).mean(1)
        err = err / DRAWS
        return [lat_cells(err[i], H, W) for i in range(B)]

    @torch.no_grad()
    def features(self, x):
        xp, H, W = pad64(x)
        Hp, Wp = xp.shape[-2:]
        out = []
        for m, vae in self.vae.items():
            z = vae.encode(xp).latent_dist.mode()
            rec = vae.decode(z).sample.clamp(-1, 1)
            out.append(pix_cells(((rec - xp) ** 2).mean(1)[0], H, W))
            out.append(pix_cells(self.lp(rec, xp)[0, 0], H, W))
            if m in self.unet:
                out += self.denoise(m, z * vae.config.scaling_factor, Hp, Wp, H, W)
        return torch.stack(out, -1)                                         # [G, G, F]


def pad64(x):
    H, W = x.shape[-2:]
    return F.pad(x, (0, (-W) % 64, 0, (-H) % 64), mode="reflect"), H, W


def load(path, max_side=1536):
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:                  # none in the pilot sets; resampling would break the fixed point
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="data/features/gencons")
    ap.add_argument("--groups", nargs="*", help="only these groups")
    ap.add_argument("--ae-only", action="store_true", help="autoencoder maps only (no denoising error)")
    args = ap.parse_args()
    if args.ae_only:
        global DENOISERS
        DENOISERS = ()
    rows = pd.read_csv(args.list, dtype={"key": str, "group": str}).fillna({"control": "none"})   # keep zero-padded ids
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inst = Instruments()
    print(f"models ready, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    for (grp, ctrl), sub in rows.groupby(["group", "control"], sort=False):
        if args.groups and grp not in args.groups:
            continue
        f = out / (grp + ("" if ctrl == "none" else f"__{ctrl}") + ".npz")
        if f.exists() and len(np.load(f)["ids"]) == len(sub):
            continue
        t0, P = time.time(), []
        for i, r in enumerate(sub.itertuples()):
            x = load(Path(args.root) / r.path)
            if ctrl != "none":
                x = inst.roundtrip(ctrl.split("_")[1], x)
            P.append(inst.features(x).cpu().numpy().astype(np.float16))
            if i % 50 == 0:
                print(f"  {f.stem} {i}/{len(sub)}  {(time.time() - t0) / (i + 1):.2f} s/img", flush=True)
        np.savez(f, ids=np.array(sub.key.astype(str).tolist()), patch=np.stack(P), names=np.array(names()))
        print(f"[gencons] {f.stem}: {len(sub)} imgs  {time.time() - t0:.0f}s  "
              f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
