"""Decoder-trace detector (docs/09): a fully convolutional network that marks pixels produced by a latent-image
decoder. Trained only on real photos and their autoencoder round trips pasted back inside random blobs
(scripts/remote/dtrace_train.py): the pasted region shows the same content as the photo, so the network can only learn
the traces a decoder leaves, never objects, and its matched null should stay near chance.

Decoder bank (autoencoders only, no diffusion): four 4-channel KL autoencoders (SD1.5 ft-MSE, SD2, SDXL, Playground
v2.5), three 16-channel KL ones (SD3.5, FLUX.1, CogView4), Kandinsky 2.2 MoVQ (vector quantised), the Qwen-Image
video-style VAE and Sana's DC-AE (32x compression).

Network: timm ResNet-34 encoder (ImageNet weights) on 6 channels - RGB and a high-pass residual x - blur(x) - so the
early stride does not throw the traces away, with a light FPN decoder to a logit map at 1/4 resolution.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BANK = {   # name: (kind, repo, subfolder, variant)
    "sd15": ("kl", "stabilityai/sd-vae-ft-mse", None, None),
    "sd2": ("kl", "sd2-community/stable-diffusion-2-1-base", "vae", None),
    "sdxl": ("kl", "stabilityai/stable-diffusion-xl-base-1.0", "vae", None),
    "playground": ("kl", "playgroundai/playground-v2.5-1024px-aesthetic", "vae", "fp16"),
    "sd35": ("kl", "stabilityai/stable-diffusion-3.5-large", "vae", None),
    "flux": ("kl", "black-forest-labs/FLUX.1-dev", "vae", None),
    "cogview4": ("kl", "THUDM/CogView4-6B", "vae", None),
    "movq": ("vq", "kandinsky-community/kandinsky-2-2-decoder", "movq", None),
    "qwen": ("qwen", "Qwen/Qwen-Image", "vae", None),
    "sana": ("dc", "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers", None, None),
}
SPLITS = {"A": ["sd15", "sdxl", "flux", "movq", "qwen"],          # held out: sd2, playground, sd35, cogview4, sana
          "B": list(BANK)}


def load_vae(name, device="cuda"):
    from diffusers import AutoencoderDC, AutoencoderKL, AutoencoderKLQwenImage, VQModel
    kind, repo, sub, variant = BANK[name]
    cls = {"kl": AutoencoderKL, "vq": VQModel, "qwen": AutoencoderKLQwenImage, "dc": AutoencoderDC}[kind]
    kw = dict(subfolder=sub) if sub else {}
    if variant:
        kw["variant"] = variant
    return kind, cls.from_pretrained(repo, torch_dtype=torch.float32, **kw).to(device).eval()


@torch.no_grad()
def roundtrip(kind, vae, x, chunk=None, bf16=False):
    """x [B,3,H,W] in [-1,1], H and W multiples of 32 -> D(E(x)), same size, clamped. In chunks (the video-style VAE's
    3D convolutions need several GB per 384 px crop in fp32); bf16=True decodes as most pipelines do (FLUX, SD3)."""
    chunk = chunk or (4 if kind == "qwen" else 8)
    if len(x) > chunk:
        return torch.cat([roundtrip(kind, vae, x[i:i + chunk], chunk, bf16) for i in range(0, len(x), chunk)])
    if bf16:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return roundtrip(kind, vae, x, chunk, False).float()
    if kind == "kl":
        y = vae.decode(vae.encode(x).latent_dist.mode()).sample
    elif kind == "vq":
        y = vae.decode(vae.encode(x).latents).sample
    elif kind == "dc":
        y = vae.decode(vae.encode(x).latent).sample
    else:                                               # video-style VAE: one frame
        y = vae.decode(vae.encode(x[:, :, None]).latent_dist.mode()).sample[:, :, 0]
    return y.clamp(-1, 1)


def highpass(x):
    k = torch.tensor([1, 4, 6, 4, 1], device=x.device, dtype=x.dtype)
    k = (k[:, None] * k[None]) / 256
    blur = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5), groups=3)
    return (x - blur) * 8


class TraceNet(nn.Module):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, pretrained=True):
        super().__init__()
        import timm
        self.enc = timm.create_model("resnet34", pretrained=pretrained, features_only=True, in_chans=6,
                                     out_indices=(1, 2, 3, 4))
        ch = self.enc.feature_info.channels()                       # strides 4, 8, 16, 32
        self.lat = nn.ModuleList([nn.Conv2d(c, 128, 1) for c in ch])
        self.smooth = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, 1, 1)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1))

    def forward(self, x):
        """x [B,3,H,W] in [-1,1] (H, W multiples of 32) -> logits [B,1,H/4,W/4]"""
        rgb = ((x + 1) / 2 - self.mean) / self.std
        f = self.enc(torch.cat([rgb, highpass(x)], 1))
        y = self.lat[-1](f[-1])
        for i in range(len(f) - 2, -1, -1):
            y = F.interpolate(y, size=f[i].shape[-2:], mode="bilinear", align_corners=False) + self.lat[i](f[i])
        return self.head(self.smooth(y))


class DecoderTraceMap:
    """feature backbone: per cell of the 24 x 24 grid the mean and max decoder-trace probability, and the image's
    top-5 % mean as a third channel (image-level evidence)"""
    native = True

    def __init__(self, split="B", max_side=1536):
        from ifl import OUTPUTS
        from ifl.features.backbones import DEV, GRID
        self.dev, self.grid, self.max_side = DEV, GRID, max_side
        ck = torch.load(OUTPUTS / "dtrace" / f"dtrace_{split}.pt", map_location="cpu", weights_only=False)
        self.net = TraceNet(pretrained=False)
        self.net.load_state_dict(ck["state_dict"])
        self.net = self.net.to(DEV).eval()

    @torch.no_grad()
    def __call__(self, x_uint8):
        out = []
        for x in x_uint8:
            x = x[None].to(self.dev).float() / 127.5 - 1
            H, W = x.shape[-2:]
            xp = F.pad(x, (0, (-W) % 32, 0, (-H) % 32), mode="reflect")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(self.net(xp).float())
            p = F.interpolate(p, size=xp.shape[-2:], mode="bilinear", align_corners=False)[..., :H, :W]
            mean = F.adaptive_avg_pool2d(p, self.grid)[0, 0]
            mx = F.adaptive_max_pool2d(p, self.grid)[0, 0]
            flat = p.flatten()
            top = flat.topk(max(1, flat.numel() // 20)).values.mean()
            out.append(torch.stack([mean, mx, top.expand_as(mean)], -1))
        patch = torch.stack(out).float()
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()
