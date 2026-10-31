"""Proposal 2 - round-trip vs. de-novo: second-order VAE statistics per patch.

In a fully regenerated (FR) edit, background pixels are one VAE encode->decode of camera content while
edited pixels were sampled by the denoiser and then decoded. A single round-trip error (AEROBLADE) sees
both as "on the decoder manifold". The hypothesis is that the two classes *converge differently* under
repeated round-trips (cf. double-JPEG idempotence), and that their latents differ in local statistics:

    x0 = x,  z0 = E(x0),  x1 = D(z0),  z1 = E(x1),  x2 = D(z1)

Per 24x24 cell (all in log space where it is a magnitude):
    e1_lo, e1_hi      first-pass error |x1-x0|, low / high frequency band
    e2_lo, e2_hi      second-pass error |x2-x1|
    r_lo,  r_hi       convergence ratios log(e2/e1)  (key statistic)
    zdrift            ||z1-z0|| / ||z0||  latent idempotence gap
    zstd              spatial std of z0 inside the cell (latent texture)
    zchan             channel-wise spread of z0 (effective use of latent channels)
    lad0, lad_ratio   local absolute-difference energy of x0 and log(LAD(x1)/LAD(x0))  (FLAME-style gap)
D = 11 per VAE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ifl.features.backbones import DEV, GRID

VAES = {
    "sd": ("stabilityai/sd-vae-ft-mse", None, "kl"),
    "sdxl": ("madebyollin/sdxl-vae-fp16-fix", None, "kl"),
    "qwen": ("Qwen/Qwen-Image", "vae", "qwen"),
}
FEATURE_NAMES = ["e1_lo", "e1_hi", "e2_lo", "e2_hi", "r_lo", "r_hi", "zdrift", "zstd", "zchan", "lad0",
                 "lad_ratio"]


def _gauss(x, sigma=1.5):
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-t ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(2, 3), groups=x.shape[1])


def _lad(x):
    g = x.mean(1, keepdim=True)
    return (g[..., 1:, :-1] - g[..., :-1, :-1]).abs() + (g[..., :-1, 1:] - g[..., :-1, :-1]).abs()


class RoundTrip:
    native = True

    def __init__(self, vae="sd"):
        repo, sub, kind = VAES[vae]
        self.kind = kind
        if kind == "qwen":
            from diffusers import AutoencoderKLQwenImage
            self.vae = AutoencoderKLQwenImage.from_pretrained(repo, subfolder=sub, torch_dtype=torch.float16)
        else:
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained(repo, subfolder=sub, torch_dtype=torch.float16)
        self.vae = self.vae.to(DEV).eval()

    def _enc(self, x):
        if self.kind == "qwen":
            return self.vae.encode(x[:, :, None]).latent_dist.mode()[:, :, 0]
        return self.vae.encode(x).latent_dist.mode()

    def _dec(self, z):
        if self.kind == "qwen":
            return self.vae.decode(z[:, :, None]).sample[:, :, 0]
        return self.vae.decode(z).sample

    @torch.no_grad()
    def roundtrip(self, x):
        """x in [-1,1], B3HW (H,W multiples of 8) -> x1, x2, z0, z1 (float32)"""
        z0 = self._enc(x.half())
        x1 = self._dec(z0).float().clamp(-1, 1)
        z1 = self._enc(x1.half())
        x2 = self._dec(z1).float().clamp(-1, 1)
        return x1, x2, z0.float(), z1.float()

    @torch.no_grad()
    def __call__(self, x_uint8):
        x = x_uint8.to(DEV).float() / 127.5 - 1
        H, W = x.shape[-2:]
        ph, pw = (-H) % 16, (-W) % 16
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        x1, x2, z0, z1 = self.roundtrip(xp)
        x1, x2 = x1[..., :H, :W], x2[..., :H, :W]
        pool = lambda t: F.adaptive_avg_pool2d(t, GRID)  # noqa: E731
        eps = 1e-5

        def bands(d):
            lo = _gauss(d)
            return pool(lo.abs().mean(1, keepdim=True)), pool((d - lo).abs().mean(1, keepdim=True))

        e1_lo, e1_hi = bands(x1 - x)
        e2_lo, e2_hi = bands(x2 - x1)
        # latent maps are (H+ph)/8 x (W+pw)/8; crop the padded part before pooling
        lh, lw = -(-H // 8), -(-W // 8)
        z0c, z1c = z0[..., :lh, :lw], z1[..., :lh, :lw]
        zdrift = pool((z1c - z0c).norm(dim=1, keepdim=True)) / (pool(z0c.norm(dim=1, keepdim=True)) + eps)
        zm = pool(z0c)
        zstd = (pool(z0c ** 2) - zm ** 2).clamp_min(0).sqrt().mean(1, keepdim=True)
        zchan = zm.std(1, keepdim=True)
        lad0 = pool(F.pad(_lad(x), (0, 1, 0, 1), mode="replicate"))
        lad1 = pool(F.pad(_lad(x1), (0, 1, 0, 1), mode="replicate"))
        L = lambda t: (t + eps).log()  # noqa: E731
        f = torch.cat([L(e1_lo), L(e1_hi), L(e2_lo), L(e2_hi), L(e2_lo) - L(e1_lo), L(e2_hi) - L(e1_hi),
                       L(zdrift), L(zstd), L(zchan), L(lad0), L(lad1) - L(lad0)], 1)
        patch = f.permute(0, 2, 3, 1)
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()
