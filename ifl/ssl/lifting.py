"""LiftNet - a learned, probabilistic lifting wavelet trained on REAL photos only.

Lifting view of a 2-D wavelet level: split x into polyphase components
    e = x[0::2, 0::2]      (even / coarse, the context)
    o = {x[0::2,1::2], x[1::2,0::2], x[1::2,1::2]}   (odd / detail, the targets)
A predictor P sees ONLY e (never o) and outputs a Laplace distribution (mu, b) for every odd sample;
details d = o - mu are the wavelet coefficients. Training minimises the Laplace NLL of d, i.e. it learns the
predict step that makes camera images maximally sparse under this transform (a learned, adaptive wavelet).
Repeating on e gives a multi-level decomposition.

Masked-target prediction from context makes it JEPA-like, but in a multiscale pixel domain with an explicit
likelihood. At test time the per-pixel surprisal of a region that was synthesised (or re-decoded) by a
generator - whose fine-scale statistics given the coarse scale differ from a camera pipeline - is expected
to deviate from the image's authentic parts. Closest prior work: ZED (Cozzolino et al., ECCV 2024,
arXiv 2409.15875) uses lossless-coder surprisal for image-level detection; this is dense and multiscale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LEVELS = 3


def split(x):
    return x[..., 0::2, 0::2], torch.cat([x[..., 0::2, 1::2], x[..., 1::2, 0::2], x[..., 1::2, 1::2]], 1)


class Predictor(nn.Module):
    def __init__(self, ch=3, width=48, depth=5):
        super().__init__()
        layers, c = [], ch
        for k in range(depth):
            layers += [nn.Conv2d(c, width, 3, padding=1, padding_mode="reflect"), nn.GELU()]
            c = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, 3 * ch * 2, 1)      # (mu, log b) for 3 odd components x ch

    def forward(self, e):
        # local normalisation: predict in units of the local contrast, so the net learns structure not level
        m = F.avg_pool2d(F.pad(e, (2, 2, 2, 2), mode="reflect"), 5, 1)
        s = (F.avg_pool2d(F.pad((e - m) ** 2, (2, 2, 2, 2), mode="reflect"), 5, 1) + 1e-4).sqrt()
        h = self.head(self.body((e - m) / s))
        mu, logb = h.chunk(2, 1)
        rep = lambda t: t.repeat(1, 3, 1, 1)  # noqa: E731  (same stats for the 3 odd components)
        return rep(m) + mu * rep(s), logb + rep(s).log()


class LiftNet(nn.Module):
    def __init__(self, ch=3, levels=LEVELS, width=48, depth=5):
        super().__init__()
        self.levels = levels
        self.P = nn.ModuleList([Predictor(ch, width, depth) for _ in range(levels)])

    def forward(self, x):
        """x: [B,3,H,W] in [0,1], H,W multiples of 2**levels.
        Returns per level: (nll, z, logb) maps at that level's resolution, each [B, 3*ch, h, w]."""
        out, e = [], x
        for P in self.P:
            e, o = split(e)
            mu, logb = P(e)
            b = logb.clamp(-9, 3).exp()
            z = (o - mu) / b
            nll = z.abs() + logb.clamp(-9, 3) + 0.6931  # Laplace NLL: |d|/b + log(2b)
            out.append((nll, z, logb))
        return out

    def loss(self, x):
        return sum(nll.mean() for nll, _, _ in self(x)) / self.levels


class LiftStats:
    """Frozen LiftNet -> per-cell statistics at every level:
    mean NLL (surprisal), mean |z|, mean z^2 (tail weight), mean log b (predicted uncertainty)  -> D = 4*levels"""
    native = True

    def __init__(self, ckpt=None):
        from ifl import DATA
        from ifl.features.backbones import DEV
        self.dev = DEV
        self.net = LiftNet().to(DEV).eval()
        self.net.load_state_dict(torch.load(ckpt or DATA / "checkpoints" / "liftnet.pt", map_location=DEV))

    @torch.no_grad()
    def __call__(self, x_uint8):
        from ifl.features.backbones import GRID
        x = x_uint8.to(self.dev).float() / 255
        H, W = x.shape[-2:]
        m = 2 ** self.net.levels
        x = F.pad(x, (0, (-W) % m, 0, (-H) % m), mode="reflect")
        feats = []
        for lvl, (nll, z, logb) in enumerate(self.net(x)):
            h, w = H // 2 ** (lvl + 1), W // 2 ** (lvl + 1)        # crop the padded border
            pool = lambda t: F.adaptive_avg_pool2d(t[..., :h, :w].mean(1, keepdim=True), GRID)  # noqa: E731
            feats += [pool(nll), pool(z.abs()), pool(z ** 2).clamp_max(1e3).log1p(), pool(logb)]
        patch = torch.cat(feats, 1).permute(0, 2, 3, 1)
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()
