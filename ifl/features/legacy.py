"""Streams ported from the earlier projects (docs/04_prior_code_audit.md, section B), with the defects fixed:
correct CIE Lab (no uint8 offset), zero-mean Gabor kernels with sigma tied to the wavelength, phase treated
as a circular quantity, and no per-image min-max rescaling (absolute noise level is the forensic quantity).
All run on the native image (capped by extract_features.py) and are pooled to the shared 24x24 grid.

  dtcwt_phase  DT-CWT (3 levels) on L*: per level log energy and inter-orientation circular phase variance,
               amplitude-weighted cross-scale phase difference |angle(c_j conj c_{j+1})| and log amplitude
               ratio; level-1 energy / circular variance / cross-scale phase on a*, b*            D = 16
  fft_band     windowed 32x32 block spectra of L*: log power in 6 radial bands + high-frequency share D = 7
  gabor_opp    zero-mean complex Gabor bank (4 frequencies x 4 orientations) on L*, a*, b*: log magnitude
               per channel and frequency, plus magnitude-weighted L*-a* / L*-b* phase coherence    D = 20
  medres_lab   5x5 median-filter residual of L*, a*, b*: log std, log1p kurtosis, lag-1 h/v autocorr D = 12
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ifl.features.backbones import DEV, GRID

EPS = 1e-8


def pool(t):
    return F.adaptive_avg_pool2d(t, GRID)


def srgb_to_lab(x_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 B,3,H,W sRGB -> float B,3,H,W CIE Lab (D65): L in [0,100], a/b roughly [-128,127]."""
    v = x_uint8.to(DEV).float() / 255
    lin = torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]], device=DEV)
    xyz = torch.einsum("ij,bjhw->bihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], device=DEV).view(1, 3, 1, 1)
    d = 6 / 29
    f = torch.where(xyz > d ** 3, xyz.clamp_min(1e-12) ** (1 / 3), xyz / (3 * d ** 2) + 4 / 29)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def _out(f):
    patch = f.permute(0, 2, 3, 1)
    return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


class DTCWTPhase:
    native = True

    def __init__(self, J=3):
        from pytorch_wavelets import DTCWTForward
        self.J = J
        self.xfm = DTCWTForward(J=J, biort="near_sym_b", qshift="qshift_b").to(DEV)

    def _levels(self, ch):
        _, yh = self.xfm(ch)
        return [(h[:, 0, ..., 0], h[:, 0, ..., 1]) for h in yh]              # (re, im) B,6,Hj,Wj

    @staticmethod
    def _circvar(re, im):
        """1 - |mean_o exp(i phi_o)| per position (phase only, unit vectors)."""
        m = torch.sqrt(re ** 2 + im ** 2) + EPS
        return 1 - torch.sqrt((re / m).mean(1, keepdim=True) ** 2 + (im / m).mean(1, keepdim=True) ** 2)

    @staticmethod
    def _cross(fine, coarse):
        """amplitude-weighted mean |angle(c_f * conj(up(c_c)))| and mean log amplitude ratio, per position."""
        (fr, fi), (cr, ci) = fine, coarse
        ur = F.interpolate(cr, size=fr.shape[-2:], mode="bilinear", align_corners=False)
        ui = F.interpolate(ci, size=fr.shape[-2:], mode="bilinear", align_corners=False)
        pr, pi_ = fr * ur + fi * ui, fi * ur - fr * ui
        w = torch.sqrt(fr ** 2 + fi ** 2) * torch.sqrt(ur ** 2 + ui ** 2) + EPS
        dphi = torch.atan2(pi_, pr).abs()
        ratio = (torch.sqrt(fr ** 2 + fi ** 2) + EPS).log() - (torch.sqrt(ur ** 2 + ui ** 2) + EPS).log()
        return (dphi * w).sum(1, keepdim=True), w.sum(1, keepdim=True), ratio.mean(1, keepdim=True)

    @torch.no_grad()
    def __call__(self, x_uint8):
        lab = srgb_to_lab(x_uint8)
        feats = []
        for c, levels in ((0, self.J), (1, 2), (2, 2)):
            ch = lab[:, c:c + 1] / 100
            lv = self._levels(ch)[:levels]
            for j, (re, im) in enumerate(lv):
                if c == 0 or j == 0:
                    feats += [(pool(re ** 2 + im ** 2).mean(1, keepdim=True) + EPS).log(), pool(self._circvar(re, im))]
            for j in range(len(lv) - 1 if c == 0 else 1):
                num, den, ratio = self._cross(lv[j], lv[j + 1])
                feats.append(pool(num) / (pool(den) + EPS))
                if c == 0:
                    feats.append(pool(ratio))
        return _out(torch.cat(feats, 1))


class FFTBand:
    native = True

    def __init__(self, block=32, bands=6):
        self.block, self.bands = block, bands
        b = block
        win = torch.hann_window(b, periodic=False, device=DEV)
        self.win = (win[:, None] * win[None]).view(1, 1, b, b)
        fy = torch.fft.fftfreq(b, device=DEV).view(-1, 1)
        fx = torch.fft.rfftfreq(b, device=DEV).view(1, -1)
        r = torch.sqrt(fx ** 2 + fy ** 2) / 0.5                               # 0 .. sqrt(2), 1 = Nyquist
        edges = torch.linspace(0, 1, bands + 1, device=DEV)
        self.masks = [((r > edges[k]) & (r <= edges[k + 1])).float() for k in range(bands)]
        self.hf = (r > 0.5).float()

    @torch.no_grad()
    def __call__(self, x_uint8):
        L = srgb_to_lab(x_uint8)[:, :1] / 100
        b = self.block
        H, W = L.shape[-2:]
        L = F.pad(L, (0, (-W) % b, 0, (-H) % b), mode="reflect")
        tiles = L.unfold(2, b, b).unfold(3, b, b)                             # B,1,nh,nw,b,b
        nh, nw = tiles.shape[2:4]
        t = tiles.reshape(-1, 1, b, b)
        t = (t - t.mean((-1, -2), keepdim=True)) * self.win
        P = torch.view_as_real(torch.fft.rfft2(t)).pow(2).sum(-1)             # N,1,b,b/2+1 (no complex kernels)
        tot = P.sum((-1, -2)) + EPS
        f = [((P * m).sum((-1, -2)) / (m.sum() + EPS) + EPS).log() for m in self.masks]
        f.append((P * self.hf).sum((-1, -2)) / tot)
        f = torch.cat(f, 1).view(L.shape[0], nh, nw, -1).permute(0, 3, 1, 2)
        return _out(pool(f))


class GaborOpp:
    native = True

    def __init__(self, freqs=(0.1, 0.2, 0.3, 0.4), n_orient=4):
        self.freqs, self.n = freqs, n_orient
        self.kernels = []
        for f in freqs:
            sigma = 0.56 / f                                                  # ~1 octave half-power bandwidth
            r = int(math.ceil(3 * sigma))
            yy, xx = torch.meshgrid(torch.arange(-r, r + 1, device=DEV, dtype=torch.float32),
                                    torch.arange(-r, r + 1, device=DEV, dtype=torch.float32), indexing="ij")
            g = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
            ks = []
            for k in range(n_orient):
                th = k * math.pi / n_orient
                u = xx * math.cos(th) + yy * math.sin(th)
                ev, od = g * torch.cos(2 * math.pi * f * u), g * torch.sin(2 * math.pi * f * u)
                ev = ev - g * (ev.sum() / g.sum())                            # zero-mean even part (no DC leak)
                ks += [ev, od]
            self.kernels.append(torch.stack(ks)[:, None])                     # 2n,1,k,k

    @torch.no_grad()
    def __call__(self, x_uint8):
        lab = srgb_to_lab(x_uint8)
        chans = [lab[:, 0:1] / 100, lab[:, 1:2] / 100, lab[:, 2:3] / 100]
        mag_feats, coh_feats = [], []
        for K in self.kernels:
            r = K.shape[-1] // 2
            resp = [F.conv2d(F.pad(c, (r, r, r, r), mode="reflect"), K) for c in chans]   # B,2n,H,W
            re = [q[:, 0::2] for q in resp]
            im = [q[:, 1::2] for q in resp]
            mag = [torch.sqrt(a ** 2 + b ** 2) for a, b in zip(re, im)]
            mag_feats += [(pool(m.mean(1, keepdim=True)) + EPS).log() for m in mag]
            for c in (1, 2):                                                  # L*-a*, L*-b* phase coherence
                cos_d = (re[0] * re[c] + im[0] * im[c]) / (mag[0] * mag[c] + EPS)
                w = mag[0] * mag[c]
                coh_feats.append(pool((cos_d.abs() * w).sum(1, keepdim=True)) /
                                 (pool(w.sum(1, keepdim=True)) + EPS))
        return _out(torch.cat(mag_feats + coh_feats, 1))


class MedResLab:
    native = True

    @torch.no_grad()
    def __call__(self, x_uint8):
        lab = srgb_to_lab(x_uint8) / 100
        B, C, H, W = lab.shape
        p = F.pad(lab, (2, 2, 2, 2), mode="reflect")
        med = torch.cat([p[:, c:c + 1].unfold(2, 5, 1).unfold(3, 5, 1).reshape(B, 1, H, W, 25).median(-1).values
                         for c in range(C)], 1)
        r = lab - med
        m1, m2 = pool(r), pool(r ** 2)
        var = (m2 - m1 ** 2).clamp_min(1e-12)
        kurt = (pool((r - F.interpolate(m1, size=r.shape[-2:], mode="nearest")) ** 4) / var ** 2).clamp(0, 50)
        ah = pool(r[..., :, 1:] * r[..., :, :-1]) / var
        av = pool(r[..., 1:, :] * r[..., :-1, :]) / var
        return _out(torch.cat([0.5 * var.log(), kurt.log1p(), ah, av], 1))
