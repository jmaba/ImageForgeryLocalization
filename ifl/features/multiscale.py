"""Sparse directional multiscale representations -> dense per-cell statistics (native resolution).

Builds on DT-CWT noise-inconsistency inpainting detection (Barglazan & Brad, arXiv 2408.06429 / 2409.17023)
and adds natural-image statistics in the spirit of Portilla & Simoncelli (2000):

for each scale j (complex subbands c_{j,o}, o = orientations), per 24x24 cell:
    logE_j     log mean |c|^2                                  (energy)
    aniso_j    std_o(log E_{j,o})                              (directional structure)
    sparse_j   mean|c| / sqrt(mean|c|^2)                       (L1/L2: 1 = flat, small = sparse/heavy-tailed)
    decay_j    logE_j - logE_{j+1}                             (scale-invariance slope, j < J)
    phcoh_j    |sum c_j conj(c~_{j+1})| / sum |c_j||c~_{j+1}|  (cross-scale phase coherence; c~ = coarse
                                                                coefficient upsampled with doubled phase)
plus  noise    log MAD(|c_1|)/0.6745                           (finest-level noise sigma, DT-CWT only)

Extractors:  dtcwt  (pytorch_wavelets, 4 levels x 6 orientations)
             shear  (FFT cone-adapted shearlet-like frame, 3 scales x 8/8/16 directions, analytic filters)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ifl.features.backbones import DEV, GRID


def _gray(x_uint8):
    x = x_uint8.to(DEV).float() / 255
    return (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]


def scale_stats(bands: list[tuple[torch.Tensor, torch.Tensor]], noise: torch.Tensor | None = None) -> torch.Tensor:
    """bands[j] = (re, im), each [B, O_j, H_j, W_j], fine -> coarse. Returns [B, D, GRID, GRID].
    Complex numbers are kept as explicit real/imag pairs: complex element-wise kernels need NVRTC/jiterator,
    which is broken on some torch/CUDA installs."""
    pool = lambda t: F.adaptive_avg_pool2d(t, GRID)  # noqa: E731
    eps = 1e-8
    feats, logE = [], []
    for j, (re, im) in enumerate(bands):
        mag = torch.sqrt(re ** 2 + im ** 2)
        e_o = pool(mag ** 2)                                   # B,O,G,G
        le = (e_o.mean(1, keepdim=True) + eps).log()
        logE.append(le)
        feats += [le, (e_o + eps).log().std(1, keepdim=True),
                  pool(mag).mean(1, keepdim=True) / (e_o.mean(1, keepdim=True) + eps).sqrt()]
    for j in range(len(bands) - 1):
        feats.append(logE[j] - logE[j + 1])
        (fr, fi), (cr, ci) = bands[j], bands[j + 1]
        if fr.shape[1] > cr.shape[1]:  # finer scale has more directions: average children of each parent
            k = fr.shape[1] // cr.shape[1]
            fr = fr.reshape(fr.shape[0], -1, k, *fr.shape[2:]).mean(2)
            fi = fi.reshape(fi.shape[0], -1, k, *fi.shape[2:]).mean(2)
        ur = F.interpolate(cr, size=fr.shape[-2:], mode="nearest")
        ui = F.interpolate(ci, size=fr.shape[-2:], mode="nearest")
        um = torch.sqrt(ur ** 2 + ui ** 2) + eps
        dr, di = (ur ** 2 - ui ** 2) / um, 2 * ur * ui / um    # coarse coefficient with doubled phase
        # f * conj(d)
        pr, pi_ = fr * dr + fi * di, fi * dr - fr * di
        num = pool(pr).sum(1, keepdim=True) ** 2 + pool(pi_).sum(1, keepdim=True) ** 2
        den = pool(torch.sqrt(fr ** 2 + fi ** 2) * torch.sqrt(dr ** 2 + di ** 2)).sum(1, keepdim=True)
        feats.append(num.sqrt() / (den + eps))
    if noise is not None:
        feats.append(noise)
    return torch.cat(feats, 1)


class DTCWTStats:
    native = True

    def __init__(self, J=4):
        from pytorch_wavelets import DTCWTForward
        self.J = J
        self.xfm = DTCWTForward(J=J, biort="near_sym_b", qshift="qshift_b").to(DEV)

    @torch.no_grad()
    def __call__(self, x_uint8):
        g = _gray(x_uint8)
        _, yh = self.xfm(g)
        bands = [(h[:, 0, ..., 0], h[:, 0, ..., 1]) for h in yh]                # (re, im), each B,6,Hj,Wj
        # robust (MAD) noise sigma of the finest level inside each cell: resample |c1| to 8x8 px per cell
        mag1 = torch.sqrt(bands[0][0] ** 2 + bands[0][1] ** 2).mean(1, keepdim=True)
        a1 = F.interpolate(mag1, size=(8 * GRID, 8 * GRID), mode="nearest")
        cells = F.unfold(a1, kernel_size=8, stride=8)                            # B, 64, G*G
        noise = (cells.median(1).values / 0.6745 + 1e-6).log().view(-1, 1, GRID, GRID)
        f = scale_stats(bands, noise)
        patch = f.permute(0, 2, 3, 1)
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


class ShearStats:
    """Cone-adapted, shearlet-like directional frame in the Fourier domain. Radial Meyer-type bands
    (dyadic), angular windows parametrised by slope (shear) inside the horizontal / vertical cones,
    one-sided in frequency so each subband is analytic (complex) and carries phase."""
    native = True

    def __init__(self, dirs=(16, 8, 8)):  # finest scale first; shearlets refine direction with scale
        self.dirs = dirs
        self._cache = {}

    @staticmethod
    def _meyer(t):  # smooth step 0->1 on [0,1]
        t = t.clamp(0, 1)
        return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)

    def _filters(self, H, W):
        key = (H, W)
        if key in self._cache:
            return self._cache[key]
        fy = torch.fft.fftfreq(H, device=DEV).view(-1, 1) * 2      # in [-1, 1), 1 = Nyquist
        fx = torch.fft.fftfreq(W, device=DEV).view(1, -1) * 2
        r = torch.sqrt(fx ** 2 + fy ** 2)
        theta = torch.atan2(fy, fx)                                 # (-pi, pi]
        S = len(self.dirs)
        filt = []
        for s in range(S):                                          # s=0 finest
            hi, lo = 1.0 / 2 ** s, 1.0 / 2 ** (s + 1)
            radial = torch.sqrt(self._meyer((r - lo / 2) / (lo / 2)) * (1 - self._meyer((r - hi) / hi)))
            radial = torch.where(r > hi * 1.5, torch.zeros_like(r), radial)
            n = self.dirs[s]
            fs = []
            for k in range(n):                                      # directions over a half-plane (analytic)
                c = -math.pi / 2 + (k + 0.5) * math.pi / n
                d = torch.remainder(theta - c + math.pi, 2 * math.pi) - math.pi
                w = torch.cos(d.clamp(-math.pi / n, math.pi / n) * n / 2) ** 2
                w = torch.where(d.abs() < math.pi / n, w, torch.zeros_like(w))
                fs.append(radial * w)
            filt.append(torch.stack(fs))                            # n,H,W
        self._cache = {key: filt}
        return filt

    @torch.no_grad()
    def __call__(self, x_uint8):
        g = _gray(x_uint8)
        g = g - g.mean((-1, -2), keepdim=True)
        H, W = g.shape[-2:]
        Gr = torch.view_as_real(torch.fft.fft2(g[:, 0]))            # B,H,W,2 (no complex element-wise ops)
        bands = []
        for s, fl in enumerate(self._filters(H, W)):
            spec = Gr[:, None] * fl[None, ..., None]                # B,n,H,W,2
            c = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(spec.contiguous())))
            step = 2 ** s                                           # decimate coarse scales (cheaper pooling)
            c = c[:, :, ::step, ::step] * (2 * step)
            bands.append((c[..., 0].contiguous(), c[..., 1].contiguous()))
        f = scale_stats(bands)
        patch = f.permute(0, 2, 3, 1)
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()
