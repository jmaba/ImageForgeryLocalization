"""Local (per-cell) texture statistics from complex wavelet frames, after two lines of work:

  psteer  Complex steerable pyramid + Portilla & Simoncelli (2000) statistics. Analytic one-sided Fourier
          filters: octave raised-cosine radial bands x cos^(K-1) angular windows, K = 4 orientations, J = 4
          scales on L*. Per cell and scale: log energy, sparsity log(E|c|^4 / (E|c|^2)^2), mean inter-orientation
          magnitude correlation; between scales j, j+1: magnitude correlation and relative-phase coherence
          |E c_j conj(c~_{j+1})| / E|c_j||c_{j+1}| (c~ = coarse coefficient with doubled phase).        D = 18

  wph     Scattering spectra (Morel et al. 2022; Cheng et al., PNAS Nexus 2024) and wavelet phase harmonics
          (Mallat, Zhang & Rochette 2020; Brochard, Zhang & Mallat, ICLR 2022), computed in local windows.
          Analytic log-Gabor bank (J = 4, L = 4). Per cell:
            sparsity            E|W_j x| ^2 / E|W_j x|^2                                   j = 1..4        (4)
            scattering ratios   E|W_j' |W_j x|| / E|W_j x|           (j, j') = (1,2) (1,3) (2,3) (2,4) (3,4) (5)
            envelope corr.      corr(|W_j x|, |W_j+1 x|)                                   j = 1..3        (3)
            phase harmonic k=2  |E [W_j x]^2 conj(W_j+1 x)| / E|W_j x||W_j+1 x|             j = 1..3        (3)
            colour coupling k=1 |E W_j L conj(W_j c)| / E|W_j L||W_j c|,  c in {a*, b*}     j = 1, 2        (4)
            chroma log energy   log E|W_j c|^2                                              j = 1, 2        (4)
                                                                                                       D = 23
          [z]^k = |z| exp(i k arg z) is the phase harmonic; k = 2 aligns the phase of an octave-finer coefficient
          with the coarser one, the dependency that edges create and that energy statistics cannot see.

Complex arithmetic is done on explicit (re, im) pairs: complex element-wise CUDA kernels need NVRTC, which is
broken on some torch installs (see multiscale.py).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ifl.features.backbones import DEV, GRID
from ifl.features.legacy import srgb_to_lab

EPS = 1e-8


def pool(t):
    return F.adaptive_avg_pool2d(t, GRID)


def _out(f):
    patch = f.permute(0, 2, 3, 1)
    return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


def _freqs(H, W):
    fy = torch.fft.fftfreq(H, device=DEV).view(-1, 1) * 2           # [-1, 1), 1 = Nyquist
    fx = torch.fft.fftfreq(W, device=DEV).view(1, -1) * 2
    return torch.sqrt(fx ** 2 + fy ** 2), torch.atan2(fy, fx)


def _filter(x, filt):
    """x B,1,H,W real; filt n,H,W real Fourier filters -> (re, im) each B,n,H,W."""
    X = torch.view_as_real(torch.fft.fft2(x[:, 0]))[:, None]          # B,1,H,W,2
    c = torch.view_as_real(torch.fft.ifft2(torch.view_as_complex((X * filt[None, ..., None]).contiguous())))
    return c[..., 0].contiguous(), c[..., 1].contiguous()


def _corr(a, b):
    """per-cell Pearson correlation of two maps B,1,H,W (pooled moments)."""
    ma, mb = pool(a), pool(b)
    va = (pool(a * a) - ma ** 2).clamp_min(EPS)
    vb = (pool(b * b) - mb ** 2).clamp_min(EPS)
    return (pool(a * b) - ma * mb) / torch.sqrt(va * vb)


def _phase_coh(fr, fi, cr, ci, k=2):
    """|E [f]^1 conj([c_up]^k ...)|: coherence between a fine coefficient and the coarser one with its phase
    multiplied by k (k = 2 for an octave). Inputs B,n,H,W at the same size; returns per-cell value B,1,G,G."""
    cm = torch.sqrt(cr ** 2 + ci ** 2) + EPS
    ang = torch.atan2(ci, cr) * k
    dr, di = cm * torch.cos(ang), cm * torch.sin(ang)                 # [c]^k (modulus kept)
    pr, pi_ = fr * dr + fi * di, fi * dr - fr * di                    # f * conj([c]^k)
    num = torch.sqrt(pool(pr).sum(1, keepdim=True) ** 2 + pool(pi_).sum(1, keepdim=True) ** 2)
    den = pool(torch.sqrt(fr ** 2 + fi ** 2) * cm).sum(1, keepdim=True)
    return num / (den + EPS)


class _Bank:
    """Analytic directional filters (one-sided in frequency => complex coefficients), cached per image size."""

    def __init__(self, J, K, kind):
        self.J, self.K, self.kind, self._cache = J, K, kind, {}

    def __call__(self, H, W):
        if (H, W) in self._cache:
            return self._cache[(H, W)]
        r, th = _freqs(H, W)
        bank = []
        for j in range(self.J):                                       # j = 0 finest
            fc = 0.5 / 2 ** j                                         # centre frequency (1 = Nyquist)
            if self.kind == "steer":                                  # octave raised-cosine in log2 radius
                lr = torch.log2(r.clamp_min(1e-6) / fc)
                radial = torch.where(lr.abs() < 1, torch.cos(lr * math.pi / 2), torch.zeros_like(lr))
            else:                                                     # log-Gabor, ~1 octave bandwidth
                radial = torch.exp(-(torch.log(r.clamp_min(1e-6) / fc)) ** 2 / (2 * math.log(1.5) ** 2))
            radial = torch.where(r > 0, radial, torch.zeros_like(radial))
            fs = []
            for k in range(self.K):                                   # orientations over a half plane
                c = k * math.pi / self.K
                d = torch.remainder(th - c + math.pi, 2 * math.pi) - math.pi
                ang = torch.cos(d).clamp_min(0) ** (self.K - 1)       # one-sided => analytic
                fs.append(2 * radial * ang)
            bank.append(torch.stack(fs))
        if len(self._cache) > 6:
            self._cache.clear()
        self._cache[(H, W)] = bank
        return bank


class SteerPyrPS:
    native = True

    def __init__(self, J=4, K=4):
        self.J, self.K, self.bank = J, K, _Bank(J, K, "steer")

    @torch.no_grad()
    def __call__(self, x_uint8):
        L = srgb_to_lab(x_uint8)[:, :1] / 100
        L = L - L.mean((-1, -2), keepdim=True)
        bands = [_filter(L, f) for f in self.bank(*L.shape[-2:])]
        feats = []
        for re, im in bands:
            m2 = re ** 2 + im ** 2
            e2 = pool(m2.mean(1, keepdim=True))
            e4 = pool((m2 ** 2).mean(1, keepdim=True))
            mag = torch.sqrt(m2 + EPS)
            oc = torch.stack([_corr(mag[:, a:a + 1], mag[:, b:b + 1])
                              for a in range(self.K) for b in range(a + 1, self.K)]).mean(0)
            feats += [(e2 + EPS).log(), (e4 / (e2 ** 2 + EPS) + EPS).log(), oc]
        for j in range(self.J - 1):
            (fr, fi), (cr, ci) = bands[j], bands[j + 1]
            fm, cm = torch.sqrt(fr ** 2 + fi ** 2 + EPS), torch.sqrt(cr ** 2 + ci ** 2 + EPS)
            feats.append(torch.stack([_corr(fm[:, o:o + 1], cm[:, o:o + 1]) for o in range(self.K)]).mean(0))
            feats.append(_phase_coh(fr, fi, cr, ci, k=2))
        return _out(torch.cat(feats, 1))


class WPHScat:
    native = True

    def __init__(self, J=4, K=4):
        self.J, self.K, self.bank = J, K, _Bank(J, K, "loggabor")

    @torch.no_grad()
    def __call__(self, x_uint8):
        lab = srgb_to_lab(x_uint8) / 100
        lab = lab - lab.mean((-1, -2), keepdim=True)
        bank = self.bank(*lab.shape[-2:])
        L = lab[:, :1]
        W = [_filter(L, f) for f in bank]                             # J x (re, im) B,K,H,W
        mod = [torch.sqrt(re ** 2 + im ** 2 + EPS) for re, im in W]   # |W_j x|
        feats = []
        s1 = [pool(m.mean(1, keepdim=True)) for m in mod]             # E|W_j x|
        for j, m in enumerate(mod):                                   # sparsity (first-order scattering spectra)
            feats.append(s1[j] ** 2 / (pool((m ** 2).mean(1, keepdim=True)) + EPS))
        env = [m.mean(1, keepdim=True) for m in mod]                  # orientation-averaged envelopes
        for j, jp in ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3)):        # second-order scattering ratios
            er, ei = _filter(env[j], bank[jp])
            s2 = pool(torch.sqrt(er ** 2 + ei ** 2 + EPS).mean(1, keepdim=True))
            feats.append(s2 / (s1[j] + EPS))
        for j in range(self.J - 1):                                   # envelope correlation across scales
            feats.append(_corr(env[j], env[j + 1]))
        for j in range(self.J - 1):                                   # phase harmonic k = 2 across an octave
            (fr, fi), (cr, ci) = W[j], W[j + 1]
            feats.append(_phase_coh(fr, fi, cr, ci, k=2))
        for c in (1, 2):                                              # L*-chroma coupling (k = 1) and chroma energy
            Wc = [_filter(lab[:, c:c + 1], bank[j]) for j in range(2)]
            for j in range(2):
                (lr, li), (cr, ci) = W[j], Wc[j]
                feats.append(_phase_coh(lr, li, cr, ci, k=1))
                feats.append((pool((cr ** 2 + ci ** 2).mean(1, keepdim=True)) + EPS).log())
        return _out(torch.cat(feats, 1))
