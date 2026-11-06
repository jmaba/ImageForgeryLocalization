"""Frozen feature extractors. Every extractor maps a batch of images to
    global  [B, D]          (image-level descriptor)
    patch   [B, G, G, D]    (dense descriptor on a shared G x G grid, default G = 24)

Spec strings: ``<name>`` or ``<name>:l<k>`` to read hidden layer k instead of the last one
(intermediate layers often carry more low-level / forensic signal than the last).

Foundation models (semantic latents, resized input):
  dinov2      facebook/dinov2-with-registers-base    336px / p14 -> 24x24
  dinov3      facebook/dinov3-vitb16-pretrain-lvd1689m 384px / p16 (gated: needs HF token + licence)
  siglip2     google/siglip2-base-patch16-384        384px / p16
  clip        openai/clip-vit-base-patch16           384px / p16 (pos-emb interpolated)
Low-level streams (native resolution, no learned weights / generic VAE):
  srm         SRM + Laplacian residual statistics per cell
  vaerec      AEROBLADE-style SD-VAE reconstruction error per cell
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

GRID = 24
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def parse_spec(spec: str):
    name, _, layer = spec.partition(":")
    return name, (int(layer[1:]) if layer else None)


class _HFViT:
    native = False

    def __init__(self, repo, size, patch, n_prefix, layer, kind="auto", mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        from transformers import AutoModel
        self.size, self.patch, self.n_prefix, self.layer, self.kind = size, patch, n_prefix, layer, kind
        try:
            m = AutoModel.from_pretrained(repo, dtype=torch.float16, use_safetensors=True)
        except OSError:  # repo without safetensors (needs torch>=2.6 for .bin)
            m = AutoModel.from_pretrained(repo, dtype=torch.float16)
        if kind == "clip":
            m = m.vision_model
        elif kind == "siglip":
            m = m.vision_model
        self.m = m.to(DEV).eval()
        self.mean = torch.tensor(mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=DEV).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x_uint8):
        x = ((x_uint8.to(DEV).float() / 255 - self.mean) / self.std).half()
        kw = dict(output_hidden_states=True)
        if self.kind in ("clip", "siglip"):
            kw["interpolate_pos_encoding"] = True
        out = self.m(pixel_values=x, **kw)
        hs = out.hidden_states[self.layer] if self.layer is not None else out.last_hidden_state
        g = self.size // self.patch
        tok = hs[:, self.n_prefix:, :]
        patch = tok.reshape(tok.shape[0], g, g, -1)
        if self.n_prefix:  # CLS token
            glob = hs[:, 0]
        else:  # SigLIP has no CLS: mean-pool patches (pooler head is trained for text alignment only)
            glob = tok.mean(1)
        if g != GRID:
            patch = F.adaptive_avg_pool2d(patch.permute(0, 3, 1, 2).float(), GRID).permute(0, 2, 3, 1)
        return glob.float().cpu().numpy(), patch.float().cpu().numpy()


# --------------------------------------------------------------------------- low-level streams
def _srm_bank():
    k1 = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4
    k2 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]]) / 12
    k3 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2
    k4 = k3.T
    return torch.tensor(np.stack([k1, k2, k3, k4]), dtype=torch.float32)[:, None]


class SRMStats:
    """Per-cell statistics of high-pass residuals on the grey native image:
    for each of 4 SRM kernels: log-std, kurtosis, mean |r|/std, lag-1 horizontal & vertical autocorr.
    Plus per-cell luminance mean/std (content control) -> D = 4*5 + 2 = 22."""
    native = True

    def __init__(self, layer=None):
        self.k = _srm_bank().to(DEV)

    @torch.no_grad()
    def __call__(self, x_uint8):
        x = x_uint8.to(DEV).float() / 255
        y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])[:, None]
        r = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), self.k)  # B,4,H,W
        pool = lambda t: F.adaptive_avg_pool2d(t, GRID)  # noqa: E731
        m1, m2 = pool(r), pool(r ** 2)
        var = (m2 - m1 ** 2).clamp_min(1e-10)
        m4 = pool((r - F.interpolate(m1, size=r.shape[-2:], mode="nearest")) ** 4)
        kurt = (m4 / var ** 2).clamp(0, 50)
        mabs = pool(r.abs()) / var.sqrt()
        ah = pool(r[..., :, 1:] * r[..., :, :-1]) / var
        av = pool(r[..., 1:, :] * r[..., :-1, :]) / var
        lum = torch.cat([pool(y), (pool(y ** 2) - pool(y) ** 2).clamp_min(0).sqrt()], 1)
        f = torch.cat([0.5 * var.log(), kurt.log1p(), mabs, ah, av, lum], 1)  # B,22,G,G
        patch = f.permute(0, 2, 3, 1)
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


class VAERecon:
    """AEROBLADE-style: reconstruct with the SD VAE (8x latent) and summarise the error per cell.
    Images produced/edited by latent-diffusion models lie close to the VAE manifold -> lower error.
    Features per cell: log mean |err| per RGB channel, log high-pass err, err/local-contrast ratio,
    log latent-norm -> D = 6."""
    native = True

    def __init__(self, layer=None, repo="stabilityai/sd-vae-ft-mse"):
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(repo, torch_dtype=torch.float16).to(DEV).eval()
        self.lap = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=DEV).view(1, 1, 3, 3)

    @torch.no_grad()
    def __call__(self, x_uint8):
        x = x_uint8.to(DEV).float() / 127.5 - 1
        H, W = x.shape[-2:]
        ph, pw = (-H) % 8, (-W) % 8
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect")
        z = self.vae.encode(xp.half()).latent_dist.mean
        rec = self.vae.decode(z).sample.float()[..., :H, :W].clamp(-1, 1)
        err = (rec - x).abs()
        pool = lambda t: F.adaptive_avg_pool2d(t, GRID)  # noqa: E731
        g = x.mean(1, keepdim=True)
        hp = F.conv2d(F.pad((rec - x).mean(1, keepdim=True), (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        contrast = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="reflect"), self.lap).abs()
        zn = F.interpolate(z.float().norm(dim=1, keepdim=True), size=(H + ph, W + pw), mode="nearest")[..., :H, :W]
        f = torch.cat([(pool(err) + 1e-4).log(), (pool(hp) + 1e-4).log(),
                       (pool(hp) / (pool(contrast) + 1e-3)), pool(zn).log()], 1)
        patch = f.permute(0, 2, 3, 1)
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


class PixelStats:
    """CONTROL stream: per-cell RGB mean/std + a 3x3 thumbnail of the cell (no forensic or semantic
    knowledge). Any backbone that cannot beat this on localization is reading colour/layout only.
    D = 6 + 27 = 33."""
    native = False
    size = 216  # 24 cells x 9 px

    def __init__(self, layer=None):
        pass

    @torch.no_grad()
    def __call__(self, x_uint8):
        x = x_uint8.float() / 255
        cells = x.unfold(2, 9, 9).unfold(3, 9, 9)  # B,3,G,G,9,9
        mean, std = cells.mean((-1, -2)), cells.std((-1, -2))
        thumb = F.adaptive_avg_pool2d(cells.reshape(-1, 1, 9, 9), 3).reshape(*cells.shape[:4], 9)
        f = torch.cat([mean, std, thumb.permute(0, 1, 4, 2, 3).reshape(x.shape[0], 27, GRID, GRID)], 1)
        patch = f.permute(0, 2, 3, 1)
        return patch.mean((1, 2)).numpy(), patch.numpy()


def build(spec: str):
    name, layer = parse_spec(spec)
    if name == "pixels":
        return PixelStats()
    if name == "dinov2":
        return _HFViT("facebook/dinov2-with-registers-base", 336, 14, 5, layer)
    if name == "dinov3":
        return _HFViT("facebook/dinov3-vitb16-pretrain-lvd1689m", 384, 16, 5, layer)
    if name == "siglip2":
        return _HFViT("google/siglip2-base-patch16-384", 384, 16, 0, layer, kind="siglip",
                      mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    if name == "clip":
        return _HFViT("openai/clip-vit-base-patch16", 384, 16, 1, layer, kind="clip",
                      mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))
    if name == "srm":
        return SRMStats()
    if name == "vaerec":
        return VAERecon()
    if name == "lift":  # self-supervised real-only learned lifting wavelet (ifl/ssl/lifting.py)
        from ifl.ssl.lifting import LiftStats
        return LiftStats()
    if name in ("dtcwt", "shear"):  # sparse directional multiscale statistics (ifl/features/multiscale.py)
        from ifl.features.multiscale import DTCWTStats, ShearStats
        return DTCWTStats() if name == "dtcwt" else ShearStats()
    if name in ("dtcwt_phase", "fft_band", "gabor_opp", "medres_lab"):  # ported streams (docs/04)
        from ifl.features import legacy
        return {"dtcwt_phase": legacy.DTCWTPhase, "fft_band": legacy.FFTBand,
                "gabor_opp": legacy.GaborOpp, "medres_lab": legacy.MedResLab}[name]()
    if name in ("psteer", "wph"):  # steerable-pyramid PS stats / scattering spectra + phase harmonics
        from ifl.features.scattering import SteerPyrPS, WPHScat
        return SteerPyrPS() if name == "psteer" else WPHScat()
    if name == "physics":  # lighting/shading + defocus-vs-depth consistency (ifl/features/physics.py)
        from ifl.features.physics import PhysicsStats
        return PhysicsStats()
    if name in ("trufor", "mesorch"):  # published supervised localizers as baselines (ifl/features/baselines.py)
        from ifl.features.baselines import MesorchMap, TruForMap
        return TruForMap() if name == "trufor" else MesorchMap()
    if name == "flame":  # FLAME (arXiv:2606.02178), local high-frequency energy + SAM2 adapter
        from ifl.features.baselines import FlameMap
        return FlameMap()
    if name in ("bfree", "bfreeimg"):  # B-Free (CVPR 2025) detector; "bfree" adds a sliding-window map
        from ifl.features.baselines import BFreeMap
        return BFreeMap(local=name == "bfree")
    if name in ("dtraceA", "dtraceB"):  # decoder-trace detector trained on autoencoder round trips (docs/09)
        from ifl.features.dtrace import DecoderTraceMap
        return DecoderTraceMap(name[-1])
    if name.startswith("rt2-"):  # Proposal 2: double round-trip through the named VAE (rt2-sd, rt2-qwen ...)
        from ifl.features.roundtrip import RoundTrip
        return RoundTrip(name[4:])
    raise KeyError(spec)


def input_size(extractor):
    return None if extractor.native else extractor.size
