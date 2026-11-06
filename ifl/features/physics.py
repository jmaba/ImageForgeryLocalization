"""Physical-consistency cell features: lighting/shading and defocus-vs-depth, from frozen geometry/intrinsics models.
They do not read sensor noise or compression, so they are candidates for the edits where trace forensics is blind
(full regeneration, LLM instruction edits) and for re-shared images (low-frequency cues).

Per image (native image capped at 1024 px, models at --processing resolution 512):
  normals N   prs-eth/marigold-normals-v1-1      (diffusers MarigoldNormalsPipeline)
  shading S,  prs-eth/marigold-iid-lighting-v1-1 (diffusers MarigoldIntrinsicsPipeline: albedo, shading, residual)
  albedo A
  disparity   depth-anything/Depth-Anything-V2-Small-hf (transformers; relative inverse depth)
All maps are resampled to a 144 x 144 working grid (6 x 6 px per cell of the 24 x 24 grid).

Lighting (the Johnson-Farid lighting-environment model with automatic normals): shading luminance s = Y(N) l with
Y the 9 second-order spherical harmonics, l fitted ONCE per image by Huber-IRLS over all pixels, then per cell
    0 light_res      mean residual s - Y(N) l
    1 light_abs      mean |residual|
    2 light_ang      angle (rad) between the cell's own first-order light direction (ridge fit on its 36 px)
                     and the image's global first-order direction
    3 light_cond     smallest eigenvalue of the cell's normal covariance (how well its direction is determined)
    4,5 illum_r/g    shading chromaticity (illuminant colour) minus its image median
    6 shade_mean     mean shading luminance
    7-9 albedo rgb   mean albedo (content control)
Defocus vs depth (thin lens: blur radius ~ a |D - D_f| + b in disparity D): blur sigma by the re-blur gradient
ratio (Zhuo & Sim 2011) on the strongest 10 % gradients of the grey image (long side 512); per cell median;
a, b, D_f fitted per image (grid over D_f, least squares), then
   10 blur_sigma     11 disparity     12 blur_res (sigma - fit)     13 blur_abs     14 blur_valid (>= 5 edge px)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ifl.features.backbones import DEV, GRID

W = 144                                   # working grid, 6 px per cell
CELL = W // GRID


def _sh9(n):
    """n [...,3] unit normals -> [...,9] real SH basis up to order 2 (constants dropped, absorbed in l)"""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    return torch.stack([torch.ones_like(x), x, y, z, x * y, x * z, y * z, x * x - y * y, 3 * z * z - 1], -1)


def _cells(t):
    """[H=W, W, C] -> [G, G, CELL*CELL, C]"""
    C = t.shape[-1]
    return t.reshape(GRID, CELL, GRID, CELL, C).permute(0, 2, 1, 3, 4).reshape(GRID, GRID, CELL * CELL, C)


class PhysicsStats:
    native = True

    def __init__(self, processing=512, steps=4):
        from diffusers import MarigoldIntrinsicsPipeline, MarigoldNormalsPipeline
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        dt = torch.float16 if DEV == "cuda" else torch.float32
        kw = dict(torch_dtype=dt, variant="fp16") if DEV == "cuda" else {}
        self.normals = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", **kw).to(DEV)
        self.iid = MarigoldIntrinsicsPipeline.from_pretrained("prs-eth/marigold-iid-lighting-v1-1", **kw).to(DEV)
        dummy = Image.new("RGB", (processing, processing), (128, 128, 128))
        for p in (self.normals, self.iid):
            p.set_progress_bar_config(disable=True)
            # Marigold encodes the empty prompt once and caches it; after that the CLIP text encoder is dead
            # weight. Moving it off the GPU keeps both pipelines inside 6 GB (otherwise the driver spills to RAM)
            p(dummy, num_inference_steps=1, processing_resolution=processing, output_type="pt")
            p.register_modules(text_encoder=None)     # (moving it to the CPU would make the pipeline run there)
        torch.cuda.empty_cache()
        names = list(self.iid.target_properties["target_names"])
        self.ia, self.isd = names.index("albedo"), names.index("shading")
        rid = "depth-anything/Depth-Anything-V2-Small-hf"
        self.dproc = AutoImageProcessor.from_pretrained(rid)
        self.depth = AutoModelForDepthEstimation.from_pretrained(rid).to(DEV).eval()
        self.processing, self.steps = processing, steps

    def _work(self, t):
        """[B,C,h,w] -> [W,W,C] float32 on DEV (area/bilinear resample, aspect not preserved like the mask grid)"""
        return F.interpolate(t.float(), size=(W, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    @torch.no_grad()
    def _maps(self, pil):
        g = torch.Generator(DEV).manual_seed(0)
        kw = dict(num_inference_steps=self.steps, processing_resolution=self.processing, output_type="pt",
                  generator=g)
        n = self._work(self.normals(pil, **kw).prediction)                        # [-1, 1]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        iid = self.iid(pil, **kw).prediction                                       # [targets,3,h,w] in [0,1]
        alb, sh = self._work(iid[self.ia:self.ia + 1]), self._work(iid[self.isd:self.isd + 1])
        inp = self.dproc(images=pil, return_tensors="pt").to(DEV)
        disp = self.depth(**inp).predicted_depth[:, None]                          # relative inverse depth
        disp = self._work(disp)[..., 0]
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        return n, sh, alb, disp

    @staticmethod
    def _lighting(n, sh):
        s = sh.mean(-1).reshape(-1)
        Y = _sh9(n).reshape(-1, 9)
        w = torch.ones_like(s)
        for _ in range(5):                                                         # Huber IRLS
            A = Y * w[:, None]
            l = torch.linalg.lstsq(A.T @ Y + 1e-4 * torch.eye(9, device=DEV), A.T @ s[:, None]).solution[:, 0]
            r = s - Y @ l
            c = 1.345 * (r.abs().median() / 0.6745 + 1e-6)
            w = torch.where(r.abs() <= c, torch.ones_like(r), c / r.abs())
        r = r.reshape(W, W, 1)
        gdir = l[1:4] / (l[1:4].norm() + 1e-8)
        nc, sc = _cells(n), _cells(sh.mean(-1, keepdim=True))[..., 0]              # [G,G,36,3], [G,G,36]
        X = torch.cat([torch.ones_like(nc[..., :1]), nc], -1)                      # [G,G,36,4]
        XtX = X.transpose(-1, -2) @ X + 1e-2 * torch.eye(4, device=DEV)
        b = torch.linalg.solve(XtX, X.transpose(-1, -2) @ sc[..., None])[..., 0]   # [G,G,4]
        ldir = b[..., 1:] / (b[..., 1:].norm(dim=-1, keepdim=True) + 1e-8)
        ang = torch.arccos((ldir * gdir).sum(-1).clamp(-1, 1))
        ncen = nc - nc.mean(2, keepdim=True)
        cond = torch.linalg.eigvalsh(ncen.transpose(-1, -2) @ ncen / nc.shape[2])[..., 0]
        rc = _cells(r)[..., 0]
        chrom = sh / (sh.sum(-1, keepdim=True) + 1e-6)
        ch = _cells(chrom[..., :2]).mean(2)
        ch = ch - ch.reshape(-1, 2).median(0).values
        return [rc.mean(-1), rc.abs().mean(-1), ang, cond, ch[..., 0], ch[..., 1], sc.mean(-1)]

    @staticmethod
    def _defocus(grey, disp):
        """grey [h,w] float in [0,1] (long side 512), disp [W,W] -> blur features per cell"""
        k = torch.tensor([[-1.0, 0, 1]], device=DEV)
        def grad(t):
            gx = F.conv2d(t[None, None], k[None, None], padding=(0, 1))
            gy = F.conv2d(t[None, None], k.T[None, None], padding=(1, 0))
            return (gx ** 2 + gy ** 2).sqrt()[0, 0]
        s0 = 1.0
        x = torch.arange(-4, 5, device=DEV, dtype=torch.float32)
        g1 = torch.exp(-x ** 2 / (2 * s0 ** 2))
        g1 = g1 / g1.sum()
        blur = F.conv2d(F.conv2d(grey[None, None], g1[None, None, None], padding=(0, 4)),
                        g1[None, None, :, None], padding=(4, 0))[0, 0]
        G0, G1 = grad(grey), grad(blur)
        edge = G0 > torch.quantile(G0.flatten()[::7], 0.9)
        R = (G0 / (G1 + 1e-6)).clamp(1.0 + 1e-3, 50)
        sig = s0 / torch.sqrt(R ** 2 - 1)
        sig = torch.where(edge, sig.clamp(0, 10), torch.full_like(sig, float("nan")))
        h, w = sig.shape
        sig = sig[: h - h % GRID, : w - w % GRID]
        cs = sig.reshape(GRID, sig.shape[0] // GRID, GRID, sig.shape[1] // GRID).permute(0, 2, 1, 3).reshape(GRID, GRID, -1)
        valid = (~torch.isnan(cs)).sum(-1) >= 5
        med = torch.nanmedian(cs, dim=-1).values
        D = _cells(disp[..., None])[..., 0].mean(-1)
        m, d = med[valid], D[valid]
        best, fit = None, torch.zeros_like(med)
        if valid.sum() >= 10:
            for df in torch.linspace(0, 1, 21, device=DEV):
                X = torch.stack([(d - df).abs(), torch.ones_like(d)], -1)
                coef = torch.linalg.lstsq(X, m[:, None]).solution[:, 0]
                coef[0] = coef[0].clamp_min(0)
                loss = ((X @ coef - m).abs()).mean()
                if best is None or loss < best:
                    best, cbest, dbest = loss, coef, df
            fit = cbest[0] * (D - dbest).abs() + cbest[1]
        res = torch.where(valid, med - fit, torch.zeros_like(med))
        med = torch.where(valid, med, torch.zeros_like(med))
        return [med, D, res, res.abs(), valid.float()]

    @torch.no_grad()
    def __call__(self, x_uint8):
        out = []
        for x in x_uint8:
            pil = Image.fromarray(x.permute(1, 2, 0).cpu().numpy())
            n, sh, alb, disp = self._maps(pil)
            feats = self._lighting(n, sh) + [a for a in _cells(alb).mean(2).unbind(-1)]
            s = 512 / max(pil.size)
            grey = torch.from_numpy(np.asarray(pil.convert("L").resize(
                (max(GRID, round(pil.width * s)), max(GRID, round(pil.height * s))), Image.BICUBIC),
                np.float32) / 255).to(DEV)
            feats += self._defocus(grey, disp)
            out.append(torch.stack([f.float() for f in feats], -1))
        patch = torch.stack(out)
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()
