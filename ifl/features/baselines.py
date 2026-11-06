"""Published, supervised forgery localizers as cell "features", so they go through the same protocol as our
scorers (per-image cell AUC, interior AUC, matched transplant null, pixel control, perturbed copies).

    trufor    TruFor (Guillaro et al., CVPR 2023), official code + weights (grip-unina/TruFor)
              patch D=3: cell mean of the localization map, cell max, cell mean of the confidence map
    mesorch   Mesorch (AAAI 2025), official code + checkpoint mesorch-98.pth (scu-zjz/Mesorch), 512x512 input
              patch D=2: cell mean, cell max of the predicted mask

Code and weights live outside the repo, in $IFL_BASELINES (default M:/SourceCodeM/baselines):
    TruFor/TruFor_train_test/pretrained_models/weights/trufor.pth.tar   (TruFor_weights.zip, MD5 7bee48f3...)
    Mesorch/ + weights/mesorch/mesorch-98.pth                            (Google Drive link in the Mesorch README)
Both were TRAINED on CASIA2, FantasticReality, IMD2020 and tampCOCO (TruFor also compRAISE): their IMD2020 cells
are not out-of-distribution. Licences: TruFor non-commercial research only; Mesorch MIT.

    flame     FLAME (Wang et al., arXiv:2606.02178), code github.com/phoenixnir/FLAME at commit 5d3d42f, released
              checkpoint flame_g2_ladmulti_sam2.pth (SHA256 9b6c67e3...) on SAM2.1 hiera base+ (facebook/
              sam2.1-hiera-base-plus). Input resized to 512x512 in [0,1], no perturbation streams, as its test.py.
              patch D=3: cell mean, cell max of the predicted mask, image detection probability.
              Trained on MagicBrush and SID-Set: its SID-Set cells are not out-of-distribution. No licence file.
    bfree     B-Free (Guillaro et al., CVPR 2025), github.com/grip-unina/B-Free at commit c6a9f89, weights
              BFREE_dino2reg4 (zip MD5 f3f53fa6...): an image-level detector (ViT-B/14 + 4 registers, DINOv2 init,
              native resolution, 5 crops of 504 px averaged). Trained on COCO vs SD2.1 self-reconstructions of the
              same photos + local inpaintings. Licence: non-profit research use with attribution.
              patch D=3: our sliding-window ADAPTATION as a map (224 px windows, stride 168, each scored with
              B-Free's own small-input handling - patch embeddings tiled to its 36 x 36 grid - logits averaged per
              pixel, sigmoid; cell mean and max), and the official image-level probability as the third channel.
    bfreeimg  the official image-level probability only (third channel repeated), cheap enough for every condition.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from ifl.features.backbones import DEV, GRID

ROOT = Path(os.environ.get("IFL_BASELINES", "M:/SourceCodeM/baselines"))


def _pool(m):
    """m [B,1,H,W] probability map -> (mean, max) per cell, each [B,G,G]"""
    return F.adaptive_avg_pool2d(m, GRID)[:, 0], F.adaptive_max_pool2d(m, GRID)[:, 0]


class TruForMap:
    native = True

    def __init__(self):
        code = ROOT / "TruFor" / "TruFor_train_test"
        sys.path.insert(0, str(code))
        from lib.config import config
        from lib.utils import get_model
        cwd = os.getcwd()
        os.chdir(code)                    # the yaml points at pretrained_models/... relative to the code folder
        try:
            config.defrost()
            config.merge_from_file(str(code / "lib" / "config" / "trufor_ph3.yaml"))
            config.freeze()
            self.model = get_model(config)
            ck = torch.load(code / "pretrained_models" / "weights" / "trufor.pth.tar", map_location=DEV,
                            weights_only=False)
            self.model.load_state_dict(ck["state_dict"])
        finally:
            os.chdir(cwd)
        self.model = self.model.to(DEV).eval()

    @torch.no_grad()
    def __call__(self, x_uint8):
        out = []
        for x in x_uint8:                 # batch 1 per image (arbitrary sizes), as in the official test.py
            rgb = x[None].to(DEV).float() / 256.0
            pred, conf, det, _ = self.model(rgb, save_np=False)
            m = F.softmax(pred, 1)[:, 1:2]
            mean, mx = _pool(m)
            c = F.adaptive_avg_pool2d(torch.sigmoid(conf), GRID)[:, 0]
            out.append(torch.stack([mean, mx, c], -1))
        patch = torch.cat(out).float()
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


class MesorchMap:
    native = True
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # IMDLBenCo albu.Normalize defaults

    def __init__(self, ckpt="mesorch-98.pth"):
        code = ROOT / "Mesorch"
        # mesorch.py only needs IMDLBenCo for its model registry; a stub avoids installing the package
        reg = types.ModuleType("IMDLBenCo.registry")
        reg.MODELS = types.SimpleNamespace(register_module=lambda *a, **k: (lambda c: c))
        sys.modules.setdefault("IMDLBenCo", types.ModuleType("IMDLBenCo"))
        sys.modules.setdefault("IMDLBenCo.registry", reg)
        sys.path.insert(0, str(code))
        from mesorch import MesorchFull
        self.model = MesorchFull(image_size=512)
        ck = torch.load(ROOT / "weights" / "mesorch" / ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        missing, unexpected = self.model.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()},
                                                         strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Mesorch checkpoint mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        self.model = self.model.to(DEV).eval()
        self.mean = torch.tensor(self.MEAN, device=DEV)[:, None, None]
        self.std = torch.tensor(self.STD, device=DEV)[:, None, None]

    @torch.no_grad()
    def __call__(self, x_uint8):
        out = []
        for x in x_uint8:
            im = x[None].to(DEV).float() / 255
            im = F.interpolate(im, size=(512, 512), mode="bilinear", align_corners=False)   # --if_resizing
            im = (im - self.mean) / self.std
            res = self.model(im, torch.zeros_like(im[:, :1]))
            mean, mx = _pool(res["pred_mask"].float())
            out.append(torch.stack([mean, mx], -1))
        patch = torch.cat(out).float()
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


class FlameMap:
    native = True

    def __init__(self):
        import importlib.util
        import json
        from PIL import Image  # noqa: F401  (FLAME's test.py needs it at import)
        code = ROOT / "FLAME" / "FLAME"
        sys.path.insert(0, str(code))
        cwd = os.getcwd()
        os.chdir(code)                    # FLAME resolves its SAM2 config and checkpoint relative to the code folder
        try:
            # test_dataset.py's builder: unlike test.py it passes the checkpoint's forensic operator (lad_multi, 4
            # scales) to the model, so the released weights load; importing it also initialises Hydra for SAM2
            spec = importlib.util.spec_from_file_location("flame_test_dataset", code / "test_dataset.py")
            ft = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ft)
            cfg = json.loads((code / "checkpoints" / "model_params.json").read_text())
            self.model = ft.load_and_initialize_model(cfg, str(code / "checkpoints" / "flame_g2_ladmulti_sam2.pth"),
                                                      torch.device(DEV), use_ema=False)
        finally:
            os.chdir(cwd)
        self.size = int(cfg["training_config"].get("img_size", 512))

    @torch.no_grad()
    def __call__(self, x_uint8):
        from PIL import Image
        from torchvision.transforms import functional as TF
        out = []
        for x in x_uint8:
            # exactly FLAME's evaluation input: PIL bilinear resize to img_size x img_size, ToTensor ([0,1]), no streams
            pil = Image.fromarray(x.permute(1, 2, 0).cpu().numpy()).resize((self.size, self.size), Image.BILINEAR)
            im = TF.to_tensor(pil)[None].to(DEV)
            with torch.autocast(device_type="cuda", enabled=DEV == "cuda", cache_enabled=False):
                res = self.model(im, [], output_extras=True)
            logits, extras = res if isinstance(res, tuple) else (res, {})
            p = torch.sigmoid(logits.float())
            if p.shape[-2:] != (self.size, self.size):
                p = F.interpolate(p, size=(self.size, self.size), mode="bilinear", align_corners=False)
            mean, mx = _pool(p)
            det = extras.get("detection_logit") if isinstance(extras, dict) else None
            d = torch.sigmoid(det.float()).reshape(1, 1, 1).expand_as(mean) if det is not None else torch.zeros_like(mean)
            out.append(torch.stack([mean, mx, d], -1))
        patch = torch.cat(out).float()
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()


class BFreeMap:
    """B-Free image-level logit (official 5-crop path) and, with local=True, a sliding-window map (our adaptation)"""
    native = True
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # norm_type 'resnet' in its config

    def __init__(self, local=True, window=224, stride=168):
        import yaml
        code = ROOT / "B-Free" / "code"
        sys.path.insert(0, str(code))
        from networks import get_network, load_weights
        wdir = code / "weights" / "BFREE_dino2reg4"
        cfg = yaml.safe_load((wdir / "config.yaml").read_text())
        self.model = load_weights(get_network(cfg["arch"]), str(wdir / cfg["weights_file"])).to(DEV).eval()
        self.mean = torch.tensor(self.MEAN, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=DEV).view(1, 3, 1, 1)
        self.local, self.window, self.stride = local, window, stride

    @staticmethod
    def _logit(out):
        return out[:, 0] if out.shape[1] == 1 else out[:, 1] - out[:, 0]

    def _window_logits(self, crops):
        """crops [n,3,w,w] -> one logit each; small inputs are handled as B-Free does (patch embeddings tiled to the
        504 px grid), but with a single crop of the tiled grid instead of five identical-content crops"""
        from networks.wrapper5crops import replicate_wrap
        w = self.model
        grid = w.patch_embed.grid_size
        emb = replicate_wrap(w.patch_embed.proj(crops), grid)[..., :grid[0], :grid[1]]
        if w.patch_embed.flatten:
            emb = emb.flatten(2).transpose(1, 2)
        return self._logit(w.model(w.patch_embed.norm(emb)))

    @torch.no_grad()
    def __call__(self, x_uint8):
        out = []
        for x in x_uint8:
            t = (x[None].to(DEV).float() / 255 - self.mean) / self.std
            img = torch.sigmoid(self._logit(self.model(t)).float())             # official image-level score
            H, W = t.shape[-2:]
            if self.local:
                win = min(self.window, H, W)
                ys = list(range(0, max(H - win, 0) + 1, self.stride)) or [0]
                xs = list(range(0, max(W - win, 0) + 1, self.stride)) or [0]
                ys = sorted(set(ys + [max(H - win, 0)]))
                xs = sorted(set(xs + [max(W - win, 0)]))
                boxes = [(y, xx) for y in ys for xx in xs]
                crops = torch.cat([t[..., y:y + win, xx:xx + win] for y, xx in boxes])
                logits = torch.cat([self._window_logits(crops[i:i + 32]) for i in range(0, len(crops), 32)]).float()
                acc = torch.zeros(1, 1, H, W, device=DEV)
                cnt = torch.zeros(1, 1, H, W, device=DEV)
                for (y, xx), lg in zip(boxes, logits):
                    acc[..., y:y + win, xx:xx + win] += lg
                    cnt[..., y:y + win, xx:xx + win] += 1
                p = torch.sigmoid(acc / cnt.clamp_min(1))
                mean, mx = _pool(p)
            else:
                mean = mx = img.view(1, 1, 1).expand(1, GRID, GRID)
            out.append(torch.stack([mean, mx, img.view(1, 1, 1).expand_as(mean)], -1))
        patch = torch.cat(out).float()
        return patch.mean((1, 2)).cpu().numpy(), patch.cpu().numpy()
