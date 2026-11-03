"""Manifest-backed dataset: resized views for foundation models + native view for low-level streams."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ifl import DATA

Image.MAX_IMAGE_PIXELS = None


def load_manifest(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA / "samples" / "manifest.csv")
    df["mask_path"] = df["mask_path"].fillna("")
    return df.reset_index(drop=True)


def harmonize(im: Image.Image, mode: str | None) -> Image.Image:
    """Optional re-encoding applied before feature extraction.
    'jpegQ' / 'webpQ' re-encode at quality Q (jpeg95 also removes the JPEG-vs-PNG shortcut), 'resizeP' downscales
    to P% of the side (kept small, as when an image is shared). 'sr*' modes are precomputed files, see
    scripts/make_perturbed.py, and are resolved in ManifestImages."""
    if not mode or mode.startswith("sr"):
        return im
    if mode.startswith(("jpeg", "webp")):
        buf = io.BytesIO()
        im.save(buf, "JPEG" if mode.startswith("jpeg") else "WEBP", quality=int(mode[4:] or 95))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if mode.startswith("resize"):
        s = int(mode[6:]) / 100
        return im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BICUBIC)
    raise ValueError(f"unknown harmonize mode {mode!r}")


def perturbed_path(img_path: str, mode: str):
    return DATA / "perturbed" / mode / (str(img_path) + ".png")


def mask_grid(row, grid: int) -> np.ndarray:
    """Fraction of manipulated pixels inside each of grid x grid cells (area-averaged)."""
    if row.mask_path:
        m = np.asarray(Image.open(DATA / row.mask_path).convert("L"), dtype=np.float32) / 255.0
    else:
        m = np.full((int(row.height), int(row.width)), 1.0 if row.label == 1 else 0.0, np.float32)
    return area_resize(m, grid)


def area_resize(a: np.ndarray, grid: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(a))[None, None]
    return torch.nn.functional.adaptive_avg_pool2d(t, grid)[0, 0].numpy()


class ManifestImages(torch.utils.data.Dataset):
    """Returns uint8 CHW tensor resized to `size` x `size` (aspect not preserved so the mask grid
    covers the whole image) or, with size=None, the native-resolution image."""

    def __init__(self, df: pd.DataFrame, size: int | None, harmonize_mode: str | None = None,
                 max_side: int | None = None):
        self.df, self.size, self.hm, self.max_side = df, size, harmonize_mode, max_side

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        src = perturbed_path(r.img_path, self.hm) if self.hm and self.hm.startswith("sr") else DATA / r.img_path
        im = harmonize(Image.open(src).convert("RGB"), self.hm)
        if self.size:
            im = im.resize((self.size, self.size), Image.BICUBIC)
        elif self.max_side and max(im.size) > self.max_side:
            s = self.max_side / max(im.size)
            im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1), i
