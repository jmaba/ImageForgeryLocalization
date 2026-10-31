"""Load cached features aligned with the manifest."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ifl import DATA
from ifl.data.dataset import load_manifest

FEAT = DATA / "features"


@dataclass
class FeatureSet:
    spec: str
    meta: pd.DataFrame          # manifest rows (index aligned with arrays)
    G: np.ndarray               # [N, D]       global
    P: np.ndarray               # [N, g, g, D] patch (memmap-backed concat)
    M: np.ndarray               # [N, g, g]    manipulated fraction per cell

    @property
    def grid(self):
        return self.M.shape[1]


def load_multi(spec_dirnames: list[str], datasets: list[str] | None = None) -> FeatureSet:
    """Concatenate several cached feature specs along the channel axis (same manifest order)."""
    sets = [load(s, datasets) for s in spec_dirnames]
    ids = [tuple(fs.meta.dataset + "/" + fs.meta.sample_id.astype(str)) for fs in sets]
    if any(i != ids[0] for i in ids[1:]):
        raise RuntimeError(f"specs {spec_dirnames} cover different images - extract them on the same datasets")
    return FeatureSet("+".join(spec_dirnames), sets[0].meta, np.concatenate([fs.G for fs in sets], 1),
                      np.concatenate([fs.P for fs in sets], -1), sets[0].M)


def available_specs() -> list[str]:
    return sorted(p.name for p in FEAT.iterdir() if p.is_dir() and not p.name.startswith("_"))


def load(spec_dirname: str, datasets: list[str] | None = None) -> FeatureSet:
    df = load_manifest()
    root = FEAT / spec_dirname
    metas, Gs, Ps, Ms = [], [], [], []
    for ds in sorted(datasets or df.dataset.unique()):
        d = root / ds
        if not (d / "ids.txt").exists():
            continue
        ids = (d / "ids.txt").read_text().split("\n")
        sub = df[df.dataset == ds].reset_index(drop=True)
        if list(sub.sample_id.astype(str)) != ids:
            raise RuntimeError(f"{spec_dirname}/{ds}: cache out of sync with manifest - re-extract")
        metas.append(sub)
        Gs.append(np.load(d / "global.npy").astype(np.float32))
        Ps.append(np.load(d / "patch.npy", mmap_mode="r"))
        Ms.append(np.load(FEAT / "_masks" / f"{ds}.npy").astype(np.float32))
    if not metas:
        raise FileNotFoundError(f"no cached features under {root}")
    return FeatureSet(spec_dirname, pd.concat(metas, ignore_index=True), np.concatenate(Gs),
                      np.concatenate(Ps), np.concatenate(Ms))  # P stays fp16 (~2.5 GB for 3k imgs, D=768)
