"""Random sub-sampling of large forgery datasets hosted on the Hugging Face Hub.

Every sampler writes the ORIGINAL image bytes (no re-encode) to
``<out_dir>/<dataset>/images`` and a binary mask (255 = manipulated) to
``<out_dir>/<dataset>/masks``, and returns manifest rows (dicts).

Supported source kinds
  hf_parquet : HF parquet shards with ``image``/``mask``/``label`` columns. Only a few random
               row groups are fetched over HTTP range requests - never the full shard set.
  hf_zip     : one big zip on the Hub; the central directory is read remotely and only the
               sampled members are fetched.
  hf_tar     : small tar(.gz) with an index txt of ``forged/x.png,gt/x_mask.png`` lines;
               downloaded once to data/raw and sampled locally.
"""
from __future__ import annotations

import io
import random
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from ifl import DATA, FULL_SYNTH, LOCAL_EDIT, REAL

Image.MAX_IMAGE_PIXELS = None


def _sniff_ext(b: bytes) -> str:
    fmt = (Image.open(io.BytesIO(b)).format or "bin").lower()
    return {"jpeg": "jpg"}.get(fmt, fmt)


def _binarize_mask(b: bytes | None, size: tuple[int, int]) -> np.ndarray | None:
    if not b:
        return None
    m = Image.open(io.BytesIO(b))
    if m.mode in ("RGBA", "LA"):
        m = m.convert("RGB")
    m = np.asarray(m)
    if m.ndim == 3:
        m = m.max(axis=2)
    # 0/255 (or soft 0..255) masks -> threshold at half; small-int label maps (e.g. DiffSeg30k stores
    # the editor id 1..8 per pixel) -> any non-zero class is manipulated
    m = m > 127 if m.max() >= 32 else m > 0
    if m.shape[::-1] != size:  # some datasets ship masks at a different resolution
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    return m


def _write(out: Path, ds: str, subset: str, sid: str, img_bytes: bytes, mask: np.ndarray | None,
           label: int, extra: dict | None = None) -> dict:
    img_dir, mask_dir = out / ds / "images", out / ds / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    sid = sid.replace("/", "__").replace("\\", "__")
    ext = _sniff_ext(img_bytes)
    ip = img_dir / f"{sid}.{ext}"
    ip.write_bytes(img_bytes)
    im = Image.open(io.BytesIO(img_bytes))
    w, h = im.size
    row = dict(dataset=ds, subset=subset, sample_id=sid, label=label, img_path=str(ip.relative_to(DATA)),
               mask_path="", width=w, height=h, format=ext, mode=im.mode, mask_frac=np.nan,
               jpeg_q=_jpeg_quality(im))
    if mask is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        mp = mask_dir / f"{sid}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mp)
        row.update(mask_path=str(mp.relative_to(DATA)), mask_frac=float(mask.mean()))
    elif label == REAL:
        row["mask_frac"] = 0.0
    elif label == FULL_SYNTH:
        row["mask_frac"] = 1.0
    row.update(extra or {})
    return row


def _jpeg_quality(im: Image.Image) -> float:
    """Rough IJG quality estimate from the luma quantisation table (NaN for non-JPEG)."""
    q = getattr(im, "quantization", None)
    if not q:
        return np.nan
    t = np.asarray(q[0], dtype=float)
    base = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)
    s = np.clip(100.0 * t.mean() / base.mean(), 1, 5000)  # IJG scale factor
    return float(np.clip((200 - s) / 2 if s <= 100 else 5000 / s / 2, 1, 100))


def _unified_label(raw_label, has_mask: bool, real_labels) -> int:
    if raw_label in real_labels:
        return REAL
    return LOCAL_EDIT if has_mask else FULL_SYNTH


# --------------------------------------------------------------------------- hf_parquet
def sample_hf_parquet(name: str, cfg: dict, out: Path, rng: random.Random) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    fs, files = HfFileSystem(), HfApi().list_repo_files(cfg["repo"], repo_type="dataset")
    rows = []
    for subset, prefix in cfg["shard_prefixes"].items():
        shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
        picks = []  # (shard, row_group)
        for shard in rng.sample(shards, min(len(shards), cfg["row_groups_per_subset"])):
            with fs.open(f"datasets/{cfg['repo']}/{shard}") as fh:
                picks.append((shard, rng.randrange(pq.ParquetFile(fh).num_row_groups)))
        per_rg = int(np.ceil(cfg["per_subset"] / len(picks)))
        for shard, rg in picks:
            with fs.open(f"datasets/{cfg['repo']}/{shard}", block_size=2 ** 22) as fh:
                tab = pq.ParquetFile(fh).read_row_group(rg).to_pylist()
            for r in rng.sample(tab, min(per_rg, len(tab))):
                img = r["image"]["bytes"]
                w, h = Image.open(io.BytesIO(img)).size
                mask = _binarize_mask((r.get("mask") or {}).get("bytes"), (w, h))
                if mask is not None and not mask.any():
                    mask = None
                lab = _unified_label(r["label"], mask is not None, cfg.get("real_labels", [0]))
                rows.append(_write(out, name, subset, str(r[cfg["id_col"]]), img, mask, lab,
                                   dict(source_file=f"{shard}#rg{rg}", raw_label=r["label"])))
            print(f"  {name}/{subset}: {shard} rg{rg} -> {len(rows)} total", flush=True)
    return rows


# --------------------------------------------------------------------------- hf_zip
def sample_hf_zip(name: str, cfg: dict, out: Path, rng: random.Random) -> list[dict]:
    from huggingface_hub import HfFileSystem

    rows = []
    with HfFileSystem().open(f"datasets/{cfg['repo']}/{cfg['file']}", block_size=2 ** 20) as fh:
        z = zipfile.ZipFile(fh)
        names = set(z.namelist())
        imgs = sorted(n for n in names if n.endswith(cfg["image_suffix"]))
        for n in rng.sample(imgs, min(cfg["n"], len(imgs))):
            stem = n[: -len(cfg["image_suffix"])]
            img = z.read(n)
            mn = stem + cfg["mask_suffix"]
            w, h = Image.open(io.BytesIO(img)).size
            mask = _binarize_mask(z.read(mn), (w, h)) if mn in names else None
            lab = LOCAL_EDIT if (mask is not None and mask.any()) else REAL
            rows.append(_write(out, name, "all", Path(stem).name, img, mask if lab == LOCAL_EDIT else None, lab,
                               dict(source_file=n)))
            if len(rows) % 50 == 0:
                print(f"  {name}: {len(rows)}", flush=True)
    return rows


# --------------------------------------------------------------------------- hf_tar
def sample_hf_tar(name: str, cfg: dict, out: Path, rng: random.Random) -> list[dict]:
    from huggingface_hub import hf_hub_download

    raw = DATA / "raw" / cfg["repo"].split("/")[-1]
    path = hf_hub_download(cfg["repo"], cfg["file"], repo_type="dataset", local_dir=raw)
    rows = []
    with tarfile.open(path) as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        index = [m for m in members if m.endswith(".txt") and m.count("/") == 1]
        root = index[0].split("/")[0]
        pairs = [ln.strip().split(",") for ln in tf.extractfile(members[index[0]]).read().decode().splitlines()
                 if ln.strip()]
        for p in rng.sample(pairs, min(cfg["n"], len(pairs))):
            img = tf.extractfile(members[f"{root}/{p[0]}"]).read()
            w, h = Image.open(io.BytesIO(img)).size
            mask = _binarize_mask(tf.extractfile(members[f"{root}/{p[1]}"]).read(), (w, h)) if len(p) > 1 else None
            lab = LOCAL_EDIT if (mask is not None and mask.any()) else FULL_SYNTH
            rows.append(_write(out, name, "all", Path(p[0]).stem, img, mask if lab == LOCAL_EDIT else None, lab,
                               dict(source_file=p[0])))
    return rows


SAMPLERS = {"hf_parquet": sample_hf_parquet, "hf_zip": sample_hf_zip, "hf_tar": sample_hf_tar}
