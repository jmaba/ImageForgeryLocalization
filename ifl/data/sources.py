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


# --------------------------------------------------------------------------- tgif2 (Nextcloud WebDAV)
_NC = "https://cloud.ilabt.imec.be/public.php/webdav/"
_NC_H = {"X-Requested-With": "XMLHttpRequest"}  # public WebDAV answers 401 without it


def _nc_get(token, path, stream=False):
    import requests
    r = requests.get(_NC + path, auth=(token, ""), headers=_NC_H, stream=stream, timeout=120)
    r.raise_for_status()
    return r


def _nc_tar_members(token, path, keep):
    """Stream a remote .tar.gz once and yield (name, bytes) for members where keep(name) is True."""
    r = _nc_get(token, path, stream=True)
    r.raw.decode_content = True
    with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
        for m in tf:
            if m.isfile() and keep(m.name):
                yield m.name, tf.extractfile(m).read()


def sample_tgif2(name: str, cfg: dict, out: Path, rng: random.Random) -> list[dict]:
    """TGIF2 random-mask release. The SAME (coco_id, variant) pairs are drawn for every subset so that
    FR / SP / original images of one edit are matched (enables the INP-X and aligned-real controls).
    Writes one manifest dataset per subset: tgif2_<subset>, plus tgif2_orig (label 0)."""
    import pandas as pd

    split, tok = cfg["split"], cfg["token"]
    rows = []
    for model, subsets in cfg["models"].items():
        meta = pd.read_csv(io.BytesIO(_nc_get(tok, f"metadata/{model}_{split}.csv").content))
        meta = meta[meta.mask_variation == "random"].drop_duplicates("image_id")
        pick = meta.sample(min(cfg["n"], len(meta)), random_state=rng.randrange(2 ** 31))
        # sd2/sdxl metadata repeat image_id in gen_variation_id -> variant unknown, fixed while streaming
        var_known = bool((pick.gen_variation_id != pick.image_id).all())
        chosen = {str(r.image_id): (str(r.gen_variation_id) if var_known else None) for r in pick.itertuples()}
        ids = set(chosen)
        mfolder = cfg["mask_folder"][model]
        masks = dict(_nc_tar_members(tok, f"{mfolder}/{mfolder}_{split}.tar.gz",
                                     lambda n: Path(n).name.split("_")[0] in ids and "_mask_random" in n))
        masks_by_id = {}
        for n, b in masks.items():
            masks_by_id.setdefault(Path(n).name.split("_")[0], []).append(b)

        def best_mask(cid, size):
            cands = masks_by_id.get(cid, [])
            for b in cands:
                if Image.open(io.BytesIO(b)).size == size:
                    return b
            return cands[0] if cands else None

        for sub in subsets:
            ds = f"tgif2_{sub.replace('-', '_')}"
            got = 0

            done = set()

            def keep(n):
                stem = Path(n).name
                cid, var = stem.split("_")[0], stem.rsplit("_", 1)[-1][:-4]
                if "_mask_random" not in stem or cid not in chosen or cid in done:
                    return False
                if chosen[cid] is None:  # first subset of this model fixes the variant for the others
                    chosen[cid] = var
                return chosen[cid] == var

            for n, img in _nc_tar_members(tok, f"{sub}/{sub}_{split}.tar.gz", keep):
                stem = Path(n).name
                cid, var = stem.split("_")[0], stem.rsplit("_", 1)[-1][:-4]
                done.add(cid)
                w, h = Image.open(io.BytesIO(img)).size
                mb = best_mask(cid, (w, h))
                mask = _binarize_mask(mb, (w, h)) if mb else None
                if mask is None or not mask.any():
                    continue
                rows.append(_write(out, ds, sub, f"{cid}_{var}", img, mask, LOCAL_EDIT,
                                   dict(source_file=n, pair_id=f"{cid}_{var}", category=Path(n).parent.name)))
                got += 1
            print(f"  {ds}: {got}", flush=True)
        if cfg.get("orig") and model == cfg["orig"]["model"]:
            otok, ofolder = cfg["orig"]["token"], cfg["orig"]["folder"]
            got = 0
            for n, img in _nc_tar_members(otok, f"{ofolder}/{ofolder}_{split}.tar.gz",
                                          lambda n: Path(n).name.split("_")[0] in ids and n.endswith("_orig.png")):
                cid = Path(n).name.split("_")[0]
                rows.append(_write(out, "tgif2_orig", "orig", cid, img, None, REAL,
                                   dict(source_file=n, pair_id=cid, category=Path(n).parent.name)))
                got += 1
            print(f"  tgif2_orig: {got}", flush=True)
    return rows


# --------------------------------------------------------------------------- local (classical benchmarks on disk)
def _local_items(layout: str, root: Path):
    """-> list of (edited_image, mask, invert_mask, key, subset, authentic_image or None)."""
    items = []
    if layout == "diverse":                      # <method>/NNNNNN.png + NNNNNN_mask.png (IID-Net, 11 methods)
        for m in sorted(p for p in root.iterdir() if p.is_dir()):
            items += [(p, p.with_name(p.stem + "_mask.png"), False, f"{m.name}_{p.stem}", m.name, None)
                      for p in sorted(m.glob("*.png")) if not p.stem.endswith("_mask")]
    elif layout == "imd2020_inpaint":            # images/x.jpg + masks/x_mask.jpg
        items = [(p, root / "masks" / f"{p.stem}_mask.jpg", False, p.stem, "all", None)
                 for p in sorted((root / "images").glob("*.jpg"))]
    elif layout == "imd2020_real":               # <id>/<id>_orig.jpg + <id>/<edit>_0.jpg + <edit>_0_mask.png
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            orig = next(iter(d.glob("*_orig.*")), None)
            items += [(p, p.with_name(p.stem + "_mask.png"), False, f"{d.name}_{p.stem}", "all", orig)
                      for p in sorted(d.iterdir()) if p.suffix.lower() in (".jpg", ".png", ".jpeg")
                      and "_orig" not in p.stem and not p.stem.endswith("_mask")
                      and p.with_name(p.stem + "_mask.png").exists()]
    elif layout == "korus":                      # <cam>/tampered-realistic/X.TIF, ground-truth/X.PNG, pristine/X.TIF
        for cam in sorted(p for p in root.iterdir() if (p / "tampered-realistic").is_dir()):
            items += [(p, cam / "ground-truth" / f"{p.stem}.PNG", False, f"{cam.name}_{p.stem}", cam.name,
                       cam / "pristine" / p.name) for p in sorted((cam / "tampered-realistic").iterdir())]
    elif layout == "nc2016":                     # reference csv: IsTarget=Y rows; masks are 0 = manipulated
        import pandas as pd
        ref = pd.read_csv(root / "reference" / "manipulation" / "NC2016-manipulation-ref.csv", sep="|")
        ref = ref[(ref.IsTarget == "Y") & ref.ProbeMaskFileName.notna()].drop_duplicates("ProbeFileID")
        for r in ref.itertuples():
            mask = root / "reference" / "manipulation" / "mask" / Path(r.ProbeMaskFileName).name
            base = root / r.BaseFileName if isinstance(r.BaseFileName, str) else None
            items.append((root / r.ProbeFileName, mask, True, r.ProbeFileID, "manipulation", base))
    elif layout == "defacto_inpaint":            # img/x.tif + inpaint_mask/x.tif (COCO-based)
        items = [(p, root / "inpaint_mask" / p.name, False, p.stem, "inpaint", None)
                 for p in sorted((root / "img").glob("*.tif"))]
    elif layout == "cmfd_velo":                  # <cam>/X_copy.png forgery, X.png original (lossless)
        # X_N.png / X_N_alpha.png are the snippet LAYERS (object on white), not forgeries. The ground truth is
        # where the forgery differs from the original (the pasted targets), written to data/_cache/cmfd_velo.
        from scipy import ndimage
        cache = DATA / "_cache" / "cmfd_velo"
        cache.mkdir(parents=True, exist_ok=True)
        for cam in sorted(p for p in root.iterdir() if p.is_dir()):
            for p in sorted(cam.glob("*_copy.png")):
                scene = p.stem[:-len("_copy")]
                orig = cam / f"{scene}.png"
                mp = cache / f"{cam.name}_{scene}.png"
                if orig.exists() and not mp.exists():
                    a = np.asarray(Image.open(p).convert("RGB"), np.int16)
                    b = np.asarray(Image.open(orig).convert("RGB"), np.int16)
                    m = ndimage.binary_opening(ndimage.binary_closing(np.abs(a - b).sum(-1) > 24, iterations=3), iterations=2)
                    Image.fromarray(m.astype(np.uint8) * 255).save(mp)
                items.append((p, mp, False, f"{cam.name}_{scene}", cam.name, orig))
    else:
        raise KeyError(layout)
    return [it for it in items if it[1].exists()]


def sample_local(name: str, cfg: dict, out: Path, rng: random.Random) -> list[dict]:
    """Datasets already on disk. Samples `n` edited images (stratified over subsets) and, where the dataset
    ships the untouched original, writes it as a REAL row with the SAME pair_id (matched transplant null)."""
    import os
    root = Path(os.environ.get("IFL_BENCHMARKS", "M:/Benchmarks")) / cfg["dir"]
    if not root.is_dir():
        raise FileNotFoundError(f"{root} (set IFL_BENCHMARKS to the folder holding the classical benchmarks)")
    items = _local_items(cfg["layout"], root)
    by_sub = {}
    for it in items:
        by_sub.setdefault(it[4], []).append(it)
    per = int(np.ceil(cfg["n"] / len(by_sub)))
    picks = [it for sub in sorted(by_sub) for it in rng.sample(by_sub[sub], min(per, len(by_sub[sub])))]
    picks = rng.sample(picks, min(cfg["n"], len(picks)))
    rows, seen = [], set()
    for img_p, mask_p, invert, key, sub, orig in picks:
        b = img_p.read_bytes()
        w, h = Image.open(io.BytesIO(b)).size
        m = np.asarray(Image.open(mask_p).convert("L"))
        # > 64: keeps Korus' grey "collateral damage" (128, tampered by the dataset's own convention) and
        # drops JPEG noise in IMD2020's .jpg masks; NC2016 masks are inverted (0 = manipulated)
        mb = Image.fromarray(np.where(m > 64 if not invert else m < 128, 255, 0).astype(np.uint8))
        buf = io.BytesIO()
        mb.save(buf, "PNG")
        mask = _binarize_mask(buf.getvalue(), (w, h))
        if mask is None or not mask.any() or mask.all():
            continue
        pid = f"{name}:{key}"
        rows.append(_write(out, name, sub, key, b, mask, LOCAL_EDIT, dict(source_file=str(img_p), pair_id=pid)))
        if orig is not None and orig.exists() and str(orig) not in seen:   # one original may have several edits
            seen.add(str(orig))
            rows.append(_write(out, name, sub, f"{key}__orig", orig.read_bytes(), None, REAL,
                               dict(source_file=str(orig), pair_id=pid)))
        if len(rows) % 50 == 0:
            print(f"  {name}: {len(rows)}", flush=True)
    return rows


SAMPLERS = {"hf_parquet": sample_hf_parquet, "hf_zip": sample_hf_zip, "hf_tar": sample_hf_tar,
            "tgif2": sample_tgif2, "local": sample_local}
