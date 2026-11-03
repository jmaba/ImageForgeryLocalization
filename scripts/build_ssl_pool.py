"""Real-photo pool for self-supervised (real-only) models. Deliberately NOT COCO, because most test edits
(CocoGlide, TGIF2, GPT-Image-2, Nano-Banana-2, OpenSDI) start from COCO photos.

    python scripts/build_ssl_pool.py            -> data/ssl_real/{images/*, pool.csv}
    python scripts/build_ssl_pool.py --div2k 800 --imagenet 20000      # bigger pool (DIV2K train has 800)
The defaults reproduce the original 1300-image pool exactly; larger counts read additional shards.
"""
import argparse
import io
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifl import DATA  # noqa: E402

SOURCES = [  # repo, image column, shards to read, images to keep, why
    ("bdanko/DIV2K_train_HR", "hr", 3, 300, "2K PNG, high-quality camera photos (clean acquisition statistics)"),
    ("mrm8488/ImageNet1K-val", "image", 2, 1000, "JPEG at mixed qualities (realistic compression history)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--div2k", type=int, default=SOURCES[0][3])
    ap.add_argument("--imagenet", type=int, default=SOURCES[1][3])
    args = ap.parse_args()
    keep = {SOURCES[0][0]: args.div2k, SOURCES[1][0]: args.imagenet}
    out = DATA / "ssl_real"
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)
    fs, api = HfFileSystem(), HfApi()
    rows = []
    for repo, col, n_shards, n_keep, _ in SOURCES:
        n_keep = keep[repo]
        shards = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
        first = rng.sample(shards, n_shards)                   # same draw as the original pool
        order = first + random.Random(repo).sample([s for s in shards if s not in first], len(shards) - n_shards)
        recs = []
        for k, sh in enumerate(order):
            if k >= n_shards and len(recs) >= n_keep:
                break
            with fs.open(f"datasets/{repo}/{sh}", block_size=2 ** 22) as fh:
                recs += [r[col]["bytes"] for r in pq.read_table(fh, columns=[col]).to_pylist()]
            print(f"  {repo}: {sh} ({len(recs)} so far)", flush=True)
        for k, b in enumerate(rng.sample(recs, min(n_keep, len(recs)))):
            im = Image.open(io.BytesIO(b))
            ext = {"JPEG": "jpg"}.get(im.format, (im.format or "png").lower())
            p = out / "images" / f"{repo.split('/')[0]}_{k:05d}.{ext}"
            p.write_bytes(b)
            rows.append(dict(source=repo, path=str(p.relative_to(DATA)), width=im.width, height=im.height, format=ext))
    pd.DataFrame(rows).to_csv(out / "pool.csv", index=False)
    print(pd.DataFrame(rows).groupby(["source", "format"]).size())


if __name__ == "__main__":
    main()
