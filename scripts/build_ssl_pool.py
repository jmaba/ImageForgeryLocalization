"""Real-photo pool for self-supervised (real-only) models. Deliberately NOT COCO, because most test edits
(CocoGlide, TGIF2, GPT-Image-2, Nano-Banana-2, OpenSDI) start from COCO photos.

    python scripts/build_ssl_pool.py            -> data/ssl_real/{images/*, pool.csv}
"""
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
    out = DATA / "ssl_real"
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)
    fs, api = HfFileSystem(), HfApi()
    rows = []
    for repo, col, n_shards, n_keep, _ in SOURCES:
        shards = sorted(f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet"))
        recs = []
        for sh in rng.sample(shards, n_shards):
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
