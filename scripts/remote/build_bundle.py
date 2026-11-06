"""Pack what the remote physics run needs into one tar: code (tracked + untracked, not ignored), the sampled
test images (data/samples, manifest rewritten with forward slashes), and the cell mask grids.

    python scripts/remote/build_bundle.py M:/SourceCodeM/ifl_gpu_bundle.tar
"""
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(out):
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.splitlines()
    files = [f for f in files if (ROOT / f).is_file()]
    m = pd.read_csv(ROOT / "data/samples/manifest.csv")
    for c in ("img_path", "mask_path"):
        m[c] = m[c].astype("string").str.replace("\\", "/", regex=False)
    missing = [p for p in m.img_path if not (ROOT / "data" / p).exists()]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    with tarfile.open(out, "w") as tar:
        for f in files:
            tar.add(ROOT / f, arcname=f"ifl/{f}")
        for p in sorted((ROOT / "data/samples").rglob("*")):
            if p.is_file() and p.name != "manifest.csv":
                tar.add(p, arcname=f"ifl/{p.relative_to(ROOT).as_posix()}")
        buf = m.to_csv(index=False).encode()
        ti = tarfile.TarInfo("ifl/data/samples/manifest.csv")
        ti.size = len(buf)
        tar.addfile(ti, io.BytesIO(buf))
        tar.add(ROOT / "data/features/_masks", arcname="ifl/data/features/_masks")
    print(f"{out}: {len(files)} code files, {len(m)} images, {Path(out).stat().st_size / 2**30:.2f} GB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT.parent / "ifl_gpu_bundle.tar"))
