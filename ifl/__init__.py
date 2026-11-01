"""ifl - image forgery localization with frozen self-supervised latents + light discriminators."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# both default to folders inside the repo (git-ignored); override to put them on another disk
DATA = Path(os.environ.get("IFL_DATA", ROOT / "data")).resolve()
OUTPUTS = Path(os.environ.get("IFL_OUTPUTS", ROOT / "outputs")).resolve()

# unified labels across every dataset
REAL, FULL_SYNTH, LOCAL_EDIT = 0, 1, 2
LABEL_NAMES = {REAL: "real", FULL_SYNTH: "full_synthetic", LOCAL_EDIT: "local_edit"}
