"""ifl - image forgery localization with frozen self-supervised latents + light discriminators."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# unified labels across every dataset
REAL, FULL_SYNTH, LOCAL_EDIT = 0, 1, 2
LABEL_NAMES = {REAL: "real", FULL_SYNTH: "full_synthetic", LOCAL_EDIT: "local_edit"}
