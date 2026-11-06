#!/usr/bin/env bash
# Physics-stream feature extraction on a larger GPU (docs/07). Self-contained: run from the unpacked bundle root.
#
#   bash scripts/remote/run_physics_remote.sh            # all conditions
#   SMOKE=1 bash scripts/remote/run_physics_remote.sh    # one small dataset (cmfd_velo, 96 imgs) to measure speed
#   SUFFIXES="clean jpeg75" bash scripts/remote/run_physics_remote.sh
#
# Settings are identical to the local run (Marigold processing 512, 4 steps, 1024 px cap), so the features are
# interchangeable with ones made on the RTX 2060. Conditions run in parallel when the GPU has room (~7 GB each).
# Result: results_physics.tar (data/features/physics*/ + logs) -> untar in the repo root on the home machine.
set -euo pipefail
cd "$(dirname "$0")/../.."
SUFFIXES="${SUFFIXES:-clean jpeg75 webp80 resize50}"
# keep everything on the persistent volume (RunPod: /workspace survives a pod stop; the container disk does not):
# model cache, pip cache and a venv that reuses the template's torch
WS="${WS:-/workspace}"
if [ -d "$WS" ]; then
  export HF_HOME="$WS/hf_cache" PIP_CACHE_DIR="$WS/pip_cache" TORCH_HOME="$WS/torch_cache"
  [ -x "$WS/venv/bin/python" ] || python3 -m venv --system-site-packages "$WS/venv"
  PY="${PY:-$WS/venv/bin/python}"
fi
PY="${PY:-python}"
mkdir -p logs

echo "== environment"
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
  $PY -c "import torch" 2>/dev/null || $PY -m pip install -q "torch>=2.3" torchvision
  $PY -m pip install -q "diffusers==0.37.1" "transformers==5.0.0" "accelerate==1.12.0" safetensors \
      huggingface_hub numpy scipy "pandas>=2.1" pillow tqdm scikit-learn pyyaml
fi
$PY - <<'EOF'
import torch, diffusers, transformers
assert torch.cuda.is_available(), "no CUDA GPU visible"
p = torch.cuda.get_device_properties(0)
print(f"torch {torch.__version__}  diffusers {diffusers.__version__}  transformers {transformers.__version__}")
print(f"GPU {p.name}  {p.total_memory / 2**30:.1f} GB")
EOF

echo "== download models once (avoids parallel download races)"
$PY -c "from ifl.features.backbones import build; build('physics'); print('models ready')"

run() {   # $1 = condition
  local h=""; [ "$1" != "clean" ] && h="--harmonize $1"
  local ds=""; [ "${SMOKE:-0}" = "1" ] && ds="--datasets cmfd_velo"
  local t0=$(date +%s)
  if $PY -u scripts/extract_features.py --backbones physics --workers 2 $h $ds > "logs/phys_$1.log" 2>&1; then
    echo "$1 done in $(( $(date +%s) - t0 )) s" | tee -a logs/timing.txt
  else
    echo "$1 FAILED after $(( $(date +%s) - t0 )) s, see logs/phys_$1.log" | tee -a logs/timing.txt
  fi
}

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NPAR=$(( VRAM_MB / 7000 )); [ "$NPAR" -lt 1 ] && NPAR=1
[ "${SMOKE:-0}" = "1" ] && SUFFIXES="clean" && NPAR=1
echo "== extracting: $SUFFIXES  ($NPAR in parallel, ${VRAM_MB} MiB)"
T0=$(date +%s)
for s in $SUFFIXES; do
  while [ "$(jobs -rp | wc -l)" -ge "$NPAR" ]; do wait -n; done
  run "$s" &
done
wait
echo "total $(( $(date +%s) - T0 )) s" | tee -a logs/timing.txt

echo "== check"
$PY - <<'EOF'
from pathlib import Path
import pandas as pd
m = pd.read_csv("data/samples/manifest.csv")
n = m.groupby("dataset").size()
for d in sorted(Path("data/features").glob("physics*")):
    done = {p.parent.name for p in d.glob("*/ids.txt")
            if len(p.read_text().split()) == n.get(p.parent.name, -1)}
    print(f"{d.name:22s} {len(done)}/{len(n)} datasets complete")
EOF
tar -cf results_physics.tar data/features/physics* logs
echo "== wrote results_physics.tar ($(du -h results_physics.tar | cut -f1)); copy it home and untar in the repo root"
