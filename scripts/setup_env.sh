#!/usr/bin/env bash
# Create the project environment with uv, assuming only a system Python (>= 3.8) is installed.
# uv downloads Python 3.11 itself, so the system Python version does not matter.
#
#   bash scripts/setup_env.sh            # CUDA 12.4 wheels (default; fine for RTX 30xx/40xx, A100, L40S, H100)
#   bash scripts/setup_env.sh cu121      # older drivers
#   bash scripts/setup_env.sh cpu        # no GPU
#   source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_FLAVOR="${1:-cu124}"

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_FLAVOR}"
uv pip install --python .venv/bin/python -r requirements.txt

.venv/bin/python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
EOF
echo "Done. Activate with:  source .venv/bin/activate"
