#!/usr/bin/env bash
# Download every dataset the experiments use (random, seeded subsets; ~10 GB of network traffic,
# ~3 GB kept on disk). The only setting is where to put them:
#
#   bash scripts/download_data.sh                  # -> <repo>/data   (default, git-ignored)
#   bash scripts/download_data.sh /mnt/vol/ifl     # -> anywhere; then export IFL_DATA=/mnt/vol/ifl
#                                                  #    before running the other scripts
#   bash scripts/download_data.sh "" sid_set tgif2 # only some test sets (names from configs/datasets.yaml)
#
# What it fetches:
#   1. test sets (configs/datasets.yaml): OpenSDI, SID-Set, DiffSeg30k, TGIF2, CocoGlide, AutoSplice,
#      GPT-Image-2, Nano-Banana-2  ->  $IFL_DATA/samples/{<dataset>/,manifest.csv}
#   2. real-only training pool (DIV2K + ImageNet-val, no COCO)  ->  $IFL_DATA/ssl_real/{images/,pool.csv}
# Model weights (DINOv2, VAEs, inpainting) are pulled from the Hugging Face Hub on first use.
# Some Hub repos may need `hf auth login` (or HF_TOKEN) after accepting their terms on the website.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${1:-}"
shift || true
if [ -n "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    export IFL_DATA="$(cd "$DATA_DIR" && pwd)"
fi
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
echo "data dir: ${IFL_DATA:-$(pwd)/data}   python: $PY"

if [ "$#" -gt 0 ]; then
    "$PY" -u scripts/sample_datasets.py --only "$@"
else
    "$PY" -u scripts/sample_datasets.py
    "$PY" -u scripts/build_ssl_pool.py
fi
