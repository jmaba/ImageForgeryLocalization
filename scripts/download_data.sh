#!/usr/bin/env bash
# Download every dataset the experiments use (random, seeded subsets). The only required setting is where to
# put them:
#
#   bash scripts/download_data.sh                  # -> <repo>/data   (default, git-ignored)
#   bash scripts/download_data.sh /mnt/vol/ifl     # -> anywhere; then export IFL_DATA=/mnt/vol/ifl
#                                                  #    before running the other scripts
#   bash scripts/download_data.sh "" sid_set tgif2 # only some test sets (names from configs/datasets.yaml)
#
# Optional environment variables:
#   SAMPLES_TAR=/path/ifl_samples.tar.gz   use the EXACT test subsets packed on another machine
#                                          (scripts/pack_samples.sh) instead of sampling again. This is the only
#                                          way to bring the licence-restricted classical benchmarks along.
#   IFL_BENCHMARKS=/path/Benchmarks        folder holding the classical benchmarks (DiverseInpainting, IMD2020,
#                                          Korus, NC2016, DEFACTO, CMFD) to sample from; see below
#   POOL_DIV2K=800 POOL_IMAGENET=20000     size of the real-only training pool (default 300 + 1000)
#
# What it fetches:
#   1. test sets (configs/datasets.yaml): OpenSDI, SID-Set, DiffSeg30k, TGIF2, CocoGlide, AutoSplice,
#      GPT-Image-2, Nano-Banana-2 (~10 GB of traffic, ~2 GB kept)  ->  $IFL_DATA/samples/{<dataset>/,manifest.csv}
#   2. classical benchmarks, only if IFL_BENCHMARKS points at them (they cannot be fetched automatically:
#      Korus, DEFACTO and NC2016 need an access request / licence; IMD2020 is at staff.utia.cas.cz/novozada/db,
#      the FAU copy-move benchmark at cs1.tf.fau.de, DiverseInpainting from the IID-Net authors)
#   3. real-only training pool (DIV2K + ImageNet-val, no COCO)  ->  $IFL_DATA/ssl_real/{images/,pool.csv}
# Model weights (DINOv2, VAEs, inpainting, SR) are pulled from the Hugging Face Hub on first use.
# Some Hub repos may need `hf auth login` (or HF_TOKEN) after accepting their terms on the website.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${1:-}"
shift || true
if [ -n "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    export IFL_DATA="$(cd "$DATA_DIR" && pwd)"
fi
DATA_ROOT="${IFL_DATA:-$(pwd)/data}"
mkdir -p "$DATA_ROOT"
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
echo "data dir: $DATA_ROOT   python: $PY"

PUBLIC="opensdi sid_set diffseg30k tgif2 gpt_image_2 nanobanana2 cocoglide autosplice"
LOCAL="diverse_inpaint imd2020_inpaint imd2020_real korus nc2016 defacto_inpaint cmfd_velo"

if [ "$#" -gt 0 ]; then
    "$PY" -u scripts/sample_datasets.py --only "$@"
    exit 0
fi

if [ -n "${SAMPLES_TAR:-}" ]; then
    echo "unpacking test subsets from $SAMPLES_TAR"
    tar -xzf "$SAMPLES_TAR" -C "$DATA_ROOT"
else
    "$PY" -u scripts/sample_datasets.py --only $PUBLIC
    if [ -n "${IFL_BENCHMARKS:-}" ] && [ -d "$IFL_BENCHMARKS" ]; then
        "$PY" -u scripts/sample_datasets.py --only $LOCAL
    else
        echo "IFL_BENCHMARKS not set: classical benchmarks skipped (use SAMPLES_TAR to bring them along)"
    fi
fi
"$PY" -u scripts/build_ssl_pool.py --div2k "${POOL_DIV2K:-300}" --imagenet "${POOL_IMAGENET:-1000}"
