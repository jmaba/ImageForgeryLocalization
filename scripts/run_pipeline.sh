#!/usr/bin/env bash
# End-to-end run on a fresh (bigger) machine, after scripts/setup_env.sh and scripts/download_data.sh.
# Every step skips work that already exists (features) or is re-run cheaply, so the script can be restarted.
#
#   bash scripts/run_pipeline.sh                       # small settings (what ran on the RTX 2060)
#   PRED_D=384 PRED_LAYERS=8 PRED_STEPS=40000 PRED_BATCH=64 SPLICE_LIMIT=5000 bash scripts/run_pipeline.sh
#
# Knobs (environment variables):
#   WORKERS=4        DataLoader workers for feature extraction (each holds a copy of torch: ~2-3 GB RAM)
#   SPLICE_LIMIT=    pool images used for the comparator's splices (default: whole pool; 4 splices each)
#   PRED_D/PRED_LAYERS/PRED_HEADS/PRED_STEPS/PRED_BATCH   context-predictor size (default 192/4/6/6000/32)
#   STEPS=           comma-free list of steps to run, e.g. STEPS="pool predictor" (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}"
export PYTHONIOENCODING=utf-8 TRANSFORMERS_VERBOSITY=error PYTHONWARNINGS=ignore
W="${WORKERS:-4}"
LIM="${SPLICE_LIMIT:+--limit $SPLICE_LIMIT}"
SUF="'' __jpeg75 __webp80 __resize50 __sr4"
TGIF2="tgif2_orig tgif2_sd2_fr tgif2_sd2_sp tgif2_sdxl_fr tgif2_flux1dev_fr tgif2_flux1dev_sp"
ALL="features perturbed splices pool comparator predictor"
run() { case " ${STEPS:-$ALL} " in *" $1 "*) echo "[$(date +%H:%M)] === $1"; return 0;; *) return 1;; esac; }

if run features; then      # test-set cell features (clean)
    $PY -u scripts/extract_features.py --backbones dinov2:l6 srm pixels dtcwt_phase --workers "$W"
fi
if run perturbed; then     # re-shared test copies: JPEG / WebP / downscale, and LDM 4x super-resolution (TGIF2)
    for m in jpeg75 webp80 resize50; do
        $PY -u scripts/extract_features.py --backbones dinov2:l6 srm pixels --harmonize $m --workers "$W"
    done
    $PY -u scripts/make_perturbed.py --mode sr4 --datasets $TGIF2
    $PY -u scripts/extract_features.py --backbones dinov2:l6 srm pixels --harmonize sr4 --workers "$W" --datasets $TGIF2
fi
if run splices; then       # real-only self-made splices for the comparator (exact commands of the reported runs)
    $PY -u scripts/make_splices.py --shard vae --versions 4 $LIM \
        --types vae:sd vae:sdxl vae:qwen vae:taesd vae:movq proc real none --probs 1 1 1 1 1 1 0.5 0.7
    $PY -u scripts/make_splices.py --shard inpaint --versions 1 --max-side 512 $LIM \
        --types inpaint_sp inpaint_fr --probs 2 1
fi
if run pool; then          # real pool cell features, clean + globally laundered (context predictor)
    $PY -u scripts/pool_features.py --variants clean jpeg resize
fi
if run comparator; then    # reference: learned same-source comparator, clean + perturbed test copies
    eval $PY -u scripts/train_comparator.py --shards vae inpaint --exclude-types inpaint_fr --tag base \
        --test-suffixes $SUF
fi
if run predictor; then     # predictive self-consistency (FoJEPA-lite)
    eval $PY -u scripts/train_predictor.py --tag ctx_d${PRED_D:-192}_l${PRED_LAYERS:-4} \
        --d "${PRED_D:-192}" --layers "${PRED_LAYERS:-4}" --heads "${PRED_HEADS:-6}" \
        --steps "${PRED_STEPS:-6000}" --batch "${PRED_BATCH:-32}" --test-suffixes $SUF
fi
echo "[$(date +%H:%M)] done; results in ${IFL_OUTPUTS:-outputs}/comparator/results_*.md"
