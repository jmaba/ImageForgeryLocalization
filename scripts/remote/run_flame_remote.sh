#!/usr/bin/env bash
# FLAME (arXiv:2606.02178) as a baseline under the project's protocol (docs/10): its cell maps for every dataset in
# the clean, JPEG-75, WebP-80 and resize-50 % conditions, then - once the robustness run has produced the other
# features - the harness with FLAME next to Mesorch, TruFor, the comparator and the decoder-trace detectors.
#   nohup bash scripts/remote/run_flame_remote.sh > logs/flame.log 2>&1 &
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" IFL_BASELINES="${IFL_BASELINES:-/workspace/baselines}" \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/workspace/venv/bin/python}"
step() { echo "== $(date +%H:%M) $*"; }

ALL=$($PY -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('data/samples/manifest.csv').dataset.unique())))")
for c in "" jpeg75 webp80 resize50; do
  step "features flame ${c:-clean}"
  $PY -u scripts/extract_features.py --backbones flame --datasets $ALL ${c:+--harmonize $c} --workers 8 2>&1 \
    | grep --line-buffered -E "Error|cannot|Traceback" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED features flame ${c:-clean}"
done

step "waiting for the robustness run (other features and its harness)"
until grep -q ROBUST_ALL_DONE logs/robust.log 2>/dev/null; do sleep 60; done

step "harness with FLAME: clean + 3 conditions in parallel"
for s in "" __jpeg75 __webp80 __resize50; do
  ( $PY -u scripts/eval_dtrace.py localization image --with-flame --suffix "$s" > "logs/eval_flame${s:-_clean}.log" 2>&1 \
      || echo "FAILED evaluate ${s:-clean}" ) &
done
wait
for s in "" __jpeg75 __webp80 __resize50; do
  $PY scripts/summarize_dtrace.py "$s" dtflame > /dev/null 2>&1 || echo "FAILED summarize ${s:-clean}"
done
$PY scripts/summarize_robust.py --flame > /dev/null 2>&1 || echo "FAILED robust summary"
step "FLAME_ALL_DONE"
