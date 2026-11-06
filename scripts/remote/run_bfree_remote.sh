#!/usr/bin/env bash
# B-Free (CVPR 2025) as a baseline (docs/10): its image-level score for every dataset in all four conditions, its
# sliding-window map (our adaptation) on the clean images, then - after the robustness and FLAME runs - the clean
# harness with FLAME and B-Free next to everything else, and the image-level table with both for every condition.
#   nohup bash scripts/remote/run_bfree_remote.sh > logs/bfree.log 2>&1 &
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" IFL_BASELINES="${IFL_BASELINES:-/workspace/baselines}" \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/workspace/venv/bin/python}"
step() { echo "== $(date +%H:%M) $*"; }

ALL=$($PY -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('data/samples/manifest.csv').dataset.unique())))")
step "features bfree (map, clean)"
$PY -u scripts/extract_features.py --backbones bfree --datasets $ALL --workers 8 2>&1 \
  | grep --line-buffered -E "Error|cannot|Traceback" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED features bfree"
for c in "" jpeg75 webp80 resize50; do
  step "features bfreeimg ${c:-clean}"
  $PY -u scripts/extract_features.py --backbones bfreeimg --datasets $ALL ${c:+--harmonize $c} --workers 8 2>&1 \
    | grep --line-buffered -E "Error|cannot|Traceback" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED features bfreeimg ${c:-clean}"
done

step "waiting for the FLAME run and the robustness reruns"
until grep -q FLAME_ALL_DONE logs/flame.log 2>/dev/null && grep -q RERUN_DONE logs/rerun_robust.log 2>/dev/null; do
  sleep 60
done

step "clean harness with FLAME and B-Free; image level with both for every condition"
$PY -u scripts/eval_dtrace.py localization --with-flame --with-bfree > logs/eval_fb_clean.log 2>&1 || echo "FAILED evaluate fb"
for s in "" __jpeg75 __webp80 __resize50; do
  $PY -u scripts/eval_dtrace.py image --with-flame --with-bfree --suffix "$s" > "logs/eval_fb_image${s:-_clean}.log" 2>&1 \
    || echo "FAILED image level ${s:-clean}"
done
$PY scripts/summarize_dtrace.py "" dtflamebfree > /dev/null 2>&1 || echo "FAILED summarize fb"
step "BFREE_ALL_DONE"
