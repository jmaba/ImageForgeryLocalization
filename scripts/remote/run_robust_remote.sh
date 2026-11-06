#!/usr/bin/env bash
# Robustness of the decoder-trace detector and the published localizers on the GPU pod (docs/09): every feature for
# every dataset under JPEG-75, WebP-80 and resize-50 %, then the harness for clean + the three conditions (in
# parallel on the CPU), the tables and the cross-condition summary. Only the tables need to leave the pod.
#   nohup bash scripts/remote/run_robust_remote.sh > logs/robust.log 2>&1 &
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" IFL_BASELINES="${IFL_BASELINES:-/workspace/baselines}" \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/workspace/venv/bin/python}"
step() { echo "== $(date +%H:%M) $*"; }

step "GPT-Image-2 / Nano-Banana-2 paired with their COCO originals (matched nulls)"
$PY - <<'EOF'
import pandas as pd
man = "data/samples/manifest.csv"
m = pd.read_csv(man, dtype={"sample_id": str, "pair_id": str})
for ds in ("gpt_image_2", "nanobanana2"):
    for d in (ds, "coco_orig_" + ds):
        sel = m.dataset == d
        m.loc[sel, "pair_id"] = ds + ":" + m.loc[sel, "sample_id"]
m.to_csv(man, index=False)
print("pair ids set")
EOF

ALL=$($PY -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('data/samples/manifest.csv').dataset.unique())))")
for c in jpeg75 webp80 resize50; do
  for b in pixels dinov2:l6 srm mesorch trufor dtraceA dtraceB; do
    step "features $b __$c"
    $PY -u scripts/extract_features.py --backbones "$b" --datasets $ALL --harmonize "$c" --workers 8 2>&1 \
      | grep --line-buffered -E "Error|cannot" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED features $b __$c"
  done
done

step "harness: clean + 3 conditions in parallel"
for s in "" __jpeg75 __webp80 __resize50; do
  ( $PY -u scripts/eval_dtrace.py localization image --suffix "$s" > "logs/eval_dtrace${s:-_clean}.log" 2>&1 \
      || echo "FAILED evaluate ${s:-clean}" ) &
done
wait
for s in "" __jpeg75 __webp80 __resize50; do
  $PY scripts/summarize_dtrace.py "$s" > /dev/null 2>&1 || echo "FAILED summarize ${s:-clean}"
done
$PY scripts/summarize_robust.py > /dev/null 2>&1 || echo "FAILED robust summary"
step "ROBUST_ALL_DONE"
