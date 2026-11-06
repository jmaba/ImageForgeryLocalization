#!/usr/bin/env bash
# Final evaluation on the GPU pod within its 31 GB container memory limit (the parallel harness runs were OOM-killed):
# after B-Free's features, at most two harness processes at a time. The FLAME harness contains every scorer of the
# plain robustness harness, and the clean FLAME + B-Free harness contains everything, so four harness runs suffice.
#   nohup bash scripts/remote/run_final_remote.sh > logs/final.log 2>&1 &
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" IFL_BASELINES="${IFL_BASELINES:-/workspace/baselines}"
PY="${PY:-/workspace/venv/bin/python}"
step() { echo "== $(date +%H:%M) $*"; }

step "waiting for B-Free's features"
until grep -q "waiting for the FLAME run" logs/bfree.log 2>/dev/null; do sleep 60; done
pkill -f "[r]un_bfree_remote.sh"                     # its own harness step waited for markers that will not come

run() {   # $1 = log name, rest = eval_dtrace arguments
  local name=$1; shift
  $PY -u scripts/eval_dtrace.py "$@" > "logs/final_${name}.log" 2>&1 || echo "FAILED $name"
}
step "harness: clean with FLAME + B-Free, JPEG-75 with FLAME"
run fb_clean localization --with-flame --with-bfree &
run flame_jpeg75 localization --with-flame --suffix __jpeg75 &
wait
step "harness: WebP-80 and resize-50 with FLAME"
run flame_webp80 localization --with-flame --suffix __webp80 &
run flame_resize50 localization --with-flame --suffix __resize50 &
wait
step "image level with FLAME + B-Free, every condition"
for s in "" __jpeg75 __webp80 __resize50; do run "img${s:-_clean}" image --with-flame --with-bfree --suffix "$s"; done

step "tables"
$PY scripts/summarize_dtrace.py "" dtflamebfree > /dev/null 2>&1 || echo "FAILED summarize clean"
cp outputs/dtrace/localization_dtflamebfree_summary.csv outputs/dtrace/localization_flame_summary.csv
for s in __jpeg75 __webp80 __resize50; do
  $PY scripts/summarize_dtrace.py "$s" dtflame > /dev/null 2>&1 || echo "FAILED summarize $s"
done
for s in "" __jpeg75 __webp80 __resize50; do
  cp "outputs/dtrace/image_level_flame_bfree${s}.csv" "outputs/dtrace/image_level_flame${s}.csv" 2>/dev/null \
    || echo "FAILED copy image level ${s:-clean}"
done
$PY scripts/summarize_robust.py --flame > /dev/null 2>&1 || echo "FAILED robust summary"
step "FINAL_ALL_DONE"
