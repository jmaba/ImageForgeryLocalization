#!/usr/bin/env bash
# Full benchmark run on the GPU pod (docs/08): generation (one process per editor, so GPU memory is released between
# them), the complete manifest, then the autoencoder maps of every image.
#   nohup bash scripts/remote/run_benchmark_remote.sh > logs/bench.log 2>&1 &
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" HF_HUB_OFFLINE=1
PY="${PY:-/workspace/venv/bin/python}"
N="${N:-120}"
OUT=data/samples/bench26
for ed in sdxlinp fluxfill kontext; do
  echo "== $(date +%H:%M) $ed"
  $PY -u scripts/remote/make_benchmark.py --n "$N" --editors "$ed" --out "$OUT" || echo "EDITOR_FAILED $ed"
done
echo "== $(date +%H:%M) manifest"
$PY -u scripts/remote/make_benchmark.py --n "$N" --editors sdxlinp fluxfill kontext --out "$OUT"
$PY - <<'EOF'
import pandas as pd
m = pd.read_csv("data/samples/bench26/manifest.csv", dtype=str)
pd.DataFrame({"group": m.dataset, "key": m.sample_id, "path": m.img_path, "control": "none"}).to_csv(
    "bench_list.csv", index=False)
print(len(m), "images listed")
EOF
echo "== $(date +%H:%M) autoencoder maps"
$PY -u scripts/remote/gen_consistency.py --list bench_list.csv --out data/features/gencons_bench --ae-only \
  || echo "MAPS_FAILED"
echo "== $(date +%H:%M) BENCH_ALL_DONE"
