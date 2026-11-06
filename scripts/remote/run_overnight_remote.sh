#!/usr/bin/env bash
# Overnight queue on the GPU pod, started after run_benchmark_remote.sh (waits for its BENCH_ALL_DONE marker):
#   1. baseline features (pixels, DINOv2-L6, SRM, Mesorch, TruFor) for the benchmark            -> morning analysis
#   2. fully synthetic set (SDXL, FLUX Fill text-to-image) + autoencoder maps + baseline features -> triage class
#   3. benchmark extension: the next 200 photos, same 3 editors + maps + features                -> held-out data
# Every step is resumable and saves per image; each step logs "== HH:MM <step>" and failures as "FAILED <step>".
#   nohup bash scripts/remote/run_overnight_remote.sh > logs/overnight.log 2>&1 &
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" IFL_BASELINES="${IFL_BASELINES:-/workspace/baselines}"
PY="${PY:-/workspace/venv/bin/python}"
step() { echo "== $(date +%H:%M) $*"; }

until grep -q BENCH_ALL_DONE logs/bench.log 2>/dev/null; do sleep 60; done
step "benchmark finished; overnight queue starts"

merge() {   # append a generated manifest to data/samples/manifest.csv (replacing earlier rows of its datasets)
  $PY - "$1" <<'EOF'
import sys
import pandas as pd
man = "data/samples/manifest.csv"
m = pd.read_csv(man, dtype={"sample_id": str, "pair_id": str})
b = pd.read_csv(sys.argv[1], dtype={"sample_id": str, "pair_id": str})
b["pair_id"] = b["pair_id"].where(~b["pair_id"].str.isdigit(), "p" + b["pair_id"])   # never numeric
m = pd.concat([m[~m.dataset.isin(set(b.dataset))], b], ignore_index=True)
m.to_csv(man, index=False)
print(f"merged {len(b)} rows: {b.dataset.value_counts().to_dict()}")
EOF
}
datasets() { $PY -c "import pandas as pd,sys; print(' '.join(sorted(pd.read_csv(sys.argv[1]).dataset.unique())))" "$1"; }
features() {   # $1 = manifest of the new images
  local ds; ds=$(datasets "$1")
  for b in pixels dinov2:l6 srm mesorch trufor; do
    step "features $b ($ds)"
    HF_HUB_OFFLINE=0 $PY -u scripts/extract_features.py --backbones "$b" --datasets $ds --workers 2 2>&1 \
      | grep -vE "it/s\]|s/it\]" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED features $b"
  done
}
maps() {   # $1 = manifest, $2 = output folder
  step "autoencoder maps -> $2"
  $PY - "$1" <<'EOF'
import sys
import pandas as pd
m = pd.read_csv(sys.argv[1], dtype=str)
pd.DataFrame({"group": m.dataset, "key": m.sample_id, "path": m.img_path, "control": "none"}).to_csv(
    "maps_list.csv", index=False)
EOF
  HF_HUB_OFFLINE=1 $PY -u scripts/remote/gen_consistency.py --list maps_list.csv --out "$2" --ae-only 2>&1 \
    | grep -E "^\[gencons\]|Error|models ready" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED maps $2"
}

# 1. baseline features for the pre-registered benchmark
merge data/samples/bench26/manifest.csv
features data/samples/bench26/manifest.csv
step "TASK1_DONE"

# 2. fully synthetic set
step "synthetic generation"
HF_HUB_OFFLINE=1 $PY -u scripts/remote/make_synthetic.py --n 300 --out data/samples/synth26 2>&1 \
  | grep -vE "Loading|Fetching|it/s\]" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED synthetic"
if [ -f data/samples/synth26/manifest.csv ]; then
  merge data/samples/synth26/manifest.csv
  maps data/samples/synth26/manifest.csv data/features/gencons_synth
  features data/samples/synth26/manifest.csv
fi
step "TASK2_DONE"

# 3. benchmark extension: photos 121-320 of the same seeded order, same editors, one process per editor
for ed in sdxlinp fluxfill kontext; do
  step "extension $ed"
  HF_HUB_OFFLINE=1 $PY -u scripts/remote/make_benchmark.py --skip 120 --n 200 --prefix benchx --editors "$ed" \
    --out data/samples/bench26x 2>&1 | grep -E "^\[|ready|resized|Error" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED extension $ed"
done
HF_HUB_OFFLINE=1 $PY -u scripts/remote/make_benchmark.py --skip 120 --n 200 --prefix benchx \
  --editors sdxlinp fluxfill kontext --out data/samples/bench26x 2>&1 | grep -E "manifest|Error"
if [ -f data/samples/bench26x/manifest.csv ]; then
  merge data/samples/bench26x/manifest.csv
  maps data/samples/bench26x/manifest.csv data/features/gencons_benchx
  features data/samples/bench26x/manifest.csv
fi
step "TASK3_DONE"
step "OVERNIGHT_ALL_DONE"
