#!/usr/bin/env bash
# Decoder-trace detector on the GPU pod (docs/09): train split A (5 autoencoders, 5 held out) and split B (all 10),
# extract every feature for every dataset (old datasets, benchmark, extension, synthetic set, the COCO originals of
# GPT-Image-2 / Nano-Banana-2), then evaluate through the shared harness. Only result tables need to leave the pod.
#   nohup bash scripts/remote/run_dtrace_remote.sh > logs/dtrace.log 2>&1 &
cd "$(dirname "$0")/../.."
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" IFL_BASELINES="${IFL_BASELINES:-/workspace/baselines}" \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-/workspace/venv/bin/python}"
step() { echo "== $(date +%H:%M) $*"; }

step "COCO originals of GPT-Image-2 / Nano-Banana-2 as datasets"
$PY - <<'EOF'
import pandas as pd
from PIL import Image
L = pd.read_csv("pilot_list.csv", dtype=str)
L = L[L.group.str.startswith("coco_orig") & (L.control == "none")]
rows = []
for r in L.itertuples():
    w, h = Image.open("data/" + r.path).size
    rows.append(dict(dataset=r.group, subset="all", sample_id=r.key, label=0, img_path=r.path, mask_path="",
                     width=w, height=h, format="jpg", mode="RGB", mask_frac=0.0, pair_id="coco" + r.key, category=""))
b = pd.DataFrame(rows)
man = "data/samples/manifest.csv"
m = pd.read_csv(man, dtype={"sample_id": str, "pair_id": str})
pd.concat([m[~m.dataset.isin(set(b.dataset))], b], ignore_index=True).to_csv(man, index=False)
print("added", b.dataset.value_counts().to_dict())
EOF

step "train A and B (shared round-trip cache)"
$PY -u scripts/remote/dtrace_train.py train --splits A B --steps "${STEPS:-6000}" 2>&1 \
  | grep -E '^\{"split|^cache|Error|Traceback' ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED train"

ALL=$($PY -c "import pandas as pd; print(' '.join(sorted(pd.read_csv('data/samples/manifest.csv').dataset.unique())))")
for b in pixels dinov2:l6 srm mesorch trufor dtraceA dtraceB; do
  step "features $b"
  $PY -u scripts/extract_features.py --backbones "$b" --datasets $ALL --workers 4 2>&1 \
    | grep -vE "it/s\]|s/it\]" | grep -E "imgs|Error|cannot" ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED features $b"
done

step "evaluate"
$PY -u scripts/eval_dtrace.py 2>&1 | grep -vE "it/s\]|s/it\]" | tail -60 ; [ "${PIPESTATUS[0]}" = 0 ] || echo "FAILED evaluate"
step "DTRACE_ALL_DONE"
