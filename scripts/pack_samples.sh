#!/usr/bin/env bash
# Pack the sampled test subsets (data/samples: images, masks, manifest) so another machine evaluates on the
# EXACT same images, including the classical benchmarks that cannot be downloaded automatically.
#
#   bash scripts/pack_samples.sh                      # -> ifl_samples.tar.gz in the repo root (~3.5 GB)
#   bash scripts/pack_samples.sh /path/out.tar.gz
# On the new machine:  SAMPLES_TAR=/path/out.tar.gz bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT="${IFL_DATA:-$(pwd)/data}"
OUT="${1:-$(pwd)/ifl_samples.tar.gz}"
tar -czf "$OUT" -C "$DATA_ROOT" samples
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
