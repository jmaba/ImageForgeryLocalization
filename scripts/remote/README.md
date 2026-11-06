# Physics features on a bigger GPU

The physics stream (`ifl/features/physics.py`) runs three models per image: Marigold normals, Marigold intrinsics
and Depth Anything V2. On the 6 GB RTX 2060 it runs at over 20 s per image. On a 24 GB card it should take about
1 s per image, with several conditions running in parallel.

## What to rent

- **GPU:** any single NVIDIA card with at least 16 GB (RTX 4090, L4, A10G, A100, or a university cluster node).
  - With 24 GB, three conditions run in parallel.
  - With 40 GB or more, all four run in parallel.
- **Disk:** at least 30 GB. The bundle is 4.2 GB; the models and Python packages take about 10 GB.
- **Software:** Linux with CUDA, Python 3.10–3.12, and PyTorch 2.3 or newer. Any "PyTorch" cloud image has these.
  The script installs the rest, pinned to the versions used locally (diffusers 0.37.1, transformers 5.0.0).
- **Internet:** needed to download the models from Hugging Face on the first run (about 5 GB, all public, no token).

## Steps

```bash
# 1. copy the bundle up and unpack
scp ifl_gpu_bundle.tar user@gpu-host:/workspace/
ssh user@gpu-host
cd /workspace && tar -xf ifl_gpu_bundle.tar && rm ifl_gpu_bundle.tar && cd ifl

# 2. a 3-5 minute check: installs packages, downloads models, runs one small dataset and prints the time
SMOKE=1 bash scripts/remote/run_physics_remote.sh
cat logs/timing.txt          # "clean done in N s" for 96 images -> N/96 s per image

# 3. the full run: clean, jpeg75, webp80, resize50 (4,601 images each); expect roughly 2-3 h on a 24 GB card
nohup bash scripts/remote/run_physics_remote.sh > logs/run.log 2>&1 &
tail -f logs/run.log         # progress per condition: logs/phys_<condition>.log
```

- **Setting conditions:** set `SUFFIXES` to choose which ones run, for example `SUFFIXES="clean jpeg75"`.
- **Skipping the install:** `SKIP_INSTALL=1` skips the package install.
- **Re-running:** a re-run skips datasets that are already complete.
- **The full run starts from the smoke output:** it reuses the cmfd_velo features from the smoke check, which
  were made with the same settings.

## Bring the results back

```bash
scp user@gpu-host:/workspace/ifl/results_physics.tar .
tar -xf results_physics.tar -C M:/SourceCodeM/NewImageForgeryLocalization
```

This fills `data/features/physics/` and `data/features/physics__jpeg75/` (plus webp80 and resize50). The scoring and
the ensemble then run locally, because they need the other cached features (25 GB) that are not in the bundle:

```bash
python scripts/eval_physics.py --test-suffixes "" __jpeg75 __webp80 __resize50
python scripts/ensemble.py --suffixes "" __jpeg75
```

## Licences

- **Code:** the bundle holds only this repository's code and the sampled test images.
- **Model weights:** the models download from their official Hugging Face pages.
  - Marigold v1.1: Apache-2.0.
  - Depth Anything V2 Small: Apache-2.0.
- **Not needed:** TruFor and Mesorch are not part of this run.
