# NewImageForgeryLocalization

Localizing **diffusion / LLM-editor manipulations** (inpainting, instruction edits, full regeneration
by GPT-Image-2, Nano-Banana-2, FLUX ...) with the 2025-26 recipe:

> **frozen self-supervised latents  ->  a deliberately simple discriminator**
> (linear patch probe, one-class Gaussian on real patches, or training-free intra-image consistency)

Phase 0 (this commit) is the measurement layer: **are the latents representative of "manipulated"
at all, and are they representative for the right reasons?** Nothing gets trained on top of a
latent until it passes these tests. See [docs/01_reanalysis_and_direction.md](docs/01_reanalysis_and_direction.md)
for the review of the earlier projects, recent literature and the plan.

## Layout

```
configs/datasets.yaml        which datasets / how many random samples
ifl/data/sources.py          remote random sampling (parquet row groups, remote zip, tar index)
ifl/data/dataset.py          manifest loader, mask -> 24x24 grid, optional JPEG harmonisation
ifl/features/backbones.py    frozen extractors on a shared 24x24 grid
                             dinov2[:lK] dinov3 siglip2 clip | srm vaerec | pixels (control)
ifl/analysis/representativeness.py   image- & patch-level metrics (see module docstring)
ifl/analysis/viz.py          UMAP, patch-PCA galleries, score maps, transfer heatmaps
scripts/sample_datasets.py   -> data/samples/<dataset>/{images,masks}, data/samples/manifest.csv
scripts/extract_features.py  -> data/features/<backbone>/<dataset>/{global,patch}.npy
scripts/analyze_features.py  -> outputs/reports/<backbone>/report.html, outputs/reports/index.html
```

`data/` and `outputs/` are git-ignored (as are all images, weights, archives, arrays).

## Setup (Linux / cloud GPU)

Only a system Python is assumed. `setup_env.sh` installs [uv](https://github.com/astral-sh/uv), which fetches
Python 3.11, then installs a CUDA build of torch and `requirements.txt` into `.venv/`.

```bash
git clone https://github.com/jmaba/ImageForgeryLocalization.git && cd ImageForgeryLocalization
bash scripts/setup_env.sh                  # or: bash scripts/setup_env.sh cu121 | cpu
source .venv/bin/activate
bash scripts/download_data.sh              # test sets + real-only training pool -> ./data
```

Data goes to `./data` and results to `./outputs` by default. To use another disk, pass a path to
`download_data.sh` and export the same path afterwards:

```bash
bash scripts/download_data.sh /mnt/volume/ifl-data
export IFL_DATA=/mnt/volume/ifl-data       # read by every script (ifl/__init__.py)
export IFL_OUTPUTS=/mnt/volume/ifl-outputs # optional
```

The classical benchmarks (DiverseInpainting, IMD2020, Korus, NC2016, DEFACTO, FAU copy-move) are licence- or
request-gated and cannot be downloaded by script. Either point `IFL_BENCHMARKS` at a local copy, or pack the
already-sampled test subsets on a machine that has them and unpack them on the new one (same images exactly):

```bash
bash scripts/pack_samples.sh                                   # old machine -> ifl_samples.tar.gz (~3.5 GB)
SAMPLES_TAR=ifl_samples.tar.gz POOL_DIV2K=800 POOL_IMAGENET=20000 bash scripts/download_data.sh   # new machine
bash scripts/run_pipeline.sh                                   # features -> splices -> comparator -> predictor
PRED_D=384 PRED_LAYERS=8 PRED_STEPS=40000 PRED_BATCH=64 STEPS="pool predictor" bash scripts/run_pipeline.sh
```

Conda / mamba alternative: `mamba env create -f environment.yml && conda activate ifl`.
Model weights download from the Hugging Face Hub on first use. Gated ones (DINOv3, FLUX VAE) need the
licence accepted on the website and `hf auth login`.

## Pipeline

```bash
python scripts/extract_features.py --backbones pixels dinov2:l6 srm      # frozen 24x24 cell features of the test sets
python scripts/analyze_features.py                                      # representativeness reports
python scripts/region_tests.py --specs srm dinov2_l6 --controls         # training-free region consistency (docs/03 s.1)
python scripts/region_null.py --specs srm dinov2_l6                     # transplant null + image-level FPR (docs/03 s.2)
python scripts/make_splices.py --shard vae --versions 4     --types vae:sd vae:sdxl vae:qwen vae:taesd vae:movq proc real none # self-made real-only splices
python scripts/make_splices.py --shard inpaint --versions 1 --max-side 512 --types inpaint_sp inpaint_fr
python scripts/train_comparator.py --shards vae inpaint --tag all       # learned comparator (docs/03 s.5)
python scripts/train_comparator.py --shards vae inpaint --exclude-types vae:qwen --tag loo_qwen
python scripts/pool_features.py --variants clean jpeg resize            # real pool cell features (+ launderings)
python scripts/train_predictor.py --tag ctx --test-suffixes "" __jpeg75 # predictive self-consistency (real-only)
```
`scripts/run_pipeline.sh` runs the reported configuration end to end.

Open `outputs/reports/index.html` for the backbone comparison.

## Unified labels

`0` real, `1` fully synthetic, `2` locally manipulated (binary mask, 255 = manipulated).
Original image bytes are stored untouched; format, size and estimated JPEG quality are in the
manifest so format shortcuts can be measured (`confound_image`) or removed (`--harmonize jpeg95`).

## Notes

* DINOv3 is gated on the Hub: accept the licence and `hf auth login` to enable `dinov3`.
* Foundation-model streams see a resized square (336/384 px); `srm`/`vaerec` see the native image
  (capped at 1024 px long side) because resizing destroys the traces they read.
