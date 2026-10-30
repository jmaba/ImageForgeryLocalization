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

## Quick start (Windows, existing conda env)

```bash
set PY=C:\Users\jm_ab\miniconda3\envs\dm-image-detection\python.exe
%PY% scripts/sample_datasets.py                     # ~3k images, ~1 GB, random but seeded
%PY% scripts/extract_features.py --backbones pixels dinov2 dinov2:l6 siglip2 clip srm vaerec
%PY% scripts/analyze_features.py
```

Open `outputs/reports/index.html` for the backbone comparison.

## Unified labels

`0` real, `1` fully synthetic, `2` locally manipulated (binary mask, 255 = manipulated).
Original image bytes are stored untouched; format, size and estimated JPEG quality are in the
manifest so format shortcuts can be measured (`confound_image`) or removed (`--harmonize jpeg95`).

## Notes

* DINOv3 is gated on the Hub: accept the licence and `hf auth login` to enable `dinov3`.
* Foundation-model streams see a resized square (336/384 px); `srm`/`vaerec` see the native image
  (capped at 1024 px long side) because resizing destroys the traces they read.
