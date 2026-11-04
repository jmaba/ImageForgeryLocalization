# Robustness, predictive self-consistency and fusion: results (2026-11-04)

Scripts:
- `scripts/train_comparator.py`: the learned comparator; `--save` writes the weights.
- `scripts/pool_features.py`
- `scripts/train_predictor.py`: options `--pool-frac`, `--targets` and `--fuse`.

Rows are in `outputs/comparator/rows_<tag>[__<perturbation>].csv`.

Protocol (as in docs/03 s.5):
- per-image cell AUC on 24x24 cells, 150 images per dataset;
- interior-only AUC: a one-cell boundary band is dropped;
- matched transplant null: an authentic image is scored with its own edit's mask, and must be 0.5;
- pixel-colour control `PIX_loc_cf`.

Re-shared test copies: JPEG-75, WebP-80, 50 % downscale (`__resize50`), and LDM 4x super-resolution (`__sr4`, TGIF2 only).

## 1. Two data/protocol bugs found and fixed

**cmfd_velo was not a copy-move test set.**
- The sampler took the benchmark's snippet layers (`X_N.png`: the copied object on a white RGBA canvas) as the "forged" images.
- Every edited cell was therefore an outlier: cell colour norm gave AUC 0.99 and a random projection gave 0.96.
- `ifl/data/sources.py` now uses the real forgery `X_copy.png`. The mask is where it differs from the lossless original `X.png`.
- The originals are the matched authentic images, so cmfd_velo is now in the transplant null.
- Result: 48 forgeries + 48 originals, median mask 3.2 %.
- **All earlier cmfd_velo numbers are void.**
- On the corrected set, every scorer is at its own null:
  - comparator: edited 0.52 vs authentic 0.60;
  - predictor: 0.62 vs 0.61.
- This is expected. A copy-move region comes from the same camera and pipeline, so a source-consistency method has nothing to find.

**The SR4 "detection jump" on SD2-FR is a resolution confound.**
- SD2-FR images are 512 px; their originals and every other TGIF2 set are 1024 px.
- `make_perturbed.py --mode sr4` synthesises at most 512 px and resizes back. So only the 1024-px images are upsampled 2x afterwards, and they come out blurrier than SD2-FR.
- Under SR4, image-level AUC falls to about 0.5 on every 1024-px set and rises to 0.98 on SD2-FR.
- The clean SD2-FR image-level numbers (TPR 77–99 %, AUC 0.90–0.97) are partly inflated by the same 512-vs-1024 gap. SDXL-FR, which matches its originals in size, reaches 0.80–0.87.
- Image-level claims on SD2-FR need a size-matched control: originals downscaled to 512.

## 2. Predictive self-consistency (FoJEPA-lite)

The design:
- A real-only transformer sees the 24x24 frozen cell tokens (DINOv2-L6 + SRM, PCA-256 per stream) plus each cell's pixel statistics.
- It predicts hidden 3–8-cell blocks.
- At test time a sliding 4x4 block (stride 2) hides every cell, and the score is the mean prediction error per cell.
- Training data: 1,300 real pool images (DIV2K + ImageNet-val, no COCO), each clean plus a JPEG and a resize laundering.
- No edit, generator or mask is seen in training.

### 2.1 First run (`ctx_small`: d=192, 4 layers, 6,000 steps, all streams predicted)

Mean AUC over 18 datasets:

| condition | predictor `ctx_all` | pixel control | comparator `pair_ref` |
|---|---|---|---|
| clean | 0.620 | 0.612 | 0.681 |
| JPEG-75 | 0.613 | 0.610 | 0.657 |
| WebP-80 | 0.610 | 0.611 | 0.646 |
| resize 50 % | 0.616 | 0.612 | 0.612 |

- **Re-sharing barely moves it** (-0.01 at most). The comparator loses up to 0.07.
- **It wins on manual, real-life edits**, the comparator's blind spot:
  - NC2016: 0.703 vs 0.649;
  - Korus: 0.649 vs 0.635;
  - SID-Set: 0.719 vs 0.704.
- **It loses on diffusion splices:** SD2 splice 0.623 vs 0.791.
- **As a stand-alone detector it is barely above the colour control.**

### 2.2 Data scaling: flat

Training images are nested subsets; the validation images are the same in every run.

| real pool used | training images | val masked MSE | `ctx_all` clean | `ctx_all` JPEG-75 | null |
|---|---|---|---|---|---|
| 25 % | 828 | 0.736 | 0.618 | 0.610 | 0.542 |
| 50 % | 1,656 | 0.705 | 0.621 | 0.613 | 0.542 |
| 100 % | 3,315 | 0.699 | 0.620 | 0.613 | 0.542 |

The prediction loss improves with data, but localisation does not move, on any single dataset either. **More data or compute for this design is not justified.**

### 2.3 Trace-only targets and content-normalised error: negative

`--targets srm`: only the SRM residual stream is hidden and predicted. DINOv2 and the pixel statistics of every cell, hidden ones included, are context.

`ctx_norm`: the log error minus the log error expected for a REAL cell with the same content token. The expectation is a ridge regression fitted on 300 held-out pool images. Content explains 27 % of the real-cell log-error variance with trace targets, and 21 % with all streams.

| mean AUC | pixel control | all streams `ctx_all` | all streams `ctx_norm` | SRM only `ctx_all` | SRM only `ctx_norm` |
|---|---|---|---|---|---|
| clean | 0.612 | 0.620 | 0.593 | 0.606 | 0.551 |
| JPEG-75 | 0.610 | 0.613 | 0.584 | 0.580 | 0.522 |
| WebP-80 | 0.611 | 0.611 | 0.580 | 0.556 | 0.504 |
| resize 50 % | 0.612 | 0.616 | 0.587 | 0.577 | 0.521 |
| null (clean) | 0.533 | 0.546 | 0.538 | 0.527 | 0.509 |

- **Trace targets** help a little on diffusion splices (SD2 splice 0.663 vs 0.623) and lose elsewhere. The re-sharing robustness disappears, because SRM is exactly what JPEG and WebP destroy. The robust signal of section 2.1 came from the semantic stream.
- **Removing the part of the error that content predicts** removes most of the localisation (down to about the null with trace targets). What the predictor localises is mostly content that the context does not explain, not acquisition traces.

The pre-registered pass criterion was ≥ 0.65 mean and null ≤ 0.55. **No variant passes.** Self-consistency by feature prediction on frozen DINOv2/SRM tokens is not pursued further at scale.

## 3. Fusion with the comparator

The comparator is retrained with the same configuration as `rbc_base` and saved (`--save`; it reproduces 0.689 vs 0.690). Scores are fused per image:

- `cmp_z2`: the comparator's own two heads, z-scored `pair_ref` + `point`. This is the control: any fusion gain must beat it.
- `fuse_z_point` = z(`ctx_all`) + z(`pair_ref`) + z(`point`), with the all-stream predictor.
- `fuse_norm`: the same, with `ctx_norm` in place of `ctx_all`.

Mean AUC over 19 datasets (cmfd_velo corrected):

| condition | comparator `pair_ref` | `cmp_z2` (control) | `fuse_z_point` | `fuse_norm` |
|---|---|---|---|---|
| clean | 0.679 | 0.679 | **0.699** | 0.688 |
| JPEG-75 | 0.655 | 0.667 | **0.684** | 0.674 |
| WebP-80 | 0.644 | 0.660 | **0.678** | 0.668 |
| resize 50 % | 0.611 | 0.610 | **0.639** | 0.627 |
| interior, clean | 0.689 | 0.688 | 0.700 | 0.693 |
| null, clean | 0.536 | 0.536 | 0.550 | 0.546 |

Part of the gain is matched by a higher null. Per domain, against the control and net of the null change (clean):

| domain | edited `cmp_z2` → fusion | null `cmp_z2` → fusion | net gain |
|---|---|---|---|
| NC2016 | 0.645 → 0.691 | 0.542 → 0.548 | **+0.04** |
| IMD2020-real | 0.704 → 0.763 | 0.533 → 0.550 | **+0.04** |
| SID-Set | 0.704 → 0.752 | 0.569 → 0.575 | **+0.04** |
| Korus | 0.621 → 0.665 | 0.534 → 0.562 | +0.02 |
| OpenSDI | 0.654 → 0.669 | 0.506 → 0.508 | +0.01 |
| TGIF2 (all) | 0.670 → 0.679 | 0.533 → 0.542 | 0.00 |

JPEG-75 gives the same picture. The predictor adds a real, re-sharing-robust +0.04 on manual and real-life edits, and nothing on diffusion splices, where the comparator is already strong.

**Under re-sharing**, ensembling the comparator's own two heads adds +0.01 to +0.02. The fusion results of the previous run (`fuse`, without this control) overstated the predictor's share by that amount.

## 4. Verdict

- **Current best system:** the real-only comparator, its own two heads, and the all-stream context predictor, z-fused per image (0.699 clean, 0.684 JPEG-75, 0.678 WebP-80, 0.639 resize 50 %).
  - It reaches 0.699 clean and 0.684 at JPEG-75, against 0.612 for the colour control.
  - Its null is 0.55.
- **Not worth scaling:** the context predictor in this form. It doesn't improve with data, and its residual signal after content normalisation is small.
- **Open:**
  - a size-matched control for SD2-FR image-level detection;
  - the comparator's weakness on WebP and resize;
  - copy-move, which is out of scope for source consistency and needs a duplicate-matching component if it matters.
