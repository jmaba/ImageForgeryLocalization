# Published baselines under our protocol (2026-11-04)

How the baselines were run:
- TruFor (CVPR 2023) and Mesorch (AAAI 2025), with their official code and released weights.
- Wrapper: `ifl/features/baselines.py`. Evaluation: `scripts/eval_baselines.py`.
- Each heatmap is pooled to the 24x24 grid (cell mean).
- Everything else is identical to docs/05: same images and masks, interior AUC, matched transplant null, colour control, and the same JPEG-75 / WebP-80 / resize-50 / SR4 copies.
- Input is capped at 1024 px for every method.
- Rows are in `outputs/comparator/rows_bl_{trufor,mesorch}*.csv`.

Caveats:
- Both baselines are supervised. They were trained on CASIA2, FantasticReality, IMD2020 and tampCOCO (TruFor also on compRAISE), so the IMD2020 test sets are in-distribution for them. Means without them are reported separately.
- SAFIRE could not be evaluated: its weights are restricted on Google Drive.
- Our system is the docs/05 best: real-only comparator + context predictor, z-fused (`fuse2` / `fuse_z_point`). It never sees a manipulated image with a real mask.

## 1. Mean cell AUC (19 datasets)

| condition | TruFor | Mesorch | ours: comparator | ours: fused | pixel control |
|---|---|---|---|---|---|
| clean | **0.780** | 0.775 | 0.679 | 0.699 | 0.612 |
| clean, excl. IMD2020 | **0.770** | 0.762 | 0.672 | 0.691 | 0.603 |
| JPEG-75 | 0.702 | **0.748** | 0.655 | 0.684 | 0.610 |
| WebP-80 | 0.685 | **0.742** | 0.644 | 0.678 | 0.611 |
| resize 50 % | 0.697 | **0.752** | 0.611 | 0.639 | 0.612 |
| interior, clean | **0.785** | 0.780 | 0.689 | 0.700 | 0.593 |
| null, clean (must be 0.5) | 0.553 | **0.589** | 0.536 | 0.550 | 0.533 |

- **Both published models are clearly better on average**: about +0.08 over our fused system on clean images.
- **Under re-sharing, Mesorch keeps most of its accuracy** (-0.03). TruFor loses 0.08–0.10, about twice what we lose.
- **Mesorch's null is the highest** (0.59 overall, 0.67 on IMD2020-real, 0.64 on cmfd_velo). Part of its score comes from where edits usually sit and what edited objects look like, not from the edit itself.
  - Net of the null (edited minus null, clean): TruFor 0.227, Mesorch 0.186, ours 0.149, pixel control 0.079.
  - Under JPEG-75: TruFor 0.163, Mesorch 0.157, ours 0.132.

## 2. Where each method wins (clean / JPEG-75)

| dataset | TruFor | Mesorch | ours: fused |
|---|---|---|---|
| TGIF2 SD2 splice | **0.999** / 0.732 | 0.754 / 0.754 | 0.802 / **0.755** |
| TGIF2 FLUX splice | **0.913** / 0.585 | 0.746 / **0.705** | 0.677 / 0.639 |
| TGIF2 SDXL full regeneration | 0.625 / 0.641 | 0.689 / 0.688 | **0.709 / 0.727** |
| TGIF2 FLUX full regeneration | 0.554 / 0.538 | 0.636 / 0.623 | **0.642 / 0.627** |
| GPT-Image-2 | 0.515 / 0.519 | 0.531 / 0.531 | **0.564 / 0.577** |
| Nano-Banana-2 | 0.689 / 0.675 | 0.679 / 0.664 | **0.698 / 0.680** |
| CocoGlide | 0.891 / 0.837 | **0.935 / 0.840** | 0.738 / 0.691 |
| NC2016 | 0.801 / 0.800 | **0.852 / 0.856** | 0.691 / 0.701 |
| SID-Set | 0.816 / 0.816 | **0.875 / 0.880** | 0.752 / 0.752 |
| Korus | 0.756 / 0.747 | **0.762 / 0.763** | 0.665 / 0.665 |

- **TruFor is near-perfect on clean paste-back diffusion splices, but that depends on Noiseprint++.** JPEG removes it: FLUX splice drops 0.913 → 0.585, below our 0.639. SD2 splice drops 0.999 → 0.732, level with our 0.755.
- **Our real-only system wins wherever the whole image was regenerated:**
  - TGIF2 full regeneration (SDXL, FLUX);
  - GPT-Image-2 and Nano-Banana-2 instruction edits;
  - the WebP / JPEG copies of the TGIF2 splices.
  - These are the newest manipulation types, and both supervised models are close to chance on the LLM editors (0.51–0.53 on GPT-Image-2).
- **The supervised models win clearly on classical and manual edits** (NC2016, Korus, SID-Set, CocoGlide, AutoSplice, DiffSeg). Those edits resemble their training data.
- **SR4** (generative re-synthesis of every pixel, TGIF2 only) brings every method down to 0.52–0.69.

## 3. Verdict

- **Our real-only system is not state of the art on average.** It is about 0.08 behind on clean images and 0.06–0.11 behind under re-sharing.
- **It does lead on fully regenerated and LLM-edited images**, where the supervised models' training data has no counterpart.
- The obvious next test is whether the two are complementary: a per-image z-fusion of Mesorch or TruFor with our comparator, using the cell maps already extracted, with the matched null checked.
- Mesorch's elevated null should be reported next to its AUC. Benchmarks that report only F1/AUC on edited images would not show it.
