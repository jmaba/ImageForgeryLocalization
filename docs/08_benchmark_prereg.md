# 08 - Controlled diffusion-edit benchmark: pre-registration

Written before any image of this benchmark was generated or scored (2026-11-05). The analysis code
(`scripts/eval_benchmark.py`) is fixed before the results are looked at; nothing below is tuned afterwards.

## Why

The generator-consistency pilot (`outputs/gencons/results_pilot.md`, docs/07) found, after the fact, that in images
fully re-decoded by an open inpainting model (TGIF2 "fr"), the **edited region reconstructs worse** through the
matched model's autoencoder than the rest of the image. The cell AUC of minus the error was 0.40 (SD2, FLUX), against
0.50 on the originals with the same masks. A likely mechanism: freely sampled latents lie outside the range of the
encoder, so they do not survive an encode-decode round trip, while latents copied from the encoded photo do. The
effect was found by looking. This benchmark tests it on images that did not exist when the hypothesis was written.

## Data

- **Sources:** 120 COCO val2017 photos. Selection is seeded (seed 2026) among images with a short side of at least
  400 px and at least one eligible object: not crowd, not a person, 3-25 % of the image, from a supercategory with at
  least two categories.
- **Edit target:** one eligible object per photo (seeded choice). Its mask is dilated by 1.5 % of the long side.
- **Prompt:** replace the object with a different category from the same supercategory (seeded), e.g. dog -> horse.
- **Resolution:** every version is stored at the same size, the photo resized to a long side of 1024 (both sides
  multiples of 16). The original at that size is the authentic control.
- **Editors** (all open weights, run locally on an RTX 4090; FLUX transformers stored in fp8, computed in bf16):

  | Editor | Model | Settings |
  |---|---|---|
  | `sdxlinp` | SDXL inpainting 0.1 (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) | mask-guided, 30 steps, guidance 7.5, strength 0.99 |
  | `fluxfill` | FLUX.1-Fill-dev | mask-guided, 28 steps, guidance 30 |
  | `kontext` | FLUX.1-Kontext-dev | instruction "Replace the <old> with a <new>. Keep everything else unchanged.", no mask, 28 steps, guidance 2.5 |

- **Variants per editor:**
  - `fr`: the editor's full output (every pixel re-decoded).
  - `sp`: that output pasted into the original inside the dilated mask only.
- **Ground truth:** the dilated object mask. For Kontext, a difference mask (|output - original| after a small blur,
  threshold 0.1) is stored as well and reported separately.

## Hypotheses (direction fixed)

**H1, primary.** On `fr` images of the mask-guided editors (`sdxlinp`, `fluxfill`), the matched model's autoencoder
LPIPS error map, scored with **higher error = edited**, localizes the edit better than chance and better than its own
null.
- Matched autoencoder: SDXL VAE for `sdxlinp`, FLUX VAE for `fluxfill` and `kontext`.
- Statistic: mean per-image cell AUC on `fr` minus mean AUC on the originals scored with the same masks (matched null).
- Test: a paired bootstrap over source photos (10,000 resamples), one-sided. Supported if the 95 % lower bound is
  above 0, and the `fr` mean AUC is above 0.55.
- Analysed per editor and pooled. Pooled is primary.

**H1b, secondary.** The same test on `kontext` `fr`, with the object mask and with the difference mask.

**H1c, secondary, opposite direction.** On `sp` images, the pasted region is a decoder output inside a real photo,
so it should reconstruct **better**: AUC of higher-error-is-edited below 0.5, null near 0.5.

**H2, image level.** Mean matched-autoencoder error over the whole image.
- Separates originals from `fr` with AUROC above 0.9.
- Separates originals from `sp` with AUROC below 0.75. Only part of a spliced image was decoded.

**H3, exploratory.** The autoencoder error maps added to the stacked ensemble (Mesorch, TruFor, comparator), with the
stacker trained on the existing datasets and applied to this benchmark as a new domain, raise the cell AUC on `fr`.

## Reporting rules

- Every generated image is scored; failed generations (NaN, uniform output) are counted and reported, not replaced.
- Also reported for every variant: Mesorch, TruFor and comparator AUC with the same masks and nulls, IoU at a fixed
  threshold, and the false-positive area on originals.
- If H1 fails, the inverted-reconstruction effect is reported as not replicated.
