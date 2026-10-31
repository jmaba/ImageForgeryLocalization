# Region self-consistency and Proposal 2: first test results (2026-10-31)

Scripts: `scripts/region_tests.py` (Tests 3/5/6), `scripts/region_null.py` (Test 1 + pixel increment),
`scripts/proposal2_roundtrip.py`. Raw rows: `outputs/region/*.csv`, `outputs/proposal2/results.md`.
Evaluation: per-image patch AUC on partial edits, 150 images per dataset, 24x24 cells (about 43 px each).

## 1. Training-free region self-consistency

Mean per-image AUC across the 12 edited datasets:

| scorer | SRM | DINOv2-L6 | DT-CWT | Lift | Shear | CTRL pixels | CTRL rand-proj |
|---|---|---|---|---|---|---|---|
| loc_raw (region vs rest) | 0.604 | 0.575 | 0.544 | 0.540 | 0.523 | 0.557 | 0.572 |
| loc_reg (ridge on all cells) | 0.602 | 0.544 | 0.552 | 0.552 | 0.520 | 0.603 | 0.483 |
| loc_cf (cross-fitted Huber) | 0.615 | 0.581 | 0.574 | 0.571 | 0.544 | 0.608 | 0.527 |
| loc_match (content kNN) | 0.611 | 0.578 | 0.565 | 0.563 | 0.541 | 0.610 | 0.597 |
| scale_cf+ (local dispersion) | 0.651 | 0.642 | 0.635 | 0.626 | 0.596 | 0.622 | 0.554 |
| energy_cf (energy distance) | 0.537 | 0.460 | 0.490 | 0.502 | 0.479 | 0.544 | 0.473 |

* Cross-fitting beats the naive regression, as predicted: edited cells leak into their own reference otherwise.
* **The negative controls fail.** Cell colour statistics alone (CTRL pixels) reach 0.61. Edited regions
  differ in content, and the content regression does not remove that difference. The descriptors beat the
  pixel control by only about 0.01–0.03 on average. The exceptions are SRM on CocoGlide (+0.10), DiffSeg
  (+0.12) and SID-Set (+0.07).
* Energy distance on 9-cell neighbourhoods is useless. Its sample is too small, as the report's sample-size
  argument predicted.

## 2. Transplant null (Test 1)

Authentic images are scored against the mask of an edited image. For TGIF2 this is the exact random mask of
the same COCO photo. Any edit-independent scorer must give 0.5 here.

| scorer (SRM) | edited TGIF2 | authentic TGIF2, same masks | edited SID-Set | authentic SID-Set |
|---|---|---|---|---|
| loc_cf | 0.558 | 0.529 | 0.746 | 0.531 |
| scale_cf+ | 0.601 | 0.567 | 0.794 | 0.570 |
| scale_cf+ \| pixels | 0.562 | 0.542 | 0.710 | 0.533 |

* The dispersion scorer is **biased**: 0.54–0.59 on authentic images, because of mask shape and position
  priors. Its apparent lead over the location scorer is mostly this bias.
* Against the matched null, the real localisation gain on TGIF2 is **+0.02 to +0.04** for every descriptor.
  On SID-Set it is about +0.18, but that is the JPEG-real / PNG-fake history: paste-back keeps JPEG traces
  outside the mask. This trace is genuine but specific to that dataset.
* **Image level** ("does this image contain an inconsistent region?", mean of the top-10 cell scores): at a
  conformal 5% false-alarm threshold the true-positive rate is 0–7% for every descriptor and dataset.
  Image AUC is 0.35–0.55. **No detection power.**

## 3. Fully regenerated vs paste-back (Test 6, TGIF2 matched pairs, Wilcoxon)

* SD2: paste-back is easier, by +0.05 to +0.085 AUC (p < 1e-4) for all descriptors, **but the pixel control
  shows the same +0.056.** The difference is a colour/content seam, not a noise trace.
* FLUX: no difference (|Δ| ≤ 0.01). FLUX's VAE leaves no seam that these descriptors can see.

## 4. Proposal 2 (double VAE round-trip): negative

* On matched TGIF2-FLUX cells, real, re-decoded and generated cells are indistinguishable (AUC 0.47–0.56).
* Leave-one-editor-out: round-trip features alone reach 0.47–0.67. Adding them to DINOv2-L6 does not help.
* INP-X check (SD2): the round-trip probe fires on restored camera pixels too (sp_out 1.11 vs fr_out
  −0.15). It reads a global image property, not the edited region.

## Verdict and direction

1. With hand-crafted or frozen descriptors at about 43 px cells, **training-free** region self-consistency
   does not beat a properly matched null by more than about 0.03 AUC. It has no image-level power. A large
   part of what earlier region-consistency work reported is plausibly content and mask-shape bias.
2. Keep the **evaluation protocol** (cross-fitting, pixel control, transplant null with matched masks,
   conformal image-level FPR). It is cheap, and it is what exposed this; it is publishable on its own.
3. The bigger test should **learn** the consistency relation rather than assume it. Train a real-only
   pairwise "same source?" comparator on DINOv2-L6 (+SRM) cell tokens, using self-made splices:
   real-in-real, VAE round-tripped regions and pasted-back inpaints (report Test 13). Compare it with TRAIL
   perturbation drift (Test 10). Success gate: beat both the pixel control and the transplant null by
   ≥ 0.10 AUC on TGIF2 random masks, with image-level TPR clearly above 5% at 5% FPR.

## 5. Bigger test: a learned, real-only comparator (`scripts/make_splices.py`, `scripts/train_comparator.py`)

**Training data.** 2,600 self-made splices built from 1,300 real non-COCO photos (DIV2K + ImageNet-val).
The masked region is replaced by one of:
- the same image after one SD-VAE pass (777) or one SDXL-VAE pass (526);
- the same image after JPEG, blur, resampling or noise (682);
- another photo (348);

with 267 left untouched. Random blob, ellipse and rectangle masks, feathered half of the time. Most types
preserve content, so the model cannot use the content shortcut.

**Models.** A 64-D cell encoder on frozen features. `pair` is a symmetric "same source?" head;
`pair_ref` scores each cell against the more-consistent half of its image. `point` is an absolute "was this
cell replaced?" head. 3,000 steps, about 2 minutes. No test editor, COCO image or test mask is seen.

**Per-image localisation AUC.** Comparator: DINOv2-L6+SRM, `pair_ref`. Training-free: best of section 1.
Supervised: E3 DINOv2 probe trained on real SD2 inpainting.

| dataset | training-free | pixel control | comparator (no fakes) | supervised (SD2 fakes) |
|---|---|---|---|---|
| CocoGlide | 0.739 | 0.611 | **0.753** | 0.656 |
| DiffSeg30k | 0.694 | 0.577 | **0.837** | 0.631 |
| Nano-Banana-2 | 0.645 | 0.636 | 0.615 | 0.579 |
| OpenSDI | 0.617 | 0.611 | **0.644** | 0.604 |
| SID-Set | 0.746 | 0.676 | 0.711 | 0.677 |
| TGIF2 FLUX FR (unseen VAE) | 0.546 | 0.526 | 0.629 | **0.661** |
| TGIF2 FLUX SP (unseen VAE) | 0.553 | 0.518 | **0.680** | 0.672 |
| TGIF2 SDXL FR | 0.572 | 0.591 | **0.710** | 0.654 |
| TGIF2 SD2 SP | 0.595 | 0.607 | **0.777** | 0.738 |
| TGIF2 SD2 FR | 0.525 | 0.552 | 0.557 | 0.708 (in-domain) |
| AutoSplice | 0.636 | **0.804** | 0.671 | 0.723 |
| GPT-Image-2 | 0.538 | 0.524 | 0.519 | 0.515 |
| mean of 12 | 0.615 | 0.601 | **0.675** | 0.659 |

**Matched transplant null (TGIF2, same masks on the original photos).**
- Gain over the null: DINOv2+SRM +0.106, DINOv2 +0.097, SRM +0.077–0.084. The pixel control gains +0.025.
  This meets the ≥ 0.10 gate set in the verdict above.
- DINOv2-based scorers still carry an object prior: their null is 0.56–0.60. The SRM-only comparator is
  nearly unbiased (null 0.50–0.52).

**Image-level detection (TPR at a conformal 5% false-alarm rate).**
- The SRM comparator catches 87% of SD2-FR and 39% of SDXL-FR images, but only 12% of FLUX-FR.
- Image-level power therefore exists only for the VAE families seen in training. It does not yet transfer.

**What this says.**
1. Learning the consistency relation from real-only splices works where assuming it did not. It gives
   +0.06 over training-free and +0.07 over the pixel control on average. It matches or beats a probe trained
   on real SD2 fakes on 8 of 11 unseen datasets.
2. GPT-Image-2 stays at chance, as expected for whole-frame re-decoding. AutoSplice is solved better by
   colour than by any trace: its edits differ in content, not process.
3. The FR vs SP gap (SD2 FR 0.56 vs SP 0.78) shows that the model reads the pasted-back seam and process
   change. It does not read the generation itself.

**Next steps, in order of expected value.**
- Scale the splice generator: more decoder families (Qwen/FLUX VAE held out, then rotated), real inpainting
  pipelines as a source type, and object-shaped masks drawn from non-COCO segmentations to break the object
  prior.
- Use finer cells: 48×48 for SRM, native-resolution tokens.
- Add leave-one-decoder-out training to measure transfer honestly.
- Replace the hard majority reference with a spectral two-cluster score, and add a conformal image-level
  decision.
