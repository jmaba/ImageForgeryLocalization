# 11 - Claims and evidence (2026-11-06)

Every number: mean per-image cell AUC on the 24 x 24 grid, minus the matched null (the same map on the edit's own
original with the edit's mask) unless marked raw or image-level. Clean = original files; re-shared = every image of
both sides re-encoded (JPEG-75, WebP-80) or resized to 50 %. Tables: outputs/dtrace/ (localization_*, image_level_*,
robustness_flame.md), outputs/benchmark/results.md (pre-registered), docs/08 (pre-registration).

## Supported

| # | Claim | Evidence | Holds under re-sharing? |
|---|---|---|---|
| C1 | Supervised localizers' "localization" of fully regenerated edits is object prior | Nulls: TruFor 0.59, Mesorch 0.62, FLAME 0.61. AUC - null on controlled regenerations (benchmark / extension): TruFor -0.07 / -0.04, Mesorch -0.05 / -0.05, FLAME +0.03 / +0.04 | yes, all conditions |
| C2 | Content-matched training removes the object prior | Nulls 0.46 (B-Free, our sliding-window adaptation) and 0.49 (our decoder-trace detector) - two independent implementations | - |
| C3 | Content-matched detectors localize regenerated edits above the null | B-Free +0.35 / +0.36; ours +0.12 / +0.14 (clean) | ours: +0.11..+0.13 resize, +0.07 WebP, +0.02..+0.04 JPEG; B-Free's map not tested re-shared |
| C4 | Training on many decoder families catches closed editors | Image AUROC vs own originals: GPT-Image-2 0.94, Nano-Banana-2 0.97-0.99 (ours) vs B-Free 0.59 / 0.61, TruFor 0.47 / 0.45, Mesorch 0.58 / 0.64, FLAME 0.54 / 0.60. Localization: GPT-Image-2 +0.10 (ours) vs <= +0.01 for all others | resize yes (0.88 / 0.84-0.87); **JPEG / WebP no (0.47-0.76)** |
| C5 | Decoder traces generalize to unseen autoencoders | held-out autoencoders (SD2, Playground, SD3.5, CogView4, Sana DC-AE): pixel AUC 0.94, image AUC 1.00 | validation splices only |
| C6 | FLAME's high raw AUC on regenerations is prior | raw 0.746 (best of all) but +0.03 above its null; negative on GPT-Image-2 (-0.05) | yes |
| C7 | Experts are complementary | equal-weight fusion of TruFor + Mesorch + comparator + ours: AUC - null 0.219 clean (best single 0.20); + FLAME + B-Free 0.235 | 0.144-0.149 re-shared (best single 0.11-0.12) |
| C8 | Pre-registered benchmark (docs/08) | H1 (inverted reconstruction) not replicated; H1c (matched-AE map on splices, ~0.90) and H2 (image-level re-decode, AUROC 0.98-0.996) supported | - |

## Withdrawn

- "Decoder-trace detection of GPT-Image-2 / Nano-Banana-2 holds after JPEG re-encoding" (0.88-0.99). That control
  re-encoded only the edited images; when the originals are re-encoded too (real re-sharing) the AUROC falls to
  0.47-0.66 - JPEG artefacts look like decoder traces to the detector.

## Limitations to state

1. Our detector is fragile to lossy re-compression (see C3, C4). B-Free, trained with heavy augmentation, stays at
   0.91-0.98 on open-model edits - so robustness is learnable (v2 below).
2. Per-source calibration: some camera datasets score high at image level (NC2016 0.99, Korus 0.87 mean trace score).
3. B-Free's map is our adaptation (sliding 224 px windows scored with its own small-input handling), not its output.
4. B-Free trained on COCO train2017; TGIF2's and GPT/Nano's COCO originals may overlap it. Our benchmark uses val2017.
5. FLAME trained on SID-Set (its SID-Set numbers are in-distribution); FLAME's test.py cannot load its released
   checkpoint (we used its test_dataset.py builder).
6. GPT/Nano masks are the benchmark's predicted masks; image-level results do not depend on them.
7. One training seed; benchmark 120 + 200 photos x 3 editors; IoU / F1 tables and bootstrap intervals still to add.

## What it supports as a paper

A forensics paper (IEEE TIFS, or WIFS short) on: (i) the matched-null protocol and the negative result for supervised
localizers and FLAME on regenerated edits (C1, C6, C8); (ii) content-matched, multi-decoder trace training as the way
to remove the prior and reach closed editors (C2-C5); (iii) complementary experts (C7). Robustness to re-sharing is the
main weakness a reviewer will attack.

## v2 of the detector (one pod run, ~2 h training + ~1.5 h evaluation)

Keep: multi-decoder, pixel-level, content-matched round trips. Add from B-Free's flow: real diffusion inpainting
reconstructions (denoiser / flow traces), heavy augmentation applied to both real and fake sides (JPEG 50-100, WebP,
resize, blur, noise), and a three-class pixel target (real / re-decoded / generated). Success criteria, fixed now:
image AUROC on GPT-Image-2 / Nano-Banana-2 >= 0.85 under JPEG-75, AUC - null on benchmark regenerations >= +0.15 clean
and >= +0.08 under JPEG-75, null <= 0.5.
