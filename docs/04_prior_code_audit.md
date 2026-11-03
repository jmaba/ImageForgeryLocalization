# Audit of the earlier forgery-localization code (2026-11-01)

Two earlier code bases were inspected so that what they found can be re-tested under the protocol of
`docs/03` (pixel control, matched transplant null, conformal image-level FPR):

* `C:\Users\jm_ab\source\repos\demoAppDataset` (Python only): v1 DT-CWT localiser (`NeuralNetworkDualTree/`),
  v2 multi-colour-space framework (`NeuralNetworkDualTree/v2/`: datasets, transforms, 3 supervised and
  3 unsupervised "approaches", the HPO explorer, fingerprint analyses), `BlockBasedInpaintingDetection/`.
  164 files, about 40k lines. Read in full: data loading, transforms, colour spaces, all six approaches'
  feature code and detectors, the HPO runner, the anomaly core, the fingerprint analysis, the v1 dataset,
  model and training loop. Pattern-swept: model architectures, trainers, UI, scratch scripts.
* `C:\Users\jm_ab\Downloads\FeatureExtractSuggestionsAnalysis` (March 2026): SIF-Net v1-v3, directions
  B and C v3-v8, `safa_features.py`, the feature-atlas repo and `RESEARCH_HISTORY.md`.

## A. Defects that change the reported numbers

| # | Where | What | Effect |
|---|---|---|---|
| A1 | v2 `approaches/u1,u2,u3,s1,s2,s3` (8 call sites) | `cv2.cvtColor(uint8, RGB2Lab)` returns a, b offset by +128; the code computes C* = sqrt(a^2+b^2) and h = atan2(b, a) **without removing the offset** | "C*" is 1.28 for black, grey and white alike: it is the distance from (a*, b*) = (-128, -128), roughly a linear red/yellow axis, not chroma. "h" is roughly b*-a*, not hue. Every "C* is the HPO winner" conclusion is about a different channel. (`explorer/core/color.py` does it correctly.) |
| A2 | all Gabor code (u1, u2, s1-s3, explorer) | sigma = 2 for every frequency, 13-px kernels, f in {0.05, 0.1, 0.2, 0.4}; the even kernel is not zero-mean | At f = 0.05 the even kernel is a pure low-pass (DC/L1 = 1.00), at f = 0.1 0.68. For half of the 32 channels "phase" = atan(local gradient / local mean value): a contrast-over-brightness ratio, i.e. content. |
| A3 | v2 `explorer/batch/runner.py` | best of up to ~1,944 configurations (colour space x channel x transform x detector) ranked by AUC on the same images it reports | The 0.97 "Gabor-phase" and 0.788 "DT-CWT real" figures are maxima selected on the test images, with no held-out set and no pixel-colour control. |
| A4 | v2 `explorer/app.py` | per-channel AUC reported as `max(auc, 1-auc)` | Optimistic by construction; flips sign per image. |
| A5 | v2 `analyze_fingerprints.py` -> `RESEARCH_REVIEW_AND_ROADMAP.md` | "discriminative channels" = channels whose whole-image mean differs most between METHODS; images per method come from the first category folders only | Measures method identity and content, not edited vs authentic pixels. The roadmap's "phase channels are 3x more discriminative / Cb, Cr phase is the key signal" does not follow from it. |
| A6 | v2 `datasets/base_dataset.py`, `s3/dataset.py` | `random_split` / shuffle over samples, not over source photos | TGIF has up to 3 mask types x 3 variants per COCO photo; the same scene lands in train and validation, which inflates per-method validation IoU. (`diagnose_split.py` tested method balance, which a random 58k split already guarantees; it could not find this.) |
| A7 | v2 `base_dataset.py` | `val_ds.dataset_augment_override = False` sets an attribute nothing reads | Validation and test images are flipped/rotated at random. |
| A8 | v1 `InpaintingCustomDataset`, v2 `DTCWTExtractor`, s2/s3 SWT/DT-CWT | every channel min-max rescaled per image; DT-CWT phase fed as a scalar in [0, 1] | Absolute noise level (the forensic quantity) is discarded per image; phase wraps at +-pi, so equal angles can map to 0 and 1. |
| A9 | u1-u3, s1/s2 `predict.py` | pixels of all images concatenated before `roc_auc_score` | Pooled AUC mixes between-image differences; with per-image [0, 1] normalisation the scale is arbitrary across images. |
| A10 | `RESEARCH_HISTORY.md` (SIF-Net) | feature ranking (FFT 0.71, Gabor-Lab 0.668, SWT-dmey 0.683, DT-CWT mag+phase 0.726) | Image-held-out, but no content control and no matched null; raw Cb/Cr scoring 0.41 shows the content effect is large. |

None of these are unusual; they are exactly the failure modes the protocol in `docs/03` was built to catch.

## B. What survives and is worth re-testing

1. **DT-CWT magnitude and phase statistics** (v1 and SAFA): the best-motivated stream. Re-implemented with
   the fixes: no per-image rescaling, phase handled as a circular quantity (inter-orientation circular
   variance, amplitude-weighted cross-scale phase difference via c0 * conj(c1)), log magnitudes.
2. **Local FFT high-frequency energy** (SIF-Net's best single feature): per-cell windowed spectrum in
   radial bands, log energy, on luminance.
3. **Gabor in an opponent colour space**: zero-mean kernels with sigma tied to wavelength, magnitude and
   circular phase statistics, on correct a*/b* (or opponent) channels.
4. **Median-filter residual in Lab** (0.660 vs SRM 0.546 in SIF-Net): per-cell residual moments.
5. **Cross-channel phase coherence (u2)**: a physically sensible idea (luminance and chroma edges should
   align); worth a test once A1/A2 are fixed.
6. The **self-referencing scoring** finding (deviation from the own image beats a learned per-patch head)
   matches `pair_ref`; the **"transformer smoothing kills local signal"** diagnostic matches our choice of a
   per-cell encoder.

## C. Disagreements with our results, and why

* "SRM does not see diffusion edits" (0.546): agrees for training-free use (SRM loc_cf ~ pixel control),
  but the learned comparator on SRM is our least biased model (null 0.50-0.52). The trace exists; the
  relation has to be learned.
* "Synthetic rectangle splices don't transfer to diffusion inpainting": our splices keep the content and
  swap only the process (VAE round-trips, processing, real inpaints) with object-shaped masks, and they do
  transfer. Rectangles of other photos teach a content/boundary detector.
* "Single-method training: 66% val IoU, all four methods: 5.8%": A6 inflates the single-method number;
  the collapse is the honest cross-method result, consistent with our leave-one-decoder-out runs.

## D. Next

Port B1-B4 as cached feature specs (`dtcwt_phase`, `fft_band`, `gabor_opp`, `medres_lab`) and run them
through `region_tests.py` (training-free, with pixel control), `region_null.py` (matched null) and
`train_comparator.py` (learned, real-only).
