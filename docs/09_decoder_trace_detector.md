# 09 - Decoder-trace detector and baselines FLAME, B-Free (2026-11-06)

A fully convolutional detector trained only on real photos and their autoencoder round trips pasted back inside
random regions (content-matched by construction, no edited data). Code: ifl/features/dtrace.py (network, decoder bank
of 10 autoencoders), scripts/remote/dtrace_train.py (round-trip cache, models A and B), scripts/eval_dtrace.py
(harness with TruFor, Mesorch, comparator, FLAME, B-Free). Paper draft: "Decoder Traces, Not Objects" (Claude Docs).

| Result | Value | Table |
|---|---|---|
| Matched null (no object prior = 0.5) | 0.483 (TruFor 0.586, Mesorch 0.615, FLAME 0.610, B-Free map 0.459) | localization_clean.md |
| Regenerated edits, AUC - null (benchmark / extension) | +0.114 / +0.141 (B-Free map +0.351 / +0.364; TruFor, Mesorch, FLAME <= +0.04) | localization_clean.md |
| Held-out autoencoders (model A, validation) | pixel AUC 0.943, image AUC 1.000 | dtrace_A_validation.json |
| GPT-Image-2 / Nano-Banana-2 vs own originals, image AUROC | 0.939 / 0.987 (all others 0.45-0.64) | image_level_clean.md |
| Under JPEG-75 (both sides re-encoded) | 0.466 / 0.662 - not robust | image_level_jpeg75.md, robustness.md |

Claims, limitations and the withdrawn JPEG claim: docs/11_claims_and_evidence.md. Baseline versions: FLAME commit
5d3d42f (built with its test_dataset.py; its test.py cannot load the released checkpoint), B-Free commit c6a9f89 (map
= our sliding-window adaptation; image score official).
