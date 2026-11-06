# 07 - Physics cues, generator-consistency maps and the expert ensemble (2026-11-05/06)

Experiments that shaped the method in docs/09; all three are dead ends or partial results, recorded so they are not
repeated.

**Physics cues** (ifl/features/physics.py, scripts/eval_physics.py). Marigold normals and intrinsics plus Depth
Anything V2 per image: a global spherical-harmonics lighting fit with per-cell residual, local light direction,
illuminant colour, and defocus versus depth (thin-lens fit). Single cues reach cell AUC 0.52-0.58 against a null of
0.50-0.53; they work on copy-paste splices (AutoSplice, IMD2020-real 0.65) and sit at chance on diffusion edits, which
harmonize lighting. In the stacked ensemble they add nothing (0.805 with, 0.804 without). Dropped.

**Generator-consistency maps** (scripts/remote/gen_consistency.py, scripts/eval_gencons.py). Autoencoder round-trip
error (SD2, SDXL, SD3.5, FLUX) and diffusion denoising error per cell. Denoising error failed (0.46-0.49, content
driven). The autoencoder error identifies a fully re-decoded image at image level (AUROC 0.96-0.99 for open models)
and localizes pasted splices with the matched autoencoder (~0.90); an "inverted reconstruction" lead on regenerated
edits did not replicate in the pre-registered benchmark (docs/08). Results: docs/results/generator_consistency_pilot.md.

**Stacked ensemble** (scripts/ensemble.py). Mesorch, TruFor, comparator, predictor and physics, z-scored per image,
fused by equal weights, reliability weights, or a gated logistic stacker fitted leave-one-domain-out. Clean: stack
0.805 vs TruFor 0.780, but most of the gain is prior (AUC - null 0.231 vs 0.228); JPEG-75: 0.754 vs Mesorch 0.748
(AUC - null 0.181 vs 0.157). The comparator is the expert that matters on regenerated edits (drop-one -0.040).
Results: docs/results/ensemble_clean.md, ensemble_jpeg75.md.
