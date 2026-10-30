# Phase 0 - are frozen latents representative of "manipulated"?

_Run 2026-10-30. 2,400 images (7 datasets, random seeded samples, see `configs/datasets.yaml`); patch grid 24x24;
all numbers are out-of-fold (GroupKFold by source image). Regenerate with `scripts/analyze_features.py`;
full reports in `outputs/reports/index.html` (git-ignored)._

**Exploratory, not confirmatory.** These are single-seed estimates on a small sample, chosen before
looking at data, with no hyper-parameter search. Differences below ~0.03 AUC should not be read as real.

## Cross-backbone summary

| backbone | IMG probe AUC | IMG LODO AUC | IMG→JPEG/PNG AUC (confound) | IMG→dataset bacc (chance .14) | PATCH probe per-img AUC | PATCH transfer (off-diag) | per-img F1 @0.5 | zero-shot kNN | zero-shot proto | one-class(real) per-img | one-class det |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **pixels (control)** | .566 | .532 | .577 | .180 | .627 | .576 | .193 | .617 | .593 | .632 | .469 |
| dinov2 (last) | .766 | .720 | .792 | .440 | .773 | .668 | .431 | .553 | .666 | .403 | .404 |
| dinov2 layer 6 | .868 | .797 | .849 | .714 | .772 | **.681** | .413 | .642 | .663 | .566 | .346 |
| clip ViT-B/16 | .871 | .790 | .896 | .673 | .742 | .665 | .389 | .564 | .583 | .481 | .384 |
| siglip2 B/16 | .806 | .739 | .825 | .507 | .724 | .653 | .373 | .545 | .574 | .451 | .374 |
| srm residual stats | .798 | .753 | .877 | .642 | .641 | .540 | .263 | .545 | .610 | .544 | **.634** |
| vaerec (AEROBLADE) | .575 | .480 | .738 | .242 | .622 | .533 | .188 | .550 | .574 | .500 | .620 |

Supervised patch probe, per-image AUC by dataset:

| backbone | autosplice | cocoglide | diffseg30k | sid_set | opensdi | nanobanana2 | gpt_image_2 |
|---|---|---|---|---|---|---|---|
| pixels | .755 | .661 | .629 | .694 | .614 | .585 | .489 |
| dinov2 | .878 | .906 | .799 | .813 | .738 | .687 | .611 |
| dinov2 l6 | .908 | .893 | .828 | .849 | .695 | .686 | .584 |
| clip | .851 | .830 | .755 | .823 | .698 | .680 | .592 |

## Findings

1. **Supervised dense probes on frozen VFMs pass the gate, but only just on transfer.** Per-image patch
   AUC is .77 for DINOv2 against .63 for the pixel control. Cross-dataset transfer is .68 against .58, a
   gain of about +0.10. Intermediate DINOv2 (layer 6) matches the last layer on patches and transfers
   slightly better.
2. **Fully regenerated editors are where everything breaks.** GPT-Image-2 sits at .58–.61 for all
   backbones and Nano-Banana-2 at about .69, while local diffusion edits reach .80–.91. This matches TGIF2
   and the review: once the whole image is re-decoded there is no authentic reference left.
3. **Image-level scores are confounded.** The same global features predict JPEG vs PNG at AUC .79–.90 and
   the source dataset at up to .71 balanced accuracy (chance .14). Real images come only from OpenSDI and
   SID-Set, so image-level "real vs fake" is partly "which dataset". The within-JPEG AUC (.75–.86) shows
   format is not the whole story, but image-level numbers here should not be used as evidence.
4. **The galleries show object bias.** The probe highlights whole salient objects. Per-image F1 at a fixed
   0.5 threshold is only about .4 despite AUC .77, the calibration gap TRAIL also reports.
5. **Training-free and real-only scoring on VFM features fails.**
   - Zero-shot intra-image kNN or prototype scores (.55–.67) barely beat the pixel control (.62/.59).
   - A one-class Gaussian on real VFM patches scores *below* chance (.40–.57).
   - The low-level streams are the only ones where one-class detection works (SRM .63, VAE-recon .62
     against ≤ .40 for VFMs).
   This is the empirical case for **Proposal 1** of the report (a *learned* dense real-only residual model)
   over semantic features.
6. **Handcrafted low-level stats are too weak at patch level.** SRM and VAE-recon transfer is ≈ .54,
   barely above chance, so they need to be learned (Proposals 1 and 2), not summarised.

## Gate decision → next steps

- Keep `dinov2:l6` + `dinov2` as the semantic stream (best transfer, cheapest).
- Drop handcrafted `srm`/`vaerec` summaries as patch features. Keep them as one-class detection cues and
  as baselines.
- Next:
  - (a) add the mandatory controls: TGIF2-style random-mask split, INP-X background restoration,
    `--harmonize jpeg95` re-run;
  - (b) a semantic-nullspace probe (Proposal 4) to measure how much of the .77 is object bias;
  - (c) prototype the dense real-only residual predictor (Proposal 1) and the double-round-trip VAE
    statistic (Proposal 2), aimed specifically at the GPT-Image-2 / Nano-Banana-2 failure.
