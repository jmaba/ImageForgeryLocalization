# 10 - Research-skills review: where the project stands, what is missing, what to try

Source: the user's document "Image Forgery Localization Research Skills" (15 skills, 44 references), merged into the
Claude skill `ifl-research` (~/.claude/skills/ifl-research/SKILL.md) together with seven skills the document lacks.
This file maps the skills onto the project (state of 2026-11-06), lists the gaps, and the ideas they suggest.

## Citations checked

| Paper | Status | Why it matters here |
|---|---|---|
| FLAME, "Order within Chaos", arXiv:2606.02178 (code public) | verified | Claims diffusion suppresses local high-frequency variance; LAD map + SAM adapter, trained on synthetic instruction edits (EditStream). The closest competitor to our decoder-trace result on regenerated edits |
| GIFTBench / ForenScope, arXiv:2610.01778 | verified | 115k images, four hold-out axes; the protocol reviewers will expect next to ours |
| TGIF2, arXiv:2603.28613 | verified | States FR localization is open; random non-semantic masks reveal object bias - prior art for our matched null |
| ATAR, arXiv:2609.39066 | verified | MLLM explanations are post hoc; agentic tool use |
| SDiFL (SD3-VAE localizer), arXiv:2508.20182 | verified | Generative tokenizer repurposed as localizer |
| Detective SAM (ICLR 2026), REFORGE, DVVG, ForgDiffuser figures | not verified | Treat the document's numbers (OOD IoUs etc.) as unconfirmed |

## Skill-by-skill: done, found, missing

| Skill | What the project has | Result | Gap |
|---|---|---|---|
| 1 mechanism map | edit families: manual / diffusion splice / regenerated | regenerated is the hard family | no copy-move or retouching expert |
| 2 gap verification | literature check (reports/), arXiv checks | idea novelty table | cross-domain terms (RKHS, change-point) not searched |
| 3 dataset audit | 20 datasets + own benchmark with paired originals | GPT/Nano masks are predicted (`*_pred_bin`) | no GIFTBench-style axis splits |
| 4 feature families | A (SRM), C (DTCWT), G (DINOv2), F (physics), H (AE errors, decoder traces), D via TruFor | F adds nothing; H decoder traces strongest new family | B (phase / LAD energy) not tested as its own family |
| 5 classical revival | physics (lighting, defocus) automated | lighting harmonized by diffusion; weak | copy-move PatchMatch/FAISS, CFA GLRT, shadow geometry not done |
| 6 causal levels | - | decoder traces: causal + discriminative at image level, weakly localizable in FR | write every claim with its level |
| 7 confounders | JPEG/WebP/resize/SR conditions, format control | TruFor collapses under compression | threshold rule per paper when comparing to published numbers |
| 8 falsify axioms | VAE-redecode control; content-matched training | re-decoded originals look like regenerations (H2) | smartphone / computational-photography false-positive test |
| 9 metrics | cell AUC, interior AUC, matched null | IoU only for TruFor/Mesorch at 0.5 | IoU/F1 for every method with the three threshold rules |
| 10 access levels | gray-box (AE bank), black-box (decoder traces) | black-box detector generalizes to closed editors | state access level in the paper |
| 11 robustness | running (JPEG-75, WebP-80, resize-50) | comparator robust, TruFor fragile | generative super-resolution attack |
| 12 cross-domain | real-only comparator (anomaly-detection spirit) | complementary in fusion | conformal / GLRT calibration with bounded false alarms |
| 14 blueprint | docs/08 pre-registration | H1 failed, H1c and H2 held | - |
| 15 pre-mortem | novelty report | see classification below | FLAME must be in the comparison |

## Novelty classification of the project's contributions (skill 15)

| Contribution | Class | Closest work |
|---|---|---|
| Decoder-trace detector: content-matched round-trip training, leave-one-decoder-out, detects GPT-Image-2 / Nano-Banana regenerations (0.94-0.99) | modern extension + new evidence | DRCT, B-Free, AEROBLADE, FLAME |
| Matched transplant null + processing / format controls as a protocol | partially explored | TGIF2 random masks, TRAIL, BG-REAL controls |
| Negative result: no published method beats its null on controlled full regenerations | new experimental contribution | TGIF2 (object bias) |
| Pre-registered controlled benchmark (3 editors x fr/sp, paired originals) | new benchmarking contribution | TGIF2, GIFTBench |
| Four-outcome triage with the regenerated-edit class | partially explored | TGIF2 (SID vs IFL split) |
| Real-only comparator in fusion | incremental | DiffusionPrint |

## Ideas the review suggests (ranked by value for the paper)

1. **FLAME under our protocol** (skills 15, 16). Its code is public. Run it on the benchmark, extension, TGIF2,
   GPT-Image-2 and Nano-Banana with matched nulls. Two outcomes are both publishable: FLAME's "energy gap" survives the
   null (then compare and fuse with decoder traces), or it is largely an object / smoothness prior (then our null
   exposes it). Also test whether our detector learned the same thing: correlate its maps with FLAME's LAD map.
2. **Per-source calibration with bounded false alarms** (skills 12, 20). Conformal calibration of the decoder-trace
   image score on authentic images from each capture source; report detection at a fixed false-positive rate. Fixes
   the NC2016 / Korus false positives without retraining; training-free.
3. **Smartphone / computational-photography falsification test** (skill 8). Score authentic phone photos (HDR,
   portrait bokeh, night mode, AI denoise) with the decoder-trace detector. If neural ISPs look like decoders, that is
   both a limitation to report and a reason to add such photos as negatives in training.
4. **Perturbation stability as a second black-box channel** (skills 10, 4). Score the decoder-trace map before and
   after small perturbations (noise, blur, JPEG); synthetic regions may change differently (the Detective SAM idea,
   unverified). Cheap: no new training.
5. **Phase / high-frequency energy as its own family** (skill 4). Add a LAD-style local high-frequency variance map
   as a training-free expert in the ensemble; check orthogonality with decoder traces (drop-one, AUC - null).
6. **Copy-move expert** (skill 5). GPU PatchMatch / FAISS over DINO tokens for CMFD-like data, where every current
   expert is near chance; completes the ensemble for classical edits.
7. **Generative super-resolution attack** (skill 11). TGIF2 reports it weakens traces; decoder traces should instead
   cover the whole image (like a regeneration), so the triage should say "regenerated" - test it.
8. **Threshold-rule metrics** (skill 9). IoU / F1 for every method with fixed, transferred and oracle thresholds.

## B-Free vs our decoder-trace detector: flow comparison and gaps (added 2026-11-06)

B-Free (Guillaro et al., CVPR 2025; code github.com/grip-unina/B-Free, commit c6a9f89) is the closest prior work to the
decoder-trace detector: both make fakes *from the real photos themselves* so that content cannot be a shortcut.

| | B-Free | Decoder-trace detector (ours) |
|---|---|---|
| How fakes are made | SD2.1 inpainting pipeline: empty mask = full self-reconstruction (VAE + denoiser), object / box masks = local inpainting, optionally background restored | autoencoder round trip only (no denoiser), pasted back inside random blobs, or everywhere / nowhere |
| Generators seen | one family (SD2.1) | 10 autoencoders, 4 design families (4-ch KL, 16-ch KL, VQ, video VAE, DC-AE); 5 held out in model A |
| Labels | image level (real / fake) | pixel level (the blob), image level as a by-product |
| Content matching | same photo | same photo **and** identical pixels outside the blob |
| Augmentation | blur, JPEG, scaling, cut-out, noise, colour jitter ("inpainted++") | JPEG (p 0.3, q 75-100), down-up resize (p 0.15) |
| Network / input | ViT-B/14 + 4 registers (DINOv2 init), native resolution, 504 px crops, 5-crop average | ResNet-34 on RGB + high-pass residual, FPN to 1/4 resolution, 384 px crops, fully convolutional |
| Data | 51,517 COCO + 309,102 SD2.1 images | 3,000 COCO crops x 10 autoencoders (cached) |
| Output | one image score | pixel map + image score |

**Gaps in B-Free that ours fills.** No localization; a single decoder family, so generalization to other decoders
rests on augmentation; decoder and denoiser traces are mixed, so it is unknown which one it relies on; a local
inpainting is labelled "fake" as a whole image, which is label noise for any localization use.

**Gaps in ours that B-Free's flow suggests (improvement opportunities).**
1. *Denoiser / flow traces.* We train on autoencoder round trips only. A real edit's generated region also carries
   the denoiser's traces (FLAME's high-frequency energy suppression). Adding real inpainting reconstructions - B-Free's
   self-conditioned and local inpaintings, whose training set is public - with pixel labels from the inpainting mask
   should help where we are weakest: localization inside full regenerations (raw AUC 0.53-0.65).
2. *A three-class pixel target - the most promising new idea.* In a full regeneration every pixel is decoded, but only
   the masked region was *generated*. Train pixels as real / re-decoded / generated (labels come for free when we run
   the editors ourselves: outside the mask = re-decoded, inside = generated). Neither B-Free (image labels) nor our
   current detector (decoded vs real) learns that distinction, and it is exactly the regenerated-edit trap.
3. *Stronger augmentation* (blur, scaling, cut-out, noise, jitter), if the robustness run shows the detector degrades.
4. *Capacity and data:* a ViT at native resolution and ~17x more real photos.
5. *Calibration:* B-Free reports calibration (NLL, ECE); per-source calibration is our open weakness.
