# Novelty check: new approaches to localizing generative image manipulations (2023 to Oct 2026)

_Compiled 2026-10-30. Sources: alphaXiv semantic search, arXiv API (abstracts fetched directly), web search. Local context was read from `NewImageForgeryLocalization/docs/01_reanalysis_and_direction.md`, `JepaApproach/docs/02_proposed_method.md` and `featureAnalysis/SOTA_ANALYSIS.md`; none of these files were edited._

_Verification note: every arXiv ID marked [abs] had its abstract fetched in this session. IDs marked [mem] come from prior knowledge, and their abstracts were not re-fetched here. Check those before citing them in a paper._

_Search log (queries tried):_ alphaXiv: "re-running generator / re-inpainting / inversion reconstruction error maps for forgery localization"; "training-free / self-supervised real-only dense anomaly detection for forgery localization (kNN, masked prediction, one-class)"; "fully regenerated images, MLLM grounding, pixel-level editor attribution, multi-turn edits"; "conformal / calibration / TTA / super-resolution laundering / lighting-shadow consistency for forgery localization"; "training-free VAE reconstruction maps, latent blending artifacts (AEROBLADE, DinoLizer, HFI)"; "generative super-resolution laundering / regeneration attacks"; "physical inconsistency (shadows, lighting, depth, normals) for AI-edit detection"; "self-supervised splice localization from real images (EXIF-as-language, ZED, JEPA)". arXiv API: `inpainting/edited AND localization AND reconstruction/inversion/re-inpaint/fixed point AND diffusion`; `latent/VAE AND inpainting/blending AND forgery AND boundary/grid/mask`; `inversion/re-inpainting/reconstruction error map AND edited/tampered/inpainted region`. Web: "test-time training image manipulation localization"; "conformal prediction forgery localization pixel-level risk control"; "re-inpainting idempotence forensic"; "latent 8x8 grid VAE periodic artifact localization mask boundary".

---

## Q1. Generator-inversion / re-editing consistency (fixed-point, "re-inpainting" maps, counterfactual re-editing)

### Takeaway
Reconstruction error is now a common signal for detection. In localization it shows up as an input feature (ReGFLoW, GAP-SAM) and as a training invariance (DiffusionPrint). I found no paper that localizes edits by testing, region by region, whether a candidate editor or inverter leaves the image unchanged (an idempotence or fixed-point test), or by comparing re-editing with and without a mask. That idea still looks novel. The nearest work is ReGFLoW, which already uses diffusion reconstruction error as dense spatial guidance, so a reviewer would call the novelty incremental unless the fixed-point formulation adds something measurable.

### Cited Findings
- ReGFLoW (arXiv 2609.36882, Sep 2026) [abs] needs only image-level real/fake labels. It uses **diffusion reconstruction errors as dense spatial guidance** inside multiple-instance learning to localize diffusion-edited regions, and reports that it beats the fully supervised baselines evaluated when fully synthetic images are included and in cross-dataset tests. Each map is a single reconstruction, used as a feature. It does not test each region against a specific editor, and it does not use the mask-conditioned (re-inpaint) asymmetry. — [arXiv 2609.36882](https://arxiv.org/abs/2609.36882)
- GAP-SAM (arXiv 2608.20929, Aug 2026) [abs] encodes the image and its **frozen VAE reconstruction** into a *global* artifact token injected into SAM3 via FiLM, and reports 79.8 average Pixel-F1 over 6 datasets (+12.6 over the best prior method). It also finds that "Mask-VAE Reconstruction Alignment" training data underperforms, because **VAE reconstruction artifacts transfer poorly to local inpainting artifacts**. It names the failure mode "boundary adhesion": predictions snap to object contours. — [arXiv 2608.20929](https://arxiv.org/abs/2608.20929)
- DiffusionPrint (arXiv 2604.12443, Apr 2026) [abs] uses patch-level MoCo contrastive learning in which real patches and their latent-reconstructed counterparts are treated as positives (invariance to VAE reconstruction). The resulting fingerprint map, plugged into TruFor or MMFusion, gives up to +28% on unseen mask types. — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443); invariance detail via [web summary](https://www.themoonlight.io/review/diffusionprint-learning-generative-fingerprints-for-diffusion-based-inpainting-localization)
- AEROBLADE (CVPR 2024, arXiv 2401.17879) is training-free image-level detection from autoencoder reconstruction error (LPIPS distance). — [alphaXiv 2401.17879](https://www.alphaxiv.org/abs/2401.17879)
- "Training-free reconstruction-based AIGI detectors are inherently vulnerable to adversarial examples" (arXiv 2608.16646, Aug 2026). This is a ready-made reviewer objection to any reconstruction-based localizer. — [alphaXiv 2608.16646](https://www.alphaxiv.org/abs/2608.16646)
- Zero-shot anomaly localization by diffusion inversion exists in industrial anomaly detection (arXiv 2601.08022, Jan 2026), but not for forgery. — [alphaXiv 2601.08022](https://www.alphaxiv.org/abs/2601.08022)
- A web search for re-inpainting or idempotence as a forensic signal found nothing direct. The closest results were DinoLizer (2511.20722) and an ECCV 2022 paper on artifacts inside inpainted holes. — [DinoLizer](https://arxiv.org/abs/2511.20722); [ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890145.pdf)

### Inferences
- **Remaining novel core:** for a candidate editor E, the edited region should be close to a fixed point of "mask the region, then re-inpaint with E". Authentic regions are not. The method would use the *asymmetry* d(x, E(x, M_region)) vs. d(x, E(x, M_complement)), or a multi-step "editability" map, as the localization statistic. Two framings could make this publishable: (a) a theoretical link to idempotence or contraction of the editor's denoise-decode map, and (b) an editor-agnostic version that uses a bank of open editors (SD, SDXL, FLUX-Fill) and shows transfer to closed editors.
- **Strongest reviewer objections:** (1) cost (N editors × M masks × inversion steps per image); (2) it reduces to AEROBLADE/ReGFLoW-style reconstruction error plus a mask search; (3) closed editors (GPT-Image-2, Nano-Banana-2) cannot be re-run, so the method depends on a proxy editor transferring; (4) adversarial fragility (2608.16646); (5) GAP-SAM's evidence that VAE-reconstruction cues transfer poorly to local edits.

### Gaps
- I could not confirm whether any 2026 workshop paper (e.g. CVPR 2026 SAFE workshop, HIDE) uses re-inpainting. The HIDE supplement reports latent h-space self-similarity, not re-inpainting. — [HIDE supp.](https://openaccess.thecvf.com/content/CVPR2026W/SAFE/supplemental/Ji_HIDE_Detecting_Diffusion-Based_CVPRW_2026_supplemental.pdf)

---

## Q2. Dense self-supervised forensic pretraining on real images only (JEPA / masked prediction; prediction error = anomaly map)

### Takeaway
Real-only SSL now works for **image-level** detection (Forensic Twins, camera-metadata SSL, Forensic Self-Descriptions) and for classic splicing localization (EXIF-as-language). I found **no** paper that trains a dense, masked-prediction (JEPA-style) objective on forensic residual embeddings from real images only and uses the per-patch prediction error as a localization map for *generative* edits. FoJEPA's central claim therefore looks unpublished. The gap is closing fast, though: Forensic Twins (Sep 2026) is one "dense" step away.

### Cited Findings
- Forensic Twins (arXiv 2609.31514, Sep 2026) [abs] trains redundancy-reduction SSL on two pixel-disjoint crops of a frozen residual extractor, on real images only. A GMM on real embeddings gives **97.99% AUC across 27 unseen generators**, and source attribution reaches 56.61%. The output is **image-level only**. — [arXiv 2609.31514](https://arxiv.org/abs/2609.31514)
- Self-supervised AIGI detection from camera metadata (arXiv 2512.05651, Dec 2025) is real-only SSL for image-level detection. — [alphaXiv 2512.05651](https://www.alphaxiv.org/abs/2512.05651)
- Forensic Self-Descriptions (arXiv 2503.21003, 2025) learns real-only forensic self-descriptions for zero-shot detection, open-set attribution and clustering. Localization is not mentioned in the listing. — [alphaXiv 2503.21003](https://www.alphaxiv.org/abs/2503.21003)
- Zero-shot color image manipulation localization via noise-residual artifact pattern analysis (arXiv 2608.20558, Aug 2026) is training-free and camera-trace based. It targets classic manipulations. — [alphaXiv 2608.20558](https://www.alphaxiv.org/abs/2608.20558)
- [mem] EXIF as Language (Zheng, Ma, Owens, CVPR 2023, arXiv 2301.04647) learns image–EXIF cross-modal embeddings from real photos and localizes splices through patch-embedding inconsistency. This is the main prior art for "real-only SSL → dense consistency map". Its target is splicing and camera traces, not diffusion edits.
- [mem] ZED (Cozzolino et al., ECCV 2024, arXiv 2409.15875) uses a lossless-coding model trained on real images only. Its per-pixel coding surprisal can be visualized spatially, but it is evaluated as an image-level detector.
- The user's local plan already frames this as A3 ("real-only SSL residual encoder ... with a dense (patch) objective") and FoJEPA §1/§6. — local notes (`docs/01_reanalysis_and_direction.md`, `JepaApproach/docs/02_proposed_method.md`)

### Inferences
- **Remaining novelty:** (i) a *dense* predictive objective on a residual or trace stream, (ii) trained on real images only, (iii) evaluated zero-shot on diffusion and instruction-edit localization (OpenSDI, DiffSeg30k, TGIF2, GPT-Image-2 edits). That combination has not been published. The strongest framing is "Forensic Twins / EXIF-as-language made dense and predictive".
- **Strongest reviewer objections:** (1) *Fully regenerated* edits: the VAE decoder rewrites the whole image, so the residual stream is globally synthetic and context prediction may not separate regions (TGIF2; INP-X shows a pervasive spectral shift). (2) Semantic edges and texture changes also break predictability, which raises false positives (cf. GAP-SAM's "boundary adhesion"). (3) The method must beat simple baselines such as TRAIL (Q4) and a one-class GMM on Forensic Twins patch features.

### Gaps
- I could not check whether Forensic Twins' code release includes a patch-level variant. The abstract says code "will be made publicly available".

---

## Q3. Latent-blending boundary physics (8×/16× latent grid alignment, latent feathering)

### Takeaway
The *generation* community has documented that linear latent-mask blending leaves seam artifacts and global colour shifts (PELC/DecFormer). Forensics has documented a global VAE spectral shift (INP-X). I found **no forensic paper** that explicitly models VAE-grid-aligned boundary statistics or latent-feathering signatures as a localization cue. This is a real but narrow gap, and the methods that would fix the artifacts already exist.

### Cited Findings
- "Your Latent Mask is Wrong" / PELC (arXiv 2512.05198, Dec 2025) [abs]: latent inpainting "relies almost universally on linearly interpolating VAE latents under a downsampled mask", which "produces large artifacts at mask seams and global degradation and color shifts". DecFormer reduces edge errors by up to 53%. This shows the artifact exists and that a fix is available. — [arXiv 2512.05198](https://arxiv.org/abs/2512.05198)
- INP-X (arXiv 2602.00192, Jan 2026) [abs]: VAE reconstruction induces a pervasive spectral shift across the whole image. Restoring the original pixels outside the mask drops SOTA detectors from 91% to 55%. — [arXiv 2602.00192](https://arxiv.org/abs/2602.00192)
- DinoLizer (arXiv 2511.20722, Nov 2025) [abs] treats VAE-reconstructed areas outside the mask as a separate class, using 14×14 patch predictions with LoRA on DINOv2, and reports +20% IoU over the second-best model. — [arXiv 2511.20722](https://arxiv.org/abs/2511.20722)
- A web search for "8×8 latent grid periodic artifact localization" found no direct forensic match. One theoretical analysis argues that hard latent replacement creates a "piece-wise latent manifold" seam, without empirical forensic validation. — [web results incl. 2602.18842](https://arxiv.org/pdf/2602.18842.pdf)

### Inferences
- **Novel angle:** a test of how edge statistics change with grid phase (boundary gradients/residuals binned by position mod 8 or 16) and of the latent-feather kernel shape. This would be interpretable and training-free. It also applies only to pipelines that paste back pixels (spliced, SP); FR pipelines have no pixel boundary but may still show latent-grid-aligned transitions inside the decoded image.
- **Reviewer objections:** fragile to resizing, cropping or JPEG at the 8-px phase (a Tier-3 trace by the user's own taxonomy in `SOTA_ANALYSIS.md`); closed editors use unknown or larger strides (DiT patch 2 × VAE 8 = 16); DecFormer-style compositors erase the cue; likely too narrow for TPAMI as a stand-alone paper, but a good component or ablation.

### Gaps
- I could not find measurements of grid-phase-aligned statistics for FLUX-Fill, GPT-Image or Gemini edits.

---

## Q4. Training-free patch anomaly detection on frozen foundation-model features (kNN / Mahalanobis / GMM, PatchCore-style)

### Takeaway
This is **already partially taken**. TRAIL (Aug 2026) shows that frozen DINO patch tokens localize edits without training (0.903 patch AUROC on CocoGlide vs 0.912 for supervised Detective SAM). It uses a perturbation-drift score rather than a memory bank. A PatchCore/GMM memory bank of *real* patches for generative-edit localization did not appear in my searches, but it is a small step from TRAIL and the FSAD literature, so a paper built on it alone would be judged incremental.

### Cited Findings
- TRAIL, "Frozen DINO Localizes Image Edits Without a Localizer" (arXiv 2608.18968, Aug 2026) [abs]: one global Haar perturbation plus cosine drift of patch tokens gives **.903 patch AUROC** on 80 CocoGlide images vs **.912** for mask-supervised Detective SAM. Dice is .619 at a fixed threshold and .790 at an oracle threshold, and AUROC is .855 on Poisson image interpolation. The best block sits at 0.80–0.94 normalized depth, and global context matters (.903 globally vs .735 for independently encoded crops). It does **not** use a real-image memory bank, does not test FR editors, and the test set is small (80 images). — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968)
- The FSAD literature already uses kNN over frozen DINOv2 patch features (e.g. arXiv 2510.13643; Hyper-FSAD 2605.10628), so the tooling is standard. — [2510.13643](https://www.alphaxiv.org/abs/2510.13643); [2605.10628](https://www.alphaxiv.org/abs/2605.10628)
- "Simplicity Prevails" (arXiv 2602.01738) reports that frozen VFM linear probes are blind to VAE reconstruction and localized edits (as summarized in the user's notes). This argues *against* semantic-feature memory banks working for edits. — local notes `01_reanalysis_and_direction.md` (abstract not re-fetched)
- DINO-FIL (arXiv 2610.21780, Oct 2026) and DinoLizer show that multi-layer DINO features carry strong supervised localization signal. — [2610.21780](https://www.alphaxiv.org/abs/2610.21780)

### Inferences
- **Remaining novelty:** a principled *one-class* density over intermediate-layer patch tokens plus a low-level stream (e.g. Forensic Twins-style residual embeddings), fit on real images only and evaluated on FR and instruction-edit benchmarks. Reported TRAIL comparisons should include that memory banks remove TRAIL's need for an oracle threshold, since densities can be calibrated (see Q6).
- **Reviewer objections:** "This is PatchCore applied to a new domain"; the semantic out-of-distribution confound (unusual objects score as anomalous); TRAIL already gives a training-free baseline at ~0.90 AUROC.

### Gaps
- I did not check whether TRAIL was tested on DiffSeg30k, OpenSDI or GPT-Image edits; the abstract mentions CocoGlide and Poisson only.

---

## Q5. Physical / semantic consistency (lighting, shadow, depth, normals, reflections) via VLM or 3D/intrinsic models

### Takeaway
There are benchmarks and diagnostic studies (a physical-violation counterfactual benchmark, a lighting-understanding study for editors, semantic manipulation localization). I found **no method that localizes real generative edits by checking physics** (e.g. intrinsic/shadow-direction or depth–normal consistency) on standard IFL benchmarks. This direction is open, but evidence is weak that physics violations are common in 2026 editors.

### Cited Findings
- Counterfactual physical-consistency benchmark (arXiv 2610.09205, Oct 2026) [abs]: 5,500 Mitsuba-rendered images with shadow, reflection, support, surface-response and occlusion violations, plus masks. VLM baselines reach 64–68% category macro-F1, and LLaVA drops to 40.8% under intervention shift, which suggests the baselines rely on rendering shortcuts. The data is synthetic; it does not test real editor outputs. — [arXiv 2610.09205](https://arxiv.org/abs/2610.09205)
- "Do Image Editing Models Understand Lighting?" (arXiv 2606.26738, Jun 2026) studies editor lighting knowledge, not detection. — [alphaXiv 2606.26738](https://www.alphaxiv.org/abs/2606.26738)
- Semantic Manipulation Localization / TRACE (arXiv 2604.10132, Apr 2026) [abs] targets subtle meaning-altering edits with no low-level artifacts, using semantic anchoring, perturbation sensing and reasoning on a new benchmark. — [arXiv 2604.10132](https://arxiv.org/abs/2604.10132)
- A specular-reflection inconsistency method exists for faces (arXiv 2602.06452), and universal physical descriptors exist for image-level detection (arXiv 2604.04608). — [2602.06452](https://www.alphaxiv.org/abs/2602.06452); [2604.04608](https://www.alphaxiv.org/abs/2604.04608)

### Inferences
- **Novel angle:** zero-shot localization from disagreement between intrinsic and geometry estimators (e.g. shading predicted from normals plus a global light estimate vs. observed shading; shadow-to-light direction consistency per object). This survives FR regeneration and SR laundering because it is semantic or physical rather than trace-based, so it complements trace methods well.
- **Reviewer objections:** low recall (most edits are physically plausible: attribute changes, removals with good shadow handling); monocular estimators are themselves error-prone on edited content; no benchmark with real-editor physical-violation labels exists. You might have to build one, which is itself a contribution.

### Gaps
- I found no statistics on how often current editors (FLUX-Kontext, GPT-Image-2, Nano-Banana-2) produce physics violations.

---

## Q6. Calibrated / uncertainty-aware / conformal localization maps with guarantees

### Takeaway
**No prior work found** that applies conformal prediction or conformal risk control to pixel-level forgery localization. The searches covered conformal + forgery localization, conformal + manipulation mask, and risk control + segmentation forensics. Related pieces exist: uncertainty-aware state updating in EG-Mamba, TruFor-style confidence maps, calibrated *image-level* authentication, and conformal methods for detection boxes. This is a clear but "methodological transfer" gap.

### Cited Findings
- Web search found no paper combining conformal prediction with pixel-level forgery masks. The nearest is sequential conformal risk control for object detection (arXiv 2505.24038). Conformal risk control (arXiv 2208.02814) bounds monotone losses such as FNR. — [2505.24038](https://arxiv.org/pdf/2505.24038); [2208.02814](https://tldr.takara.ai/p/2208.02814)
- EG-Mamba (arXiv 2607.29156, Jul 2026) [abs] updates the localization state progressively using uncertainty and boundary evidence, but without formal guarantees. — [arXiv 2607.29156](https://arxiv.org/abs/2607.29156)
- "Certification of Real Images through Calibrated Content Authentication" (arXiv 2610.05870, Oct 2026) addresses image-level calibrated certification. — [alphaXiv 2610.05870](https://www.alphaxiv.org/abs/2610.05870)
- TRAIL's Dice rises from .619 (fixed threshold) to .790 (oracle threshold), so **threshold selection is a large error source** that calibration could address. — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968)
- The user's own SOTA notes flag metric and threshold inconsistency (Gap 5). — local `featureAnalysis/SOTA_ANALYSIS.md`

### Inferences
- **Novel angle:** conformal risk control over the mask threshold that guarantees pixel FNR or FDR (or "the edited region is inside the predicted set") under exchangeability, plus **conditional/Mondrian conformal by editor family or post-processing** to address covariate shift. A useful twist: calibrate on *real* images only (FDR control on authentic pixels), which needs no fake labels.
- **Reviewer objections:** exchangeability breaks under a new editor, so guarantees fail exactly where they are needed. Report the coverage degradation honestly, or use weighted conformal. Alone it is a TIFS-level contribution, not TPAMI. It is best paired with Q2 or Q4.

### Gaps
- [mem] General conformal semantic segmentation exists (e.g. Mossina et al. 2024). The exact ID was not verified in this session.

---

## Q7. Localization in fully regenerated (FR) images: reference-free change detection, MLLM reasoning with dense grounding

### Takeaway
FR localization is acknowledged as **open** (TGIF2). Proactive watermark-style solutions now exist (APT: FR IoU 0.92), and MLLM grounding datasets and models exist (EditSleuth, LEGION, and SIDA/FakeShield [mem]). **Passive**, reference-free FR localization that explicitly reasons about "what would the generator have produced if it had not edited here" (two-generation or diff-based reasoning) was not found. This is the most impactful open slot, and also the hardest one.

### Cited Findings
- TGIF2 (arXiv 2603.28613, Mar 2026) [abs]: IFL methods localize spliced edits but struggle on FR. Fine-tuning on FR helps, but random non-semantic masks reveal **object bias**, and generative SR "significantly weakens forensic traces". — [arXiv 2603.28613](https://arxiv.org/abs/2603.28613)
- APT (arXiv 2608.30656, Aug 2026) [abs] is *proactive*: a semi-fragile latent perturbation aligns features to an anchor, and FR tampering is localized from alignment disparity (IoU 0.92 vs WAM 0.84; passive methods ~0.5 AUC). It requires embedding before distribution. — [arXiv 2608.30656](https://arxiv.org/abs/2608.30656)
- EditSleuth (arXiv 2605.08695, May 2026) [abs]: 257,725 edit triplets (source, edited, mask, 12-class taxonomy, six-step grounded reasoning chain) from Pico-Banana and MagicBrush. A Qwen2-VL-2B pilot does classification only. — [arXiv 2605.08695](https://arxiv.org/abs/2605.08695)
- LEGION (arXiv 2503.15264, 2025) [abs]: an MLLM that detects, segments and explains artifacts in *fully synthetic* images (SynthScars, 12k), not edit localization. — [arXiv 2503.15264](https://arxiv.org/abs/2503.15264)
- ReGFLoW reports gains when evaluation includes fully synthetic images. — [arXiv 2609.36882](https://arxiv.org/abs/2609.36882)
- PAL / EditGroup-45K (arXiv 2610.07916, Oct 2026) [abs] uses **pairwise edit relations** (source-anchored edit groups) to disentangle artifacts during training. This is the closest to "two-generation" reasoning, but only as a training signal. — [arXiv 2610.07916](https://arxiv.org/abs/2610.07916)
- [mem] SIDA (CVPR 2025, arXiv 2412.04292) and FakeShield (ICLR 2025, arXiv 2410.02761) are MLLM detection+localization+explanation frameworks.

### Inferences
- **Novel angle:** in FR images both regions are synthetic, but they are synthetic in *different ways*. Unedited areas are VAE round-trips of camera content (high-frequency attenuation only, per INP-X), while edited areas are de-novo samples. A passive "round-trip vs. de-novo" contrast could localize without a reference: compare the local statistics of *E-roundtrip(x)* with *x*, or estimate per-patch "how many VAE passes" (a generation-count estimator). Combined with MLLM proposals and dense verification, this is a strong CVPR/TPAMI story.
- **Reviewer objections:** closed FR editors (GPT-Image-2, Nano-Banana-2) re-render with unknown decoders and may re-synthesize textures everywhere, which makes the round-trip/de-novo distinction vanish. Object bias (TGIF2) also needs random-mask controls.

### Gaps
- Searches for "two-generation" or "double-generation" reasoning returned nothing directly relevant. I could not check how GPT-Image-2 handles unedited regions (exact pixels vs. regeneration).

---

## Q8. Test-time adaptation / per-image self-supervised fitting

### Takeaway
This direction is **already published** for classic IML: ForgeryTTT (2024) does per-image test-time training. I found nothing specifically for *generative* edits that uses a real-only self-supervised loss (e.g. JEPA prediction loss) at test time. Novelty would have to come from coupling TTA with Q2's pretext task. ForgeryTTT will be the main prior art reviewers cite.

### Cited Findings
- ForgeryTTT (arXiv 2410.04032, Oct 2024) [abs] uses a ViT encoder with localization and classification heads. At test time, the predicted mask splits tokens into manipulated and genuine groups, and a self-supervised classification loss updates the encoder per image. It reports +20.1% localization accuracy vs zero-shot methods and +4.3% vs non-zero-shot methods on five classic benchmarks. — [arXiv 2410.04032](https://arxiv.org/abs/2410.04032)
- Forensic-Aware Continual Adaptation for IFL (arXiv 2609.38251, Sep 2026) is continual (dataset-level) rather than per-image adaptation. — [alphaXiv 2609.38251](https://www.alphaxiv.org/abs/2609.38251)
- A single-image zero-shot anomaly localization method (arXiv 2509.18354, 2025) exists in industrial anomaly detection. — [alphaXiv 2509.18354](https://www.alphaxiv.org/abs/2509.18354)

### Inferences
- **Novel angle:** at test time, fit the JEPA or Forensic Twins predictor *on the suspect image itself* (internal statistics, like Deep Image Prior or ZSSR for forensics). Edited regions remain unpredictable after fitting because they are a minority distribution. This is a natural, cheap extension of Q2.
- **Reviewer objection:** "ForgeryTTT already does per-image TTT". The difference must be shown as a label-free, real-only pretext loss and evaluated on generative/FR benchmarks.

### Gaps
- I did not verify whether ForgeryTTT was evaluated on diffusion-edit datasets (CocoGlide etc.); the abstract says "five standard benchmarks".

---

## Q9. Pixel-level generative-model attribution and multi-edit / sequential edits

### Takeaway
DiffSeg30k already frames localization as **semantic segmentation over editor identities**, including up to three sequential edits. Multi-tool attribution exists for faces (MIEA, image-level). What remains open is **open-set** pixel-level attribution (editors unseen at training time) and **edit-order** recovery (which edit came first), which I did not find.

### Cited Findings
- DiffSeg30k (arXiv 2511.19111, Nov 2025) [abs]: 30k COCO images, 8 diffusion editors, up to 3 sequential edits, and per-pixel labels for both "where" and "which editor". Segmentation baselines struggle with distortions but are strong whole-image classifiers. — [arXiv 2511.19111](https://arxiv.org/abs/2511.19111)
- Multi-Tool Image Editing Attribution, MIEA / MultiEdit / DPEC (arXiv 2609.02751, Sep 2026) [abs]: 500k+ faces, 6 tool types, attribution of multiple tools up to 5 steps. It predicts *which tools*, not per-pixel maps, and covers faces only. — [arXiv 2609.02751](https://arxiv.org/abs/2609.02751)
- DiffusionPrint uses a generator-aware head and same-model fingerprint consistency (a building block for attribution). — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443)
- Forensic Twins: real-only embeddings attribute generator sources at 56.61% accuracy (image-level). — [arXiv 2609.31514](https://arxiv.org/abs/2609.31514)

### Inferences
- **Novel angle:** open-set per-pixel attribution by clustering dense real-only forensic embeddings (Forensic Twins made dense, then patch clustering → "number of distinct sources" + map), plus edit-order recovery from occlusion and overwriting of fingerprints.
- **Reviewer objection:** a supervised closed-set version already exists (DiffSeg30k), and open-set clustering accuracy may be low (image-level attribution is only 56.6%).

### Gaps
- No search surfaced work on recovering edit *order*. Only two queries touched this, so confidence is moderate.

---

## Q10. Robustness to generative super-resolution / laundering

### Takeaway
TGIF2 shows that generative SR "significantly weakens forensic traces". Laundering through foundation models is well documented for **watermarks**. I found **no localization method designed for, or certified against, generative-SR laundering**. This is an open evaluation and method gap, but it is mostly an "add a robustness axis" contribution.

### Cited Findings
- TGIF2 (arXiv 2603.28613) [abs]: generative SR significantly weakens forensic traces for both IFL and SID. — [arXiv 2603.28613](https://arxiv.org/abs/2603.28613)
- Watermark laundering via foundation image models (arXiv 2609.01249) and diffusion editing defeating invisible watermarks (2602.20680, 2511.05598) show regeneration erases low-level signals. — [2609.01249](https://www.alphaxiv.org/abs/2609.01249); [2602.20680](https://www.alphaxiv.org/abs/2602.20680); [2511.05598](https://www.alphaxiv.org/abs/2511.05598)
- GAP-SAM reports the best results at all tested JPEG, blur and resize severities, but not generative SR. — [arXiv 2608.20929](https://arxiv.org/abs/2608.20929)
- WildFakeToolBench (arXiv 2610.31152, Oct 2026) benchmarks *detectors* (image-level) under web-sourced processing tools. — [alphaXiv 2610.31152](https://www.alphaxiv.org/abs/2610.31152)

### Inferences
- Laundering through generative SR is effectively an FR regeneration of the *whole* image. Methods that rely on semantic or physical consistency (Q5) or on "round-trip vs. de-novo" statistics (Q7) are the natural candidates to survive it. A paper on this could present the regenerated-image "trace hierarchy" (which cues survive which regenerations) and connect Q7 and Q10.

### Gaps
- No quantitative per-method numbers from TGIF2's SR experiment were extracted; only the abstract was read.

---

## Additional directions encountered (not in the original list)

- **Perturbation-response maps** (TRAIL): a training-free, model-probing signal. It could be combined with the Q1 fixed-point test, e.g. using perturbation sensitivity of the *generator* instead of DINO. — [2608.18968](https://arxiv.org/abs/2608.18968)
- **Weak supervision (image labels only)** is now a direction (ReGFLoW), which raises the bar for "self-supervised" claims. — [2609.36882](https://arxiv.org/abs/2609.36882)
- **Content-aware evaluation controls**: INP-X (restore pixels outside the mask) and TGIF2 random masks should be mandatory controls for any new method. Otherwise reviewers will attribute gains to global VAE artifacts or object bias. — [2602.00192](https://arxiv.org/abs/2602.00192); [2603.28613](https://arxiv.org/abs/2603.28613)
- **Energy-gap / high-frequency suppression** (FLAME): a theoretical statistic for diffusion regions with a SAM adapter. It is a competitor to Q3 and Q4. — [2606.02178](https://arxiv.org/abs/2606.02178)

## Overall ranking (my judgment, based on the findings above)

1. **Q2 + Q8 (dense real-only predictive SSL + per-image fitting)**: unpublished in this exact form. It has a direct lineage (Forensic Twins, EXIF-as-language, ForgeryTTT) and fits the user's FoJEPA plan. Risk: FR images. Must beat TRAIL and the INP-X/random-mask controls.
2. **Q7 (passive FR localization via round-trip vs. de-novo statistics)**: highest impact and open. Only a proactive solution (APT) exists. Highest technical risk.
3. **Q1 (re-editing fixed-point maps)**: novel formulation, but ReGFLoW and GAP-SAM are close. Cost and closed-editor transfer are the main objections.
4. **Q6 (conformal risk control for masks)**: no prior work found. Cheap add-on contribution, better as a component than a stand-alone TPAMI paper.
5. **Q5 (physics)**: open, but recall and benchmark problems. Q3 (latent-grid) is narrow and fragile. Q4 is largely pre-empted by TRAIL. Q9 is partially pre-empted by DiffSeg30k. Q10 is mainly an evaluation axis.
