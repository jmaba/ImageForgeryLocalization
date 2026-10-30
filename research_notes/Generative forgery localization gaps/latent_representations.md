# Frozen / Self-Supervised Latent Representations + Simple Discriminators for AI-Generated Image Forensics, and Their Transfer to Pixel-Level Localization (2023 – Oct 2026)

Note on verification: arXiv abstracts for 2602.01738, 2610.08639, 2609.31514, 2512.05651, 2603.29386, 2608.18968, 2604.16083, 2608.12155, 2608.20929, 2604.12443, 2511.22471, 2604.26772, 2503.21003, 2409.15875, 2402.19091 and 2408.09647 were retrieved in this session. Full-text reports (via alphaXiv) were read for 2602.01738 and 2608.18968. Items marked "[not re-verified]" are cited from well-known IDs without re-reading them in this session. No numbers were reported for those items.

## Q1. Linear probes / light heads on frozen vision foundation models (VFMs): what works, and which layers carry forensic information?

### Takeaway
By 2026 a linear probe on frozen modern VFMs (DINOv3, Perception Encoder, MetaCLIP-2) is the strongest image-level baseline in the wild, and adapters or LoRA often make it worse. The same papers find these probes blind to VAE-only reconstruction and to localized edits, because global pooling washes out local traces. The layer evidence is mixed. For CLIP, intermediate blocks help image-level detection (RINE). For DINO-family encoders, late blocks (normalized depth 0.80–0.94) carry the strongest training-free localization signal (TRAIL). DINOv3 is reported to rely on global low-to-mid-frequency cues, not high-frequency generator artifacts.

### Cited Findings
- UnivFD (Ojha et al., CVPR 2023) showed that a linear classifier, or nearest neighbour, on frozen CLIP ViT-L/14 features generalizes across generators better than CNNs trained from scratch. This is the origin of the paradigm. — [arXiv 2302.10174](https://arxiv.org/abs/2302.10174) [not re-verified]; its role as the "turning point" is also described in [Simplicity Prevails, arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
- **Simplicity Prevails** (Zhou et al., Feb 2026) trains only a linear head on frozen PE, MetaCLIP-2, SigLIP2 and DINOv3, using GenImage SD1.4 training data, AdamW, lr 1e-3, 2 epochs and no augmentation — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
  - DINOv3-Linear reaches 96.5% average accuracy on GenImage, against 94.6% for OMAT. On the in-the-wild sets (Chameleon, WildRF, SocialRF, CommunityAI) it reaches 94.0% on average, about 10 points above DDA (85.0%) and more than 30 points above traditional detectors. PE-CLIP reaches 97.8% and DINOv3 97.2% on AIGIHolmes. MetaCLIP2 reaches 90.7% on AIGI-Now — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
  - The gains depend on the model generation. DINOv3 beats DINOv2 by 30.4 points in the wild, and MetaCLIP-2 beats MetaCLIP by 18.8 points. A DINOv3 pretrained on satellite imagery only (DINOv3-Sat) gets 12.1% accuracy on Chameleon fakes, against 91.4% for DINOv3-Web. The authors attribute the capability to synthetic content in the pretraining data — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
  - **Limitations:** the probes are "largely blind" to pure VAE reconstruction (DDA-COCO), where detection rates fall to negligible levels. On localized diffusion edits (BR-Gen) they reach about 50–60%, which the authors attribute to global pooling suppressing subtle local traces. They also degrade under recapture and transmission, with MetaCLIP2 at about 72% on RRDataset — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
  - LoRA hurts generalization. With PE at r=8, Chameleon accuracy falls from 95.9% to 63.5%. Moving specialized heads (Effort, AIDE, DDA) onto modern VFMs underperforms the plain linear probe, and DDA collapses to 50–60% — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
- **RINE** (Koutlis & Papadopoulos, ECCV 2024) maps intermediate CLIP image-encoder blocks into a forgery-aware space, with a learned importance weight for each block. It reports a +10.6% average absolute improvement across 20 test sets, after training for one epoch (about 8 minutes). Its motivation is that final-layer CLIP features "mostly encapsulate high-level visual semantics" — [arXiv 2402.19091](https://arxiv.org/abs/2402.19091)
- **C2P-CLIP** (Aug 2024) injects category-common prompts into CLIP to improve generalization in deepfake/AIGC detection — [arXiv 2408.09647](https://arxiv.org/abs/2408.09647)
- **AIDE** combines low-level patch statistics with high-level CLIP semantics — [arXiv 2406.19435](https://arxiv.org/abs/2406.19435) [not re-verified]. Simplicity Prevails reports that AIDE, given a modern VFM backbone, still trails the plain linear probe — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
- **Huang et al., "Rethinking Cross-Generator Image Forgery Detection through DINOv3"** (Nov 2025) report that frozen DINOv3 detects across generators without fine-tuning. It relies on "global, low-frequency structures" rather than high-frequency generator-specific artifacts. A training-free token-ranking step selects a subset of authenticity-relevant tokens for a linear probe — [arXiv 2511.22471](https://arxiv.org/abs/2511.22471)
- **Cozzolino, Poggi & Verdoliva** (Aug 2026), using a DDIM-inversion protocol, report the following — [arXiv 2608.12155](https://arxiv.org/abs/2608.12155)
  - Semantically identical regenerated copies receive very different detector scores, so foundation-model detectors are "not primarily driven by semantic failures."
  - The discriminative cues lie in the low-to-mid frequency range.
  - Regenerated images have lower latent variance and lower effective dimensionality.
- **TAP** (Apr 2026) benchmarks many VFM families out of the box on fully generated and inpainted images. The best model beats original CLIP by more than 12% accuracy. Replacing CLS/linear readout with tunable attention pooling over patch tokens sets state of the art on two in-the-wild benchmarks that include inpainted images — [arXiv 2604.26772](https://arxiv.org/abs/2604.26772)
- **Forensic Reserve / RGE** (Oct 2026) uses a "Forensic Lens" to decompose activations across layers and token groups into independent components. It screens them by real-versus-generated response difference to find sparse "reserve sites." It then trains only Forensic Reserve Adapters constrained to those subspaces. The setup uses 500 labeled images and less than 0.2% of the backbone parameters, and is competitive on three benchmarks. It improves over the frozen detectors for 8 encoders, both SSL and VLM, which suggests a "latent forensic capacity broadly shared across pretrained vision models" — [arXiv 2610.08639](https://arxiv.org/abs/2610.08639)
- Other related 2026 work that was not read in full:
  - Anchor-regularized DINOv3 adaptation — [arXiv 2608.15196](https://arxiv.org/abs/2608.15196)
  - "Unleashing VLMs for generalizable AIGI detection", which starts from the linear-probe result — [arXiv 2608.04935](https://arxiv.org/abs/2608.04935)
  - "Foundation Models are Implicit Deepfake Detectors" — [arXiv 2608.09427](https://arxiv.org/abs/2608.09427)
  - COMPASS — [arXiv 2610.18034](https://arxiv.org/abs/2610.18034)
- **Layer evidence for localization:** in TRAIL, across 16 DINO / DINOv2 / DINOv2-reg / DINOv3 encoders (ViT-S to 7B), the block selected for localization always lay at normalized depth 0.80–0.94 — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968)

### Inferences
- Image-level evidence points to a shared, non-semantic, low-to-mid-frequency distributional cue in modern VFMs. Its strength depends on whether the pretraining data contained synthetic content. That makes it partly a memorization of the web's synthetic distribution, which is a risk for future generators and for local edits.
- The "late layers carry it" (DINO, TRAIL) and "intermediate layers carry it" (CLIP, RINE) findings are not in conflict. They concern different objectives (perturbation drift versus a supervised probe) and different backbones. No paper found here sweeps layers for dense, patch-level linear probing of real versus inpainted patches.
- Frozen-feature image-level wins come from pooled tokens. The "blind to localized editing" result shows that the pooling is the bottleneck, not necessarily the patch tokens. TAP's attention pooling and TRAIL's patch maps point the same way.

### Gaps
- No systematic layer-by-layer study of patch-level (dense) linear probes for diffusion-inpainting localization across PE, MetaCLIP-2 and DINOv3 was found.
- Exact metrics for C2P-CLIP and AIDE were not re-read in this session.
- Whether the Simplicity Prevails BR-Gen failure is specific to pooling, or persists with patch-token probes, was not tested in that paper.

## Q2. Self-supervised forensic representation learning on real images only

### Takeaway
Real-only SSL has matured into strong zero-shot image-level detectors, all scored with one-class density models (GMM, entropy/likelihood). Three works:
- Forensic Twins: residual-based, Barlow-style SSL with disjoint crops, 97.99% AUC across 27 unseen generators.
- Forensic Self-Descriptions: predictive-filter residual models.
- EXIF-supervised SSL: camera-metadata pretext tasks.

All of them operate on stationary acquisition-pipeline statistics, which in principle are spatially local and should support localization. None of the 2024–2026 papers found here reports pixel-level localization of diffusion edits.

### Cited Findings
- **Forensic Twins** (Muñoz-Haro et al., Sep 2026) — [arXiv 2609.31514](https://arxiv.org/abs/2609.31514)
  - Method: a frozen off-the-shelf forensic residual extractor, two spatially disjoint crops sharing no pixel, and a redundancy-reduction objective. Because the crops share no content, the remaining common signal is the "stationary fingerprint of the image acquisition pipeline." It trains only on real images.
  - Results: 56.61% generator attribution accuracy, 6.13 points above the previous zero-shot SoTA at 375x lower latency. A GMM fitted on real embeddings gives 97.99% AUC across 27 unseen generators (GANs, diffusion, commercial).
  - The authors argue that standard SSL augmentations "overwrite the micro-statistics of image formation."
- **Forensic Self-Descriptions** (Nguyen, Azizpour, Stamm, CVPR 2025) learn diverse predictive filters on real images only. The filters extract residuals capturing "forensic microstructures," which are modeled jointly across scales. The resulting compact per-image model is used for zero-shot detection, open-set attribution and clustering — [arXiv 2503.21003](https://arxiv.org/abs/2503.21003)
- **Camera-metadata SSL** (Zhong et al., Dec 2025) — [arXiv 2512.05651](https://arxiv.org/abs/2512.05651)
  - Pretext tasks: classify categorical EXIF tags (camera model, scene type) and pairwise-rank ordinal tags (focal length, aperture), on real photos only.
  - Two detection modes: (a) one-class, with a GMM on photo features where low likelihood means AI-generated; (b) binary, with the SSL extractor as regularizer for a classifier on high-frequency residuals of spatially scrambled patches.
  - The abstract claims SoTA, in-the-wild generalization and robustness, but gives no numbers.
- **ZED** (Cozzolino et al., ECCV 2024) is a zero-shot, entropy-based detector. A multi-resolution lossless image encoder trained on real images estimates per-pixel probabilities, and the score is the image's "surprise." It reports an average accuracy gain of more than 3% over SoTA — [arXiv 2409.15875](https://arxiv.org/abs/2409.15875). The per-pixel likelihood structure is inherently dense, but the paper reports image-level detection only.
- Classic real-only forensic SSL that these works build on [not re-verified]:
  - Noiseprint (Cozzolino & Verdoliva), a camera-model fingerprint learned with Siamese training on same/different-camera patches — [arXiv 1808.08396](https://arxiv.org/abs/1808.08396)
  - Noiseprint++, inside TruFor — [arXiv 2212.10957](https://arxiv.org/abs/2212.10957)
  - "Fighting Fake News: Image Splice Detection via Learned Self-Consistency" (Huh et al., ECCV 2018), with EXIF-consistency between patches used for splice localization — [arXiv 1805.04096](https://arxiv.org/abs/1805.04096)
  - EXIF-as-language (Zheng, Owens et al., CVPR 2023), with a contrastive image–EXIF embedding used zero-shot for splice localization — [arXiv 2301.04647](https://arxiv.org/abs/2301.04647)
- **DiffusionPrint** (Giakoumoglou & Papadopoulos, Apr 2026) — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443)
  - Patch-level MoCo-style contrastive learning, using the consistency of a generator's fingerprint across inpainted regions as the self-supervisory signal.
  - Uses inpainted data, so it is not real-only.
  - Plugged into TruFor or MMFusion as an extra modality, it gives up to +28% on unseen mask types.
  - Its motivation: full-regeneration inpainting pipelines rebuild the whole image through the latent decoder, which "disrupt[s] the camera-level noise patterns" that Noiseprint-style methods rely on.
- No JEPA- or MAE-style pretraining designed for forensics was surfaced by the searches in this session.

### Inferences
- The two building blocks for real-only localization already exist: patch-level real-only embeddings (Forensic Twins crops, Noiseprint patches, ZED per-pixel likelihoods) and one-class scoring (GMM). The combination, patch-wise GMM/likelihood maps on diffusion-inpainted images, appears unreported for modern diffusion edits.
- DiffusionPrint warns that full-image VAE regeneration erases the camera traces that real-only methods model. Real-only acquisition fingerprints may therefore flag the whole image as anomalous, not the edited region, when the inpainting pipeline re-decodes the full image. Compositing the generated region back into the original pixels avoids this, but not all datasets do it.
- Forensic Twins' disjoint-crop design suppresses semantics by construction. That is relevant to the semantic-confound problem in localization (Q5).

### Gaps
- Forensic Twins, Self-Descriptions and the EXIF-SSL paper report no localization numbers.
- No forensic JEPA or MAE pretraining paper was verified in this session.
- Robustness of real-only GMM detectors to JPEG format and resizing confounds is not documented in the abstracts.

## Q3. Anomaly-detection / one-class / training-free LOCALIZATION of diffusion edits

### Takeaway
The clearest training-free localization result is TRAIL ("Frozen DINO Localizes Image Edits Without a Localizer," Aug 2026):
- Method: drift of frozen DINOv3-7B patch tokens under a global Haar high-frequency attenuation, plus a 3x3 median filter.
- CocoGlide: patch AUROC 0.903 versus 0.912 for mask-supervised Detective SAM (difference not significant). Dice is 0.619 against 0.709.
- Limits: coarse (28x28 grid), and weaker on TGIF2 semantic masks.

Training-free localization based on diffusion reconstruction exists, but needs a generative model at inference. Real-only (one-class) patch-level localization of diffusion edits with a learned forensic SSL encoder plus kNN/GMM/Mahalanobis scoring was not found with reported numbers.

### Cited Findings
- **TRAIL** (Kumar, Jain, Kainz, Aug 2026) — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968)
  - Method: the drift score per patch is 1 − cos(f(x), f(T(x))), with T(x) = x − 0.2·HF₂(x) (two-level Haar). Tokens come from DINOv3 ViT-7B/16, block 36, at 448x448 input.
  - CocoGlide test split, 80 images:
    - Patch AUROC 0.903 against Detective SAM 0.912 (difference −0.009, 95% CI [−0.049, 0.033]).
    - AUPRC 0.788 against 0.841.
    - Dice 0.619 against 0.709 at a fixed threshold, and 0.790 with a per-image oracle threshold.
  - Controls on the same data:
    - Raw-pixel RMS control: 0.641 AUROC.
    - Aligned-real control (score the unedited source, then apply the mask): 0.697.
    - Fake-minus-aligned-real increment: +0.206.
    - Raw drift before the median filter: 0.843.
  - Transfer to classical edits: PII 0.855 against 0.864; FPI 0.850 against 0.894.
  - TGIF2 semantic masks: the AUROC deficit against Detective SAM is 0.092, and the increment is only +0.068 (+0.164 on random masks). This suggests the drift partly responds to semantic objects before any edit.
  - Context matters:
    - Haar perturbation: 0.903 global, 0.857 local-in-canvas, 0.735 for independently encoded crops.
    - Gaussian noise: 0.798 falling to 0.527.
    - Gaussian blur: 0.830 falling to 0.610.
  - Scaling: for DINOv3, the edit-specific increment grows from 0.089 (ViT-S) to 0.206 (7B), while raw AUROC saturates after ViT-B.
- The TRAIL paper also positions related training-free work, as described by the TRAIL authors — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968)
  - RIGID (DINOv2 under Gaussian noise), MINDER and WaRPAD (Haar high-frequency on crops) are perturbation-based training-free detectors that pool patch responses into one image score.
  - DRIFT produces patch-wise perturbation maps but trains one-class projection heads for image-level detection.
  - Zhang et al. do training-free localization by comparing conditional and unconditional diffusion reconstructions, which needs a generative model at inference.
- **Zero-shot colour manipulation localization via noise-residual artifact pattern analysis** (Aug 2026) exploits camera demosaicing, processing and compression traces — [arXiv 2608.20558](https://arxiv.org/abs/2608.20558). Only the abstract snippet was seen. Its diffusion-edit numbers were not verified.
- Classic self-consistency localization (Huh et al. 2018; EXIF-as-language 2023) targets splicing — [arXiv 1805.04096](https://arxiv.org/abs/1805.04096), [arXiv 2301.04647](https://arxiv.org/abs/2301.04647) [not re-verified]. No numbers were checked here for diffusion inpainting.
- Weakly supervised direction: "Less Supervision, Better Generalization" (Sep 2026) localizes diffusion-edited regions with weak supervision and argues that artifacts are "misaligned with semantic or object boundaries" — [arXiv 2609.36882](https://arxiv.org/abs/2609.36882) (abstract snippet only).
- "Look Closer: Patch-wise Supervision for AIGI detection" (Sep 2026) reports that small RGB patches keep synthesis evidence even when they show little of the scene — [arXiv 2609.37937](https://arxiv.org/abs/2609.37937) (snippet only).

### Inferences
- TRAIL shows that frozen VFM patch tokens do carry spatial edit information, even though image-level probes on the same models are "blind to localized editing" (Simplicity Prevails). This supports pooling, not representation, as the failure point.
- TRAIL's needs point to what dense, one-class methods should adopt:
  - Global context, since independently encoded crops lose 0.17 AUROC.
  - An aligned-real control.
  - Calibration, since fixed-threshold Dice is well below oracle Dice.
- The TGIF2 semantic-mask weakness, and GAP-SAM's "boundary adhesion" (Q5), show the same confound: frozen semantic encoders respond to objects.
- Test-time training/adaptation for forensic localization was not found in the searches.

### Gaps
- No paper was found reporting patch-level kNN/GMM/Mahalanobis scoring on real-only forensic SSL embeddings (Forensic Twins, Noiseprint++, EXIF-SSL) for diffusion-inpainting localization with pixel-F1/IoU.
- No test-time-training forensic localization numbers were found.
- TRAIL is evaluated on small test sets (80 CocoGlide images, 720 TGIF2 splices) and excludes fully regenerated images.

## Q4. Contrastive intra-image objectives for localization

### Takeaway
Intra-image contrastive learning (tampered versus authentic patches in the same image) is an established supervised IML regularizer, starting with NCL-IML (ICCV 2023). In 2026 it was carried over to prompt-based or diffusion edits: ICL-Net in the PromptForge-350k paper, and DiffusionPrint's cross-image patch contrast. These all still need masks or generated data. The real-only analogue is self-consistency (Huh 2018, EXIF-as-language), which contrasts patches by acquisition metadata rather than by edit label.

### Cited Findings
- **NCL-IML** (Zhou et al., ICCV 2023) — [arXiv 2309.14900](https://arxiv.org/abs/2309.14900), [GitHub](https://github.com/Knightzjz/NCL-IML), DOI 10.1109/ICCV51070.2023.02042
  - Contrastive learning between tampered and authentic patches, with no pre-training.
  - Handles "non-mutually exclusive" contour patches through a dual-branch pivot structure that switches them between positive and negative roles, plus a pivot-consistent loss.
  - Reports SoTA on five benchmarks.
- **ICL-Net** — the ID arXiv 2603.29386 is the paper "PromptForge-350k: A Large-Scale Dataset and Contrastive Framework for Prompt-Based AI Image Forgery Localization" (Wang et al., Mar 2026) — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)
  - Triple-stream backbone with intra-image contrastive learning.
  - 62.5% IoU on PromptForge-350k, 5.1 points above SoTA. Degradation costs less than 1% IoU.
  - Average IoU on unseen editing models is 41.5%, which shows a large generalization drop.
  - Masks are auto-annotated through keypoint alignment and semantic similarity.
- **DiffusionPrint** — patch-level MoCo with cross-category hard negatives — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443)
- **"Can We Model the Artifacts Explicitly? Pairwise Edit Relations for IML"** (Oct 2026) disentangles artifacts through pairwise edit relations — [arXiv 2610.07916](https://arxiv.org/abs/2610.07916) (snippet only, not read).
- A paper titled "Rethinking IML via contrastive learning" could not be located or verified in this session.

### Inferences
- The intra-image contrastive objectives (NCL, ICL-Net) and the real-only self-consistency objectives (Huh, Forensic Twins' disjoint crops) share one structure: "patches from the same source should embed together." The difference is the source of the positive-pair label, a mask versus the acquisition pipeline. A real-only, intra-image consistency objective on a frozen VFM, or on a Forensic-Twins-style encoder, scored by a one-class model, is the obvious unexplored combination.
- ICL-Net's drop from 62.5 to 41.5 IoU on unseen editors repeats the supervised-localization generalization gap that image-level frozen probes avoided.

### Gaps
- "Rethinking IML via contrastive learning" was not found. Its reference should be confirmed by the coordinator.
- No direct comparison of contrastive-IML objectives under real-only training was found.

## Q5. Evidence that latents are confounded by semantics, JPEG format or resolution; controls and diagnostics used

### Takeaway
There is substantial evidence of confounding:
- JPEG and size biases in GenImage-type datasets.
- Semantic shortcuts in pixel-supervised localization (boundary adhesion, mask geometry).
- Frozen-encoder drift that responds to semantic objects before any edit.
- Detector capability that depends on synthetic data in pretraining.

The diagnostics used so far:
- Debiased re-encoding.
- Semantics-aligned regeneration (DDIM-inversion depth sweeps, ControlNet-aligned COCO).
- Frequency swapping.
- Aligned-real and raw-pixel controls.
- Counterfactual pretraining (DINOv3-Sat).
- AIGI-Now's pixel-versus-semantic splits.

### Cited Findings
- **Fake or JPEG?** (Grommelt et al., ECCVW 2024) — [arXiv 2403.17608](https://arxiv.org/abs/2403.17608)
  - In GenImage, real images are mostly JPEG and fakes PNG, and generated images come in fixed sizes. Detectors learn these shortcuts, and accuracy drops sharply when fakes are JPEG-compressed.
  - Debiasing improves cross-generator performance by more than 11 points for ResNet50 and Swin-T. This figure comes from a secondary summary and should be checked in the PDF.
  - Debiased data: unbiased-genimage.org.
- **GAP-SAM** (Aug 2026) — [arXiv 2608.20929](https://arxiv.org/abs/2608.20929)
  - Finds that "pixel supervision entangles forensic evidence with dataset-specific mask geometry and semantic boundaries."
  - Names a **boundary adhesion** failure: fine-tuned segmenters snap to object contours, not manipulation boundaries.
  - A semantics- and geometry-aligned dataset (COCO-ControlNet, built from source Canny edges and depth) improves OOD localization across several localizers.
  - Mask-VAE reconstruction alignment underperforms, so "VAE reconstruction artifacts transfer poorly to local diffusion-inpainting artifacts."
  - GAP-SAM itself (frozen VAE-reconstruction global artifact token into SAM3 through FiLM) reaches 79.8 average Pixel-F1 over six datasets, 12.6 points above prior methods.
- **Cozzolino et al. 2026** use a DDIM-inversion-depth sweep to hold semantics fixed, plus frequency swapping. Foundation-model detectors respond to non-semantic low-to-mid-frequency discrepancies — [arXiv 2608.12155](https://arxiv.org/abs/2608.12155)
- **Simplicity Prevails** diagnostics — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)
  - Text–image alignment probes: modern VLMs align fakes with "AI generated" or with platform names like "midjourney_images," so VLM detection is partly an explicit semantic concept of forgery.
  - A counterfactual pretraining domain (DINOv3-Sat, 12.1% on fakes).
  - AIGI-Now pixel-versus-semantic splits; specialized detectors fall to about 50% on the semantic splits.
  - Resolution handling was native-resolution resize plus center-crop. A resize/crop sensitivity sweep was not reported in the material read.
- **TRAIL controls** — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968)
  - Raw-pixel control: 0.641.
  - Aligned-real control (unedited source, same mask region): 0.697 AUROC. The non-trivial baseline drift is content or semantic sensitivity.
  - The semantic-versus-random-mask split on TGIF2 (increment +0.068 versus +0.164) exposes semantic confounding.
- **Huang et al.**: DINOv3 relies on global, low-frequency structure — [arXiv 2511.22471](https://arxiv.org/abs/2511.22471). This is a cue that resizing and resampling could affect. No resolution-controlled study was verified.
- **Forensic Twins** removes shared content by using spatially disjoint crops, and avoids augmentations that destroy micro-statistics. This is a design-level control against semantic confounding in SSL — [arXiv 2609.31514](https://arxiv.org/abs/2609.31514)
- **"Unbiased"/aligned training** for image-level detection appears as "careful training data alignment … removing spurious correlations" in the DINOv3 anchor-regularization paper — [arXiv 2608.15196](https://arxiv.org/abs/2608.15196) (snippet only). DDA (dual data alignment) is the VAE-reconstruction-aligned detector that survives DDA-COCO — [arXiv 2602.01738](https://arxiv.org/abs/2602.01738)

### Inferences
- For dense or patch-level use, the main confounds are:
  - (a) Semantic or object response in frozen encoders: boundary adhesion and TGIF2 semantic-mask drift.
  - (b) Format mismatch between the edited region and the rest of the image: regeneration through the VAE decoder, and different JPEG histories inside and outside the mask.
  - (c) Resolution and resampling effects on low-to-mid-frequency cues.
- A rigorous dense evaluation should include:
  - Aligned-real (pre-edit) controls.
  - Random-mask versus semantic-mask splits.
  - Format-matched compositing, so inside and outside the mask share the same JPEG history.
  - VAE-reconstruction-only negatives, where the whole image is re-decoded with no edit.
  - Resolution sweeps.

### Gaps
- No study was found that quantifies JPEG or resolution confounds specifically for patch-level, frozen-feature localization maps.
- Exact debiasing gains in Fake-or-JPEG were not checked in the PDF.

## Cross-cutting synthesis: what is missing for dense / patch-level use

### Takeaway
The paradigm has transferred only partially. Frozen VFM patch tokens contain a training-free edit signal: TRAIL reaches about 0.90 patch AUROC on CocoGlide. Frozen backbones with light decoders dominate supervised IML: DINOv3 + LoRA gives +17.0 pixel-F1 under the CAT-Net protocol. But real-only SSL forensic encoders have not yet been turned into localization maps.

### Cited Findings
- DINOv3 with LoRA and a lightweight conv decoder (9.1M trainable parameters on a frozen ViT-L) improves average pixel-F1 by 17.0 points over prior SoTA on four benchmarks (CAT-Net protocol). Under the data-scarce MVSS-Net protocol it reaches F1 0.774 against 0.530. LoRA beats full fine-tuning, which "becomes highly unstable" — [arXiv 2604.16083](https://arxiv.org/abs/2604.16083), [GitHub](https://github.com/Irennnne/DINOv3-IML)
- FUSED (Aug 2026) is a forensic-semantic mixture-of-experts for inpainting detection and localization. It notes that many AIGI detectors "rely on global artifacts and do not localize" — [arXiv 2608.28302](https://arxiv.org/abs/2608.28302) (snippet only)
- In supervised localization, generalization to unseen editors falls from 62.5 to 41.5 IoU (ICL-Net) — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)

### Inferences
These are the gaps for dense or patch-level use:
1. **No real-only dense localizer.** No paper was found that fits a one-class density model (GMM, kNN, Mahalanobis) per patch on real-only forensic SSL embeddings, then reports pixel-F1/IoU on diffusion inpainting. Candidates: Forensic Twins, EXIF-SSL, ZED per-pixel likelihood, Noiseprint++.
2. **No dense layer-selection study** across modern VFMs (PE, MetaCLIP-2, DINOv3) for patch-level probes. The evidence is split between late layers (DINO, training-free) and intermediate layers (CLIP, supervised).
3. **Resolution.** Training-free maps are coarse (28x28 tokens). No upsampling or refinement that stays training-free was reported.
4. **Calibration.** TRAIL's fixed-threshold Dice is 0.619 against 0.790 with an oracle. One-class scores need per-image or per-dataset calibration that nobody has solved.
5. **Semantic confounding at patch level.** This shows up as boundary adhesion and as semantic-mask drift. Remedies so far either need supervision (GAP-SAM's global token) or are untested for real-only methods (Forensic-Twins-style disjoint crops).
6. **Full-image VAE regeneration.** Real-only acquisition fingerprints may flag the whole image (DiffusionPrint's argument), and VFM probes are blind to VAE-only reconstruction. A dense method must separate "re-decoded everywhere" from "generated here."
7. **Benchmarks lack confound controls.** Missing: aligned-real pairs, format-matched compositing, random-versus-semantic masks, VAE-reconstruction-only negatives.

### Gaps
- Test-time training/adaptation and JEPA/MAE forensic pretraining for localization: no verified sources found.
- FUSED, 2610.07916, 2609.36882, 2608.20558 and 2609.37937 were seen only as search snippets. Their methods and numbers need checking before they are cited quantitatively.
