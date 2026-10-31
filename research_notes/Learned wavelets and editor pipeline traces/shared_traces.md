# Forensic traces left by stages shared across modern generative editors and generators, and their use for localizing edits

Scope note: every claim has a citation. Papers I verified directly in this session (abstract and/or HTML full text fetched) are marked [V]. Papers cited from prior knowledge, with arXiv IDs I am confident of but did not re-open this session, are marked [K]. The report writer should treat [K] numbers as "reported by the paper, not re-checked here". Hypotheses are kept under "Inferences".

---

## Q1. Decoder and tokenizer fingerprints (VAE spectral peaks, 8x/16x upsampling grids, VQ quantisation, AR-image detection, diffusion-decoder artifacts)

### Takeaway
Decoder fingerprints are very well established for image-level detection: periodic spectral peaks from up-convolution and VAE decoders, high-frequency attenuation, and VQ quantisation error in AR models. But in a full regeneration the decoder runs over every pixel, so its fingerprint is spatially uniform. The 2026 evidence (INP-X, GAP-SAM, the processing-matched-controls benchmark) shows that this global trace is the main thing current detectors use, and that on its own it carries almost no localization signal.

### Cited Findings
- **Upsampling periodicity (origin of the grid peaks).** Transposed-convolution and up-sampling layers leave periodic spectral replicas. A detector trained on these spectra generalizes across GANs [K] — [Zhang, Karaman & Chang 2019, arXiv:1907.06515](https://arxiv.org/abs/1907.06515). Up-convolution also fails to reproduce the natural high-frequency power-spectrum decay [K] — [Durall et al., CVPR 2020, arXiv:2003.01826](https://arxiv.org/abs/2003.01826).
- **Latent-diffusion and VAE spectral peaks.** LDM/Stable-Diffusion images show grid-like peaks in the Fourier spectrum of noise residuals. Pixel-space diffusion (e.g., GLIDE/ADM) shows much weaker peaks. Detectors trained on LDM generalize to other LDMs [K] — [Corvi et al., ICASSP 2023, arXiv:2211.00680](https://arxiv.org/abs/2211.00680); [Ricker et al., arXiv:2210.14571](https://arxiv.org/abs/2210.14571).
- **VAE reconstruction shifts the spectrum of the whole image, including unedited pixels [V].** INP-X shows that "VAE-based reconstruction induces a subtle but pervasive spectral shift across the entire image, including unedited regions". It reports grid-like high-frequency peaks under a cross-difference high-pass + FFT analysis. Spectral-difference MSE (×1000) between real and inpainted images falls when the original pixels outside the mask are pasted back (standard → INP-X): OpenImages 17.13 → 1.92, SUN-RGBD 23.42 → 1.94, CityScapes 11.47 → 5.89, CelebA-HQ 15.31 → 12.87 — [INP-X, arXiv:2602.00192](https://arxiv.org/abs/2602.00192) ([HTML](https://arxiv.org/html/2602.00192v1)).
- **High-frequency attenuation theory [V].** Theorem 3.2 of INP-X shows that, for sensor-noise-dominated high frequencies, the reconstruction power spectrum is at most the original's (S_x̃(ω) ≤ S_x(ω)). This assumes MSE-dominant training and that sensor noise is not encoded in the latent. The authors concede (Remark 3.3) that perceptual and adversarial VAE losses break this assumption, and only argue that attenuation persists. For SD v1.4 real images, VAE loss correlates with high-frequency content at r = 0.941 per image. Per pixel, the correlations are VAE↔HF r = 0.52±0.08 and VAE↔inpainting loss r = 0.45±0.23 — [arXiv:2602.00192](https://arxiv.org/html/2602.00192v1).
- **Detectors rely on this global decoder trace [V].** Image-level accuracy falls as follows, standard inpainting → INP-X (decoder trace removed outside the mask):

  | Detector | Standard inpainting | INP-X |
  |---|---|---|
  | Corvi2023 | 0.942 | 0.554 |
  | Hive (commercial) | 0.914 | 0.548 |
  | Sightengine (commercial) | 0.926 | 0.550 |
  | SPAI | 0.661 | 0.542 |
  | DNF | 0.710 | 0.604 |

  The same detectors stay at 0.80–0.89 under JPEG q=80 (Corvi 0.803, Hive 0.885, Sightengine 0.831). So the decoder trace survives JPEG, but it is spatially global — [arXiv:2602.00192](https://arxiv.org/html/2602.00192v1).
- **Diffusion autoencoder passes alone trigger AI-image detectors [V].** A 2026 background-replacement benchmark pairs each edit with authentic controls that went through the same processing (resampling, harmonization, diffusion-autoencoder passes) but had no semantic edit. An AI-image detector "flags most authentic images that passed through a diffusion autoencoder". Seven zero-shot detectors reach AUROC ≤ 0.745. Manipulation localisers highlight the preserved subject rather than the replaced background — [arXiv:2607.26232](https://arxiv.org/abs/2607.26232).
- **DiffusionPrint: the latent decoder destroys camera noise in fully regenerated (FR) images [V].** "Full regeneration pipelines reconstruct the entire image via a latent decoder, disrupting the camera-level noise patterns." The authors instead learn a patch-level contrastive (MoCo) fingerprint of the generator. It works because inpainted regions from the same model share a consistent fingerprint. Used as an extra modality in TruFor or MMFusion, it gives up to +28% localization on unseen mask types — [DiffusionPrint, arXiv:2604.12443](https://arxiv.org/abs/2604.12443).
- **VQ/tokenizer quantisation traces in AR images [V].** D³QE uses the quantisation error and the codebook-frequency distribution bias between real and AR-generated images. It feeds them through a "discrete distribution discrepancy-aware transformer". The ARForensics dataset covers 7 visual AR models. The paper claims strong cross-AR generalization and robustness to perturbations; I did not extract the specific numbers — [D³QE, arXiv:2510.05891](https://arxiv.org/abs/2510.05891). This is image-level detection only; no pixel-level localization.
- **AR tokenizers are lossy at 16× downsampling [V].** Tokenizer papers state that "aggressive downsampling and quantization often discard the fine-grained structures" for text and faces (16k codebook, 16× downsampling) — [InsightTok, arXiv:2605.14333](https://arxiv.org/abs/2605.14333). This supports a 16×16 token grid as the plausible scale for AR traces.
- **The AR detection literature is thin [V].** An arXiv abstract search for "autoregressive" AND "generated image detection" (2025+) returned only 2 papers: D³QE and [arXiv:2608.07863](https://arxiv.org/abs/2608.07863) (low/high/semantic-level vision-language AIGI detection). I found no paper titled "D^3" other than D³QE, and none on token-level localization.

### Inferences
- In a fully regenerated image (GPT-Image, Nano-Banana, FLUX-Kontext, Qwen-Image-Edit), the decoder fingerprint (8× grid for SD/FLUX-type VAEs, likely 16× for AR tokenizers) covers every pixel. Edited and unedited regions differ only where the content does. That is consistent with your SRM / VAE-error / wavelet streams sitting at AUC ≈ 0.50 on GPT-Image-2. (Hypothesis, but strongly supported by the INP-X and processing-matched-control results.)
- A residual difference could exist if the decoder's spectral response depends on content. Freshly synthesized latents are "on-manifold" for the decoder, while re-encoded real content carries encoder-quantisation error. This has not been measured for proprietary pipelines (see Q2).
- GPT-Image's architecture is undisclosed ([arXiv:2504.05979](https://arxiv.org/abs/2504.05979) calls it "mysterious and unpublished"). Whether its decoder is VQ, continuous VAE or a diffusion decoder is unknown, so grid scale and quantisation traces cannot be assumed.

### Gaps
- No paper found that uses decoder or tokenizer spectral periodicity for pixel-level localization in fully regenerated images.
- No published spectral analysis of GPT-Image-1/2 or Nano-Banana outputs (peak positions, grid period) was found.
- No numbers extracted for D³QE accuracy or robustness; the full text was not opened.

---

## Q2. Reconstruction-based asymmetries (AEROBLADE, LaRE², DIRE, ZED, FIRE, B-Free, manifold methods), and do generated vs re-decoded regions differ in reconstruction error?

### Takeaway
Reconstruction error separates generated images from real ones at image level. Re-encoding real content through a VAE, however, produces the same kind of trace as generation. GAP-SAM shows that VAE-reconstruction training data transfers poorly to local inpainting (OOD F1 44.6 vs 61.4). INP-X shows that VAE-loss maps only moderately track inpainted pixels (pixel r ≈ 0.45). Reconstruction error is useful as a global conditioning token (GAP-SAM) or as weak supervision (ReGFLoW), not as a direct per-pixel map. No paper reports it working on GPT-Image or Gemini full regenerations.

### Cited Findings
- **DIRE [K].** Uses the error of DDIM inversion plus reconstruction as a feature; generated images reconstruct better — [Wang et al., ICCV 2023, arXiv:2303.09295](https://arxiv.org/abs/2303.09295).
- **AEROBLADE [K].** Training-free. Uses the LPIPS distance between an image and its LDM-autoencoder reconstruction: generated images have lower error. The paper reports high average precision (≈0.99 mAP over SD/Midjourney generators) and notes the error map is spatial and could aid qualitative localization — [Ricker, Lukovnikov & Fischer, CVPR 2024, arXiv:2401.17879](https://arxiv.org/abs/2401.17879).
- **LaRE² [K].** Uses latent reconstruction error with error-guided feature refinement (CVPR 2024) — [arXiv:2403.17465](https://arxiv.org/abs/2403.17465). (ID from memory; verify.)
- **ZED [K].** Zero-shot detection using the coding cost (surprisal) of a lossless neural model trained only on real images; synthetic images show anomalous coding cost — [Cozzolino et al., ECCV 2024, arXiv:2409.15875](https://arxiv.org/abs/2409.15875).
- **FIRE [K].** Frequency-guided reconstruction error for diffusion-image detection (CVPR 2025) — [arXiv:2412.07140](https://arxiv.org/abs/2412.07140).
- **B-Free [K].** Trains on real images and their SD-inpainted/self-conditioned regenerations to remove semantic bias (CVPR 2025) — [Guillaro et al., arXiv:2412.17671](https://arxiv.org/abs/2412.17671).
- **B-Free and PatchCraft on TGIF2 FR images [V].** B-Free reaches AUC 1.00 on SD2/SDXL fully regenerated images and ≈0.85–0.88 on FLUX.1 subsets. PatchCraft averages 0.95. Only 4 of 11 SID methods stay above 0.8 on all FLUX.1 subsets — [TGIF2, arXiv:2603.28613](https://arxiv.org/html/2603.28613v1).
- **Re-decoded vs generated, direct evidence 1 (GAP-SAM) [V].** Training localizers on "Mask-VAE Reconstruction Alignment" data (real image with a masked region replaced by its VAE reconstruction) generalizes much worse than training on COCO-ControlNet inpainting (F1/IoU):

  | Localizer | Train data | In-domain COCO-CN | OOD average |
  |---|---|---|---|
  | CoDE | Mask-VAE | 41.0 / 25.8 | 44.6 / 29.8 |
  | CoDE | COCO-CN | 92.7 / 86.4 | 61.4 / 46.5 |
  | MaskCLIP | Mask-VAE | 35.6 / 21.7 | 48.7 / 33.6 |
  | MaskCLIP | COCO-CN | 91.3 / 84.1 | 55.2 / 41.7 |

  The authors conclude that "VAE reconstruction artifacts transfer poorly to local diffusion-inpainting artifacts". Note this is inferred from the transfer gap, not from a direct error-map measurement — [GAP-SAM, arXiv:2608.20929](https://arxiv.org/html/2608.20929v1).
- **GAP-SAM uses reconstruction only globally [V].** It encodes the image and its frozen VAE reconstruction into a single global artifact token, injected into SAM3 through zero-gated FiLM, "without prescribing a spatial region". Average Pixel-F1 is 79.8 over 6 datasets (+12.6 over TruFor's 67.2). Per dataset: OpenSDID 90.5, SID-Set 92.7, AutoSplice 78.3, UltraEdit 49.8, DiffSeg30K 69.0. Note that UltraEdit (instruction editing) stays low for all methods (best 49.8). GPT-Image and Gemini are not evaluated. Robustness to JPEG, blur and resize is claimed best at every severity, but numbers appear only in a figure — [arXiv:2608.20929](https://arxiv.org/html/2608.20929v1).
- **Re-decoded vs generated, direct evidence 2 (INP-X) [V].** Pixel-level correlation between the VAE reconstruction-loss map and the inpainting-loss map is only r = 0.45 ± 0.23 (SD v1.4). The VAE loss correlates more with local high-frequency content (r = 0.52). Training on INP-X gives only "modest but significant" localization gains, e.g., EfficientNet mIoU 0.476 → 0.486 and mAP 0.352 → 0.408 — [arXiv:2602.00192](https://arxiv.org/html/2602.00192v1).
- **ReGFLoW: weakly supervised localization from diffusion reconstruction error [V].** Uses diffusion reconstruction errors as dense spatial guidance with image-level labels only, plus artifact-centric multiple-instance learning. The authors observe that "artifacts frequently extend beyond annotated regions, while out-of-mask pixels are treated as authentic". They claim it beats fully supervised baselines when evaluation mixes partially edited and fully synthetic images, and cross-dataset. The abstract gives no numbers — [ReGFLoW, arXiv:2609.36882](https://arxiv.org/abs/2609.36882).
- **TGIF2: FR images break classical localization; fine-tuning recovers but with object bias [V].** On fully regenerated images, zero-shot F1 is TruFor 0.14, MMFusion 0.13, CAT-Net 0.02. After fine-tuning on FR it rises to TruFor 0.75 and MMFusion 0.74 (TruFor only 0.49/0.39 on SD2 semantic/random masks). Fine-tuning only on semantic masks drops TruFor on SD2 random masks to 0.23 vs 0.50 on semantic masks: the model learns objects, not traces — [arXiv:2603.28613](https://arxiv.org/html/2603.28613v1).
- **Diffusion-autoencoder passes defeat reconstruction-based detectors [V].** In the processing-matched-controls benchmark, AI-image detectors flag authentic images that passed through a diffusion autoencoder — [arXiv:2607.26232](https://arxiv.org/abs/2607.26232).

### Inferences
- The TGIF2 recovery of fine-tuned FR localization (F1 ≈ 0.75) happens on open-source inpainting pipelines that keep the known mask region and blend latents. Your AUC ≈ 0.50 suggests GPT-Image-2 behaves more like INP-X-inverted: everything is equally "generated". A small residual of reconstruction asymmetry might remain where content is novel and off-distribution for the encoder. (Hypothesis.)
- Because AEROBLADE-style error tracks high-frequency content (r = 0.94 image-level in INP-X), any reconstruction-error map has to be normalized for local texture. Otherwise it localizes textured regions, not edits. (Inference from INP-X.)
- Manifold or curvature-based zero-shot detectors would probably inherit the same global behaviour. No localization evidence found.

### Gaps
- Not verified this session: "manifold-induced bias" / curvature zero-shot methods (e.g., Brokman et al., ICLR 2025). I could not confirm an arXiv ID, so no claim is made.
- No paper measures reconstruction error inside vs outside the edit on GPT-Image or Gemini outputs.
- ReGFLoW numbers were not extracted; the abstract only gives qualitative claims.

---

## Q3. Local statistical traces of synthesis (variance and energy suppression, noise/PRNU/CFA absence, colour statistics, texture-frequency contrast, seams, alignment shifts)

### Takeaway
Two local traces have direct localization evidence on instruction or MLLM editors:
- **Local high-frequency energy suppression (FLAME's LAD map).** It works on splice-style diffusion edits (IoU ≈ 0.47–0.58) but drops sharply on Nano-Banana (IoU 0.216 / F1 0.295), FLUX-Kontext (0.285 / 0.391) and Qwen-Image-Edit (0.321 / 0.408). It also degrades under JPEG.
- **Colour statistics.** On Qwen-Image-Edit, edited regions show significantly lower colorfulness and brightness than the background.

Absence of camera noise, PRNU and CFA traces is a strong cue only for splices. In fully regenerated images the decoder erases these traces everywhere.

### Cited Findings
- **FLAME / LAD energy-suppression map [V].** LAD(p) is the mean over a hollow 3×3 neighbourhood of tanh(‖I(p)−I(q)‖²/τ²). Edited regions appear as "low-energy voids bounded by spikes at the boundary". The key inequality (authentic local energy > edited) is an assumption (Assumption 3.1), not a theorem; no numeric energy-gap values are given. Localization (IoU / F1):

  | Dataset | IoU | F1 |
  |---|---|---|
  | MagicBrush | 0.538 | 0.650 |
  | SID | 0.580 | 0.677 |
  | CoCoGLIDE | 0.469 | 0.576 |
  | AutoSplice | 0.501 | 0.624 |
  | Nano-Banana | 0.216 | 0.295 |
  | Qwen-Image-Edit | 0.321 | 0.408 |
  | FLUX-Kontext | 0.285 | 0.391 |

  The FLAME-F variant averages 0.460 IoU / 0.565 F1 OOD. Ablation: removing the LAD map (raw RGB) drops IoU 0.567 → 0.294. Robustness: average IoU 0.416 → 0.329 (JPEG q=75), 0.379 (blur), 0.270 (noise); resize not tested. GPT-Image is not evaluated. Listed failure modes: smooth or repetitive authentic regions, small edits, object removal, heavy post-processing — [FLAME, arXiv:2606.02178](https://arxiv.org/html/2606.02178v1).
- **Colour, brightness and spatial-information traces in instruction edits [V].** On Qwen-Image-Edit, "edited regions exhibit significantly reduced colorfulness and brightness" with a slight increase in spatial information (SI). The same paper finds methods using only low-level anomaly signals perform worse than semantic-feature methods. Supervised in-domain localizers do well on its four open editors (Qwen-Image-Edit, FLUX-Kontext, Step1X-Edit, BAGEL; no GPT-Image or Gemini):
  - SegFormer pixel F1: 87.1 / 88.7 / 85.6 / 86.5.
  - DINOv3 pixel F1: 84.9 / 87.2 / 83.0 / 84.1.
  - BAGEL is the hardest editor and FLUX-Kontext the easiest.

  GT masks are SAM3 object masks, so object bias is possible — [LocateEdit-Bench, arXiv:2602.05577](https://arxiv.org/html/2602.05577v1).
- **Texture-frequency contrast [K].** PatchCraft contrasts rich-texture and poor-texture patches; inter-pixel correlation differs in generated images — [Zhong et al., arXiv:2311.12397](https://arxiv.org/abs/2311.12397). It is image-level, but its FR AUC is 0.95 on TGIF2 [V] — [arXiv:2603.28613](https://arxiv.org/html/2603.28613v1).
- **Camera noise absence [V].** DiffusionPrint states that latent decoding disrupts camera-level noise patterns over the whole image in FR inpainting — [arXiv:2604.12443](https://arxiv.org/abs/2604.12443). TGIF2 shows camera-trace localizers collapse on FR (CAT-Net F1 0.02, TruFor 0.14). On splices they work: CAT-Net 0.89, TruFor 0.75 — [arXiv:2603.28613](https://arxiv.org/html/2603.28613v1).
- **CocoGlide [K].** The CocoGlide benchmark comes from TruFor, which fuses RGB with a learned camera-noise fingerprint (Noiseprint++) — [Guillaro et al., CVPR 2023, arXiv:2212.10957](https://arxiv.org/abs/2212.10957). CocoGlide is splice-style: GLIDE output pasted into a real image, so camera traces survive outside the edit.
- **Boundary and seam traces.** FLAME's LAD shows boundary spikes [V] — [arXiv:2606.02178](https://arxiv.org/html/2606.02178v1). Multiple 2026 papers report the opposite problem: localizers snapping to semantic contours ("boundary adhesion") rather than true manipulation boundaries. GAP-SAM gives only qualitative evidence for this [V] — [arXiv:2608.20929](https://arxiv.org/html/2608.20929v1). ReGFLoW notes artifacts extend beyond annotated masks [V] — [arXiv:2609.36882](https://arxiv.org/abs/2609.36882).
- **Alignment shifts in MLLM editors [V].** GPT-4o-class native image generation shows "misalignment between input and output" among its failure modes — [Ramachandran et al., arXiv:2507.01955](https://arxiv.org/abs/2507.01955). Nano-Banana Pro "struggle[s] to maintain the strict pixel-level consistency" in reference-based low-level tasks — [arXiv:2512.15110](https://arxiv.org/abs/2512.15110). PromptForge-350k needed "keypoint alignment and semantic space similarity" just to derive GT masks for prompt-based editors. It reports ICL-Net IoU 62.5% in-domain, 41.5% on unseen editors, and <1% IoU drop under degradations — [PromptForge-350k, arXiv:2603.29386](https://arxiv.org/abs/2603.29386).
- **Agentic editing frameworks show that commercial editors over-edit [V].** Agent Banana reports that editors "over-edit, modifying content beyond the user's intent" and works around it with layer decomposition — [arXiv:2602.09084](https://arxiv.org/abs/2602.09084).

### Inferences
- Of the local traces, colour/brightness shift and local-energy suppression are the only ones with published separation on MLLM-class editors. Both are content-correlated and fragile: FLAME's IoU falls by more than half from splice datasets to Nano-Banana. A 24×24-pooled low-level stream would likely average away LAD's boundary spikes, which are the most discriminative part of its signal. (Hypothesis.)
- Geometric misalignment between source and output is a trace that exists only when the original is available (reference-based). For blind localization it shows up only as GT-mask noise. Your GT masks for GPT-Image-2 may themselves contain misregistration noise that caps achievable AUC. (Inference from PromptForge's need for keypoint alignment.)
- PRNU and CFA absence should be uninformative inside a fully regenerated image, since both regions lack them. They become informative only if the output is composited back onto the original, as in INP-X.

### Gaps
- Not found this session: a verified colour-statistics detector paper (e.g., "Secret lies in color", CVPR 2025) with an arXiv ID; omitted.
- No paper measures local-variance suppression specifically on GPT-Image outputs.
- No quantitative boundary-adhesion numbers exist (GAP-SAM gives only a figure).

---

## Q4. Semantic and physical traces (lighting, shadows, perspective, text) for MLLM edits

### Takeaway
Physics-based checks (shadows, perspective, lighting) are documented as failing in diffusion images and can be localized geometrically. Semantic features (DINO, CLIP, SegFormer) are the strongest signal for instruction-edit localization in recent benchmarks. However, semantic localizers show strong object or subject bias: they highlight the edited object class, or the preserved subject, rather than the edit itself.

### Cited Findings
- **Perspective and shadows [K].** Text-to-image outputs violate perspective and shadow geometry — [Farid, arXiv:2206.14617](https://arxiv.org/abs/2206.14617). "Shadows don't lie and lines can't bend!" shows generated images have projective-geometry inconsistencies (vanishing points, shadow-object correspondences) that a learned classifier can detect and that can be visualized spatially — [Sarkar et al., CVPR 2024, arXiv:2311.17138](https://arxiv.org/abs/2311.17138).
- **Semantic backbones beat low-level ones on instruction edits [V].** On LocateEdit-Bench, SegFormer/DINOv3/Mesorch (semantic) outperform low-level methods. The authors state low-level-only methods perform worse — [arXiv:2602.05577](https://arxiv.org/html/2602.05577v1).
- **Object bias [V].** TGIF2: TruFor fine-tuned on semantic masks drops to F1 0.23 on random masks (vs 0.50 semantic) — [arXiv:2603.28613](https://arxiv.org/html/2603.28613v1). Background-replacement benchmark: "full-background localisation largely reduces to subject segmentation" — [arXiv:2607.26232](https://arxiv.org/abs/2607.26232).
- **MLLM/VLM localizers [V].** A VLM-based detector reports an average IoU of 0.47 over 8 datasets including AIGC editing (search snippet only; not opened) — [arXiv:2603.12930](https://arxiv.org/html/2603.12930). GPT-4o as a zero-shot forgery judge on Gemini/Ideogram document forgeries scores 0.509, "essentially at chance". On the same data TruFor gets AUC 0.751 and DocTamper pixel IoU 0.020 — [AIForge-Doc, arXiv:2602.20569](https://arxiv.org/abs/2602.20569).
- **Explainable MLLM forensics [K].** Examples include FakeShield ([arXiv:2410.02761](https://arxiv.org/abs/2410.02761)) and SIDA ([arXiv:2412.04292](https://arxiv.org/abs/2412.04292)). On GAP-SAM's benchmark SIDA averages 56.2 Pixel-F1 [V], with 67.0 on DiffSeg30K and 36.9 on UltraEdit — [arXiv:2608.20929](https://arxiv.org/html/2608.20929v1).

### Inferences
- Your DINOv2 results (0.83–0.91 on spliced diffusion sets, 0.58 on GPT-Image-2) fit the published picture. Semantic features partly work because splices create contextual and semantic discontinuities and are evaluated on object-shaped masks. GPT-Image-2 harmonizes the whole scene, which removes much of that discontinuity. (Hypothesis consistent with TGIF2 and the background-replacement findings.)
- Physical-consistency cues (shadow direction, vanishing lines, text glyph fidelity) are the only trace class whose mechanism does not depend on the pipeline. No paper was found that applies them to pixel-level localization of GPT-Image or Gemini edits.

### Gaps
- No published numbers on lighting, shadow or perspective localization for GPT-Image or Nano-Banana edits.
- Text-rendering traces: tokenizers are known to lose glyph detail ([arXiv:2605.14333](https://arxiv.org/abs/2605.14333)). No forensic localization use was found.

---

## Q5. Which traces survive in GPT-Image / Nano-Banana full regenerations? Any localization numbers on GPT-Image-1/2 or Gemini edits?

### Takeaway
Published pixel-level localization numbers on MLLM/commercial editors are scarce and low:
- **Nano-Banana:** FLAME IoU 0.216 / F1 0.295.
- **Gemini 2.5 Flash Image** document forgeries: TruFor AUC 0.751, DocTamper IoU 0.020.

I found no peer-reviewed or arXiv paper reporting pixel-level localization on GPT-Image-1 or GPT-Image-2 edits. Your result (low-level ≈ 0.50, DINOv2 0.58) is therefore consistent with, and slightly beyond, the published state of the art.

### Cited Findings
- **Nano-Banana (Gemini 2.5) [V].** FLAME IoU 0.216 / F1 0.295. On the same table: Qwen-Image-Edit 0.321 / 0.408, FLUX-Kontext 0.285 / 0.391 — [arXiv:2606.02178](https://arxiv.org/html/2606.02178v1).
- **Gemini 2.5 Flash Image and Ideogram v2 document forgeries [V].** TruFor AUC 0.751 (vs 0.96 on NIST16). DocTamper AUC 0.563, IoU 0.020. GPT-4o judge 0.509 — [arXiv:2602.20569](https://arxiv.org/abs/2602.20569).
- **Open MLLM-class editors with supervised in-domain training [V].** On LocateEdit-Bench (Qwen-Image-Edit, FLUX-Kontext, Step1X-Edit, BAGEL), pixel F1 reaches 84–89 — [arXiv:2602.05577](https://arxiv.org/html/2602.05577v1). PromptForge-350k: IoU 62.5% in-domain, 41.5% on unseen editors — [arXiv:2603.29386](https://arxiv.org/abs/2603.29386). UltraEdit instruction edits: best Pixel-F1 49.8 (GAP-SAM) — [arXiv:2608.20929](https://arxiv.org/html/2608.20929v1).
- **GPT-Image (ChatGPT Images) [V, snippet only].** A 2026 study on "ChatGPT Images 2.5 on Forgery Tasks" reports that "localisation remains weak for both generations". I did not open it, so its role (forger vs detector) and numbers are unverified — [arXiv:2609.13617](https://arxiv.org/abs/2609.13617).
- **GPT-Image-generated training corpora exist and could serve as data sources [V].** GPT-IMAGE-EDIT-1.5M ([arXiv:2507.21033](https://arxiv.org/abs/2507.21033)), ShareGPT-4o-Image ([arXiv:2506.18095](https://arxiv.org/abs/2506.18095)) and Pico-Banana-400K for Nano-Banana ([arXiv:2510.19808](https://arxiv.org/abs/2510.19808)). None provides forensic masks.
- **Generative super-resolution weakens traces [V].** TGIF2 reports that generative SR (Real-ESRGAN) "significantly weakens forensic traces". The text says IFL drops by 0.51–0.80 on splices vs 0.02–0.11 for bicubic, but the table labels appear swapped — [arXiv:2603.28613](https://arxiv.org/html/2603.28613v1).

### Trace-by-trace evidence map (summary)

| Trace | Produced by | Exists? (evidence) | Robustness | Edited ≠ unedited in full regeneration? | Used for pixel localization? |
|---|---|---|---|---|---|
| Up-conv / VAE spectral grid peaks (8×; likely 16× for AR) | Decoder upsampling | Yes: [1907.06515](https://arxiv.org/abs/1907.06515), [2211.00680](https://arxiv.org/abs/2211.00680), [2602.00192](https://arxiv.org/abs/2602.00192) | Survives JPEG q80 for image-level (Corvi 0.80, Hive 0.89) [2602.00192] | **No**: global; removing it outside the mask drops detectors to ≈0.55 [2602.00192] | Only via INP-X-trained models, small gains (mIoU +0.01) [2602.00192] |
| HF attenuation (VAE bottleneck) | Encoder/decoder bottleneck | Yes, theorem under assumptions; image-level r = 0.94 with VAE loss [2602.00192] | Not measured | Weak; pixel r = 0.45 with inpaint loss [2602.00192] | Indirectly (FLAME's energy idea) |
| VQ quantisation error / codebook bias | AR tokenizer | Yes, image-level D³QE [2510.05891] | Claimed robust | Unknown | No paper found |
| Reconstruction error (AEROBLADE/DIRE/LaRE²/FIRE) | Generated content sits closer to the AE manifold | Yes, image level (AEROBLADE ≈0.99 AP) [2401.17879] | Diffusion-AE passes on real images cause false positives [2607.26232] | Mask-VAE data transfers poorly (OOD F1 44.6 vs 61.4) [2608.20929] | As a global token (GAP-SAM F1 79.8) [2608.20929]; as weak guidance (ReGFLoW) [2609.36882] |
| Learned generator fingerprint | Model-specific | Yes [2604.12443] | n/a | Yes for open inpainting FR (+28% unseen masks) | Yes, DiffusionPrint [2604.12443] |
| Local energy / variance suppression | Diffusion spectral bias | Assumed; ablation IoU 0.567 vs 0.294 [2606.02178] | JPEG q75: IoU 0.416 → 0.329 | Partially: Nano-Banana IoU 0.216 | Yes, FLAME [2606.02178] |
| Colour / brightness shift | Editor generation prior | Yes on Qwen-Image-Edit (qualitative significance) [2602.05577] | Not measured | Yes, reported for an MLLM editor | Not as a stand-alone method |
| Camera noise / PRNU / CFA absence | Non-camera synthesis | Yes | Classic | **No** in FR (CAT-Net F1 0.02) [2603.28613] | Splices only (CAT-Net 0.89) |
| Texture-contrast (PatchCraft) | Generator texture prior | Yes, FR AUC 0.95 [2603.28613] | Not measured | Unknown | No |
| Geometric misalignment | MLLM re-rendering | Yes [2507.01955], [2603.29386] | n/a | Global, with local residuals | Only for GT construction |
| Physics (shadows / perspective) | Lack of a 3-D model | Yes [2311.17138], [2206.14617] | Pipeline-independent | Plausibly yes | Not on MLLM edits |
| Semantic features (DINO/SegFormer) | Content and context | Strong in-domain (F1 85–89) [2602.05577] | Robust | Partly; object bias [2603.28613], [2607.26232] | Yes, most current methods |

### Inferences
- For GPT-Image or Nano-Banana full regeneration, the published evidence says pipeline-level low-level traces are global. Only content-dependent traces remain candidates for localization: colour/brightness shift, local-energy suppression, physical inconsistency, and learned generator fingerprints of fresh vs re-rendered content. All published numbers on these editors are low (IoU ≤ 0.32). (Synthesis.)
- A promising but untested direction is weak or MIL supervision with reconstruction guidance (ReGFLoW) plus random, non-semantic masks to suppress object bias (TGIF2). Your 24×24 pooling may also hide boundary-spike traces (FLAME). (Hypothesis.)

### Gaps
- No pixel-level localization numbers on GPT-Image-1 or GPT-Image-2 were found in any arXiv paper. arXiv:2609.13617 may contain some but was not verified.
- No robustness-to-resize/upscaling numbers for FLAME. GAP-SAM gives robustness numbers only in a figure.
- No study tests whether GPT-Image keeps unedited pixels bit-exact or re-renders them. This is critical for whether any edited-vs-unedited low-level asymmetry can exist.
