# Generation-side mechanics of generative editing and the forensic traces they leave or destroy

Scope note: researcher notes, compiled 2026-10-30. arXiv API access timed out during this session, so sources come from web search and fetches of arXiv HTML pages. Items marked **[prior knowledge, not re-fetched]** are well-known papers cited from memory by arXiv ID; check them before final citation. Labels used throughout: **ESTABLISHED** means a primary source shows it experimentally. **HYPOTHESIS** means my mechanistic inference.

---

## Q1. Which editors paste original pixels back outside the mask (SP), and which re-decode or regenerate the whole image (FR)?

### Takeaway
There are two families. In latent inpainting, the default code path VAE-decodes the whole canvas, so the output is FR: every pixel outside the mask is a VAE reconstruction. Paste-back (SP) happens only when the pipeline or UI adds an explicit overlay step (diffusers `apply_overlay` / `padding_mask_crop`, BrushNet blurred blending, Photoshop/Firefly, ComfyUI crop-and-stitch). Instruction editors (FLUX.1 Kontext, Qwen-Image-Edit, Step1X-Edit, BAGEL, GPT-Image, Nano-Banana) generate the whole output image. Their outputs are FR by construction, often with geometric misalignment such as crop, rescale or pixel shift, so the "unedited" region carries no camera traces.

### Cited Findings
- **Latent-masked denoising.** At each reverse step the latent is composited: the generated latent inside the mask, the forward-noised reference latent outside it (FLAME's Eq. 2). This compositing happens in the latent domain, not on pixels, and the final image is a full VAE decode — [FLAME, arXiv 2606.02178](https://arxiv.org/html/2606.02178)
- **diffusers default (ESTABLISHED, from the docs):**
  - Inpaint-specific checkpoints "are more likely to change your unmasked area."
  - Preserving the unmasked area requires an explicit `VaeImageProcessor.apply_overlay()`, which the docs say costs "more unnatural transitions between the masked and unmasked areas."
  - `padding_mask_crop` crops the masked area, upscales it, inpaints it, then overlays it on the original. This is an SP variant with a resampling step.
  - `mask_processor.blur(blur_factor=…)` gives a soft transition.
  - [diffusers inpainting guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint)
- **BrushNet** blends the generated image with the masked original after denoising, using a blurred mask. The reference app uses `cv2.GaussianBlur(mask*255,(21,21),0)/255`. This makes it SP with a roughly 10-px feathered alpha ramp — [BrushNet paper 2403.06976](https://arxiv.org/pdf/2403.06976v1.pdf); [BrushNet HF Space app.py](https://huggingface.co/spaces/TencentARC/BrushNet/blame/main/app.py)
- **TGIF (2024)** contains about 75k forgeries from SD2, SDXL and Adobe Firefly.
  - It separates spliced from fully regenerated images and supplies localization masks for FR.
  - Classical IFL methods localize SP but fail on FR. SID methods detect FR images but cannot localize the edit.
  - Both degrade under stronger compression, and more so under WEBP.
  - [TGIF, arXiv 2407.11566](https://arxiv.org/html/2407.11566v2)
- **TGIF2 (March 2026)** adds FLUX.1 schnell, dev and Fill [dev].
  - FR is defined as "the entire input image is used to condition the diffusion process," with the non-masked area subtly altered. SP is the regenerated region pasted back with the mask.
  - SD2 has SP and FR. SDXL has FR only: splicing "discolors the whole image," i.e. SDXL inpainting shifts global color. Firefly (Photoshop) has SP only, with an automatically added gradient border (feathered paste). FLUX.1 has SP and FR.
  - IFL on SP, average F1: CAT-Net 0.89, TruFor 0.75, MMFusion 0.64.
  - IFL on FR: best about 0.14. After fine-tuning on FR it rises to TruFor 0.75 and MMFusion 0.74.
  - SID: B-Free reaches AUC 1.00 on SD2/SDXL but 0.85–0.88 on FLUX. Only 4 of 11 strong SID methods stay above 0.8 on all FLUX subsets.
  - [TGIF2, arXiv 2603.28613](https://arxiv.org/html/2603.28613v1)
- **DinoLizer** classifies datasets by whether the background is kept.
  - SP (background kept): Beyond the Brush, CocoGlide, TGIF-SP, SAGI-SP, FOSSIL-SP.
  - FR (background auto-encoded): SAGI-FR, FOSSIL-FR, SIDA, MagicBrush.
  - [DinoLizer, arXiv 2511.20722](https://arxiv.org/html/2511.20722)
- **Instruction editors and misalignment.** PromptForge-350k (Nano-Banana, BAGEL, FLUX.1 Kontext, Step1X-Edit) needed ORB+RANSAC affine alignment before deriving masks, because some editors "internally apply pixel cropping to input images."
  - Masks come from DINOv3 cosine similarity with an Otsu threshold. This is chosen to "circumvent pixel-level variances introduced by prompt-based methods in non-edited regions," which makes the unedited regions pixel-different by construction.
  - Leave-one-out OOD IoU: Nano-Banana held out gives 18.95, other editors 36–57.
  - [PromptForge-350k, arXiv 2603.29386](https://arxiv.org/html/2603.29386)
- **Qwen-Image-Edit pixel shift** is documented by the community, not in peer-reviewed work. ComfyUI's reference node silently rescaled inputs, producing about 8 px shifts per edit. Alibaba says the 2511 release targeted the shifts. Community fixes re-warp the output with a homography, or use Inpaint-Crop → edit → Stitch to guarantee unchanged pixels — [lilting.ch write-up](https://lilting.ch/en/articles/qwen-image-edit-pixel-perfect-referencelatent); [ComfyUI Realign node](https://comfy.icu/node/AUSBOSS_NODES_RealignToSource)
- **FLUX.1 Kontext [dev]** advertises "minimal visual drift" across successive edits. That is a model claim, not a forensic measurement — [HF model card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- **Synthetic-data practice.** A 2026 data-augmentation paper notes that latent inpainters push even unmasked pixels through a VAE round trip. It applies a pixel-exact composite to undo this — [arXiv 2606.31603](https://arxiv.org/pdf/2606.31603)
- **Pixel-space inpainters** (e.g. RePaint) avoid VAE reconstruction of the background but take about 9 minutes per 256² image on a P100 — [INP-X, arXiv 2602.00192](https://arxiv.org/html/2602.00192v1)

### Inferences
- **HYPOTHESIS:** Approximate deployment mapping. Not systematically verified per product.

  | Mode | Editors |
  |---|---|
  | SP (with feathering) | Photoshop Generative Fill/Firefly; BrushNet with blending on; diffusers with `apply_overlay`/`padding_mask_crop`; ComfyUI/A1111 "inpaint only masked" + stitch |
  | FR (VAE-reconstructed background) | Raw diffusers SD/SDXL/SD3/FLUX.1 Fill output; PowerPaint default output |
  | FR (fully regenerated, possibly geometrically warped) | Kontext, Qwen-Image-Edit, Step1X-Edit, GPT-Image, Nano-Banana |

- **Pixel classes.** The forensic problem has at least four pixel classes, not two:
  1. Authentic camera pixels (SP background).
  2. VAE-reconstructed authentic content (FR background).
  3. Generated content.
  4. A blend band where the output is α·generated + (1−α)·original, with α from a Gaussian-blurred mask. This band holds a convex mixture of two noise models.

  DinoLizer's 3-class labeling is a first step toward this.
- **Blend band as a signal (HYPOTHESIS, testable).** In SP with feathering, local noise variance across the band should follow σ²(x) ≈ α(x)²σ_gen² + (1−α(x))²σ_cam². With σ_gen < σ_cam (see Q2), it dips below both endpoints mid-band (it is not monotone between them), and the predicted profile has a known Gaussian ramp width (e.g. 21-px kernel in BrushNet). Fitting this profile is a possible boundary detector.
- **Instruction editors.** Any localization method that assumes pixel alignment with an unseen original, or camera-noise continuity in the background, does not apply. Localization must rely on inside-vs-outside differences within one fully synthetic image, for example the generator "copying" conditioning tokens for unedited areas versus synthesizing new content.

### Gaps
- I found no published per-product audit (GPT-Image, Gemini/Nano-Banana, Firefly versions, Canva, Samsung/Google Pixel Magic Editor) of whether background pixels are bit-exact, VAE-reconstructed or regenerated. This could be measured directly with PSNR/bit-exactness on outside-mask regions.
- No quantitative PSNR or SSIM figures for FR background drift per editor (Kontext, Qwen-Image-Edit, Step1X-Edit). PromptForge does not report them.
- The DinoLizer and DiffusionPrint PDFs could not be parsed. DiffusionPrint (arXiv 2604.12443, mever-team) evidently covers SD/SDXL/FLUX/BrushNet/PowerPaint/HD-Painter, but its findings are unverified here.
- Whether PowerPaint's released pipeline pastes back by default was not verified.

---

## Q2. What is known about VAE decoder fingerprints, spectral artifacts, upsampling periodicity, local HF-variance suppression, and reconstruction-error asymmetry, and how do they behave at mask boundaries?

### Takeaway
The best-supported trace is that the VAE bottleneck low-pass filters the image. It attenuates high-frequency sensor noise, so outside-mask regions in FR images carry a "VAE-only" signature and inside-mask regions carry "VAE + generation". Many detectors key on this global VAE trace rather than the local edit. Restoring original background pixels (INP-X) drops commercial and academic detectors from about 92% to about 55% accuracy. Local HF-energy (FLAME) and reconstruction-error methods exploit the same low-entropy property. Boundary behavior is theorized (FLAME predicts a boundary energy spike) but hardly measured.

### Cited Findings
- **INP-X (arXiv 2602.00192), ESTABLISHED.** Restoring outside-mask pixels from the original while keeping the generated region drops:
  - Hive 91.4% → 54.8%
  - Sightengine 92.6% → 55.0%
  - Corvi2023 94.2% → 55.4%
  - DNF 71.0% → 60.4%
  - SPAI 66.1% → 54.2%

  Other results:
  - Theorem 3.2 states that the VAE bottleneck attenuates high-frequency sensor noise, so reconstructed spectral power falls below the original's. Remark 3.3 argues this persists in modern VAEs with hybrid losses at 8× compression.
  - Spectral MSE between real and manipulated images falls sharply with INP-X, e.g. OpenImages 17.13 → 1.92 (×1000).
  - Image-level correlation of VAE loss with inpainting loss is r = 0.600, and of VAE loss with HF content r = 0.941.
  - Soft alpha blending of edges gave similarly low detection, so detectors do not rely on seams.
  - Models tested: Kandinsky 2.2, OpenJourney, SD v1.4.
  - [arXiv 2602.00192](https://arxiv.org/html/2602.00192v1); [code](https://github.com/emirhanbilgic/INP-X)
- **DinoLizer.** Inpainted regions carry generation and VAE artifacts; outside-mask FR regions carry only VAE artifacts. Labeling the latter "real" creates "conflicting supervisory signals," so it trains with Real/VAE/Inpainted classes on a DINOv2 + LoRA backbone.
  - Average IoU about 0.55, F1 about 0.64, at least 20% F1 above other detectors.
  - Robust to double JPEG; sensitive to Gaussian noise.
  - [arXiv 2511.20722](https://arxiv.org/html/2511.20722)
- **FLAME (ICML 2026, arXiv 2606.02178), local HF-variance suppression.** Assumption 3.1 is a "Spectral-Energy Inequality": denoising "tends to suppress high-frequency local variance," and the VAE decoder's Lipschitz continuity "acts as a low-pass filter." Real images carry high-entropy shot/thermal noise.
  - LAD map: per-pixel mean of tanh(‖I(p)−I(q)‖²/τ²) over a hollow 3×3 neighborhood.
  - Theorem 3.3 predicts an interior energy gap plus a **boundary spike**. The proof is in the appendix, not verified here.
  - Small 3×3 windows work best (IoU 0.567 vs 0.439–0.502 for other operators); larger windows degrade.
  - OOD average IoU 0.358 / F1 0.459 vs SAFIRE 0.221 / 0.273. On FLUX Kontext, IoU 0.285.
  - Under JPEG-75, IoU falls 0.416 → 0.329. Under Gaussian noise (var 5), 0.270, which suggests the signal is a noise-level cue and can be laundered by adding noise.
  - The paper has internal table inconsistencies: Table 1 vs Table 3 averages, and the Table 2 ACC mean.
  - [FLAME HTML](https://arxiv.org/html/2606.02178); [ICML poster](https://icml.cc/virtual/2026/poster/66704)
- **Decoder-reconstruction training.** Rajan et al. (ICLR 2025) train detectors on real images vs. their LDM-autoencoder reconstructions with no denoising, which forces focus on decoder artifacts at about 10× lower cost. Vesnin et al. (arXiv 2411.06441) do the same — [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/9ead108421b202494d01b5060d12aa34-Paper-Conference.pdf); [arXiv 2411.06441](https://arxiv.org/html/2411.06441v1)
- **B-Free (CVPR 2025).** Fakes are self-conditioned SD2.1 reconstructions of real COCO images, removing content and format bias.
  - "inpainted++" augmentation: local object replacement plus blur, JPEG, scaling, cut-out, noise and jitter; a background-restoration variant also exists.
  - Over 80% balanced accuracy in the hardest online-sharing setting.
  - [CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Guillaro_A_Bias-Free_Training_Paradigm_for_More_General_AI-generated_Image_Detection_CVPR_2025_paper.pdf); [project](https://grip-unina.github.io/B-Free)
  - In TGIF2 it reaches 1.00 AUC on SD2/SDXL but 0.85–0.88 on FLUX, consistent with FLUX's 16-channel VAE leaving a different or weaker decoder trace — [TGIF2](https://arxiv.org/html/2603.28613v1)
- **AEROBLADE** (CVPR 2024, arXiv 2401.17879) is training-free. It uses LPIPS reconstruction error through the LDM autoencoder; generated images reconstruct with lower error — [arXiv listing](https://arxiv.deeppaper.ai/papers/2401.17879v2/similar)
- **Reconstruction-error relatives** [prior knowledge, not re-fetched]:
  - DIRE (arXiv 2303.09295): DDIM inversion/reconstruction error.
  - LaRE² (CVPR 2024, arXiv 2403.17465): latent reconstruction error, guided refinement.
  - ZED (ECCV 2024, arXiv 2409.15875): zero-shot coding-cost/surprisal of a lossless model trained on real images only.
  - Corvi et al. (ICASSP 2023, arXiv 2211.00680): spectral peaks from upsampling in LDM/GAN outputs.
  - "Fake or JPEG?" (arXiv 2403.17608): dataset bias in DIRE-like detectors.

### Inferences
- **Reconstruction-error asymmetry (HYPOTHESIS, strongly motivated).** Re-encode/decode a suspect image with the candidate VAE:
  - Pixels that already lie on the decoder manifold (generated or VAE-reconstructed) have near-zero error.
  - Camera pixels have higher error.

  So in SP images the per-pixel AEROBLADE map should localize the edit. In FR images it separates nothing between inside and outside, because both are on the manifold, unless a second-order cue is used: the inside was generated by the denoiser at full manifold density, while the outside is an encode-decode of camera content. A *double*-round-trip error ratio (error after one more cycle vs. the first) might separate them, by analogy to double-JPEG quantization idempotence. Untested as far as I found.
- **Latent-grid alignment (HYPOTHESIS).** VAE encoders downsample by 8. Generated content is synthesized on an 8×8 latent grid, while the mask boundary is snapped to that grid by latent-mask downsampling (diffusers nearest/area interpolation of the mask). Expect:
  1. Edit boundaries in the latent-blended region quantized to 8-px blocks before any pixel feathering.
  2. Decoder upsampling periodicity (period 8, spectral peaks at multiples of 1/8 cycles/px), present over the whole image in FR and only inside the mask in SP.

  A blockwise 8×8 periodicity map could localize in SP. In FR it only confirms "fully decoded."
- **FLAME's predicted boundary spike** is a joint-texture discontinuity. If the editor uses Gaussian feathering, the spike should be smeared over the blur width. LAD with 3×3 windows would then show a "valley" rather than a spike (see the Q1 blend-band variance model).
- **SDXL "discoloration."** TGIF2 found SDXL splicing discolors the whole image. FR SDXL outputs therefore carry a global color/tone shift, which is an image-level cue but not a localization cue.
- The DiT/MM-DiT editors (FLUX.1, SD3, Qwen-Image) use 16-channel VAEs with higher reconstruction PSNR than SD1/2's 4-channel VAE. The VAE trace is weaker but the low-pass/denoising energy gap remains. This is consistent with lower SID AUC on FLUX in TGIF2. HYPOTHESIS for the causal link.

### Gaps
- No paper found that measures per-pixel VAE-trace behavior across a feathered mask boundary, or the width and shape of the boundary transition, in real editor outputs.
- No direct measurements found of VAE decoder fingerprints for the FLUX/SD3/Qwen-Image VAEs vs SD-VAE, such as model attribution from decoder traces alone.
- I did not verify any study of upsampling periodicity specific to DiT patchify/unpatchify (2×2 patches over the 8× VAE, giving a 16-px effective period). This is a plausible new signal.
- MLLM/autoregressive editors (GPT-Image, Nano-Banana): no public architecture details on whether outputs pass through a VAE, a VQ tokenizer or a diffusion decoder, so their decoder traces cannot be modeled from documentation.

---

## Q3. Anti-forensics and laundering: what has been shown to defeat current detectors?

### Takeaway
The simplest effective "attack" is benign. Pasting back original pixels (SP) removes the global VAE trace that most AI-image detectors rely on (INP-X: about 92% → 55%). The converse, a whole-image VAE/diffusion round trip, turns SP into FR and kills classical splicing localizers (TGIF/TGIF2: F1 from about 0.89 to about 0.14 or lower without FR fine-tuning). Generative super-resolution degrades but does not destroy SP localization in TGIF2. Noise addition hurts the newest energy-based localizers most.

### Cited Findings
- **SP → FR conversion (any whole-image VAE round trip).** Classical IFL drops from F1 0.64–0.89 (SP) to at most about 0.14 (FR) unless retrained on FR data — [TGIF2](https://arxiv.org/html/2603.28613v1); [TGIF](https://arxiv.org/html/2407.11566v2)
- **Background restoration (FR → SP)** defeats SID detectors (INP-X numbers in Q2) — [arXiv 2602.00192](https://arxiv.org/html/2602.00192v1)
- **Generative super-resolution (TGIF2).** Real-ESRGAN ×2/×4 then downsampled to the original size; bicubic as baseline.
  - Table 9 (SP IFL F1): CAT-Net 0.87 → 0.76/0.73, TruFor 0.78 → 0.76, MMFusion 0.69 → 0.65/0.64.
  - The text reverses the roles, saying Real-ESRGAN drops F1 by 0.51–0.80 and bicubic only by 0.02–0.11. **Text and tables conflict; check the PDF.**
  - [TGIF2](https://arxiv.org/html/2603.28613v1)
- **Compression.** TGIF reports both IFL and SID degrade with stronger compression, and WEBP is especially damaging — [TGIF](https://arxiv.org/html/2407.11566v2)
- **FLAME** drops to IoU 0.270 under Gaussian noise (var 5) and 0.329 under JPEG-75 — [FLAME](https://arxiv.org/html/2606.02178)
- **DinoLizer** is stable under double JPEG but sensitive to Gaussian noise — [DinoLizer](https://arxiv.org/html/2511.20722)
- **PromptForge ICL-Net** is reported robust to JPEG-80 plus 20% crop (F1 −0.58 points) — [PromptForge](https://arxiv.org/html/2603.29386)
- **Adversarial attacks on localizers.** Anti-forensic datasets built with six adversarial attack algorithms against IFL models, plus an inference-time noise-suppression defense — [arXiv 2506.12871](https://arxiv.org/html/2506.12871)
- **Diffusion-based counter-forensics** (diffusion purification as a laundering step) — [arXiv 2311.13629](https://arxiv.org/pdf/2311.13629) (method details not verified in this session)
- **Regeneration attacks on watermarks** (noise and denoise with a diffusion model or VAE) provably remove bounded-L2 pixel watermarks — [Zhao et al., NeurIPS 2024, arXiv 2306.01953](https://arxiv.org/abs/2306.01953). CtrlRegen regenerates from clean noise — [arXiv 2410.05470](https://www.alphaxiv.org/abs/2410.05470)

### Inferences
- **HYPOTHESIS:** Because SP↔FR conversion is cheap and defeats each detector family in turn, a robust localizer should combine both cues:
  - (a) a camera-trace consistency branch for SP;
  - (b) a generation-vs-reconstruction branch (DinoLizer-style 3 classes) for FR.

  Training should include both conversions as augmentation.
- **HYPOTHESIS:** Noise-level cues (FLAME LAD, PRNU/Noiseprint-like) are fragile to additive noise, because an adversary can re-inject synthetic sensor-like noise matched to σ_cam. Cues based on *structure* of the noise (spatial correlation, demosaicing/CFA periodicity, JPEG grid alignment) are harder to forge but are themselves destroyed in FR.
- **HYPOTHESIS:** A full-image diffusion purification at low strength (SDEdit, t≈0.1–0.3) would convert any SP forgery into a near-FR image. It should defeat SP localizers and leave only low-entropy/energy gaps. Untested in the papers found.

### Gaps
- No paper found that systematically attacks *generative-inpainting localizers* (TruFor, DinoLizer, FLAME, SAFIRE) with VAE round trips of *different* VAEs (e.g. laundering FLUX edits through the SD-VAE) or with diffusion purification.
- The TGIF2 SR text/table conflict is unresolved.
- No public evidence found on how social-media pipelines (resizing plus JPEG) affect the SP/FR distinction for real-world GPT-Image or Nano-Banana edits.

---

## Q4. Semantic and physical consistency cues editors still violate (lighting, shadows, reflections, perspective, text)

### Takeaway
There is solid prior evidence (2022–2024) that diffusion models do not respect projective geometry, shadows or lighting, and that these inconsistencies can be detected. I found little 2025–2026 forensic work that applies these cues specifically to *local* edits by modern DiT/MLLM editors. This looks like a genuine gap.

### Cited Findings
- **Projective geometry.** Generated images have "distinctive geometric features" from imperfect projective geometry: inconsistent vanishing points and object–shadow relations that reliably differ from real images. Detectors trained on these cues work — [arXiv 2311.17138 ("Shadows Don't Lie and Lines Can't Bend!")](https://ar5iv.labs.arxiv.org/html/2311.17138)
- **Farid, paint-by-text (2022).** DALL-E-2 outputs show perspective inconsistencies and lighting inconsistencies — [Perspective (In)consistency of Paint by Text, arXiv 2206.14617](https://bohrium.dp.tech/paper/arxiv/2206.14617); [Lighting (In)consistency of Paint by Text, arXiv 2207.13744](https://arxiv.org/abs/2207.13744v2)
- **Classical shadow-consistency test.** Linear-programming feasibility over shadow constraints; infeasibility is evidence of tampering — [Kee, O'Brien, Farid](https://history.siggraph.org/?p=108529)
- **MLLM-agent forensics** (ForenAgent/FABench) uses GPT-4o, FLUX-Kontext, Nano-Banana and Qwen-Image forgeries. It reports that Nano-Banana and Qwen-Image edits are the most photorealistic and hard to discriminate — [arXiv 2512.16300](https://arxiv.org/html/2512.16300v1)

### Inferences
- **HYPOTHESIS:** Local edits are *more* vulnerable to physical-consistency checks than full generations. The inserted object must agree with lighting, shadows and perspective that are *fixed* by authentic (or VAE-reconstructed) surroundings, while the editor only sees the context through attention. Object *removal* leaves no shadow-caster but can leave orphan shadows, reflections or contact occlusions. Object *addition* often lacks cast shadows or has a mismatched shadow direction. These are also mask-independent and survive SP↔FR laundering and noise, which makes them complementary to low-level cues.
- **HYPOTHESIS:** Instruction editors that regenerate the whole frame (Kontext, Nano-Banana) can re-light the whole scene consistently. Physical cues may therefore be weaker for them than for masked inpainters that are constrained by a fixed background.

### Gaps
- No 2025–2026 paper found that quantifies shadow, lighting or perspective violations for FLUX.1 Fill/Kontext, Qwen-Image-Edit, GPT-Image or Nano-Banana edits specifically.
- Text rendering: modern editors (Qwen-Image, GPT-Image) are strong at text, and I found no forensic evaluation of text-edit artifacts (font or kerning inconsistency with surrounding text).
- Reflection-consistency forensics for generative edits was not found in this search.

---

## Q5. Provenance alternatives (C2PA, SynthID, Stable Signature, Tree-Ring, EditGuard, OmniGuard): localization capability and weaknesses

### Takeaway
Only *fragile/semi-fragile localized* watermarks (EditGuard, OmniGuard, and newer schemes) give pixel-level tamper maps, and they need the image to be watermarked *before* editing. Generator-side watermarks (Stable Signature, Tree-Ring, SynthID) label generated content but are not designed for localizing partial edits. Pixel-level invisible watermarks are provably removable by diffusion regeneration. C2PA metadata gives edit history, not pixel localization, and is stripped by re-encoding. Status: established for the watermark-removal results; partly inferred for C2PA.

### Cited Findings
- **EditGuard (CVPR 2024)** jointly embeds a copyright bit-string and a localization template ("united image-bit steganography"), so it can localize tampering from both classical and AIGC edits without training on the edit type — [arXiv 2312.08883](https://www.alphaxiv.org/abs/2312.08883)
- **OmniGuard (CVPR 2025)** uses a natural-image template instead of a solid-color one, a degradation-aware tamper extractor, and an AIGC-editing simulation layer so the copyright watermark survives global and local AI edits — [arXiv 2412.01615](https://arxiv.org/html/2412.01615); [CVPR PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_OmniGuard_Hybrid_Manipulation_Localization_via_Augmented_Versatile_Deep_Image_Watermarking_CVPR_2025_paper.pdf)
- **Competing claim (2026):** EditGuard's localization collapses under full-image semantic editing (F1 about 34%, ownership accuracy about 0.08%). This comes from a rival method's authors, so treat it with caution — [arXiv 2601.06639](https://arxiv.org/pdf/2601.06639)
- **Mask Image Watermarking** is a related localized-watermark direction — [arXiv 2504.12739](https://arxiv.org/pdf/2504.12739) (not examined in depth)
- **Regeneration attacks** provably remove pixel-level invisible watermarks (bounded L2 perturbation) while preserving quality. The authors suggest semantic watermarks instead — [arXiv 2306.01953](https://arxiv.org/abs/2306.01953); [code](https://github.com/XuandongZhao/WatermarkAttacker)
- **Generator-side watermarks** [prior knowledge, not re-fetched]:
  - Stable Signature (arXiv 2303.15435) fine-tunes the LDM *decoder* so every decoded image carries a watermark.
  - Tree-Ring (arXiv 2305.20030) embeds a Fourier-space pattern in the initial noise and detects it via DDIM inversion. It is a semantic watermark, more robust to regeneration, but image-level only.

### Inferences
- **Stable Signature in FR vs SP (HYPOTHESIS, interesting for localization).** The watermark is in the decoder, so in FR output *every* pixel (including the background) carries it. In SP output only the inpainted region does. A per-patch Stable-Signature/decoder-watermark detector would then be a localizer for SP edits made with watermarked decoders, and a whole-image flag for FR.
- **Tree-Ring** cannot localize: it is tied to the initial noise of the whole latent and needs DDIM inversion of the full image.
- **C2PA** manifests record "AI edit" actions (e.g. Photoshop Generative Fill) but not pixel masks in general, and they are lost under screenshot or re-encode. HYPOTHESIS/background knowledge; no source fetched this session.

### Gaps
- No primary source fetched on SynthID's localization ability or its robustness to inpainting. Google's public material describes image-level detection; not verified here.
- No study found on whether OmniGuard/EditGuard tamper maps survive SP→FR conversion (whole-image VAE round trip) or generative SR. Given the regeneration-attack theory, one would expect the localization template to be destroyed (HYPOTHESIS).
- C2PA: I found no source this session documenting mask-level ("region of interest") assertions in deployed manifests, though the C2PA spec may support regions. Needs checking.
