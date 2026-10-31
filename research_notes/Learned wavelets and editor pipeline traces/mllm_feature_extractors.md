# MLLMs, unified models and generator internals as frozen feature extractors for generative-edit detection and localization

Scope note: about 17 tool calls (arXiv API plus web). "Abstract-only" means only the arXiv abstract was read. "HTML read" means the arXiv HTML full text was read through an automated extractor, so table cells may be garbled. Items marked [UNVERIFIED-prior] come from background knowledge and were not re-checked in this session. Date: 2026-10-31.

User context, for comparison: frozen DINOv2 L6 patch tokens plus a linear probe give 0.77 per-image AUC, 0.68 cross-dataset, and 0.58 on GPT-Image-2. CLIP-B/16 gives 0.74 / 0.67 and SigLIP2-B gives 0.72 / 0.65.

## Q1. Vision encoders of MLLMs (Qwen-VL ViT, InternViT, SigLIP2-so400m, PE, CLIP-L/336, Molmo, Gemma-3 SigLIP) as forensic features

### Takeaway
Direct evidence exists only for the *standalone* versions of these encoders: PE-CLIP-L/336, SigLIP2, MetaCLIP2, DINOv2/v3. With a linear probe on pooled features, they are strong *whole-image* AIGI detectors (PE 0.90, DINOv3 0.94 avg accuracy in-the-wild). On *localized edits* they sit near chance (0.45 to 0.62), and on VAE reconstructions they are near 0. I found no 2024-2026 paper that benchmarks the ViT pulled out of Qwen2.5/3-VL, InternVL, Molmo or Gemma-3 as a frozen forensic feature, either dense or global.

### Cited Findings
- "Simplicity Prevails" trains a linear head on frozen *pooled output* features. It does not specify CLS versus patch tokens and does no layer-wise analysis. Training data is GenImage SD1.4 with no augmentation. Backbones: MetaCLIP, MetaCLIP2, SigLIP, SigLIP2, PE-CLIP ViT-L/14-336, DINOv2, DINOv3. No MLLM-internal ViTs (Qwen, InternViT) were tested. — [arXiv 2602.01738 (HTML read)](https://arxiv.org/abs/2602.01738)
- Average in-the-wild accuracy over Chameleon, WildRF, SocialRF and CommunityAI: DINOv3 0.940, PE-CLIP 0.899, MetaCLIP2 0.842, SigLIP2 0.822, DINOv2 0.636, SigLIP(v1) 0.610, MetaCLIP(v1) 0.654. The best specialized detector, DDA, gets 0.850. The extractor flagged that individual cells may be misread, so verify against the PDF before quoting single numbers. — [2602.01738 (HTML read)](https://arxiv.org/abs/2602.01738)
- **Localized editing (BR-Gen, columns Brush / Power / SDXL):** DINOv3 0.592/0.613/0.450, PE 0.564/0.581/0.528, MetaCLIP2 0.544/0.575/0.500, SigLIP2 0.597/0.623/0.476. Effort reaches 0.767 to 0.801. The authors blame global pooling, where unedited regions dominate the pooled feature. — [2602.01738 Table 8 (HTML read)](https://arxiv.org/abs/2602.01738)
- **VAE reconstruction (DDA-COCO, SDXL/SD2/SD3.5L):** every VFM scores near 0. Examples: DINOv3 0.030/0.079/0.004 and PE 0.066/0.170/0.024, versus DDA 0.949/0.997/0.682. — [2602.01738 (HTML read)](https://arxiv.org/abs/2602.01738)
- Mechanism claim: VLMs "internalize an explicit semantic concept of forgery", while SSL models "implicitly acquire discriminative forensic features". The authors attribute both to synthetic content in the pretraining data. Modern VLMs (MetaCLIP2, PE) align fakes with the text "AI generated". SigLIP2 does not. All of the models degrade under recapture: PE falls to 54.8% on recaptured RRDataset. — [2602.01738 (abstract + HTML read)](https://arxiv.org/abs/2602.01738)
- Mechanistic probe of a full MLLM pipeline (vision encoder → adapter → LLM): layerwise linear probing for *segmentation* shows the **adapter causes a segmentation "drop-off"**. LLM layers then progressively recover spatial and semantic segmentation through attention between image tokens. Causal attention limits recovery at early image-token positions, and bidirectional attention among image tokens helps. — [arXiv 2603.17228 (abstract-only)](https://arxiv.org/abs/2603.17228)
- CheXagent's ViT and Q-former probed linearly beat task-specific models in chest X-ray. This is a general precedent that MLLM vision towers work as frozen probe features, but it is not forensics. — [arXiv 2407.08811 (abstract-only)](https://arxiv.org/abs/2407.08811)

### Inferences
- 2602.01738 measured the near-chance BR-Gen numbers with *pooled* features and a whole-image label. They therefore say nothing about patch-token probes. Your L6 DINOv2 patch probe (AUC 0.77) is already a different and better-posed regime. The finding mainly warns that CLS/pooled features are useless for partial edits.
- MLLM-internal ViTs (Qwen2.5-VL ViT ~0.67B, InternViT-300M, SigLIP-so400m inside Gemma-3 / PaliGemma) are close relatives of the encoders tested. SigLIP2 and PE results probably transfer roughly, but they are trained further with LLM-alignment objectives, and Qwen's ViT is trained at native resolution with 2D-RoPE and no fixed 224 resize. Native resolution *avoids resize interpolation*, which destroys high-frequency traces, and that is a plausible advantage for dense forensics. Untested.
- 2603.17228 implies that if you tap an MLLM, the *pre-adapter ViT output* or *mid/late LLM layers* are better dense sources than the adapter/merger output. The adapter in Qwen is a 2×2 patch-merger MLP that also *halves spatial resolution*, which is bad for localization.

### Gaps
- No published benchmark of Qwen2.5/3-VL ViT, InternViT, Molmo (CLIP/SigLIP-based), Gemma-3 SigLIP or LLaVA CLIP-L/336 *patch tokens* for edit localization was found.
- 2602.01738 reports accuracy only, with no AUC, so it does not compare directly with your AUCs.

## Q2. Probing LLM hidden states / image-token states of MLLMs for authenticity, and whether they keep spatial layout

### Takeaway
One direct layer-wise probing study exists: Deep-VRM on Qwen2.5-VL-7B. Linear probes on LLM layers show that *semantic* forgery signal converges in early-to-middle layers (~1-16, with a peak zone around 16-20). *Low-level artifact* detection stays flat (~81%) at every depth, which suggests the frozen MLLM suppresses high-frequency evidence. The well-known explainable MLLM forensics systems (FakeShield, SIDA, LEGION, So-Fake, AIGI-Holmes, FakeVLM) are all *fine-tuned*. For localization they use a [SEG] token feeding a SAM-style decoder, or boxes. None of them is a frozen probe on image-token states.

### Cited Findings
- Deep-VRM: layer-wise linear probes on pretrained Qwen-2.5-VL-7B. Semantic-forgery accuracy (D1) rises and converges in early-to-middle layers. The text says 1-16 while the figure caption says a "semantic convergence zone 16-20", so the paper is internally inconsistent here. Artifact-forgery accuracy (D2) is "ineffective throughout", at roughly 81% flat across depth. The authors argue that frozen early LLM layers "suppress high-frequency artifact signals" and that a native MLLM overlooks low-level artifacts. — [arXiv 2606.15880 (HTML read)](https://arxiv.org/abs/2606.15880)
- Deep-VRM's fix: freeze layers 1..K−1, inject a LoRA-adapted copy of the vision encoder (r=64, α=128) as a residual before layer K=16, and fuse with α=β=0.5. Results: GenImage 97.42%, AIGI-Now 95.70%, AIGIBench 90.48%, WildRF 89.51%, avg 93.28%. Detection only, no localization. — [2606.15880 (HTML read)](https://arxiv.org/abs/2606.15880)
- MAD-Guard (Qwen3-VL-8B, LoRA r=16, 2,400 FakeClue samples) compares a binary direct head on the hidden representation with autoregressive generation. The direct head gives 93.10% accuracy / 0.9795 AUC versus AR-SFT 94.90% / 0.9871, at 53 ms versus 386 ms, with better ECE (0.045 vs 0.085). A contrastive head reaches 96.55%. It fails on compressed FF++ faces (AUC 0.5913). So a head on the hidden states matches token generation for closed decisions. — [arXiv 2609.33683 (abstract-only)](https://arxiv.org/abs/2609.33683)
- IFDL-VLM: "priors from VLMs hardly benefit the detection and localization performance and even have negative effects due to their inherent biases toward semantic plausibility rather than authenticity". The authors instead feed localization masks *to* the VLM as priors, and claim SOTA on 9 benchmarks. — [arXiv 2603.12930 (abstract-only)](https://arxiv.org/abs/2603.12930)
- LEGION is an MLLM framework for artifact detection, segmentation and explanation on SynthScars (12,236 fully synthetic images with pixel masks). It beats the second-best expert by +3.31% mIoU and +7.75% F1. — [arXiv 2503.15264 (abstract-only)](https://arxiv.org/abs/2503.15264)
- AIGI-Holmes uses three stages: visual-expert pretraining, SFT, then DPO. At inference, "collaborative decoding" fuses visual-expert perception with MLLM semantic reasoning to improve generalization. Detection and explanation only. — [arXiv 2507.02664 (abstract-only)](https://arxiv.org/abs/2507.02664)
- So-Fake-R1 reports 48.6 IoU on So-Fake-Set and 47.8 IoU OOD. Baselines: SIDA 44.1 and FakeShield 33.7, both fine-tuned on So-Fake. Predicted boxes are converted to masks with SAM2 before scoring, so IoU is not comparable across papers. — [arXiv 2505.18660 (via search snippet; not fully read)](https://arxiv.org/abs/2505.18660)
- FakeShield is an explainable IFDL system with an MLLM and a mask decoder. The search snippet relayed per-dataset IoU, e.g. CASIA1+ 0.54 and Columbia 0.67. — [arXiv 2410.02761 (via secondary snippet; unverified)](https://arxiv.org/abs/2410.02761)
- Architecture details [UNVERIFIED-prior]:
  - SIDA ([2412.04292](https://arxiv.org/abs/2412.04292)) and FakeShield follow LISA: an LLaVA-style MLLM emits a [SEG] token, and its hidden state prompts a SAM decoder that runs on SAM's *own* image encoder. The dense spatial evidence therefore comes from SAM's ViT, not from the LLM's image tokens.
  - FakeVLM ([2503.14905](https://arxiv.org/abs/2503.14905)) is LLaVA-based with full fine-tuning on FakeClue.
  - AIDE ([2406.19435](https://arxiv.org/abs/2406.19435)) is not an MLLM. It combines DCT-selected high/low-frequency patches through SRM and ResNet with ConvNeXt-CLIP semantic features, and introduced the Chameleon dataset.
- Spatial layout in LLM image-token states: 2603.17228 shows segmentation information is linearly decodable from LLM-layer image tokens and *improves* with depth after an adapter drop-off. Causal attention hurts early token positions, i.e. the top-left of the raster. — [2603.17228 (abstract-only)](https://arxiv.org/abs/2603.17228)

### Inferences
- For localization, MLLM LLM-layer image-token states can in principle support patch probes, since segmentation is recoverable. However, (a) Deep-VRM finds artifact information is poor at every LLM depth; (b) Qwen's merger halves token resolution, so a 448² input gives a 16×16 grid; (c) causal attention creates raster-position bias. Expect a semantic "what object looks implausible" map, not a trace map. That is likely worse than your DINOv2-L6 patches on photorealistic edits such as GPT-Image-2.
- The consistent message from IFDL-VLM, Deep-VRM and AIGI-Holmes is that MLLM semantics need a separate low-level expert branch. Frozen MLLM features alone are not the localization signal.

### Gaps
- No paper found that trains a linear/MLP probe on *frozen* MLLM image-token hidden states for *pixel/patch-level* edit localization. A claimed "MLLM internal representations know forgery" paper was not found under that phrasing.
- Deep-VRM does not state which token positions were probed (image tokens vs last token).

## Q3. Unified models (BAGEL, Janus-Pro, Show-o, Emu3) and generator internals (U-Net/DiT features, DIFT, hyperfeatures, VAE latents, tokenizer codes) as dense forensic features

### Takeaway
Evidence is thin. No paper found uses BAGEL, Janus-Pro, Show-o or Emu3 internals as forensic features. Diffusion-model use in localization is mostly *generative*: diffusion as a mask decoder, or a frozen SD3 VAE as a mask decoder. Feature-probing uses are scarce. The best pointer is HIDE (CVPRW 2026), which uses SD1.5 U-Net h-space as frozen features to detect diffusion inpainting.

### Cited Findings
- HIDE uses a frozen SD v1.5 U-Net purely as a feature extractor. It tests three extraction points inside the U-Net, including the h-space bottleneck, to detect diffusion-based inpainting. Its classifier is not described as a linear probe. — [CVPRW 2026 SAFE supplemental (search snippet only; unverified)](https://openaccess.thecvf.com/content/CVPR2026W/SAFE/supplemental/Ji_HIDE_Detecting_Diffusion-Based_CVPRW_2026_supplemental.pdf)
- SDiFL keeps the SD3 VAE frozen and conditions on forgery traces, so the pretrained decoder outputs a binary mask. The generator is used as a *decoder*, not a feature source. — [arXiv 2508.20182 (search snippet only)](https://arxiv.org/abs/2508.20182)
- A score-based IML method uses a denoising U-Net that predicts mask latents conditioned on image features, plus a distilled SD-VAE, and claims better unseen-dataset generalization. — [arXiv 2605.16879 (search snippet only)](https://arxiv.org/abs/2605.16879)
- Other diffusion-as-decoder IFL methods: InpDiffusion ([2501.02816](https://arxiv.org/abs/2501.02816)) and CBDiff ([2510.19597](https://arxiv.org/abs/2510.19597), conditioned on DINO plus Noiseprint++). Both are search snippets only.
- A training-free DDPM method treats a pretrained DDPM as a model of authentic images, so denoising "purifies" tampered regions and the difference localizes them. — [Lacuna mirror (secondary; unverified)](https://lacuna.tiptreesystems.com/work/training-free-image-manipulation-localization-using-diffusion-models/wrk_456a4499bca391d6d0d75d6350a67b66)
- Frozen SD features linearly encode scene geometry, support, shadows and depth, but are weaker for occlusion and material. This is a precedent for probing diffusion features, not forensics. — [arXiv 2310.06836 (search snippet)](https://arxiv.org/abs/2310.06836)
- DinoLizer (DINOv2-B with LoRA on Q/V plus **1 linear layer**, 14×14 patch prediction, sliding window) treats *VAE-reconstructed areas outside the inpaint mask as a separate class*. This "significantly" improves localization, gives about 20% higher IoU than the second-best model, and is robust to double JPEG. A frozen-backbone variant with a linear head (769 params) is also described. — [arXiv 2511.20722 (abstract + search snippet)](https://arxiv.org/abs/2511.20722)
- DIFT ([2306.03881](https://arxiv.org/abs/2306.03881)) and Diffusion Hyperfeatures ([2305.14334](https://arxiv.org/abs/2305.14334)) are dense semantic-correspondence features [UNVERIFIED-prior]. No forensic use of either was found.

### Inferences
- DinoLizer's finding and 2602.01738's "VAE blindness" together suggest that whole-image pipeline editors (GPT-Image-style models that re-encode the *entire* image) leave a global VAE/re-render trace. That trace makes "authentic background" not truly authentic, which may explain your 0.58 on GPT-Image-2. Label design (mask vs. VAE-touched vs. pristine) may matter more than the choice of backbone.
- Generator-internal features are costly on 6 GB (an SD1.5 U-Net is about 860M params, SDXL about 2.6B). Their semantic layers are like DINO's, so they offer no clear advantage over DINO patches. The cheap, promising piece is the *VAE encoder* alone (~34M params for the SD VAE encoder): its latent reconstruction error or latents as per-8×8 features could target re-encoding traces. That is untested in literature I found.

### Gaps
- No evidence was found on BAGEL, Janus-Pro, Show-o or Emu3 (SigLIP/VQ tokenizers, MoT experts) as forensic features, and none on VQ tokenizer codes as forensic features.

## Q4. Forensic knowledge emerging at intermediate layers (Simplicity Prevails, Forensic Reserve, TRAIL)

### Takeaway
The layer evidence is inconsistent across tasks.
- TRAIL (training-free, perturbation-drift maps) finds the best *localization* signal in **late** DINO blocks, at normalized depth 0.80-0.94 across 16 DINO encoders.
- Forensic Reserve finds the most elicitable *detection* signal in **middle** blocks (L11-21 of DINOv3-H+), with the single strongest site at the layer-15 *register* token.
- Simplicity Prevails does no layer analysis and uses pooled final features.
- Your L6 result for a learned linear patch probe is earlier still, and is consistent with low-level traces living early.

### Cited Findings
- TRAIL applies one global Haar perturbation and maps the cosine drift between corresponding patch tokens before and after it. On 80 CocoGlide images: patch AUROC .903 versus .912 for mask-supervised Detective SAM. Dice .619 at fixed threshold versus .709, and .790 with an oracle threshold. On Poisson-interpolation edits with no generator: .855 versus .864. Best block at normalized depth .80-.94 across sixteen DINO encoders. Context matters: AUROC .903 global, .857 local-in-canvas perturbation, .735 for independently encoded crops. Code: github.com/VishalJ99/trail-image-edit-localization. — [arXiv 2608.18968 (abstract-only)](https://arxiv.org/abs/2608.18968)
- Forensic Reserve (RGE) decomposes activations per (block, token group: CLS / register / patch) with FastICA (logcosh) after whitening. Components are ranked by real/fake mean difference and the top 1% are kept. Findings:
  - Middle layers L11-21 hold most of the elicitable reserve, and restricting adaptation to them recovers about 93.5% of the full gain.
  - The layer-15 register site is strongest.
  - Knocking out the top-6 sites costs 3.60 mBA, versus 0.60-1.05 for random sites.
  - Patch-level sensitivity maps peak on generated subjects such as faces and animal heads.
  - Primary backbone is DINOv3-H+ (~840M). Families: DINOv3, DINOv2, CLIP, SigLIP2 (8 backbones).
  - Frozen baseline 91.90 mAP / 68.86 mBA on AIGIBench, versus RGE 97.58 / 93.41 (5K images). With 500 images: 90.98 mAP. Gains: DINOv2-L +11.51 mAP, DINOv2-g +21.29.
  - Detection only, whole-image. — [arXiv 2610.08639 (abstract + HTML read)](https://arxiv.org/abs/2610.08639)
- Simplicity Prevails reports no layer-wise analysis and explains its localized-edit failure by global pooling. — [2602.01738 (HTML read)](https://arxiv.org/abs/2602.01738)
- Deep-VRM (MLLM LLM layers): semantic forgery signal sits in early-to-middle LLM layers, and the artifact signal is flat. — [2606.15880 (HTML read)](https://arxiv.org/abs/2606.15880)

### Inferences
- "Which layer" depends on readout. With *learned* linear patch probes for low-level traces, earlier layers win, as in your L6. With *training-free drift under perturbation*, late layers win (TRAIL), because late layers amplify context inconsistency. Whole-image detection signal concentrates mid-depth and in register/CLS tokens (RGE). A multi-layer concatenation (e.g. L4-6 + L9-12 + a late block) is the evidence-backed hedge.
- TRAIL's crop result (.903 → .735) implies tile-wise inference hurts. Run the full image in one pass where memory allows.
- Register tokens (DINOv2-reg, DINOv3) carry forensic signal per RGE. They are global, but could be used as an image-level gate on the patch map.

### Gaps
- TRAIL has not been tested on GPT-Image/whole-image-regenerating editors (unknown). Neither TRAIL nor RGE was read in full for the CLIP/SigLIP layer curves.

## Q5. Combining MLLM semantic understanding ("what was likely edited") with low-level evidence

### Takeaway
Several systems fuse the two. AIGI-Holmes (collaborative decoding), Deep-VRM (residual injection at L16), ForenAgent (MLLM writes and executes low-level Python tools) and IFDL-VLM (masks as VLM priors) all report gains from the fusion. The recurring failure mode is the MLLM's *bias toward semantic plausibility*: realistic edits look "fine" to it. Benchmarks also show limited localization IoU, around 44-49 on So-Fake.

### Cited Findings
- VLM priors "hardly benefit" IFDL and can hurt, because of semantic-plausibility bias. — [2603.12930 (abstract-only)](https://arxiv.org/abs/2603.12930)
- Native MLLMs overlook low-level artifacts, and fine-tuning for artifacts disrupts semantic representations. That motivates leaving early layers frozen and injecting artifacts at the middle. — [2606.15880 (HTML read)](https://arxiv.org/abs/2606.15880)
- ForenAgent: an MLLM generates, runs and refines Python low-level tools in a multi-round loop (global perception → local focusing → iterative probing → adjudication). Trained with cold-start plus RFT on FABench (100k images, ~200k QA). — [arXiv 2512.16300 (abstract-only)](https://arxiv.org/abs/2512.16300)
- AIGI-Holmes collaborative decoding fuses visual expert and MLLM reasoning. — [2507.02664 (abstract-only)](https://arxiv.org/abs/2507.02664)
- Localization quality ceiling of MLLM IFDL systems on social-media forgeries: So-Fake-R1 48.6 IoU, SIDA 44.1, FakeShield 33.7. — [2505.18660 (snippet)](https://arxiv.org/abs/2505.18660)
- A failure boundary on compressed faces (FF++ AUC 0.59) for a Qwen3-VL-8B head. — [2609.33683 (abstract-only)](https://arxiv.org/abs/2609.33683)
- Hallucination: CXR work found VLMs "hallucinate with confident language". This is general evidence and not forensic-specific. — [2407.08811 (abstract-only)](https://arxiv.org/abs/2407.08811)

### Inferences
- A practical, cheap design for a 6 GB setup: use a small MLLM *offline* to propose candidate object regions ("which objects might be edited"), e.g. via grounding boxes. Use those only as a prior or attention gate over a low-level/DINO patch map, never as the decision. The literature's "semantic plausibility bias" predicts the MLLM will miss photorealistic edits (GPT-Image-2) and over-flag odd but real content (object bias).

### Gaps
- No quantitative hallucination/object-bias rates specific to edit localization were found.

## Practical feasibility on a single 6 GB GPU (estimates)

### Takeaway
Feasible taps on 6 GB in fp16 for feature caching: DINOv2/v3-B/L, CLIP-L/336, SigLIP2-so400m, PE-Core-L, InternViT-300M, and Qwen2.5-VL's ViT run *standalone* (~0.67B). Full MLLM LLM-layer taps need 3B-class models in 4-bit with short image-token sequences. A 7B model in 4-bit (~4.5-5.5 GB weights) is marginal, and the evidence (Deep-VRM) says its LLM layers carry little artifact signal anyway.

### Cited Findings
- Deep-VRM injects at LLM layer 16 of Qwen2.5-VL-7B, so the semantic zone is around layers 16-20 of 28. — [2606.15880 (HTML read)](https://arxiv.org/abs/2606.15880)
- TRAIL's best block is at normalized depth .80-.94 (for a 12-block ViT-B, blocks ~10-11; for 24-block ViT-L, ~19-23). — [2608.18968 (abstract-only)](https://arxiv.org/abs/2608.18968)
- RGE's best sites are DINOv3-H+ L11-21, with the register token at L15. — [2610.08639 (HTML read)](https://arxiv.org/abs/2610.08639)

### Inferences (parameter counts and VRAM figures are [UNVERIFIED-prior] estimates)
- Encoder sizes: ViT-B ~86M, ViT-L ~300M, SigLIP-so400m ~400M, Qwen2.5-VL ViT ~670M, DINOv3-H+ ~840M, DINOv3-7B (too big at fp16, ~14 GB).
- Recommended taps:
  - DINOv3-L/B: early (~25%), mid (~50%) and late (~85%) blocks, concatenated.
  - PE-Core-L: PE papers report that the best dense features are intermediate, not final (unverified).
  - Qwen2.5-VL ViT: last block *before* the 2×2 merger, to keep 14-px resolution.
- 4-bit (bitsandbytes NF4) is fine for *LLM* layers but should be avoided for the vision tower. Quantization noise may wash out the very low-amplitude forensic directions RGE isolates (top-1% ICA components). Keep ViTs in fp16 and cache features to disk.
- Probing recipe consistent with the literature: per-patch logistic/linear (as in DinoLizer's 769-param head), then an MLP. Add a "VAE-touched outside mask" class (DinoLizer). Run full-image single-pass inference (TRAIL context finding), and add TRAIL-style perturbation drift as a free extra channel.
