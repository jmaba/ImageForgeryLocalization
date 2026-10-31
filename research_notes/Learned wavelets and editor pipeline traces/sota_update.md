# SOTA update (June–Oct 2026): pixel-level localization of instruction/MLLM/autoregressive edits and fully regenerated inpainting

_Compiled 2026-10-31. This is a delta on top of `Generative forgery localization gaps/localization_methods.md` and `novelty_check.md`. Methods already covered there (FLAME, GAP-SAM, ICL-Net/PromptForge, LocateEdit-Bench, PANet/Impostor, ForenScope/GIFTBench, DiffusionPrint, ReGFLoW, TRAIL, DeFakerOne, ForensicsTok, APT, PAL/EditGroup, TGIF2, INP-X, DinoLizer, EditSleuth, SIGMA) are only cross-referenced here, not re-described._

_Reading-depth tags: **[full]** = full text or a structured full-paper report was read (alphaXiv AI report or extracted full text; table values copied from that report, not re-checked against the PDF); **[abs]** = abstract only; **[search]** = web-search snippet only._

_Search budget note: about 16 tool calls (arXiv API, alphaXiv semantic search, one web search). Coverage of ACM MM 2026 / NeurIPS 2026 camera-ready versions not on arXiv is NOT guaranteed._

---

## Q1. What is the best reported pixel-level performance on GPT-Image-1/2 and Gemini/Nano-Banana edits and on PIXAR / LocateEdit / PromptForge / TGIF2-FR? Under which splits?

### Takeaway
The first pixel-level numbers on **GPT-Image** edits now exist. They come from the PIXAR line of work (MBZUAI) and are very low. PIXAR-7B, trained on Qwen-Image edits only, reaches **IoU 11.7% on GPT-Image-1.5** (worst of six test editors) and 26.3% on Qwen-Image (in-domain). PIXAR-DG raises the average OOD gIoU over {GPT-Image-2.0, Gemini-3.1, FLUX.2, Seedream 4.5} to only **0.222** (13B). Measured against pixel-difference ground truth, no published method localizes closed-source MLLM edits above roughly 0.25 IoU zero-shot. This matches the earlier Nano-Banana numbers (ICL-Net 18.95, FLAME 21.6).

### Cited Findings

**PIXAR benchmark + PIXAR model (CVPR 2026 Findings per web snippet)** — [arXiv 2603.20193](https://arxiv.org/abs/2603.20193) [full]; code/data at github.com/VILA-Lab/PIXAR — [search](https://arxiv.org/abs/2603.20193)
- Ground truth is **per-pixel difference maps**, not object masks. D(x,y) = |I_orig − I_gen| after ORB+RANSAC homography alignment, binarized at τ (default 0.05); raw difference maps are also released. Filters: overlap with the instruction mask ≥0.2, and a spatial-concentration filter that removes speckle — [2603.20193](https://arxiv.org/abs/2603.20193)
- Scale and splits:
  - Over 420K image pairs, COCO sources, 8 edit types (intra- and inter-class replace, remove, add, color, motion, material, background), plus multi-object sequential edits, binned by size.
  - **Training uses Qwen-Image edits only.**
  - **Test** editors are Flux.2, Gemini 2.5, Gemini 3, GPT-image-1.5, Qwen-Image and Seedream 4.5.
  - Fidelity filtering: Qwen3 VLM score ≥9, then 10 human experts (realism ≥4/5).
  - [2603.20193](https://arxiv.org/abs/2603.20193)
- Results (PIXAR test set, as reported):
  - PIXAR-7B: IoU 18.1% overall, vs SIDA-7B 6.9% and FakeShield 9.3%. PIXAR-7B-Lite (pixel-label supervision only, data subset) reaches 14.9%.
  - Per editor: **GPT-Image-1.5 hardest at 11.7% IoU**; Qwen-Image easiest at 26.3%, a case the authors note may be in-domain.
  - Binary F1: 69.7 vs SIDA-7B 25.1.
  - [2603.20193](https://arxiv.org/abs/2603.20193)
- Human study (10 participants, 1,000 images): F1 31.0% for detection and **IoU 10.7%** for localization. Humans are roughly at GPT-Image-level model performance — [2603.20193](https://arxiv.org/abs/2603.20193)
- Lower τ for training labels is better. Raising the training τ (evaluation fixed at 0.05) consistently hurts localization, so micro-edits carry useful supervision — [2603.20193](https://arxiv.org/abs/2603.20193)

**PIXAR-DG: "Simple Domain Generalization for Strong Pixel-Level Image Tampering Detection in Modern VLMs" (Jul 2026)** — [arXiv 2607.18230](https://arxiv.org/abs/2607.18230) [full]
- Architecture: the PIXAR detector, a LISA/SIDA-style setup with a frozen CLIP vision tower, a frozen SAM image encoder, a LoRA-tuned LLM, and a fine-tuned SAM mask decoder plus heads. Outputs are a pixel map, a semantic multi-label vector and text — [2607.18230](https://arxiv.org/abs/2607.18230)
- Training split: 70K Qwen-Image tampered images as base plus **3K Gemini-2.5** as a "late-injected" new domain, 73K in total (19.2% of PIXAR's 380K). **OOD test editors: GPT-Image-2.0, Gemini-3.1, FLUX.2, Seedream 4.5** — [2607.18230](https://arxiv.org/abs/2607.18230)
- Recipe: (i) balanced real:tampered minibatches at 1:1; (ii) late injection of the small new-domain set after base convergence; (iii) a lower constant learning rate (1e-4 → 2e-5). No architectural change — [2607.18230](https://arxiv.org/abs/2607.18230)
- OOD results:
  - 13B: average gIoU 0.176 → **0.222** and cIoU 0.183 → **0.232** vs PIXAR-13B; Pixel-F1 +6.47.
  - 7B: Pixel-F1 +4.80 and Pixel Recall +19.34 vs PIXAR-7B; FLUX.2 gIoU 0.170 → 0.222.
  - Binary OOD accuracy: 79.6% (7B) and 84.0% (13B).
  - [2607.18230](https://arxiv.org/abs/2607.18230)
- Source-selection finding: with a Qwen-Image-trained detector, **Gemini-2.5 is the worst-case OOD target** (41.2% accuracy). Adding Gemini-2.5 as the companion source gives the best two-source OOD accuracy (66.3%). Under the same Qwen+Gemini data, PIXAR-DG beats PIXAR at 29.89 vs 28.59 Pixel-F1, which is a small localization gap — [2607.18230](https://arxiv.org/abs/2607.18230)
- Note: the per-editor GPT-Image-2.0 gIoU value was not visible in the report I read. Only averages and the FLUX.2 gain are verified.

**ChatGPT Images 2.5 / GPT-Image-2 forgery study (Sep 2026)** — [arXiv 2609.13617](https://arxiv.org/abs/2609.13617) [abs]
- Tests GPT-Image-2 (two tiers) and ChatGPT Images 2.5 on receipt-field alteration, repeated editing, product placement and fine print. The abstract states "**localisation remains weak for both generations**". A detector that flags 68.6% of controlled benchmark images flags only **35.9% of images posted online**. No localization numbers appear in the abstract — [2609.13617](https://arxiv.org/abs/2609.13617)
- Relevant to the "unedited pixels" question: after alignment, OCR detects changes to *surrounding* (non-target) text in 31.7% of Images 2.5 outputs vs 44.2% for the cheaper GPT-Image-2 baseline. GPT-Image editors therefore frequently alter pixels outside the intended region — [2609.13617](https://arxiv.org/abs/2609.13617)

**Nano-Banana / Gemini (update):** no new zero-shot Nano-Banana number beyond ICL-Net (IoU 18.95, leave-one-out) and FLAME (0.216/0.295) from the prior notes was found. The new Gemini evidence is PIXAR (Gemini 2.5 and Gemini 3 as test editors; per-editor IoU not extracted) and PIXAR-DG (Gemini-3.1 OOD, averaged only) — [2603.20193](https://arxiv.org/abs/2603.20193); [2607.18230](https://arxiv.org/abs/2607.18230)

**LocateEdit-Bench / PromptForge / TGIF2-FR:** no new 2026 method reporting on these benchmarks surfaced beyond what the prior notes list (LocateEdit in-domain SegFormer 87.0 F1; ICL-Net PromptForge leave-one-out average 41.5 IoU; TGIF2-FR TruFor 0.14 off-the-shelf, 0.75 fine-tuned; DiffusionPrint FR F1 up to 85). The web search confirmed the benchmark identities: PromptForge-350k has 354,258 pairs, and Nano-Banana applies internal pixel cropping that complicates alignment — [search: 2603.29386](https://arxiv.org/pdf/2603.29386); [2602.05577](https://arxiv.org/html/2602.05577v1)

**Related benchmark with GPT-Image + Nano-Banana (box-level only):** ManipBench (Nov 2025) covers 25 editors including Nano-Banana and GPT-Image, but its localization labels are **bounding boxes**, not masks — [search: 2511.14259](https://arxiv.org/pdf/2511.14259v1)

**FragFake (2025, earlier work missing from notes):** an automated benchmark of AI-edited images. Pretrained VLMs including GPT-4o "perform poorly" at edited-region localization, while fine-tuned Qwen2.5-VL is much better. GRPO/RLVR gives "modest metric gains", and cross-editor and cross-dataset generalization is limited (abs) — [arXiv 2505.15644](https://arxiv.org/abs/2505.15644) [abs]

### Inferences
- The hierarchy of hardness is now consistent across three independent groups: GPT-Image (11.7 IoU, PIXAR) < Nano-Banana (≈19–22 IoU, ICL-Net/FLAME) < open DiT editors such as Kontext and Step1X (≈45–57 IoU held out). Closed MLLM editors are the frontier.
- PIXAR changes the label definition from object masks to pixel-difference maps. Numbers on PIXAR are therefore **not comparable** with LocateEdit (ChatGPT+SAM3 object masks) or PromptForge (DINOv3-difference + Otsu). A method can look much worse on PIXAR simply because the difference-map GT includes off-mask spillover.
- PIXAR-DG's gains come from the training recipe, not from new evidence, and the localization gain under matched data is small (+1.3 Pixel-F1). This supports the prior-notes inference that MLLM localizers change the decoder, not the forensic signal.

### Gaps
- No per-editor table values for GPT-Image-2.0 or Gemini-3.1 from PIXAR-DG were extracted (averages only).
- No paper found that reports pixel-level localization on GPT-Image-1 (original) specifically, or on Nano-Banana Pro / Nano-Banana-2 beyond the Impostor dataset (which includes Nano Banana Pro edits; see prior notes).
- No 2026 TGIF2-FR or LocateEdit cross-editor numbers from new methods were found in this search.

---

## Q2. New methods (June–Oct 2026) by signal family, with best cross-editor numbers

### Takeaway
New entries since the prior notes are FUSED (token-wise forensic+semantic MoE), SegWave (learned wavelet sub-band attention plus a SAM-style prompt decoder), PIXAR/PIXAR-DG (MLLM + SAM with pixel-difference supervision), and a GRPO-reasoning VLM with SAM refinement (RPI/IBM). None is evaluated on GPT-Image or Nano-Banana except PIXAR/PIXAR-DG. The strongest *architectural* evidence is FUSED's: the frozen semantic branch is the single most important component for cross-generator transfer, and a low-level-only branch collapses.

### Cited Findings

**(a) Foundation-model features + light heads**
- **FUSED (Aug 2026, U. Amsterdam)** — [arXiv 2608.28302](https://arxiv.org/abs/2608.28302) [full]
  - Architecture: a SparseViT forensic branch (sparse attention to suppress semantics) plus a **frozen CLIP-ConvNeXt-XXL** semantic branch (16×16 tokens). They are fused by a token-wise sparse MoE (8 SwiGLU experts, top-2), and a "segmentation bridge" decoder concatenates the post-MoE semantic tokens with the forensic map so the mask loss supervises the fusion.
  - OpenSDID (train SD1.5, test SD1.5/2.1/XL/3/Flux.1): average **IoU 44.5 / pixel-F1 55.3** vs MaskCLIP 42.7/49.4. On unseen SDXL, SD3 and Flux.1 it gains +11.3, +10.9 and +11.5 F1 over MaskCLIP.
  - Held-out pixel-space edits: **AutoSplice F1 51.4, CocoGlide F1 62.8** (+18.3 and +37.2 over CAT-Net).
  - INP-X (global VAE artifact removed): accuracy drops 80.8 → 68.3. That is the smallest drop among the methods tested, but residual reliance on the global VAE trace remains.
  - So-Fake-OOD (held-out commercial generators, trained on So-Fake-Set): +14.5 pixel-F1 and +13.1 IoU over MaskCLIP. Absolute values were not in the report.
  - Ablation: **removing the frozen semantic branch drops average localization F1 to 19.1**. Scaling the frozen backbone (ConvNeXt-B → XXL) improves both tasks monotonically.
  - Limitations: residual sensitivity to global artifacts; very small edits.
- **PIXAR / PIXAR-DG** (frozen CLIP + frozen SAM encoder, LoRA LLM, tuned mask decoder): see Q1 — [2603.20193](https://arxiv.org/abs/2603.20193); [2607.18230](https://arxiv.org/abs/2607.18230)
- **RiSE (Oct 2026)**: a residual subspace expert for text-centric (document) forgery localization. Out of scope for natural-image editors; listed only for completeness (abs snippet) — [alphaXiv 2610.24480](https://www.alphaxiv.org/abs/2610.24480)

**(b) Frequency / wavelet / sparse features**
- **SegWave (Aug 2026, IISER Bhopal)** — [arXiv 2608.30714](https://arxiv.org/abs/2608.30714) [full]
  - Pipeline: a single-level Haar DWT keeps only LH/HL/HH. An **Adaptive Sub-band Attention (ASA)** reweights the sub-bands, and an inverse DWT produces an "enhanced image". That image is patch-embedded in parallel with RGB, injected into every ViT block (SAM/SAFIRE-like), and decoded with point prompts. Inference uses a 16×16 prompt grid with clustering for multi-source partitioning.
  - Training data is CASIA2 + FantasticReality only (classic forgeries, not generative).
  - Results:
    - CocoGlide: F1_fixed **0.69** (SAFIRE 0.63, TruFor 0.52). RealisticTampering: 0.36 (TruFor 0.43); a global-DCT variant scores 0.40 there.
    - SafireMS-Expert-2: pmIoU 0.821 vs SAFIRE 0.763.
    - ASA ablation: CocoGlide F1 0.69 → 0.67 without ASA, a small effect. Multi-source pmIoU 0.821 → 0.695, a large effect.
  - **Not evaluated on any instruction/MLLM editor.**
  - The authors argue DWT is compression-agnostic (better on PNG), while DCT helps on previously JPEG-compressed content.
- **FUSED's forensic branch is SparseViT** (sparse non-adjacent attention as a semantics suppressor), which is a "sparse" non-semantic stream. Alone it generalizes poorly; it helps only when fused with the frozen semantic branch — [2608.28302](https://arxiv.org/abs/2608.28302)
- No 2026 paper was found that learns wavelets (lifting schemes or learnable filter banks) specifically for localizing MLLM/autoregressive edits. Wavelet use in 2026 localization is fixed Haar with learned sub-band attention (SegWave) or face deepfakes (WAVIE, image-level) — [alphaXiv 2609.14437](https://www.alphaxiv.org/abs/2609.14437) [abs snippet]

**(c) Reference-free change detection / re-editing consistency** — see Q3 (no new method found).

**(d) MLLM reasoning with dense grounding**
- **"Can Vision-Language Models Reason about AI Edits in Images?" (Jul 2026, RPI + IBM Research)** — [arXiv 2607.28464](https://arxiv.org/abs/2607.28464) [full]
  - Method: Qwen2.5-VL-7B trained with **GRPO**, using only a format reward plus an accuracy reward (box IoU for true positives, a fixed reward w for true negatives). There is no explanation supervision. The model outputs think/answer with a bounding box. A **frozen SAM** is then prompted by a projected ⟨SEG⟩ embedding of the reasoning trace plus the box; only the prompt encoder and projection are trained.
  - Training data: AutoSplice, CASIAv2, FantasticReality, FFHQ-FM, MagicBrush and SD-inpaint, 97K samples.
  - Results:
    - Average Acc 0.808, **mIoU 0.264**, eff-IoU 0.211, vs SIDA 0.650/0.227/0.144 and FakeShield 0.507/0.263/0.136.
    - MagicBrush mIoU is only 0.141, vs SIDA 0.162.
    - Pixel-F1 is often *lower* than SIDA's (AutoSplice 0.340 vs 0.484; MagicBrush 0.157 vs 0.436).
  - The gain is mostly in detection accuracy, not mask quality.
  - Reward hacking: if w is set badly, the model collapses to predicting "authentic" (w=0.8 gives Acc 0.500, mIoU 0.000).
  - **Not evaluated on modern MLLM editors.**
- PIXAR's text head (natural-language tamper description) plus semantic class head is a form of "edit-intent" output jointly trained with masks — [2603.20193](https://arxiv.org/abs/2603.20193)
- Training-free region mining for grounded deepfake explanations (Sep 2026) addresses visual grounding of MLLM explanations, but for deepfakes (abs snippet only) — [alphaXiv 2609.35536](https://www.alphaxiv.org/abs/2609.35536)
- SPARED (Aug 2026) is reasoning-based AIGI detection trained on adversarially edited data. It is detection-focused (abs snippet) — [alphaXiv 2608.12876](https://www.alphaxiv.org/abs/2608.12876)

### Inferences
- **(a) is winning.** Across FUSED, GAP-SAM, ForenScope, FLAME and PIXAR-DG, every 2026 cross-generator gain relies on a *frozen* large foundation backbone with a light trainable head or decoder. FUSED's ablation shows that the low-level stream alone does not transfer (F1 19.1).
- **(b) is under-explored for MLLM editors.** SegWave shows learned sub-band attention matters most for *separating sources* (multi-source pmIoU +0.13), which maps naturally onto "edited vs VAE-round-tripped" regions in FR images. Nobody has tested learned wavelets on GPT-Image, Nano-Banana or Kontext edits. This is an open slot for the user's "learned wavelets" direction, provided it is fused with a frozen semantic backbone (per FUSED).
- **(d) adds detection accuracy, not pixel accuracy.** In every 2026 MLLM paper read here, pixel-level quality comes from SAM, and mIoU stays at about 0.15–0.26.

### Gaps
- FUSED absolute So-Fake-OOD numbers and per-editor values were not extracted. FUSED does not test instruction editors.
- SegWave and the GRPO-VLM do not report compute. FUSED uses a ConvNeXt-XXL backbone (large), with parameters not quantified in the report.
- I could not verify code release for FUSED, SegWave or PIXAR-DG. PIXAR code is at VILA-Lab/PIXAR per the web snippet.

---

## Q3. Are there 2026 methods that localize edits by comparing the image with a regenerated/re-edited version of itself, or by detecting semantic "edit intent" (MLLM + segmentation)?

### Takeaway
**Re-editing / regeneration consistency:** still **not found** as a localization method for instruction/MLLM editors. arXiv queries for re-editing/regeneration combined with edited-region localization returned nothing after Sep 2025. The closest remain ReGFLoW (VAE-reconstruction residual as dense guidance), GAP-SAM (global VAE token) and APT (proactive, FR) from the prior notes. **Edit intent:** a weak form exists. PIXAR jointly predicts the semantic class and a text description of the edit with the mask, and the GRPO-VLM localizes by reasoning then a box then SAM. No method was found that **reconstructs the instruction** ("what was the user's edit prompt") and then grounds it as the localization mechanism.

### Cited Findings
- arXiv query `(re-editing OR regeneration OR re-generate OR reconstruction) AND (edited/manipulated/tampered region) AND localiz`, Sep 2025–Oct 2026: **0 results** — arXiv API search (this session)
- Prior-notes items that remain the nearest neighbours: ReGFLoW, using the frozen SD VAE residual |I−D(E(I))| — [2609.36882](https://arxiv.org/abs/2609.36882); GAP-SAM — [2608.20929](https://arxiv.org/abs/2608.20929); TRAIL, using perturbation-drift of frozen DINO — [2608.18968](https://arxiv.org/abs/2608.18968); APT, proactive FR — [2608.30656](https://arxiv.org/abs/2608.30656)
- PIXAR outputs a mask, multi-label semantic class of the tampered object, and a natural-language description, trained jointly (L_sem + L_bce + L_dice + L_cls + L_text) — [2603.20193](https://arxiv.org/abs/2603.20193)
- GRPO-VLM: the reasoning trace plus box serves as the SAM prompt. Its "intent" is an implicit "where/why tampered" judgment, not a reconstructed instruction — [2607.28464](https://arxiv.org/abs/2607.28464)
- On the generation side, instruction editors make unintended out-of-region changes. GPT-Image-2 alters surrounding text in 44.2% of receipt edits (Images 2.5: 31.7%) — [2609.13617](https://arxiv.org/abs/2609.13617). Editing papers explicitly target "minimal change" because current editors make unintended changes — [2610.06021](https://arxiv.org/abs/2610.06021) [abs snippet]; [2604.20258](https://arxiv.org/abs/2604.20258) [abs snippet]. Any intent-grounding localizer would therefore miss the spillover that PIXAR-style difference GT counts as tampered.

### Inferences
- A "re-edit with an open proxy editor (Qwen-Image-Edit / FLUX Kontext), compare to input" localizer remains novel. Its asymmetry with spillover is a two-sided issue: intent grounding misses spillover, while re-edit residuals might capture it. That is an argument for combining (c) and (d).
- Because PIXAR's GT is a pixel-difference map, it is the natural benchmark for a regeneration-residual method (both measure "what pixels changed"). LocateEdit's object masks favour intent-grounding methods instead.

### Gaps
- Not checked: CVPR 2026 workshop proceedings and ACM MM 2026 papers not on arXiv. A re-editing method could exist there.

---

## Q4. Which papers report that low-level traces fail on these editors, and why?

### Takeaway
2026 evidence is consistent:
- Low-level-only streams transfer poorly across generators (FUSED ablation).
- Global VAE traces are shortcuts that vanish when out-of-mask pixels are restored (INP-X, FUSED).
- Compression or JPEG-based cues fail on PNG and regenerated outputs (SegWave motivation; TGIF2/CAT-Net in prior notes).
- Object-mask labels misalign with true edit footprints, so low-level-trained models get penalized for real off-mask changes (PIXAR).

### Cited Findings
- FUSED: "low-level forensic cues are often effective for precise localization within a known generation domain but tend to generalize poorly across different generators". Without the frozen semantic branch, average localization F1 falls to 19.1. Several forensic baselines drop to single digits on AutoSplice/CocoGlide or produce empty or fragmented masks on unseen generators — [2608.28302](https://arxiv.org/abs/2608.28302)
- FUSED on INP-X: removing the global VAE artifact reduces accuracy by 12.5 points for FUSED and 18.6 for PSCC-Net. All methods show residual reliance — [2608.28302](https://arxiv.org/abs/2608.28302)
- PIXAR: many pixels inside object masks are untouched, while relighting, color bleeding, deblurring and seam removal occur *outside* masks. Mask-only metrics penalize models that detect real off-mask artifacts — [2603.20193](https://arxiv.org/abs/2603.20193)
- PIXAR-DG: detectors trained on one VLM "overfit to the specific artifacts produced by those models". VLMs differ in architecture, editing interface, generation pipeline and post-processing — [2607.18230](https://arxiv.org/abs/2607.18230)
- SegWave: FFT/DCT methods tied to JPEG compression are "less reliable on uncompressed formats (e.g., PNG)". It also cites analyses that VLM semantic priors can reduce sensitivity to low-level artifacts, which cuts the other way — [2608.30714](https://arxiv.org/abs/2608.30714)
- GPT-Image-2 / Images 2.5: detection rate falls from 68.6% (controlled) to 35.9% (posted online); localization "remains weak" (abs) — [2609.13617](https://arxiv.org/abs/2609.13617)
- Earlier evidence (prior notes, not repeated): TGIF2 FR and generative SR; DiffusionPrint (camera noise disrupted by latent decoding); FLAME limitations; GAP-SAM boundary adhesion.

### Inferences
- The "why" has three parts: (1) every pixel passes through a VAE or AR decoder, so camera/ISP and JPEG traces are globally overwritten; (2) generator-specific decoder fingerprints differ across editors, so a learned trace does not transfer; (3) the global trace is a shortcut that does not mark *where* the edit is. A learned-wavelet stream would hit (2) and (3) unless it is trained to separate de-novo regions from VAE-round-tripped ones and fused with a frozen semantic backbone.

### Gaps
- No 2026 paper was found that measures *which* low-level statistics survive GPT-Image or Nano-Banana decoding (e.g., per-sub-band spectra of edited vs unedited regions). This is a measurable gap.

---

## Q5. Explicit open problems and limitations stated in 2026 papers and benchmarks

### Takeaway
Stated open problems:
- Cross-generator generalization to closed MLLM editors.
- Scarce labelled data for new editors.
- Label definition (object mask vs pixel difference).
- Small or subtle edits.
- Residual global-artifact reliance.
- Threshold miscalibration under transfer.
- The gap between benchmark and in-the-wild performance.

### Cited Findings
- **Data scarcity for new editors:** PIXAR-DG's late injection still requires labelled samples from new, closed VLMs, "which may not always be available"; performance is bounded by the base architecture — [2607.18230](https://arxiv.org/abs/2607.18230)
- **Label definition / metric validity:** PIXAR argues mask-only metrics over- and under-score detectors and proposes τ-thresholded pixel-difference GT plus semantic and text metrics — [2603.20193](https://arxiv.org/abs/2603.20193)
- **Weak localization even when detection works:** GPT-Image-2 study — [2609.13617](https://arxiv.org/abs/2609.13617); GRPO-VLM high accuracy but mIoU 0.264 — [2607.28464](https://arxiv.org/abs/2607.28464)
- **Threshold shift under transfer:** FUSED on CocoGlide has ROC-AUC 92.4 but accuracy at 0.5 of only 62.6 — [2608.28302](https://arxiv.org/abs/2608.28302)
- **Residual global-artifact sensitivity, small edits:** FUSED limitations — [2608.28302](https://arxiv.org/abs/2608.28302)
- **Robustness to distribution shift and perturbations; the need for better annotated AI-edit masks:** GRPO-VLM future work. Its advantage over baselines shrinks as Gaussian noise grows — [2607.28464](https://arxiv.org/abs/2607.28464)
- **Benchmark-to-wild gap:** 68.6% → 35.9% detection on posted-online images — [2609.13617](https://arxiv.org/abs/2609.13617)
- **Cross-editor limits for VLM localizers:** FragFake (abs) — [2505.15644](https://arxiv.org/abs/2505.15644)
- **Human ceiling is low:** humans reach IoU 10.7% on PIXAR, so human-annotated GT for subtle MLLM edits is itself unreliable — [2603.20193](https://arxiv.org/abs/2603.20193)
- **Multi-source separation degrades** as the number of sources grows to 4 (SegWave) — [2608.30714](https://arxiv.org/abs/2608.30714)

### Inferences
- Concrete openings that follow from these limitations:
  1. Label-free or few-shot adaptation to closed editors (avoiding PIXAR-DG's need for labels).
  2. Calibrated thresholds under transfer (FUSED's AUC-vs-accuracy gap).
  3. Localizers evaluated on pixel-difference GT (PIXAR) *and* object-mask GT (LocateEdit) to expose object bias.
  4. A low-level, learned-frequency stream whose value is shown specifically on GPT-Image and Nano-Banana, where the frozen-semantic methods plateau at about 0.22 gIoU.

### Gaps
- No unified leaderboard still. PIXAR, LocateEdit, PromptForge, Impostor and GIFTBench use different editors and GT definitions, and none of the June–Oct 2026 methods reports on more than one of them.
- ACM MM 2026 / NeurIPS 2026 non-arXiv papers were not covered.
