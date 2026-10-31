# Learned wavelets and sparse-representation networks as frozen or self-supervised feature extractors for generative-edit forgery localization

Scope: learned wavelets, wavelet-domain nets, sparse coding, SAEs on foundation models, learned-compression encoders and self-supervised forensic transforms, all judged on one question: could each one replace or add to a frozen DINOv2 layer-6 linear patch probe? (User baseline: DINOv2 0.77 per-image AUC / 0.68 cross-dataset. Pooled DT-CWT / shearlet / learned-lifting cell statistics: 0.66 / 0.56.)

Verification legend: items tagged **[verified]** had their arXiv abstract or search snippet checked this session (2026-10-31). Items tagged **[unverified-ID]** come from prior knowledge, and their arXiv ID / DOI / numbers were NOT re-checked this session. Check those before citing them.

---

## Q1. Learned wavelets / lifting / scattering / DT-CWT nets: what they learn, weights, cost, forensic evidence

### Takeaway
Learned lifting and learned wavelet transforms come mostly from compression (iWave/iWave++, learned-lifting DWT codecs). They learn invertible, near-orthogonal, rate-distortion-optimal multiscale decompositions. They are tiny (a few conv layers per lifting step) but have no general-purpose public pretrained checkpoints for natural-image features, and I found **no forensic or forgery-localization evaluations** of them. They are fixed or energy-compacting, low-level transforms with no semantic context, so they behave like the user's DT-CWT / shearlet baseline. They are not a step toward DINOv2.

### Cited Findings
- Learned lifting DWT for image compression: the lifting scheme uses learned nonlinear predict/update filters, and a learned tree-based entropy model codes the subband coefficients (Sahin & Kamisli, 2022) **[verified]** — [arXiv:2212.03616](https://arxiv.org/abs/2212.03616)
- iWave3D, a trainable 3-D wavelet transform for end-to-end brain image compression (the iWave family from USTC: Ma, Liu, Xiong et al.) **[verified]** — [arXiv:2109.08942](https://arxiv.org/abs/2109.08942)
- iWave++: "End-to-End Optimized Versatile Image Compression With Wavelet-Like Transform", TPAMI 2022. CNN-based lifting with trained predict/update, and one model covers lossy and lossless coding **[unverified-ID]** — DOI 10.1109/TPAMI.2020.3026003 (https://doi.org/10.1109/TPAMI.2020.3026003)
- Nonlinear wavelet transform learned as a normalizing flow, i.e. an invertible map that separates slow from fast components **[verified]** — [arXiv:2101.11306](https://arxiv.org/abs/2101.11306)
- Lifting-based multiresolution neural operator (LiNO, 2026). This is for PDE operator learning, not images or forensics **[verified]** — [arXiv:2607.02715](https://arxiv.org/abs/2607.02715)
- Adaptive Wavelet Distillation (AWD, NeurIPS 2021) learns a wavelet with interpretability constraints from a trained network. Deep Adaptive Wavelet Network (DAWN, WACV 2020) uses learnable lifting blocks for classification **[unverified-ID]** — AWD arXiv:2107.09145 (https://arxiv.org/abs/2107.09145); DAWN arXiv:1912.05035 (https://arxiv.org/abs/1912.05035)
- Wavelet Neural Operator (Tripura & Chakraborty), a PDE solver. No public natural-image weights relevant here **[unverified-ID]** — arXiv:2205.02191 (https://arxiv.org/abs/2205.02191)
- Kymatio, a GPU wavelet scattering library (fixed Morlet filters; translation-invariant, deformation-stable modulus averages). It has no learned weights; "learned / hybrid scattering" puts a CNN on top **[unverified-ID]** — arXiv:1812.11214 (https://arxiv.org/abs/1812.11214)
- Search for a learned-lifting transform used in forensics or AIGI detection: arXiv queries on "lifting scheme" plus learned plus compression/wavelet returned only compression, video coding, PDE and robotics uses (11 hits). None were forensic **[verified, negative result]** — [arXiv search, see 2212.03616 / 2305.16211 / 2507.04331 list](https://arxiv.org/abs/2305.16211)

### Inferences
- Scattering, DT-CWT and learned lifting are all built to be **invariant or energy-compacting**. Scattering's modulus plus low-pass averaging deliberately throws away the phase and fine-position information that would separate a locally regenerated region from its surroundings. The user's pipeline adds per-cell statistics on top, which is a second averaging step. That explains why it lands near the raw-pixel control.
- A learned-lifting transform trained for rate-distortion learns to remove *predictable* structure. For forensics, the useful signal is often in the *prediction residual* of a model trained on real images (as in Noiseprint, forensic self-descriptions and SPAI; see Q5). If the user keeps the learned-lifting model, the high-pass detail coefficients (the prediction error) at full resolution, not their pooled statistics, are the part worth probing. Better still is the per-pixel log-likelihood under a learned entropy model.

### Gaps
- I found no public, general-purpose pretrained learned-lifting / iWave checkpoint documented as a feature extractor. iWave code exists, but its availability was not verified this session.
- I found no paper evaluating scattering or DT-CWT features with a patch probe for diffusion-inpainting localization.

---

## Q2. Wavelet-domain deep nets and frequency/wavelet forensic detectors: evidence and numbers

### Takeaway
Wavelet backbones (WaveMix, Wave-ViT, WaveCNet, Haar-downsampling nets, WaveDiff) use wavelets as an **architectural downsampling or token-mixing operator** inside networks trained on ImageNet or for generation. They are not forensic features. Forensic wavelet and frequency detectors (Wavelet-CLIP, FreqNet, SAFE, FIRE, SPAI, WaRPAD) are almost all **image-level** real-vs-fake classifiers. The strongest generalizers either (a) combine frequency cues with a big pretrained backbone (Wavelet-CLIP on CLIP ViT-L/14) or (b) model the *spectral distribution of real images* self-supervised and score the deviation (SPAI). Only a few target localization. DEAL-300K's frozen-VFM plus multi-frequency prompt tuning is the clearest localization data point.

### Cited Findings
- **Wavelet-CLIP**: DWT applied to CLIP ViT-L/14 features. Average AUC 0.749 for cross-dataset deepfake generalization and 0.893 on unseen diffusion-generated faces. Code is public **[verified]** — [arXiv:2409.18301](https://arxiv.org/abs/2409.18301), [GitHub](https://github.com/lalithbharadwajbaru/Wavelet-CLIP)
- **WMamba**: wavelet-based Mamba for face forgery detection. It argues wavelets help generalization by capturing structural forgery cues **[verified, abstract snippet only]** — [arXiv:2501.09617](https://arxiv.org/abs/2501.09617)
- **WAVIE (Sep 2026)**: wavelet-augmented *intermediate* ViT embeddings for lightweight generalizable deepfake face detection. It is relevant because it mixes wavelets with intermediate foundation-model layers, much as the user probes DINOv2 L6 **[verified, title/snippet only; numbers not checked]** — [arXiv:2609.14437](https://arxiv.org/abs/2609.14437)
- **FreqNet** (AAAI 2024): forces attention to high-frequency feature content and puts conv layers on the FFT amplitude and phase. +9.8% over SOTA across 17 GANs, with fewer parameters **[verified]** — [arXiv:2403.07240](https://arxiv.org/abs/2403.07240)
- **SAFE** (KDD 2025): swaps downsampling for cropping to preserve artifacts, adds ColorJitter/rotation and patch-based random masking, and notes that the imaging mechanism of synthetic images heightens local pixel correlations. +4.5% accuracy and +2.9% AP over prior methods on 26 generators. Code is public **[verified]** — [arXiv:2408.06741](https://arxiv.org/abs/2408.06741), [GitHub](https://github.com/Ouxiang-Li/SAFE). (The DWT high-frequency input in SAFE's code is from prior knowledge **[unverified]**.)
- **FIRE** (CVPR 2025): diffusion models reconstruct the *mid-band* frequencies of real images poorly. The score is the change in reconstruction error before vs. after frequency decomposition, and it generalizes to unseen diffusion models **[verified]** — [arXiv:2412.07140](https://arxiv.org/abs/2412.07140)
- **SPAI** (CVPR 2025): masked spectral learning (frequency-reconstruction pretext) on **real images only**. Spectral reconstruction similarity is the OOD score, and spectral context attention handles any resolution. +5.5% absolute AUC over prior SOTA across 13 generators, robust to online perturbations. Code is public **[verified]** — [arXiv:2411.19417](https://arxiv.org/abs/2411.19417), [project](https://mever-team.github.io/spai)
- **WaRPAD**: training-free. It measures how sensitive self-supervised (e.g. DINO-family) embeddings are to perturbations along **Haar-wavelet high-frequency directions**, computed per patch and then averaged. Evaluated on 23 generators and applicable across SSL models **[verified]** — [arXiv:2511.14030](https://arxiv.org/abs/2511.14030)
- **DEAL-300K** (Nov 2025): over 300k diffusion-edit localization images. A **frozen Visual Foundation Model + Multi-Frequency Prompt Tuning** reaches pixel F1 82.56% on its own test split and 80.97% on CoCoGlide **[verified]** — [arXiv:2511.23377](https://arxiv.org/abs/2511.23377)
- **TGIF dataset**: diffusion inpainting either splices the inpainted region into the original or **regenerates the entire image** (VAE round-trip). "Traditional IFL methods ... fail to detect regenerated inpainted images", SID methods flag them as fake but cannot localize, and both fail under strong or WEBP compression **[verified]** — [arXiv:2407.11566](https://arxiv.org/abs/2407.11566)
- Wavelet backbones, all from prior knowledge, none forensic: WaveMix (token mixing via 2D-DWT, arXiv:2205.14375), Wave-ViT (ECCV 2022; DWT-downsampled keys/values, arXiv:2207.04978), WaveCNet (TIP 2021; DWT replaces pooling/strided conv for noise robustness, arXiv:2005.03337), Haar Wavelet Downsampling (HWD, Pattern Recognition 2023, DOI 10.1016/j.patcog.2023.109819), WaveDiff (CVPR 2023; diffusion GAN in wavelet space, arXiv:2211.16152) **[unverified-ID]**

### Inferences
- **Explanation for the user's result (pooled wavelets vs. DINOv2).** TGIF shows that full-image-regenerated inpainting leaves *global* low-level traces (the whole image goes through the VAE decoder). Inside and outside the mask then differ mainly in **content plausibility and in how the edit stitches into its context**, not in subband energy. A pooled 24x24 cell statistic of DT-CWT or shearlet energy sees the VAE trace everywhere, so it cannot localize. DINOv2 L6 tokens carry both mid-level texture and contextual information (attention mixes neighbouring patches), which matches what remains discriminative. This is an inference that fits the TGIF finding. It has not been directly tested.
- Frequency methods generalize when they **model real-image statistics and score deviation** (SPAI, FIRE, Noiseprint, self-descriptions). Methods that classify frequency features directly overfit. The user's linear probe on wavelet statistics belongs to the second, weaker category.
- WaRPAD and Wavelet-CLIP / WAVIE suggest a hybrid: inject wavelet high-frequency perturbations into DINOv2 and use the *patch-token sensitivity* as a per-patch feature. This keeps spatial structure and adds a frequency signal the probe can use.

### Gaps
- SPAI, FIRE and SAFE report image-level AUC/accuracy only. I found no published patch-level localization numbers for them.
- I did not verify per-dataset numbers for WAVIE, WMamba or DEAL-300K beyond the abstracts.

---

## Q3. Sparse representations: CSC / LISTA / unrolled nets, dictionary learning, SAEs on CLIP/DINO for AIGI detection or localization

### Takeaway
Unrolled sparse-coding networks (SDNet / CSC layers, CRATE white-box transformers) give spatially resolved sparse codes and better robustness, and pretrained ImageNet weights exist. None were found evaluated in forensics. SAEs on vision foundation models matured in 2025-2026: patch-level SAEs on DINOv2/CLIP and a 32k-concept archetypal SAE on DINOv2 have public code or models. **I found no paper that uses SAE features for AI-generated image detection or forgery localization.** The closest are an interpretability analysis of a VLM deepfake detector and a real-vs-generated concept comparison. This is an open and cheap opportunity: a **patch-level SAE on DINOv2 L6 tokens** keeps the spatial grid and sparsifies it.

### Cited Findings
- **SDNet** ("Revisiting Sparse Convolutional Model for Visual Recognition", NeurIPS 2022): differentiable convolutional-sparse-coding layers as drop-in replacements for conv layers. It matches ResNets on CIFAR / ImageNet and is "much more robust to input corruptions as well as adversarial perturbations" via a sparsity/reconstruction trade-off. Code is public **[verified]** — [arXiv:2210.12945](https://arxiv.org/abs/2210.12945), [GitHub](https://github.com/Delay-Xili/SDNet)
- **CRATE** (white-box transformer; each layer has an ISTA sparse-coding step; reported to show emergent segmentation in its attention). Pretrained ImageNet weights are released **[unverified-ID]** — arXiv:2306.01129 (https://arxiv.org/abs/2306.01129)
- **saev / patch-level SAEs**: SAEs trained on frozen ViT activations (CLIP, DINOv2) support **patch-level** causal edits for classification and segmentation without retraining the ViT or heads. Code, demos and models are public **[verified]** — [arXiv:2502.06755](https://arxiv.org/abs/2502.06755), [project](https://osu-nlp-group.github.io/saev)
- **RA-SAE on DINOv2 (32,000 concepts)**: used to compare concept prevalence between real images and SD1.5/2.1, PixArt and Kandinsky generations. It finds "suppressed" and "exaggerated" concepts, e.g. wood background texture is exaggerated in generated images **[verified]** — [arXiv:2506.19708](https://arxiv.org/abs/2506.19708). Archetypal SAE method (Fel et al., ICML 2025) **[unverified-ID]** — arXiv:2502.12892 (https://arxiv.org/abs/2502.12892)
- **"The Deepfake Detective"** (Dec 2025): SAE plus "forensic manifold" analysis on a VLM deepfake detector. Only a small fraction of latents are active per layer, and the manifold geometry varies systematically with artifact type. This is interpretability only, with no localization benchmark **[verified]** — [arXiv:2512.21670](https://arxiv.org/abs/2512.21670)
- SAE probing on DINOv2 (ICCVW 2025 Findings) covers OOD generalization and ontology construction. It reports that CLIP SAE performance degrades at high sparsity **[verified via search snippet]** — [arXiv:2508.11277](https://arxiv.org/abs/2508.11277)
- A targeted arXiv query for "sparse autoencoder" plus forensic/deepfake/synthetic-image/generated-image (2024-06 onward) returned only 7 papers. None proposed an SAE-based detector or localizer **[verified, negative result]** — [arXiv:2512.21670 among results](https://arxiv.org/abs/2512.21670). A web search confirmed no SAE + AIGI-detection paper — [ICCVW 2025 SAE probing paper](https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Olson_Probing_the_Representational_Power_of_Sparse_Autoencoders_in_Vision_Models_ICCVW_2025_paper.pdf)

### Inferences
- An SAE trained on DINOv2 L6 *patch* tokens (unlabeled, real plus generated) gives a sparse, overcomplete, spatially resolved code at the native 16x16 or 37x37 token grid. A linear probe on SAE codes will be at most as good as one on the dense tokens in-domain, because the SAE is a near-linear re-parameterization. Its value would be in (a) **cross-dataset transfer**, if a few "generator-trace" latents are dataset-agnostic, and (b) interpretability (finding the latents that fire on inpainted patches). This is a hypothesis.
- Earlier LISTA / CSC forensic work (dictionary-learning splicing detectors before 2021) is outside the date window and not catalogued here.

### Gaps
- No SAE-feature detector or localizer exists, so there are no numbers.
- Whether public saev checkpoints include DINOv2 at intermediate layers such as L6 was not verified. Those released are likely final-layer.

---

## Q4. Learned image compression encoders (hyperprior, ELIC, TCM, JPEG AI) as forensic feature extractors

### Takeaway
2024-2026 work treats learned compression as a **counter-forensic confound**, not a feature extractor. JPEG AI artifacts resemble synthesis and splicing artifacts and degrade SOTA detectors and localizers. Bergmann et al. give interpretable JPEG AI cues: colour-channel correlation, recompression convergence, and latent quantization to separate JPEG-AI-real from generated images. **I found no paper that uses a hyperprior / ELIC / TCM encoder's latents as frozen features for forgery localization.** The *rate* (per-pixel bits under a learned entropy model) is the more principled cue, by analogy with ZED-style lossless-coder detectors.

### Cited Findings
- "Is JPEG AI going to change image forensics?" JPEG AI introduces artifacts "that closely resemble those generated by image synthesis techniques and image splicing pipelines", and SOTA deepfake detectors and splicing localizers lose performance on JPEG-AI-processed content **[verified]** — [arXiv:2412.03261](https://arxiv.org/abs/2412.03261)
- "Three Forensic Cues for JPEG AI Images": (1) JPEG AI preprocessing introduces colour-channel correlations; (2) repeated compression shows diminishing distortion differences, which enables recompression detection; (3) latent-space quantization separates JPEG-AI-compressed real images from synthetic ones **[verified]** — [arXiv:2504.03191](https://arxiv.org/abs/2504.03191)
- JPEG AI standard overview **[verified]** — [arXiv:2510.13867](https://arxiv.org/abs/2510.13867)
- SLIC: a secure learned codec that watermarks in the compressed domain to defend against manipulation. This is proactive and not a passive feature extractor **[verified via search result title]** — [arXiv:2410.15075](https://arxiv.org/abs/2410.15075)
- Web search for learned-compression latents used as deepfake/forgery features found no such paper **[verified, negative result]** — [search results incl. arXiv:2404.18820](https://arxiv.org/abs/2404.18820)
- ZED (Cozzolino et al., ECCV 2024): zero-shot AIGI detection from the coding cost under a lossless learned coder trained on real images **[unverified-ID]** — arXiv:2409.15875 (https://arxiv.org/abs/2409.15875)
- Hyperprior (Ballé 2018, arXiv:1802.01436), ELIC (arXiv:2203.10886) and TCM (CVPR 2023, arXiv:2303.14978) all have public CompressAI / GitHub weights trained on natural images (CLIC / ImageNet / Flickr) **[unverified-ID]**

### Inferences
- A learned codec's **per-location bit-cost map** (−log2 p(ŷ) from the hyperprior) is spatially resolved and self-supervised on real images. Regenerated regions should be "too easy" or atypical to code, as in ZED. This keeps spatial structure for free at stride 16, with roughly 10-20 GFLOPs at 512² for a mean-scale hyperprior (estimate). It is a promising and untested localization feature.
- The risk is that JPEG AI or WEBP post-processing confounds this signal; see 2412.03261.

### Gaps
- No numbers exist for codec-latent or rate-map features in localization.

---

## Q5. Self-supervised learned transforms for forensics (Noiseprint/Noiseprint++, residual filter banks, forensic self-descriptions)

### Takeaway
These are the closest relatives of a "learned filter bank" that actually works in forensics. They learn *predictive* residual filters on **real images only** and then detect deviations. TruFor's Noiseprint++ is the standard frozen low-level extractor for localization, with public weights. Forensic self-descriptions (CVPR 2025) learn multiscale predictive filters and fit a per-image compact model, giving zero-shot detection and attribution. On fully regenerated inpainting, though, TGIF shows that camera-trace localizers such as these fail. They complement DINOv2 rather than replace it.

### Cited Findings
- **TruFor / Noiseprint++** (CVPR 2023): a learned noise-sensitive fingerprint trained self-supervised on real data only, fused with RGB in a transformer. It outputs a localization map, an integrity score and a reliability map. Code and weights are public **[verified]** — [arXiv:2212.10957](https://arxiv.org/abs/2212.10957), [project](https://grip-unina.github.io/TruFor/)
- **Forensic self-descriptions** (Nguyen, Azizpour, Stamm, CVPR 2025): self-supervised diverse predictive filters on real images. Residuals are modelled jointly across scales, and the fitted parameters form a per-image "self-description" for zero-shot detection, open-set attribution and clustering **[verified]** — [arXiv:2503.21003](https://arxiv.org/abs/2503.21003)
- TGIF: IFL methods (which include Noiseprint-family tools) detect spliced inpaintings but **fail on regenerated** inpaintings **[verified]** — [arXiv:2407.11566](https://arxiv.org/abs/2407.11566)
- Original Noiseprint (TIFS 2020) **[unverified-ID]** — arXiv:1808.08396 (https://arxiv.org/abs/1808.08396)

### Inferences
- Self-descriptions fit a per-image model. Fitting the same predictive-filter model *per window*, then measuring the window residual's likelihood under the global image model, would turn it into a localizer that keeps spatial structure. This is untested.
- For the user's partial generative edits, a Noiseprint++ map concatenated channel-wise with DINOv2 L6 tokens in the patch probe is the cheapest strong hybrid test.

### Gaps
- I found no published Noiseprint++ / self-description numbers on the user's datasets.

---

## Ranked shortlist (synthesis; each item rests on the cited findings above)

1. **Keep DINOv2 and add a spatially resolved frequency channel instead of pooled stats.** Options: (a) WaRPAD-style per-patch sensitivity of L6 tokens to Haar high-frequency perturbations ([2511.14030](https://arxiv.org/abs/2511.14030)); (b) multi-frequency prompt tuning on a frozen VFM, which has the best published diffusion-edit localization result (F1 0.81 on CoCoGlide) ([2511.23377](https://arxiv.org/abs/2511.23377)). Reason: this attacks the identified failure mode (pooling destroys spatial detail; the context cue lives in VFM tokens) at low cost.
2. **TruFor Noiseprint++ map + DINOv2 L6 concatenation in the linear/MLP probe** ([2212.10957](https://arxiv.org/abs/2212.10957)). Reason: public weights, full-resolution self-supervised residual, standard in IFL. It is complementary because it catches spliced and pasted edits, where DINOv2 may miss low-level seams.
3. **SPAI-style self-supervised spectral reconstruction, scored per patch** ([2411.19417](https://arxiv.org/abs/2411.19417)), or FIRE-style mid-band reconstruction error maps ([2412.07140](https://arxiv.org/abs/2412.07140)). Reason: these are the generalizing "model real, score deviation" paradigm, and they give a native spatial map if per-patch similarities are kept before aggregation.
4. **Patch-level SAE on DINOv2 L6 tokens** (saev tooling, [2502.06755](https://arxiv.org/abs/2502.06755)). Reason: cheap, spatial, unexplored for forensics ([2512.21670](https://arxiv.org/abs/2512.21670) is interpretability only). It may improve cross-dataset transfer by isolating dataset-agnostic latents, and it is novel enough to publish.
5. **Learned-codec rate map** (hyperprior/ELIC per-latent bits, ZED-like) as an extra channel. Reason: self-supervised on real images and spatial at stride 16. Untested and confounded by JPEG AI / WEBP ([2412.03261](https://arxiv.org/abs/2412.03261), [2504.03191](https://arxiv.org/abs/2504.03191)).
6. **Unrolled sparse-coding backbones (SDNet, CRATE)** as alternative frozen encoders. Reason: spatial sparse codes and robustness ([2210.12945](https://arxiv.org/abs/2210.12945)), but ImageNet-scale weights only and no forensic evidence. Expect them to fall below DINOv2.
7. **Lower priority:** learned lifting / iWave, scattering, DT-CWT, WaveMix / Wave-ViT / WaveCNet backbones. Reason: no forensic evidence, invariance by design, and the user's own results already show this family near the raw-pixel control. If the learned-lifting model is kept, probe its full-resolution detail (prediction-error) coefficients unpooled.

Unifying explanation (inference): partial generative edits made with modern diffusion inpainting often regenerate the whole image ([TGIF, 2407.11566](https://arxiv.org/abs/2407.11566)), so low-level multiscale statistics are altered globally and weakly separate the mask. The remaining local cues are contextual / semantic plausibility plus subtle mid-band reconstruction behaviour ([FIRE](https://arxiv.org/abs/2412.07140)). Those cues are captured by spatially resolved, context-mixed features (ViT tokens) or by per-location deviation-from-real-model scores. Per-cell pooled subband energies capture neither.
