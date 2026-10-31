# Region descriptors for region-wise self-consistency localization of generative edits

Scope: which per-region descriptors stay roughly stationary across an authentic image (once content is accounted for) but shift inside a generated region. Edit types covered: diffusion/flow inpainting and replacement, instruction/MLLM edits, and fully regenerated edits. Learned pairwise comparators are included as an alternative to hand-chosen descriptors.

Labels used below:
- **[V]** = I checked this against the primary abstract or HTML in this session.
- **[BK]** = background knowledge. The citation is standard, but I did not re-fetch it this session.
- **[H]** = hypothesis or inference, not shown in a paper.

Three pixel regimes recur throughout:
- **Camera:** real sensor/ISP pixels.
- **Gen-decoded:** pixels produced by a generator's decoder from new latents.
- **VAE-RT:** a VAE round trip (encode then decode) of camera pixels, with no new content.

The regime that matters most is a full re-decode, where the authentic region is itself a VAE round trip.

---

## 1. Noise-level and noise-structure descriptors (PCA noise level, wavelet MAD, NLF, Noiseprint/Noiseprint++, DT-CWT noise)

### Takeaway
These are the classical splicing descriptors. They assume authentic pixels share one sensor/ISP noise process. That holds for paste-back inpainting, where the generated region has a different, usually lower, noise floor. It fails when the editor re-decodes the whole image, because the decoder then replaces camera noise everywhere. That matches your observed AUC of 0.62–0.68 and near chance on GPT-Image-2. Keep noise descriptors only as a cue that works for paste-back editors.

### Cited Findings
- **(a) Blind local noise from band-pass kurtosis.** Lyu, Pan & Zhang estimate local noise variance from the regular kurtosis of natural images in band-pass domains, solved in closed form. They flag splices as regions with inconsistent local noise. The precursor paper segments tampered regions by coarse-to-fine clustering of per-block noise variance. — [Lyu et al., IJCV 110(2):202–221, 2014](https://cse.buffalo.edu/~siweilyu/papers/ijcv14.pdf); [Pan et al. 2011](https://cse.buffalo.edu/~siweilyu/papers/mmsec11.pdf) [V, via search snippet]
- **(a) MAD and PCA noise estimators.** The MAD/0.6745 estimator on the finest wavelet band comes from Donoho's soft-thresholding paper — [Donoho 1995, DOI 10.1109/18.382009](https://doi.org/10.1109/18.382009) [BK]. Applied block-wise as a forgery cue — [Mahdian & Saic 2009, DOI 10.1016/j.imavis.2009.02.001](https://doi.org/10.1016/j.imavis.2009.02.001) [BK]. PCA (minimum-eigenvalue) estimators on weakly textured patches — [Liu, Tanaka & Okutomi, TIP 2013, DOI 10.1109/TIP.2013.2283400](https://doi.org/10.1109/TIP.2013.2283400) [BK]; [Chen, Zhu & Heng, ICCV 2015](https://openaccess.thecvf.com/content_iccv_2015/html/Chen_An_Efficient_Statistical_ICCV_2015_paper.html) [BK].
- **(a) Noiseprint++ / TruFor.** TruFor extracts a learned noise fingerprint (Noiseprint++), fuses it with RGB, and outputs an anomaly map, a confidence map and an integrity score. It treats forgeries as departures from the pristine image's expected pattern. — [TruFor, arXiv 2212.10957](https://arxiv.org/abs/2212.10957) [V, via search summary]. Noiseprint itself: [arXiv 1808.08396](https://arxiv.org/abs/1808.08396) [BK].
- **(a,e) DT-CWT noise.** This approach does colour segmentation first, then analyses DT-CWT noise inconsistency per segment. It argues that the DT-CWT's shift-invariance and directional selectivity help capture inpainting artefacts, and reports beating SOTA on the authors' own inpainting benchmark. — [Barglazan & Brad, arXiv 2408.06429](https://arxiv.org/abs/2408.06429) [V]
- **(b) Full re-decode breaks camera-noise cues.** Diffusion inpainting with a full regeneration pipeline "reconstruct[s] the entire image via a latent decoder", which disrupts the camera-level noise patterns that existing IFL methods rely on. — [DiffusionPrint, arXiv 2604.12443](https://arxiv.org/abs/2604.12443) [V]
- **(b) The VAE round trip shifts the spectrum of unedited pixels too.** It causes "a subtle but pervasive spectral shift across the entire image, including unedited regions", linked theoretically to high-frequency attenuation by the VAE bottleneck. Restoring the original pixels outside the mask (INP-X) drops detector accuracy, for example from 91% to 55%, often near chance. — [Nebioglu, Bilgiç & Popescu, arXiv 2602.00192](https://arxiv.org/abs/2602.00192) [V]
- **(b) The generated interior has lower local variance.** FLAME postulates (Assumption 3.1, not derived) that expected local high-frequency energy is lower in generated regions than in authentic ones. It also argues for an energy spike at the mask boundary (Thm 3.3). — [FLAME, arXiv 2606.02178](https://arxiv.org/html/2606.02178v1) [V]
- **(e) Accuracy on generative edits.** Out of distribution, TruFor was the strongest baseline in FLAME's Table 1, at average IoU 0.247 / F1 0.324 over CoCoGLIDE, AutoSplice, NanoBanana, Qwen-Image(-Edit) and Flux Kontext. FLAME base scored 0.358 / 0.459. — [arXiv 2606.02178](https://arxiv.org/html/2606.02178v1) [V]
- **(e) Learned generator-fingerprint companion.** DiffusionPrint learns a patch-level contrastive fingerprint, based on the idea that regions from the same generator share a fingerprint. Used as a second modality in TruFor/MMFusion, it gains up to +28% on unseen mask types. Its stated failure modes are object bias and false activations on authentic objects. — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443) [V]

### Inferences
- **(b) Expected behaviour by regime:**
  - Camera pixels: a noise level function (NLF) that is a smooth function of intensity, the same across the image.
  - VAE-RT and gen-decoded pixels: lower, decoder-shaped noise (high frequencies attenuated, per 2602.00192 and 2606.02178). Under full re-decode, both authentic and edited pixels sit in this regime, so the contrast shrinks to "decoded from encoder latents" versus "decoded from denoised latents". That gap is likely much smaller. [H]
- **(c) Content dependence:** this is severe, because texture inflates every residual-variance estimator. To normalise:
  - Estimate on weakly textured patches only (the PCA/Liu selection idea).
  - Condition on intensity, comparing NLF(I) curves rather than a single σ.
  - Regress σ on local gradient energy and intensity across the image, then score the residual of that regression. [H]
- **(d) Sample size:** a variance estimate from n i.i.d. samples has relative standard error about √(2/n). Getting σ² to ±10% needs about 200 effectively independent flat-region samples. Correlated decoder noise and texture make the effective n much smaller. A practical floor is roughly 32×32 to 64×64 pixels of low-texture area per region. [H, standard statistics]
- **(f) Verdict:** keep NLF-normalised noise level as a cheap cue for paste-back and composite editors. Do not expect it to work on MLLM re-decode editors. Noiseprint++ adds little on full re-decode, because there is no camera fingerprint left to contrast against. [H, consistent with 2604.12443 and 2602.00192]

### Gaps
- I found no published per-region noise-level AUC on MLLM editors such as GPT-Image, Qwen-Image-Edit or Nano Banana.
- I found no measurement of how far apart "VAE-encoded then decoded" and "denoised-latent then decoded" noise statistics are.

---

## 2. Signal-statistics families: learned wavelets / sparse codes / scattering, spectral slope and multifractal exponents, bispectrum, RMT eigen-spectra, MDL/compression cost, chromatic aberration, CFA, JPEG grid, colour statistics, local energy (FLAME)

### Takeaway
Two signals in this group have direct 2024–2026 evidence on generative edits:
- **Coding cost:** surprisal under a lossless model of real images (ZED).
- **Local adjacency energy:** FLAME's LAD map.

Colour statistics get anecdotal support for Qwen-Image-Edit, and I found no numbers. Camera-pipeline traces (CFA, chromatic aberration, JPEG grid) are destroyed by re-decoding. That makes them useless against MLLM editors, though they work against paste-back. Bispectrum, multifractal exponents and RMT spectra have no local-edit evidence I could find. They are research bets.

### Cited Findings
- **(a,e) Coding cost (ZED).** ZED is a zero-shot, entropy-based detector trained only on real images. A multi-resolution lossless encoder estimates each pixel's probability given its context, and the detector measures "how surprising" the image is. A single feature reaches SOTA, about 3% above prior methods on average. Training uses only real images. — [Cozzolino et al., arXiv 2409.15875](https://arxiv.org/abs/2409.15875) [V]
- **(a,e) Local energy (FLAME LAD).** LAD(p) is the mean, over a hollow 3×3 neighbourhood, of tanh(‖I(p)−I(q)‖²/τ²). 3×3 beat min/max/avg operators, and 5×5 and 7×7 performed worse.
  - Removing the LAD map drops in-distribution validation IoU/F1 from 0.567/0.662 to 0.294/0.380.
  - Out of distribution (base model), IoU/F1 was 0.469/0.576 on CoCoGLIDE, 0.501/0.624 on AutoSplice, 0.216/0.295 on NanoBanana, 0.321/0.408 on Qwen-Image and 0.285/0.391 on Flux Kontext.
  - Robustness: JPEG lowers average IoU to 0.329 and Gaussian noise to 0.270 (from 0.416 in Table 3, which is inconsistent with Table 1's 0.358).
  - Stated limitations: smooth or repetitive authentic regions give false positives, object removal is hard, and ISP denoising disrupts LAD. The paper reports no numeric energy gap, only visualisations.
  — [arXiv 2606.02178](https://arxiv.org/html/2606.02178v1) [V]
- **(a,e) Colour statistics.** LocateEdit-Bench (231K images, four editors: Qwen-Image-Edit, Flux-Kontext, Step1X-Edit, BAGEL) analyses only Qwen-Image-Edit. In edited regions it finds significantly lower colourfulness and brightness and slightly higher spatial information (SI), shown in Fig. 5 with no numbers given. The supervised IML-ViT reaches pixel AUC 94.9–97.2 and F1 78.0–85.3 on these editors when trained in domain. — [arXiv 2602.05577](https://arxiv.org/html/2602.05577v1) [V]
- **(a) VAE reconstruction error.** AEROBLADE scores images by LPIPS between the image and its AE reconstruction, since generated images reconstruct more accurately. Its error map can be used qualitatively to find inpainted regions. — [Ricker et al., CVPR 2024, arXiv 2401.17879](https://arxiv.org/abs/2401.17879) [V, via search summary]
- **(a) Spectral slope and decay.** Generated images show discrepancies in high-frequency spectral decay. — [Dzanic et al., NeurIPS 2020, arXiv 1911.06465](https://arxiv.org/abs/1911.06465) [BK]. VAE high-frequency attenuation is formalised in [arXiv 2602.00192](https://arxiv.org/abs/2602.00192) [V].
- **(a) Camera-pipeline traces.** These are classical localisation cues, and all assume a camera pipeline:
  - CFA: [Ferrara et al., TIFS 2012, DOI 10.1109/TIFS.2012.2202227](https://doi.org/10.1109/TIFS.2012.2202227) [BK]
  - Chromatic aberration: [Johnson & Farid 2006, DOI 10.1145/1161366.1161368](https://doi.org/10.1145/1161366.1161368) [BK]
  - JPEG block-grained / double quantisation: [Bianchi & Piva, TIFS 2012, DOI 10.1109/TIFS.2012.2187516](https://doi.org/10.1109/TIFS.2012.2187516) [BK]
  - Bicoherence for splicing: [Ng, Chang & Sun, ISCAS 2004, DOI 10.1109/ISCAS.2004.1329954](https://doi.org/10.1109/ISCAS.2004.1329954) [BK]
- **(a) Scattering coefficients:** [Bruna & Mallat, arXiv 1203.1513](https://arxiv.org/abs/1203.1513) [BK]. I found no forensic local-edit use.
- **(a) RMT eigen-spectra:** an arXiv search for random-matrix forensics returned only a paper on diffusion-model consistency across data splits, which is not about forensics. — [arXiv 2602.02908](https://arxiv.org/abs/2602.02908) [V]

### Inferences
- **(b) Expected behaviour by regime:**
  - CFA, chromatic aberration and JPEG grid: present in camera pixels, absent in both gen-decoded and VAE-RT pixels. They therefore separate edited from authentic only when authentic pixels were pasted back. Under full re-decode they are absent everywhere. Chromatic aberration is a partial exception: it is geometric, so an in-place VAE round trip may preserve it while generated content gets a new, inconsistent misalignment field. That would make it one of the few optical cues that might survive re-decode. [H, untested]
  - Coding cost and LAD: both measure high-frequency predictability, which drops in gen-decoded pixels (less entropy, smoother). VAE-RT pixels are also smoothed, but they keep the scene's real micro-structure that was encoded in the latent. The contrast should shrink, not vanish. [H; supported only indirectly by FLAME's OOD numbers on Qwen/Flux/NanoBanana, which are much lower than on CocoGlide/AutoSplice]
- **(c) Content dependence:** coding cost and LAD are dominated by texture, so normalising for content is critical. Options:
  - Use a calibrated cost such as the ZED-style gap between actual coding cost and model-predicted entropy, rather than raw bits.
  - Regress per-region cost on content covariates (gradient energy, segment class from DINO) and use the residual.
  - Compare only between regions of similar texture class. [H]
  - Colour statistics are scene-dependent (lighting, objects), so compare them only conditioned on semantic class or illumination. [H]
- **(d) Sample size:**
  - LAD and coding cost: per-pixel quantities, so means are stable at 32×32 or more.
  - Spectral slope: needs at least 64×64 for a stable log-log fit across 2–3 octaves.
  - Bispectrum and multifractal (leader) exponents: estimators have high variance and typically need 128×128 or more. That makes them poor choices for small edit regions. [H/BK]
- **(f) Verdicts:**
  - **Promising, use:** LAD/local energy and coding-cost surprisal, both content-normalised.
  - **Paste-back only:** CFA, JPEG grid, NLF.
  - **Weak and editor-specific:** colour statistics.
  - **Speculative, low priority:** bispectrum, multifractal, RMT, scattering.
  - **Test once:** chromatic-aberration misalignment fields. [H]

### Gaps
- I found no numbers for per-region ZED surprisal used for localisation; ZED is image-level.
- LocateEdit gives its colour statistics only qualitatively.
- I found no forensic local-edit studies for wavelet leaders, RMT or learned lifting surprisal.

---

## 3. Foundation-model tokens (DINOv2/v3, CLIP) and their perturbation response (TRAIL, WaRPAD)

### Takeaway
This is the strongest training-free family in 2025–2026 and matches your best result (DINOv2 layer 6, AUC 0.77). TRAIL shows that late-layer DINO patch-token drift under a global Haar perturbation is already a localisation map. That map comes close to a mask-supervised localiser on CocoGlide. It is not yet shown on MLLM re-decode editors.

### Cited Findings
- **(a,e) TRAIL.** It applies one global Haar perturbation and maps the cosine drift of each patch token.
  - CocoGlide (80 source-disjoint test images): patch AUROC .903, against .912 for the mask-supervised Detective SAM. Dice is .619 at a fixed threshold versus .709, and .790 at an oracle threshold.
  - Poisson image interpolation, transferred unchanged: AUROC .855 against .864, so the cue survives without a generator.
  - Across 16 DINO encoders, the best block sits at normalised depth .80–.94.
  - Global context matters: AUROC .903 with a global perturbation, .857 local-in-canvas, .735 for independently encoded crops.
  — [Kumar, Jain & Kainz, arXiv 2608.18968](https://arxiv.org/abs/2608.18968) [V]
- **(a) WaRPAD.** It scores how sensitive self-supervised embeddings are to Haar high-frequency perturbations. Scores are computed per patch after rescaling the image to a multiple of the model's input size, then averaged. It is training-free, tested against 23 generators, robust to test-time corruptions, and works across SSL models. It operates at image level. — [Choi, Lee & Lee, arXiv 2511.14030](https://arxiv.org/abs/2511.14030) [V]
- **(e) Inferred from comparison set:** FLAME compares against MLLM localisers such as SIDA and FakeShield, which score below it out of distribution. This suggests semantic-only reasoning does not localise reliably. — [arXiv 2606.02178](https://arxiv.org/html/2606.02178v1) [V]

### Inferences
- **(b) Expected behaviour:** DINO tokens encode both semantics and some low-level statistics. Drift under a high-frequency perturbation measures how much the token depends on high frequencies. That dependence likely differs for gen-decoded content, which has smoother, off-manifold high frequencies. Under full re-decode the background is a VAE round trip, so the drift contrast should fall. TRAIL's Poisson result shows the cue reacts to an interpolated or smoothed patch even without a generator. That may help on MLLM edits, where the new object is synthesised but the background is only round-tripped. [H]
- **(c) Content dependence:** raw tokens are mostly semantic, so a region whose content differs looks inconsistent. That likely explains a large share of your 0.77. Perturbation drift, a difference between two forward passes, cancels much of the semantics, which makes it a better-normalised descriptor than raw tokens.
  - Your layer-6 optimum on a ViT-B/L is shallower than TRAIL's best depth of .80–.94. It is worth sweeping depth for drift specifically. [H]
- **(d) Sample size:** one 14×14 or 16×16 patch token per location. Region scores average over at least 4–16 tokens. Context is required: encode the full canvas, not crops (.903 vs .735). [V for the context effect; H for the counts]
- **(f) Verdict:** **Top priority.** Make TRAIL-style drift the main per-token descriptor. Aggregate it per region with robust statistics, and run self-consistency on drift residuals rather than raw tokens. Validate on Qwen-Image-Edit, Flux Kontext and GPT-Image, where no published number exists yet.

### Gaps
- I found no TRAIL or WaRPAD numbers on instruction/MLLM editors or on GPT-Image.
- I found no study of DINOv3 vs DINOv2 for this purpose (unverified).

---

## 4. Learned pairwise forensic similarity as the region comparator

### Takeaway
Pairwise "same source?" models fit self-consistency by construction. The comparator is learned rather than hand-picked, and real-only training is possible (EXIF consistency, FSD/Forensic Twins). The main risk is that existing real-only models learn camera/ISP identity, which full re-decode erases. A real-only model on residuals would separate paste-back edits well. Under MLLM re-decode it needs training pairs that include VAE round-trip versus generated regions to stay discriminative. That calls for a hybrid training scheme. [H]

### Cited Findings
- **(a,e) Forensic Similarity.** A CNN feature extractor plus a 3-layer similarity network decides whether two patches share a forensic trace. It works on "unknown" traces not seen in training, cuts error rates "by more than half" over prior art, and is shown for forgery detection and localisation. — [Mayer & Stamm, TIFS 15:1331–1346, 2020, arXiv 1902.04684](https://arxiv.org/abs/1902.04684) [V, abstract via search]. The graph-community extension for localisation is in [Mayer & Stamm, JSTSP 14(5), 2020](https://researchdiscovery.drexel.edu/esploro/outputs/journalArticle/Exposing-Fake-Images-With-Forensic-Similarity/991019167787804721) [V, metadata].
- **(a,e) EXIF self-consistency.** It is trained on EXIF metadata from real photos, with no manipulated training images, and predicts whether patch pairs share EXIF attributes. It reaches SOTA mAP on Columbia, Carvalho and Realistic Tampering, beating supervised FCN. — [Huh et al., ECCV 2018, arXiv 1805.04096](https://arxiv.org/abs/1805.04096) [V, exact mAP not retrieved]
- **(a,e) Forensic Self-Descriptions (FSD).** Predictive filters learned on real images only produce multi-scale residuals. A compact model fitted per image serves as its "self-description", supporting zero-shot detection, open-set attribution and clustering. — [Nguyen, Azizpour & Stamm, arXiv 2503.21003](https://arxiv.org/abs/2503.21003) [V]
- **(a,e) Forensic Twins.**
  - Method: a frozen FSD residual extractor (8-channel learned filter bank; SRM ablation 84.14% vs 93.77%). Two spatially disjoint crops of variable size, ViT-Tiny about 3M parameters (patch size 8), Barlow Twins loss. Trained on 491,690 real ImageNet+COCO images with no generated data.
  - Results: a GMM on real embeddings gives 97.99% AUC across 27 generators (GAN 98.17, diffusion 98.56, commercial 96.10), with attribution clustering at 56.61%.
  - Limits: no localisation or pairwise crop-verification experiments. Re-encoding drops AUC to 61–71. Commercial generators are the weakest family.
  — [arXiv 2609.31514](https://arxiv.org/html/2609.31514v1) [V]
- **(e) DiffusionPrint as a pairwise comparator trained on generated data.** It uses MoCo contrast between regions from the same and different generators, with hard negatives. — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443) [V]

### Inferences
- **Could a real-only same-source model on residuals be the comparator?** Partly. The Forensic Twins objective (disjoint crops of one real image should match) is exactly a stationarity prior, so the embedding distance between two regions of one image is a natural self-consistency score. [H] Four caveats:
  1. It was never evaluated pairwise or at patch level.
  2. It is sensitive to re-compression.
  3. Training only on camera images teaches it "same camera pipeline". Under full re-decode both regions come out of the same decoder, which may look like the same source. This would reproduce your near-chance result on GPT-Image-2. [H]
  4. Proposed fix: add positives made of VAE round-trip images (both crops round-tripped) and negatives made of round-trip versus generated crops. The latter can be built by running inpainting and then INP-X-style or reverse compositing. The model then learns "same decoding origin" rather than "same camera". This breaks the strict real-only rule but needs no specific editor. [H; motivated by arXiv 2602.00192]
- **(c) Content:** disjoint-crop training plus residual input suppresses semantics by design. That is a key advantage over raw DINO tokens. [V for the design intent]
- **(d) Sample size:** Forensic Similarity used fixed small patches (I did not retrieve exact sizes). Forensic Twins uses variable crops. Expect useful comparisons at 64×64 or more of residual. [H]
- **(f) Verdict:** **High priority** as the comparator. Fine-tune a Forensic Twins / FSD-residual ViT with a disjoint-crop objective, adding VAE-RT positives and composite negatives. Score region pairs by embedding distance, with an optional small similarity head as in Mayer & Stamm. Fuse with DINO-drift.

### Gaps
- Exact Forensic Similarity patch sizes and localisation numbers were not retrieved.
- EXIF-consistency mAP values were not retrieved.
- No pairwise model has been evaluated on MLLM re-decode editors.

---

## 5. What is newer or better than handcrafted descriptors in 2025–2026, and the synthesis for your pipeline

### Takeaway
The 2025–2026 frontier moves away from hand-picked residual statistics toward three kinds of method:
1. Frozen foundation-model perturbation responses (TRAIL, WaRPAD).
2. Self-supervised forensic embeddings trained only on real images (FSD, Forensic Twins).
3. Learned generator fingerprints and simple physics-motivated maps fed into SAM-style decoders (DiffusionPrint, FLAME).

All of them degrade sharply on instruction/MLLM editors compared with paste-back benchmarks.

### Cited Findings
- **Paste-back versus re-decode gap.** FLAME base IoU drops from about 0.47–0.50 on CoCoGLIDE/AutoSplice to 0.22–0.32 on NanoBanana, Qwen-Image and Flux Kontext. — [arXiv 2606.02178](https://arxiv.org/html/2606.02178v1) [V]
- **Detectors exploit global VAE artefacts.** Current detectors rely on global VAE artefacts rather than on local synthesis, and become near chance once the unedited pixels are restored. — [arXiv 2602.00192](https://arxiv.org/abs/2602.00192) [V]
- **Supervised in-domain training is still strong.** IML-ViT trained in domain reaches F1 about 0.81 on LocateEdit-Bench. — [arXiv 2602.05577](https://arxiv.org/html/2602.05577v1) [V]
- **Other 2026 localisers seen but not read in full:**
  - GAP-SAM, a global artifact prior for out-of-distribution AIGC manipulation localisation — [arXiv 2608.20929](https://arxiv.org/abs/2608.20929) [V, title only]
  - Iterative manifold-deviation amplification — [arXiv 2602.18842](https://arxiv.org/abs/2602.18842) [V, title only]

### Inferences: ranked recommendation for the region descriptor vector [H]
1. **DINO perturbation drift (TRAIL-style).** Use a full-canvas encode and sweep layers, since late layers may beat layer 6 for drift. This should be the primary cue.
2. **Learned pairwise residual embedding** (Forensic Twins / FSD backbone), fine-tuned with VAE round-trip positives. Use it as the region comparator in place of a Mahalanobis distance over hand features.
3. **Content-normalised local-energy / coding-cost residual** (LAD 3×3, per-pixel ZED-style surprisal), regressed on texture.
4. **Paste-back-only cues** (NLF noise, JPEG grid, CFA). Gate them by a global "was the image re-decoded?" test, for example AEROBLADE-style reconstruction distance on the whole image.
5. **Drop or deprioritise** bispectrum, multifractal exponents, RMT and scattering until a pilot shows signal.

For re-decoded images the authentic region's statistics are VAE round-trip statistics. The null model for "authentic" should therefore be estimated after re-encoding the image through a candidate VAE, or learned with VAE-RT augmentation. [H]

### Gaps
- No public per-region benchmark separates paste-back, partial-latent-blend and full-re-decode editors with descriptor-level ablations.
- GPT-Image-2 behaviour is undocumented in the literature I found.
- Exact numbers for several classical methods ([BK] items) were not re-verified this session.
