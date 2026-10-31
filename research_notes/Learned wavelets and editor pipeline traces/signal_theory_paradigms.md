# Signal-theoretic, statistical and information-theoretic paradigms for localizing generative edits

Scope note: this researcher covered harmonic analysis, higher-order statistics, information theory, statistical image models, geometry/topology and colour/CFA paradigms. Tool budget was about 15 calls, so many classic references are cited from prior knowledge and marked **[not verified this session]**. Items labelled **HYPOTHESIS** are untested ideas. They are not findings.

Context from the user's own experiments (the report writer should keep this in mind): pooled SRM, VAE reconstruction error, DT-CWT/shearlet statistics and a learned-lifting surprisal model reached per-image AUC of only 0.62-0.68, with about 0.55 cross-dataset and chance on GPT-Image-2 edits. DINOv2 patch features reached 0.77 (transfer 0.68).

## Q0. Framing: why decoder-fingerprint statistics fail on fully regenerated local edits

### Takeaway
When the editor re-decodes the whole image (latent inpainting with full VAE decode, instruction editors, autoregressive MLLM editors), the "camera vs. generator" signal exists everywhere in the image. The edited-vs-unedited contrast then shrinks to "re-encoded content" vs "content synthesized from noise/tokens". Published benchmarks confirm that classic IFL fails in this regime.

### Cited Findings
- TGIF (about 75k forgeries from SD2, SDXL and Adobe Firefly): traditional IFL methods detect spliced inpaintings but fail on regenerated ones. SID methods flag regenerated images as fake but cannot localize the edit. — [TGIF, arXiv 2407.11566](https://arxiv.org/abs/2407.11566)
- TGIF2 (March 2026) adds FLUX.1 inpainting and random non-semantic masks. IFL and SID both degrade on FLUX.1. Fine-tuning IFL on fully regenerated images helps, but random masks reveal **object bias**: detectors learn "where an object is" rather than forensic traces. Generative super-resolution "significantly weakens forensic traces". — [TGIF2, arXiv 2603.28613](https://arxiv.org/abs/2603.28613)
- AEROBLADE: LDM-generated images are reconstructed more accurately by the LDM autoencoder than real images. This allows training-free detection and a qualitative localization of inpainted regions. — [AEROBLADE, arXiv 2401.17879](https://arxiv.org/abs/2401.17879)
- Impostor (June 2026): a 100K-image AIGC-edit localization benchmark built with 7 recent AIGC models and 3 manipulation types. The authors propose PANet, which uses **local phase modelling** (monogenic/Riesz-based local phase cues, per the search snippet) plus semantic-forensic consistency. They claim superior performance, but numbers were not retrieved. — [Impostor, arXiv 2606.04545](https://arxiv.org/abs/2606.04545)

### Inferences
- The TGIF2 object-bias finding is consistent with DINOv2 doing best in the user's tests (0.77). Part of DINOv2's signal may be semantic "objectness", not forensic evidence. Any new paradigm should be evaluated on random-mask edits to separate the two.
- In a fully regenerated image, decoder-level statistics (spectral peaks, VAE error, CFA absence) are roughly constant across the image. Only statistics that depend on *how content was synthesized* can separate edited from unedited regions. Examples: inter-scale dependency, phase coupling, local regularity, and prior likelihood or score curvature under a model of real content. This argument rests on inference from the TGIF/TGIF2 results and is not a published result.
- For GPT-Image-2-style autoregressive editors that re-synthesize the full image, no pixel-level "camera" region survives. Local differences can only come from content-statistics mismatch, e.g. the "copied" region being reproduced at a different fidelity from the newly synthesized region. **HYPOTHESIS.**

### Gaps
- No public per-region statistics (spectral slope, kurtosis, phase coupling) for edited vs. preserved areas in fully regenerated edits were found.

## Q1. Higher-order spectra (bispectrum, bicoherence, trispectrum, phase coupling)

### Takeaway
Bicoherence is a classic splicing cue with modest accuracy (62% to 70-72%). One small 2023 paper reports about 80% for GAN-image detection. **No test on diffusion, flow or AR edits, and no dense local bicoherence map for generative edits, was found.** The area is largely untested.

### Cited Findings
- Ng, Chang & Sun (2004): plain bicoherence magnitude and phase features give **62%** accuracy (SVM) for authentic vs. spliced 128x128 blocks. Improved features reach **about 70%** (ISCAS 2004 abstract) or **72%** (technical report / Columbia project page). — [Columbia TR](https://www.ee.columbia.edu/ln/dvmm/publications/04/TR_splicingHOS_ttng.pdf); [Columbia splicing project](https://www.ee.columbia.edu/ln/dvmm/trustfoto/projs/splicing/splicing.htm)
- Columbia splicing dataset: 1845 blocks of 128x128 (933 authentic, 912 spliced). — [Dataset details](https://www.ee.columbia.edu/ln/dvmm/downloads/AuthSplicedDataSet/detailed.htm)
- Nikitenkova (2023), "Deepfake image detection using bispectral analysis": higher-order spectral correlations are *weaker* in natural images than in GAN images. Clustering identifies generated images with **80%** accuracy; dataset size is not given in the abstract. — [Sistemy i Sredstva Informatiki 33(2)](https://m.mathnet.ru/eng/ssi881)
- A 2006 patent / related work combines wavelet characteristic-function moments with 2-D phase congruency statistics for splicing detection. — [USPTO 7991185](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7991185)
- Bispectral analysis has been used to detect AI-synthesized *speech* (AlBadawy, Lyu, Farid, CVPRW 2019). **[not verified this session]**

### Inferences
- Principle: the bispectrum B(f1,f2)=E[X(f1)X(f2)X*(f1+f2)] measures quadratic phase coupling, which second-order (power-spectrum) features cannot see. Edges and natural structures produce consistent phase alignment across scales.
- **HYPOTHESIS:** Diffusion denoisers synthesize texture through learned nonlinear maps. Edited regions may therefore show a different degree of cross-frequency phase coupling than VAE-re-encoded real content, where the coupling was inherited from the camera scene. The dense version would be a local (windowed, e.g. 32-64 px) bicoherence or cross-scale phase-coupling map, e.g. from complex wavelet coefficients (DT-CWT relative phase).
- Cost: bispectrum is O(N^2) per window in 2-D (a 4-D object), so a reduced form is needed (radial/diagonal slices, or wavelet-based bicoherence). Variance is high in small windows.
- Robustness: JPEG quantization removes high-frequency phase. Resize changes frequency indexing but preserves coupling structure at lower bands. Expect moderate JPEG fragility. No numbers were found.
- The user's DT-CWT statistics were magnitude-pooled. Phase coupling (relative phase across scales) is a distinct and untested feature.

### Gaps
- No trispectrum forensics papers were found.
- No bicoherence evaluations on SD/FLUX/GPT-Image edits were found.

## Q2. Multifractal analysis, Hölder regularity, 1/f and local spectral slope

### Takeaway
Spectral decay at high frequency is a documented (but contested) GAN cue that is fragile to compression. Wavelet-leader multifractal analysis has a pixel-wise anomaly-detection framework from signal processing. **No application to AI-image or AI-edit detection was found.**

### Cited Findings
- Dzanic, Shah & Witherden (NeurIPS 2020): real images show large decay rates in the highest Fourier modes. Generated images show small decay rates that vary with the model. Detection accuracy is **up to 99.2%** across GAN/VAE models. Differences are larger at high resolution, and **lossy compression reduces them**. — [arXiv 1911.06465](https://arxiv.org/abs/1911.06465)
- Chandrasegaran, Tran & Cheung (CVPR 2021): high-frequency spectral decay discrepancies are "not intrinsic" to CNN-generated images, so caution is needed. — [arXiv 2103.17195](https://arxiv.org/abs/2103.17195)
- Durall et al. (2020) attribute GAN spectral failure to transposed-convolution upsampling. — cited via [Dzanic/Chandrasegaran search summary](https://arxiv.org/abs/2103.17195); original arXiv 2003.01826 **[not verified this session]**
- Wendt, Leon Arencibia, Tourneret, Abry (ICIP 2022): **pixel-wise multifractal anomaly detection** using a wavelet-leader statistical model, space-scale surrogates simulating hypothesized global multifractality, and multiple hypothesis testing. It is validated only on synthetic multifractal images. — [HAL hal-03735492](https://hal.archives-ouvertes.fr/hal-03735492); [PDF](https://www.tesa.prd.fr/documents/26/mf_anomaly_surrogates-vf.pdf)
- Wavelet-leader multifractal analysis of images with bootstrap confidence intervals (Wendt, Roux, Jaffard, Abry, Signal Processing 2009). — [HAL ensl-00365041](https://hal-ens-lyon.archives-ouvertes.fr/ensl-00365041)
- R implementation: wmlf. — [rdrr.io/cran/wmlf](https://rdrr.io/cran/wmlf/)
- SPAI (CVPR 2025) learns the spectral distribution of real images by masked frequency reconstruction. It reports a **+5.5% absolute AUC** over prior SOTA across 13 generators and robustness to online perturbations. It is image-level, not dense. — [arXiv 2411.19417](https://arxiv.org/abs/2411.19417)

### Inferences
- Principle: local Hölder exponent h(x) and the singularity spectrum D(h) describe how regularity varies spatially. Natural images are multifractal with characteristic log-cumulants c1 and c2 (c2 < 0). A denoiser with a learned smoothness prior may produce a narrower spectrum (|c2| smaller) or a shifted c1 in synthesized regions.
- **HYPOTHESIS (strong candidate):** Use the Wendt et al. surrogate-testing framework directly as a *dense, training-free* edit map. Estimate the global multifractal model from the whole image, then flag pixels whose local wavelet-leader log-cumulants are inconsistent with it. This method was designed for exactly "local anomaly in an otherwise homogeneous multifractal field" and has never been applied to forensics.
- The local spectral slope (local 1/f^beta exponent in sliding windows) is a cheap proxy. Dzanic shows compression fragility at the highest frequencies, so estimate beta over mid bands.
- Cost: wavelet leaders are O(N log N) and cheap. Robustness: JPEG truncates fine scales, so restrict the regression to scales above 8x8 block size. Resize shifts the scale axis but preserves the slope if it is re-indexed.

### Gaps
- No multifractal or Hölder results on any AI-image benchmark were found.
- It is unknown whether VAE re-encoding of preserved regions already homogenizes regularity enough to mask the contrast.

## Q3. Analytic-signal and adaptive decompositions (monogenic/Riesz, phase congruency, EMD/HHT, steerable/Gabor, FrFT)

### Takeaway
Monogenic local phase has just entered AIGC edit localization (Impostor/PANet 2026, learned and combined with semantics). HHT/EMD and phase congruency are 2006-era splicing features (about 86% accuracy, small datasets). Steerable pyramids underpin classical natural-image statistics. Fractional Fourier transform forensics for generative edits was not found.

### Cited Findings
- The monogenic signal generalizes the analytic signal to 2-D by replacing the Hilbert transform with the Riesz transform, giving local amplitude, phase and orientation (Felsberg & Sommer). — [Kiel notes](https://www.informatik.uni-kiel.de/inf/Sommer/doc/Publications/mfe/D2k.pdf)
- Impostor/PANet (arXiv, June 2026) uses local phase modelling for AIGC manipulation localization. Per the search snippet, it builds multi-scale band-limited responses converted to local phase maps via the Riesz transform. Quantitative results were not retrieved. — [arXiv 2606.04545](https://arxiv.org/abs/2606.04545)
- Fu, Shi & Su (IWDW 2006): HHT (EMD into IMFs plus Hilbert spectral analysis) features combined with wavelet characteristic-function moments and an SVM for splicing detection. The rationale is that splicing is nonlinear and non-stationary. A secondary source reports **85.86%** accuracy, vs 92% for a DCT Markov method; treat these figures with care. — [NJIT record](https://researchwith.njit.edu/en/publications/detection-of-image-splicing-based-on-hilbert-huang-transform-and-/); [Unpaywall DOI 10.1007/11922841_15](https://unpaywall.org/10.1007%2F11922841_15)
- 2-D EMD combined with Riesz/monogenic analysis is described by the Flandrin group (ICASSP). — [Flandrin ICASSP PDF](https://perso.ens-lyon.fr/patrick.flandrin/ICASSP_emd.pdf)
- Phase congruency statistics for splicing: USPTO 7991185 (Shi et al.). — [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7991185)

### Inferences
- Principle: phase congruency (PC) is contrast-invariant. At genuine image features, Fourier components at all scales are in phase. Real optics (diffraction-limited PSF, defocus) produce a specific relation between PC, local energy and scale. A generator's synthesized edges may be "too congruent" (over-sharp) or less congruent across scales, and the decoder's upsampling may break scale-consistent phase.
- **HYPOTHESIS:** A training-free dense map of *cross-scale phase consistency* could be built as the variance of monogenic local phase/orientation across scales, normalized by local energy. It is complementary to the user's magnitude-based DT-CWT/shearlet statistics, which discard phase. Impostor shows phase helps when learned, but there is no training-free, generator-agnostic test.
- EMD/EWT: these are adaptive and data-driven, but slow (iterative sifting, roughly O(N) per iteration times many iterations), unstable at borders, and subject to mode mixing. Low priority.
- Fractional Fourier / Gabor / steerable pyramids are mostly re-parameterizations of the same time-frequency information the user already tested with DT-CWT/shearlets. Expected marginal gain is low unless phase or higher-order dependency is used. **HYPOTHESIS.**

### Gaps
- PANet numbers and its ablation (phase vs no phase) were not retrieved.
- No FrFT image-forensics paper relevant to generative edits was found.

## Q4. Information-theoretic paradigms (lossless-coder surprisal, NCD, MDL, rate-distortion, MI across scales/channels)

### Takeaway
Lossless-coder surprisal (ZED) is the established zero-shot information-theoretic detector, at image level. The user's learned-lifting surprisal is a variant of it and underperformed. Untested directions are *conditional/relative* information measures. Examples: surprisal of a region under a coder fitted on the rest of the same image, rate-distortion curve shape, and inter-channel or inter-scale mutual information maps.

### Cited Findings
- ZED (Cozzolino, Poggi, Nießner, Verdoliva, ECCV 2024) uses a multi-resolution lossless encoder trained only on real images to measure per-pixel surprisal. A *single* feature beats SOTA by **more than 3% average accuracy** across many generators. — [arXiv 2409.15875](https://arxiv.org/abs/2409.15875)
- Forensic Self-Descriptions (Nguyen, Azizpour, Stamm, CVPR 2025) learn predictive filters on real images only. A compact per-image model of multi-scale residuals (the "self-description") enables zero-shot detection, open-set attribution and clustering. — [arXiv 2503.21003](https://arxiv.org/abs/2503.21003)

### Inferences
- Principle: under a model p of real images, the code length -log p(x) is lower for over-smooth or "typical" generator output, or higher for out-of-distribution pixels. Generators sample near high-density modes, so their outputs are often *more compressible* than camera images. ZED exploits this, inspired by LLM-text detection.
- Why it can fail on local edits: a coder trained on real images rates both a VAE-re-encoded background and a synthesized region as "generator-like". The global offset dominates and the local contrast is small. This is consistent with the user's 0.62-0.68.
- **HYPOTHESIS (promising):** Use a *self-referential* or *relative* description length. Fit (or adapt) a lightweight context model on the image's own pixels outside a window, then measure the window's code length relative to its in-image expected code length (cross-entropy minus own-entropy). This is similar in spirit to NCD between a patch and the rest: NCD(a,b) = [C(ab) - min(C(a),C(b))]/max(C(a),C(b)). It cancels the global "decoded-by-VAE" offset. Forensic Self-Descriptions shows per-image fitted residual models are informative; making them *spatially local* is untested.
- **HYPOTHESIS:** Use MDL segmentation. Choose the two-region partition that minimizes the total code length of a per-region residual model (as in classic MDL image segmentation). This gives a dense, principled map with no detector training.
- **HYPOTHESIS:** Use rate-distortion behaviour. Measure the local slope of the operational RD curve, e.g. re-encode with JPEG/JPEG-XL at several qualities and measure per-block distortion vs rate. Synthesized textures may lie on a different RD curve because their high frequencies are "procedural" rather than sensor noise. It is cheap and robust only if the input has not been heavily compressed already.
- **HYPOTHESIS:** Use inter-channel MI. Measure the local mutual information between R/G/B residuals or between wavelet scales (parent-child). Camera demosaicing plus ISP creates specific cross-channel dependence, while the decoder produces another. Transfer entropy across scales is a directional version that is more exotic, noisier and lower priority.

### Gaps
- No published localization results for NCD/MDL/RD-curve methods on AI edits were found.
- ZED localization maps on inpainting were not checked this session.

## Q5. Statistical image models (GSM, FoE/MRF, Portilla-Simoncelli, noise level and random-matrix covariance, Benford)

### Takeaway
Classic natural-image statistics distinguished CG from photos (Lyu & Farid). Benford's law on DCT detects GANs in low-data regimes. Blind local-noise estimation is a proven *dense* splicing localizer. None of these were found tested on diffusion/AR *local edits*.

### Cited Findings
- Bonettini, Bestagini, Milani, Tubaro (ICPR 2020): Benford first-digit statistics of quantized DCT coefficients form a compact feature with a simple classifier for GAN detection. It works in data-scarce settings where CNNs fail. Results are on GANs only. — [arXiv 2004.07682](https://arxiv.org/abs/2004.07682)
- Lyu, Pan, Zhang (IJCV 2014): blind local noise estimation exploits the near-constant kurtosis of natural images in band-pass domains. It has a closed-form solution, is accelerated with integral images, and detects spliced regions by segmenting local noise variance. This is a **dense** map. — [PDF](https://cse.buffalo.edu/~siweilyu/papers/ijcv14.pdf); ICCP 2012 version [PDF](https://cse.buffalo.edu/~siweilyu/papers/iccp12.pdf)
- Statistical analysis of signal-dependent noise for blind splicing localization (follow-up line). — [arXiv 2010.16211](https://arxiv.org/abs/2010.16211)
- Lyu & Farid, "How realistic is photorealistic?" (IEEE TSP 2005): wavelet higher-order statistics (marginal and linear-prediction-error statistics across scales and orientations) separate CG from photos. **[not verified this session]**
- Portilla & Simoncelli (IJCV 2000) parametric texture model: joint statistics of steerable-pyramid coefficients (auto-correlations, cross-scale magnitude/phase correlations). **[not verified this session]**
- PCA/eigenvalue-based noise level estimation from patch covariance (e.g. Chen, Zhu, Heng ICCV 2015, which uses the trailing eigenvalues of patch covariance with Marchenko-Pastur-type reasoning). **[not verified this session]**

### Inferences
- Principle (GSM): wavelet coefficients behave as x = sqrt(z)u, a Gaussian times a hidden multiplier. This yields heavy tails and magnitude dependencies between neighbours, parents and orientations. Denoisers learn approximations of these priors. Over-regularized synthesis may show *too-regular* GSM multipliers, i.e. lower kurtosis and different parent-child magnitude correlation. The user's DT-CWT stats likely captured marginals, not *joint* GSM dependencies.
- **HYPOTHESIS (strong):** Build a random-matrix local covariance spectrum map. For each window, compute the eigenvalue spectrum of the covariance of small (e.g. 7x7) patches. Camera regions have a noise floor of eigenvalues following a Marchenko-Pastur bulk set by sensor noise (signal-dependent). Generator-synthesized regions in a fully regenerated image may have *no* MP bulk, a different bulk edge, or a deficit of small eigenvalues (the decoder's low effective rank). The bulk edge, the number of spikes above the edge, and the KS distance to MP are all dense, training-free and cheap (SVD of 49x49 per window). It is directly analogous to the noise-inconsistency principle of Lyu et al., but sensitive to *rank structure*, not just variance.
- Caveat: in a fully regenerated edit, background "sensor noise" was itself re-synthesized by the VAE decoder. The contrast then becomes "VAE-reconstructed noise (from the real latent)" vs "denoiser-generated noise". Whether these differ in eigen-structure is unknown. **HYPOTHESIS.**
- Benford: re-JPEG and the final save typically impose double-quantization patterns that dominate first-digit statistics. Locally (per 64x64) the digit histograms are noisy. Low priority for dense maps, but cheap.
- Field-of-Experts/MRF priors: local energy under a learned FoE is a cheap "prior likelihood map". It is conceptually a hand-crafted version of score/likelihood methods and is untested on generative edits.

### Gaps
- No RMT-based forensic papers were found in this session. It is unknown whether any exist.
- No GSM-joint-statistics maps on diffusion edits were found.

## Q6. Geometric / topological / score-based (persistent homology, optimal transport, Fisher information, score curvature)

### Takeaway
Persistent homology has small-scale forensic use (morphing, inpainting tamper detection with LBP features, one group) but no modern-generator benchmarks. Score-function curvature from a pretrained diffusion model is a 2025 zero-shot detector at image level. Its local form is untested.

### Cited Findings
- Asaad & Jassim (Buckingham): persistent homology on LBP landmarks (Betti 0/1) for morphed-face detection, steganalysis and tamper detection. — [Buckingham repository](https://bear.buckingham.ac.uk/467/)
- Al-Jaberi, Asaad, Jassim (2020): TDA with LBP and kNN detects tampered (inpainted) images. — [Academia.edu](https://www.academia.edu/96643634/Topological_Data_Analysis_for_Image_Forgery_Detection)
- A student paper proposes TDA for AI-generated faces. Its results are framed as future work. — [CAUSEweb PDF](https://www.causeweb.org/usproc/sites/default/files/usresp/2024-1/usresp%203357%20-%20topological%20data%20analysis%20for%20classification%20of%20ai%20generated%20faces.pdf)
- Brokman et al. (ICLR 2025): use the score function of a pretrained diffusion model to approximate curvature, gradient and bias toward points on the probability manifold. This gives zero-shot and few-shot detection, outperforming approaches across **20 generative models**. — [arXiv 2504.15470](https://arxiv.org/abs/2504.15470)

### Inferences
- **HYPOTHESIS:** Build a local score/curvature map. Compute the Brokman criteria per pixel or patch, e.g. ||score|| and Hessian-trace proxies via Hutchinson estimates at small noise levels. Synthesized regions should sit closer to manifold modes, giving smaller score norm and higher curvature. This is the dense analogue of an image-level result. Cost is high (multiple U-Net/DiT passes) but feasible. Robustness to JPEG is unknown. Risk: in a fully regenerated image the whole image is near-manifold.
- **HYPOTHESIS:** Use optimal transport between local feature distributions. Compute a sliced Wasserstein distance between the distribution of residual or wavelet coefficients in a window and that of the global image (or a reference camera distribution). This gives a principled two-sample statistic that is denser and more sensitive than comparing pooled moments. It is cheap with sliced-W. It complements DINOv2 because it can run on DINOv2 patch tokens too, e.g. OT distance between a patch token neighbourhood and the image's token distribution.
- Persistent homology of local patch point clouds (Carlsson-style Klein-bottle structure of 3x3 high-contrast patches) is mathematically interesting. Generator textures could deviate from the natural-patch manifold topology. It is costly (Vietoris-Rips) and statistically noisy for small windows. Low-medium priority. **HYPOTHESIS.**

### Gaps
- No TDA or OT results on diffusion/AR edits were found.
- Brokman localization was not reported in the abstract.

## Q7. Colour science and demosaicing (CFA, chromatic aberration, inter-channel correlation)

### Takeaway
CFA and colour-correlation cues are re-emerging for AI-image detection (CVPR 2025, Jan 2026). Classic CFA and lateral chromatic aberration (LCA) methods give dense splicing maps. For *fully regenerated* edits, CFA traces are destroyed everywhere, so only inter-channel *statistics* (not periodic CFA) remain as possible local cues.

### Cited Findings
- "Secret Lies in Color" (Jia et al., CVPR 2025): synthetic images show uneven colour distribution. A colour quantization/restoration detector with 1.4M parameters reports SOTA on several benchmarks. — [summary](https://sotaverified.org/papers/secret-lies-in-color-enhancing-ai-generated) (full text not checked)
- DCCT, "Color Matters: Demosaicing-Guided Color Correlation Training" (arXiv, Jan 2026): simulates Bayer sampling and trains a model to predict the missing colour channels. It reports strong generalization across **more than 20 unseen generators**. — [arXiv 2601.22778](https://arxiv.org/abs/2601.22778)
- Popescu & Farid (IEEE TSP 2005), CFA interpolation correlations via EM: dense map. Ferrara et al. (IEEE TIFS 2012), fine-grained CFA artifact localization. **[not verified this session]**
- Johnson & Farid (2006) and Mayer & Stamm (IEEE TIFS 2018): lateral chromatic aberration inconsistency for forgery localization. **[not verified this session]**

### Inferences
- In latent inpainting with paste-back, unedited pixels keep CFA/LCA traces, so classic CFA maps should work (the TGIF finding for spliced images is consistent). In fully regenerated edits, the VAE decoder re-synthesizes all pixels and periodic CFA correlations vanish globally, giving no local contrast.
- **HYPOTHESIS:** LCA could survive full regeneration partially. LCA is a *geometric* radial misalignment of colour channels at scene edges, at a scale of several pixels. A VAE reconstructing a real latent may preserve it in unedited regions, while synthesized content will not follow the image's global radial LCA model. This is dense at edges, cheap and moderately JPEG-robust (low-frequency geometric). It is untested on generative edits.
- **HYPOTHESIS:** Use a DCCT-style "predict missing channel" residual as a *dense* map. Residual magnitude per pixel under a demosaicing-consistency model trained on real RAW-to-RGB could localize regions without camera-like colour correlation.

### Gaps
- No LCA or CFA results on TGIF/FLUX/GPT-Image edits were found.

## Q8. "Hand-crafted is back" (2024-2026)

### Takeaway
The evidence that classical statistics *beat* deep detectors on modern generators is thin. The strongest 2024-2026 results are *self-supervised models of real-image statistics* (ZED, SPAI, Forensic Self-Descriptions, DCCT) and *physics/analytic priors* (score curvature). These are "model real images, flag deviations", not handcrafted features per se.

### Cited Findings
- Nirob et al. (arXiv Jan 2026): handcrafted fusion (HOG, LBP, GLCM, wavelets, histograms) with LightGBM gives ROC-AUC **0.9878** and PR-AUC 0.9879 on **CIFAKE only**, with no cross-generator testing. This is weak evidence. — [arXiv 2601.19262](https://arxiv.org/abs/2601.19262)
- ZED: a single entropy feature, trained only on real images, beats SOTA by more than 3%. — [arXiv 2409.15875](https://arxiv.org/abs/2409.15875)
- SPAI: spectral distribution of real images as an invariant, +5.5% AUC over 13 generators. — [arXiv 2411.19417](https://arxiv.org/abs/2411.19417)
- Forensic Self-Descriptions: real-only, zero-shot. — [arXiv 2503.21003](https://arxiv.org/abs/2503.21003)
- Brokman et al. (score curvature, zero-shot, 20 generators). — [arXiv 2504.15470](https://arxiv.org/abs/2504.15470)
- Survey: established detectors often fail to extend to diffusion models even when retrained. — [arXiv 2502.19716](https://arxiv.org/html/2502.19716v1)
- Synthbuster (Bammey 2023) exploits diffusion frequency artefacts. — [IP Paris record](https://researchportal.ip-paris.fr/en/publications/synthbuster-towards-detection-of-diffusion-model-generated-images/)

### Inferences
- The common denominator of the successful methods is *one-class modelling of real-image statistics plus a divergence score*. For local edits in fully regenerated images, the reference must be the image's *own* preserved content (self-referential), not a corpus of camera images. Otherwise the global VAE offset swamps the local signal.

### Gaps
- No 2024-2026 paper was found showing classical statistics beating deep detectors on *local edit localization*.

## Q9. Ranked shortlist of most promising untested ideas (all HYPOTHESES)

### Takeaway
Rank by: (a) sensitivity to *how* content was synthesized rather than *which decoder*; (b) a natural dense/self-referential form that cancels the global regeneration offset; (c) cost; (d) robustness. The top picks are self-referential multifractal/RMT/OT tests and phase-coupling maps, plus local score curvature as a heavier option.

### Cited Findings
- Basis for ranks 1, 2 and 5: surrogate-based pixel-wise multifractal anomaly testing exists ([Wendt et al. ICIP 2022](https://hal.archives-ouvertes.fr/hal-03735492)). Noise-statistics inconsistency is a proven dense localizer ([Lyu et al. IJCV 2014](https://cse.buffalo.edu/~siweilyu/papers/ijcv14.pdf)). Real-only self-description models work at image level ([arXiv 2503.21003](https://arxiv.org/abs/2503.21003)). Local phase helps learned AIGC localization ([arXiv 2606.04545](https://arxiv.org/abs/2606.04545)). Score curvature works zero-shot at image level ([arXiv 2504.15470](https://arxiv.org/abs/2504.15470)). TGIF2 object bias argues for random-mask evaluation ([arXiv 2603.28613](https://arxiv.org/abs/2603.28613)).

### Inferences
1. **Wavelet-leader multifractal surrogate test as a dense map.** Fit a global multifractal model, flag pixels with inconsistent local log-cumulants (c1, c2). Why: training-free, inherently self-referential (cancels global VAE offset), probes regularity statistics that denoisers alter, cheap. It is completely untested in forensics. Use scales above the JPEG block size.
2. **Random-matrix local patch-covariance spectrum map.** Use MP bulk-edge fit, spike count and small-eigenvalue deficit per window. Why: generalizes the proven noise-inconsistency principle to rank structure, which captures decoder/denoiser low-rank synthesis. Cheap and dense. Untested.
3. **Cross-scale phase coupling map** (local bicoherence on DT-CWT/monogenic relative phase, or phase-congruency consistency across scales). Why: the user's failed features were magnitude-only. Phase coupling is a second-order-invisible property. Impostor suggests phase carries edit signal, but no training-free test exists. Medium cost, JPEG-fragile at fine scales.
4. **Self-referential description length / MDL two-region segmentation** (code length of a window under a context model fitted on the rest of the same image, or NCD vs rest). Why: directly fixes the failure mode of ZED-like surrogate models (global offset). Medium cost.
5. **Local score-curvature / Fisher-information map from a pretrained diffusion model.** Why: the strongest zero-shot image-level evidence (20 generators), and the dense version is natural. High cost. The risk is that fully regenerated images are near-manifold everywhere. The most likely of these to work on GPT-Image-2, because it targets content plausibility rather than decoder traces.
6. **Sliced-Wasserstein two-sample maps** on residuals *and* DINOv2 tokens. Why: a principled local-vs-global divergence that can improve the user's best feature (DINOv2) by replacing a classifier with a distribution test. This could improve transfer. Cheap.
7. **Lateral chromatic aberration consistency.** Why: a geometric, low-frequency cue that may survive VAE re-encoding of preserved regions. Edge-only density. Cheap.
- Lower priority: Benford (noisy locally, dominated by final JPEG), EMD/HHT (cost, instability), FrFT/Gabor (redundant with tested DT-CWT/shearlets), persistent homology (cost and variance), transfer entropy (estimation noise).
- Evaluation advice for any of the above: report per-image AUC on **random non-semantic masks** (TGIF2 style) as well as object masks, to separate forensic signal from object bias. Fuse with DINOv2 only after demonstrating standalone above-chance localization on random masks.

### Gaps
- None of the ranked ideas has published numbers on generative edits. All rankings are reasoned hypotheses.
- Robustness numbers (JPEG/resize) exist only for Dzanic-style spectral decay (qualitative degradation) and SPAI (claimed robust). Every other robustness statement here is an inference.
