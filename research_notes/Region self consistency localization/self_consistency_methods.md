# Region Self-Consistency Forgery Localization: History and State of the Art (2018 – Oct 2026)

Scope note: "self-consistency" here means each region or patch is treated as a sample, and regions whose forensic statistics disagree with the rest of the same image are flagged. Items I confirmed this session (abstracts or full text fetched) carry a link. Items marked **[UNVERIFIED, prior knowledge]** are bibliographic or numeric facts from background knowledge that I did not re-check here. The report writer should either drop them or re-check them.

---

## Q1. Classic and learned self-consistency methods: how each defines regions, what it compares, how it decides "inconsistent", and results

### Takeaway
The paradigm has three generations. (1) Hand-crafted local statistics (noise variance, CFA, JPEG grid/DCT zeros), each compared with an image-wide model, mostly with statistical tests. The IPOL a-contrario line (ZERO, Noisesniffer) puts this on firm ground with NFA control. (2) Learned pairwise "same source?" functions: Huh et al.'s EXIF self-consistency, Mayer & Stamm's Forensic Similarity and its graphs, EXIF-as-language. These are aggregated by clustering or community detection. (3) Learned camera fingerprints (Noiseprint, Noiseprint++), whose local deviations are read by a supervised head (TruFor). On camera-pipeline splices the generations progress clearly. On diffusion-inpainted images the unsupervised members fall to F1 ≈ 0.40–0.43 even on CocoGlide (TruFor Table 1).

### Cited Findings

**Learned self-consistency (2018–2023)**
- Huh, Liu, Owens, Efros, "Fighting Fake News: Image Splice Detection via Learned Self-Consistency" (ECCV 2018). The method is trained only on real photos with EXIF metadata. It learns whether two patches could come from the same imaging pipeline, which addresses the scarcity of labelled manipulated data. — [arXiv 1805.04096](https://arxiv.org/abs/1805.04096)
  - Region, descriptor and decision **[UNVERIFIED, prior knowledge]**: 128×128 patches; a Siamese network predicts per-EXIF-attribute agreement; the pairwise consistency is averaged into a patch × patch affinity; mean-shift or normalized-cut clustering gives the localization map.
  - Numbers as re-measured by TruFor (EXIF-SC, pixel F1 best/fixed): Columbia .880/.798, DSO-1 .577/.442, CASIAv1+ .255/.106, NIST16 .298/.227, CocoGlide .424/.293, average .437/.303. Image-level AUC on CocoGlide is .526, essentially chance. — [TruFor, arXiv 2212.10957, Tables 1–2](https://arxiv.org/html/2212.10957v3)
- Mayer & Stamm, "Forensic Similarity for Digital Images" (TIFS 2020). A learned function decides whether two patches carry the same or different forensic traces. It needs no prior training samples of the target trace. — [arXiv 1902.04684](https://arxiv.org/abs/1902.04684)
- Mayer & Stamm, "Exposing Fake Images with Forensic Similarity Graphs". Detection and localization are recast as graph community detection. Patches are nodes, and edges are weighted by forensic similarity. — [arXiv 1912.02861](https://arxiv.org/abs/1912.02861)
  - Specifics **[UNVERIFIED, prior knowledge]**: published in IEEE JSTSP 2020; uses spectral clustering or modularity-based community detection; a forgery is declared when the graph splits into more than one community.
- Fang & Stamm (2022) show that Siamese self-consistency localizers can be fooled by synthetic forensic traces. "These algorithms identify spliced content by detecting localized inconsistencies in forensic traces using Siamese neural networks." Their attack makes spliced regions look consistent. — [arXiv 2211.12314](https://arxiv.org/abs/2211.12314)
- SISL (2022): self-supervised image-signature learning for splicing localization, which needs no masks, camera IDs or metadata. — [arXiv 2203.07824](https://arxiv.org/abs/2203.07824)
- Zheng, Shrivastava, Owens, "EXIF as Language" (2023). A multimodal embedding is learned between image patches and EXIF text, the text being processed by a transformer. Splices are localized "zero shot" by clustering the visual embeddings of all patches in an image. — [arXiv 2301.04647](https://arxiv.org/abs/2301.04647). The clustering algorithm and per-dataset numbers were not retrieved (the abs page has no table); see Gaps.
- FOCAL applies the same self-consistency idea to video coding parameters. — [arXiv 2008.10454](https://arxiv.org/abs/2008.10454)

**Noise-level and camera-fingerprint inconsistency**
- Mahdian & Saic, "Using noise inconsistencies for blind image forensics", Image and Vision Computing 2009. Wavelet (HH1) noise is estimated with a median-absolute-deviation estimator in blocks; blocks are merged by homogeneous noise level. **[UNVERIFIED, prior knowledge]**
- Pan, Zhang, Lyu, "Exposing image splicing with inconsistent local noise variances", ICCP 2012. Local noise variance is estimated from band-pass kurtosis concentration, then thresholded or clustered. **[UNVERIFIED, prior knowledge]**
- Lyu, Pan, Zhang, "Exposing region splicing forgeries with blind local noise estimation", IJCV 2014. The kurtosis-based estimator is computed with integral images, and per-pixel noise variance is clustered with k-means. **[UNVERIFIED, prior knowledge]**
- Cozzolino, Poggi, Verdoliva, "Splicebuster: a new blind image splicing detector", WIFS 2015. Co-occurrence features of quantized high-pass residuals are fitted with an image-wide GMM (pristine versus outlier), learned by EM on the test image itself. **[UNVERIFIED, prior knowledge]** TruFor's re-measurement: Splicebuster average pixel F1 .460/.297; CocoGlide .434/.332; image AUC on CocoGlide .529. — [arXiv 2212.10957](https://arxiv.org/html/2212.10957v3)
- Noiseprint (Cozzolino & Verdoliva, TIFS 2020) is a Siamese-trained camera-model fingerprint extractor; at test time anomalies are found with Splicebuster-style EM. **[UNVERIFIED, prior knowledge, arXiv 1808.08396]** TruFor's re-measurement: Noiseprint average pixel F1 .521/.329; DSO-1 .811; CocoGlide .405/.318; image AUC on CocoGlide .520. — [arXiv 2212.10957](https://arxiv.org/html/2212.10957v3)
- Noiseprint++ (inside TruFor) is a DnCNN extractor trained on real images only, using InfoNCE on 64×64 patches. Positives share camera source, spatial position and editing history (512 possible processing pipelines). The anomaly map is a supervised segmentation head over RGB plus Noiseprint++. A confidence map and 8 pooled statistics give the image-level score. — [TruFor, arXiv 2212.10957](https://arxiv.org/html/2212.10957v3)
- TruFor results (pixel F1 best/fixed): average .785/.696; CocoGlide .720/.523; image AUC on CocoGlide .752. ManTraNet's image AUC on CocoGlide is higher, at .778. CocoGlide is 512 GLIDE-manipulated COCO-val images (256×256, object masks plus text prompts), paired with 512 real images. — [arXiv 2212.10957](https://arxiv.org/html/2212.10957v3). Important caveat: TruFor's decision stage is supervised. It is self-consistency-inspired rather than test-time label-free.

**A-contrario (IPOL) line**
- ZERO (Nikoukhah, Anger, Colom, Morel, Grompone von Gioi; IPOL 2021). The method counts zero DCT coefficients over 8×8 blocks to find each pixel's JPEG grid origin, then detects locally misaligned or missing grids. Two masks are produced, foreign-grid and missing-grid. Validation is a-contrario (Desolneux–Moisan–Morel): an NFA is attached to each detection, and a threshold on the NFA makes the method fully automatic. — [IPOL 2021/390](https://ipol.im/pub/art/2021/390); conference version: [CVPRW 2019](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Media%20Forensics/Nikoukhah_JPEG_Grid_Detection_based_on_the_Number_of_DCT_Zeros_CVPRW_2019_paper.pdf)
- An earlier grid detector uses blocking artifacts (Chen–Hsu cross-difference). — [IPOL 2020/283](https://www.ipol.im/pub/art/2020/283/)
- Centre Borelli reports that ZERO is integrated into the AFP InVID-WeVerify tool (over 90,000 weekly users), a deployment claim by the authors' lab. — [Centre Borelli](https://centreborelli.ens-paris-saclay.fr/en/detection-digital-image-forgery)
- Noisesniffer (Gardella, Musé, Morel, Colom; IWBF 2021) is a blind detector built on a background stochastic noise model. Local anomalies are flagged by NFA. — [HAL hal-03243928](https://hal.archives-ouvertes.fr/hal-03243928)
- The refined Noisesniffer (IPOL vol. 14, pp. 86–115, 2024) adds region growing from local deviations of the background noise model. The authors report it beats the original and the state of the art (self-reported). — [IPOL 2024/462](https://ipol.im/pub/art/2024/462)
- PhotoHolmes is a Python library that bundles classic and learned forgery detectors, including these a-contrario methods, for uniform benchmarking. — [arXiv 2412.14969](https://arxiv.org/pdf/2412.14969)
- Other a-contrario forensics: resampling detection with deep features plus a-contrario analysis [arXiv 1803.01711](https://arxiv.org/abs/1803.01711); copy-move detection with false-alarm control [arXiv 1906.00649](https://arxiv.org/abs/1906.00649).
- Double JPEG and quantization-matrix clustering: Niu et al. estimate the primary quantization matrix per block and cluster blocks to detect, localize and attribute splices. — [arXiv 2102.01439](https://arxiv.org/abs/2102.01439). Korus gives a large-scale evaluation of JPEG forgery localizers. — [arXiv 1811.12915](https://arxiv.org/abs/1811.12915)
- CFA and demosaicing local consistency: Ferrara, Bianchi, De Rosa, Piva, "Image forgery localization via fine-grained analysis of CFA artifacts", TIFS 2012. A per-2×2-block demosaicing feature is modelled with a GMM, and the local likelihood ratio gives the map. **[UNVERIFIED, prior knowledge]**

### Inferences
- All generations rest on one assumption: the pristine part of the image has one homogeneous acquisition or processing signature, and the forgery has a different one. Handcrafted methods compare local estimates against a global model with a test (EM-GMM, k-means, NFA). Learned pairwise methods compare region pairs and cluster. Fingerprint methods compare against a learned "regular pattern" but use a supervised readout.
- The a-contrario approach is the only one with principled false-alarm control, giving a per-image NFA. It is the natural framework if the researcher wants a test-time-label-free decision rule over DT-CWT/colour-segment statistics. The detector is only as good as the background model, however. If the edit and the host share the same model (full regeneration), no detection is possible by construction.
- Even unsupervised fingerprint-consistency methods (Noiseprint, Splicebuster, EXIF-SC) reach only about 0.40–0.43 F1 and about 0.52–0.53 image AUC on CocoGlide, against 0.80–0.88 on Columbia. Diffusion local edits were already the weak case in 2022.

### Gaps
- EXIF-as-language: the clustering algorithm and per-dataset numbers were not retrieved; the full text is at arxiv.org/html/2301.04647v4.
- Forensic Similarity Graph numbers and exact venue were not verified this session.
- Lyu, Mahdian–Saic, Pan, Ferrara CFA and the Splicebuster/Noiseprint venues come from background knowledge and were not re-fetched.
- I found no published evaluation of ZERO, Noisesniffer or CFA detectors on diffusion-inpainting benchmarks. By construction, JPEG-grid methods detect only grid misalignment or absence, so a VAE-decoded uncompressed output would show "missing grid" everywhere or nowhere. This is an inference, not a measured result.

---

## Q2. Newer (2023–2026) intra-image consistency, contrastive, foundation-model and training-free localizers

### Takeaway
Since 2023 "intra-image consistency" has mostly become a training-time loss, i.e. contrastive learning between authentic and tampered patches of the same image (NCL-IML, ICL-Net/PromptForge, DiffusionPrint), inside supervised localizers. Methods that are truly test-time label-free and run on foundation-model tokens remain rare. The strongest found is TRAIL (frozen DINOv3 patch-token drift under a Haar perturbation): AUROC .903 on CocoGlide, close to supervised Detective SAM at .912. TRAIL is a perturbation-response method, not a region-vs-rest comparison. Its own controls show a substantial content and raw-pixel baseline: raw-pixel .641, aligned-real .697/.743.

### Cited Findings
- **NCL-IML** (Zhou et al., 2023), "Pre-training-free Image Manipulation Localization through Non-Mutually Exclusive Contrastive Learning". Patches fall into tampered, authentic and contour classes. Contour patches (mixed pixels) are non-mutually exclusive, so a dual-branch "pivot" structure switches their role between positive and negative, with a pivot-consistent loss. The authors claim SOTA on five benchmarks with no pre-training. It is supervised: masks define the patch roles. — [arXiv 2309.14900](https://arxiv.org/abs/2309.14900); code: [GitHub NCL-IML](https://github.com/Knightzjz/NCL-IML)
- **ICL-Net / PromptForge-350k** (Wang et al., March 2026). Ground-truth masks for prompt-based editors are produced automatically by keypoint alignment plus semantic similarity. The dataset covers 4 prompt-based editing models. ICL-Net has a triple-stream backbone plus intra-image contrastive learning. It reports IoU 62.5% on PromptForge-350k (+5.1% over SOTA), an IoU drop under degradations of less than 1%, and an average IoU of 41.5% on unseen editing models. It is supervised. — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)
- **DiffusionPrint** (Giakoumoglou & Papadopoulos, April 2026). The authors state that full-regeneration inpainting pipelines "reconstruct the entire image via a latent decoder, disrupting the camera-level noise patterns" that forensic methods rely on. In response, they learn a patch-level MoCo-style generative fingerprint: inpainted regions from the same model share a fingerprint, which serves as the self-supervisory signal, with cross-category hard negatives and a generator-aware head. Used as an extra modality in TruFor, MMFusion and a light fusion baseline, it gains up to +28% on mask types unseen during fine-tuning. — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443)
- **TRAIL** (Kumar, Jain, Kainz, August 2026), "Frozen DINO Localizes Image Edits Without a Localizer". — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968), [full text](https://arxiv.org/html/2608.18968v1), [code](https://github.com/VishalJ99/trail-image-edit-localization)
  - Regions: 28×28 patch tokens of frozen DINOv3 ViT-7B/16 at 448×448, block 36. CLS and register tokens are excluded.
  - Descriptor: T(x) = x − 0.2·HF₂(x), where HF₂ is the two-level Haar high-frequency band. The score per patch is 1 − cos(token(x), token(T(x))), followed by a 3×3 median filter.
  - Decision: a fixed threshold τ = 0.0026 chosen on development data, so the method is "training-free in parameters but not validation-free".
  - Results (patch AUROC, TRAIL vs mask-supervised Detective SAM): CocoGlide (80 images) .903 vs .912, fixed-threshold Dice .619 vs .709 (oracle-threshold Dice .790). TGIF2 semantic masks .811 vs .904. TGIF2 random masks .727 vs .758. Poisson Image Interpolation .855 vs .864. Foreign Patch Interpolation .850 vs .894. All 720 TGIF2 splices .774 AUROC, .231 Dice (vs .839 / .324).
  - Controls: raw-pixel control AUROC .641; aligned-real control .697 on CocoGlide and .743 on TGIF2 semantic masks. The response is partly present before any edit.
  - Context matters: AUROC .903 with a global perturbation, .857 for local-in-canvas, .735 for independently encoded crops. Across 16 DINO encoders the best block lies at normalized depth .80–.94.
  - Failure modes: small edits covering 2–8% of the frame are associated with failure. Fully regenerated images are excluded because their masks cover most of the frame. MLLM edits, recompression, resizing and adversarial post-processing are not tested.
  - The WaRPAD DINOv2-L/14 construction (as reported by TRAIL) reaches .808 AUROC on 45 CocoGlide images.
- **DinoLizer** (Doi, Itier, Butora, Boulanger, Bas; Nov 2025) is a supervised DINOv2-based localizer for generative inpainting. Reconstructed areas outside the inpainted mask are treated as a separate class, separating VAE artifacts from diffusion artifacts. — [arXiv 2511.20722](https://arxiv.org/abs/2511.20722)
- **FUSED** (Aug 2026) is a forensic-semantic mixture-of-experts for inpainting detection and localization. It motivates the design by noting that many AI-image detectors rely on global artifacts and do not localize. — [arXiv 2608.28302](https://arxiv.org/abs/2608.28302)
- Other related supervised localizers: Pixel-Inconsistency Modeling [arXiv 2310.00234](https://arxiv.org/abs/2310.00234); SDiFL (Stable-Diffusion-driven localization) [arXiv 2508.20182](https://arxiv.org/abs/2508.20182); InpDiffusion [arXiv 2501.02816](https://arxiv.org/abs/2501.02816); Dense Feature Interaction Network for inpainting [arXiv 2408.02191](https://arxiv.org/abs/2408.02191); COCO-Inpaint benchmark [arXiv 2504.18361](https://arxiv.org/abs/2504.18361).

### Inferences
- The researcher's results fit the literature closely. Intra-image kNN/prototype distances on DINOv2 tokens reach about 0.55–0.67, barely above their raw-pixel control at 0.62. TRAIL's own raw-pixel control is .641 and its aligned-real control .697–.743. Plain DINO token-affinity statistics are therefore largely content-driven. TRAIL's gain to .90 comes from the perturbation-response (drift) signal, the late-layer choice (depth .80–.94, not layer 6) and preserved global context, not from region-vs-rest dissimilarity. A promising test is to use drift or perturbation sensitivity as the per-patch descriptor and then apply a self-consistency or a-contrario decision over it.
- TRAIL's late-layer optimum contrasts with the researcher's supervised layer-6 probe (0.77/0.68). Layer choice may matter differently for drift and for linear probing; this is worth checking.
- Modern "intra-image contrastive" methods (NCL, ICL-Net, DiffusionPrint) use the self-consistency idea only as a training-time regularizer. Their test-time decision is still a supervised segmentation head, so they do not satisfy "no external labels at test time" in the strict sense.

### Gaps
- No arXiv paper was found that does CLIP-based patch-affinity forgery localization with quantitative results on diffusion edits. The search returned anomaly-detection analogues only.
- "Relational" or graph-based foundation-model localizers after Forensic Similarity Graphs were not found in the budget.
- Detective SAM and WaRPAD are known only through TRAIL's text; their primary sources were not fetched.

---

## Q3. Robustness on diffusion inpainting, fully regenerated (FR) images and MLLM edits

### Takeaway
On spliced diffusion inpainting (generated region pasted into the original pixels), camera/compression-consistency methods still work (TGIF: CAT-Net F1 .87, TruFor .83). On fully regenerated images, where a VAE decodes the whole frame, every IFL method collapses: CAT-Net .04, TruFor .19, MMFusion .15–.18. Fine-tuning on FR helps, but random-mask tests in TGIF2 reveal object bias. On VLM/MLLM edits (PIXAR: GPT-image-1.5, Gemini, Seedream, Flux.2, Qwen-Image), specialized localizers reach only IoU of about 8–12% and AUC of about 52–56%. Humans reach IoU 10.7%.

### Cited Findings
- **TGIF** (Mareen et al., 2024): about 75k forged images from SD2, SDXL and Adobe Firefly. Traditional IFL detects spliced images but fails on regenerated ones; SID detects regenerated images but cannot localize. Both fail under stronger compression, especially WEBP. — [arXiv 2407.11566](https://arxiv.org/abs/2407.11566)
  - Table I (uncompressed F1), SD2-sp / PS-sp(Firefly) / SD2-fr / SDXL-fr: CAT-Net .87/.85/.04/.03; TruFor .83/.79/.19/.18; MMFusion .75/.74/.15/.18; Mantranet .15/.56/.03/.05; PSCC-Net .15/.38/.05/.05; ImageForensicsOSN .23/.36/.20/.18; MVSS-Net++ .07/.08/.06/.09; SPAN 0 throughout. The authors say "no method is able to detect and localize these manipulations" on FR because regeneration removes compression and camera-noise traces, acting as an attack on forgery detectors. Noiseprint and Splicebuster were not evaluated. — [TGIF full text](https://arxiv.org/html/2407.11566v2)
- **TGIF2** (Mareen et al., March 2026) adds FLUX.1 edits and random non-semantic masks. Both IFL and SID degrade on FLUX.1. Fine-tuning IFL on FR improves localization, but random masks reveal object bias. Generative super-resolution "significantly weakens forensic traces". — [arXiv 2603.28613](https://arxiv.org/abs/2603.28613). Exact tables were not fetched.
- TRAIL on TGIF2: .811 AUROC on semantic masks versus .727 on random masks. The drop is consistent with the object/semantic bias TGIF2 reports. Detective SAM drops .904 → .758. — [arXiv 2608.18968](https://arxiv.org/html/2608.18968v1)
- **CocoGlide** (GLIDE, 256×256): unsupervised fingerprint-consistency methods reach pixel F1 .405–.434 and image AUC .52–.53. TruFor reaches F1 .720 and AUC .752. — [arXiv 2212.10957](https://arxiv.org/html/2212.10957v3). TRAIL reaches patch AUROC .903. — [arXiv 2608.18968](https://arxiv.org/abs/2608.18968)
- **PIXAR** (Shang et al., March 2026) argues that object masks misalign with the true edit signal. Many in-mask pixels are untouched, while off-mask pixels are changed. It releases per-pixel tamper maps. Test editors: Flux.2, Gemini 2.5, Gemini 3, GPT-image-1.5, Qwen-Image, Seedream 4.5; training data comes from Qwen-Image. Baselines on the PIXAR test set: LISA-7B IoU 8.3 / F1 15.4 / AUC 55.0; SIDA-7B 11.8 / 21.1 / 55.0; SIDA-13B 10.8 / 19.5 / 55.6; FakeShield 9.3 / 17.0 / 52.0. Detectors are "ill-scored under mask-only protocols, especially on micro-edits and off-mask changes". No TruFor or low-level self-consistency baselines appear in the portion read. — [arXiv 2603.20193](https://arxiv.org/abs/2603.20193), [full text](https://arxiv.org/html/2603.20193v1), [code](https://github.com/VILA-Lab/PIXAR)
- **PromptForge-350k / ICL-Net** (prompt-based editors): in-domain IoU 62.5%, but 41.5% on unseen editors. — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)
- DiffusionPrint names VAE latent decoding as the mechanism that destroys camera-noise patterns. — [arXiv 2604.12443](https://arxiv.org/abs/2604.12443)
- Pandolfini et al. (Dec 2025) evaluate how synthetic-image detectors handle inpainting and localized deepfakes. — [arXiv 2512.16688](https://arxiv.org/abs/2512.16688). Numbers not fetched.

### Inferences
- There is a mechanistic explanation for the researcher's results. Pooled per-cell DT-CWT, shearlet, lifting and SRM statistics reach an AUC of about 0.66, and GPT-Image-2 edits are at chance. MLLM editors and FR inpainting re-synthesize the whole frame through a decoder, so the "rest of the image" no longer carries a distinct camera/noise signature to be inconsistent with. TGIF's CAT-Net drop from .87 to .04 is the cleanest published demonstration. Low-level self-consistency can only succeed where pristine pixels survive bit-exactly or near-exactly (splice-back) or where the decoder leaves region-dependent residue (DinoLizer's VAE vs diffusion classes).
- PIXAR's point about mask misalignment implies part of the measured "failure" on MLLM edits is label noise. Pixel-difference ground truth should be preferred when evaluating self-consistency maps on GPT-Image or Gemini edits.
- The object bias in TGIF2 (semantic vs random masks) is a warning for any DINO-based "consistency" score. A detector may be segmenting salient objects rather than edits. The researcher's raw-pixel control at 0.62 and TRAIL's aligned-real control at .70–.74 point the same way.

### Gaps
- **OpenSDI**: no numbers retrieved within budget.
- TGIF2 table numbers (FLUX.1, fine-tuned FR, super-resolution) were not fetched.
- No paper was found that evaluates classic a-contrario detectors (ZERO, Noisesniffer, Splicebuster) on MLLM or GPT-Image edits.

---

## Q4. Human visual forensics: cues and localization accuracy

### Takeaway
Humans are poor at both detection and localization. The 2017 real-world-scene study puts detection near chance (about 62% overall) and finds poor localization even when a fake is detected. On modern VLM edits (PIXAR), 10 annotators reached localization IoU of only 10.7% and F1 of 18.8%. Detection relates to how much the edit disrupts the underlying pixel structure.

### Cited Findings
- Nightingale, Wade, Watson (2017), Cognitive Research: Principles and Implications 2:30, with 659 participants. 58% of originals and 65% of manipulated photos were correctly classified (about 62% overall, "only slightly better than chance"). Even when subjects correctly detected manipulated images, they were often unable to locate the manipulation. Detection correlated positively with the extent of disruption to the underlying pixel structure. — [DOI 10.1186/s41235-017-0067-2 / PMC5514174](https://pmc.ncbi.nlm.nih.gov/articles/PMC5514174); localization figure reported as 45% by [Nieman Lab](https://www.niemanlab.org/?p=145334) and as 56% in a garbled [ScienceDaily](https://www.sciencedaily.com/releases/2017/07/170717221038.htm) summary. The sources conflict, so check the primary text.
- PIXAR human study: 10 participants, 1,000 images (500 real, 500 tampered), asked to classify and then draw boxes. Detection precision 22.2, recall 55.5, F1 31.0. Localization recall 17.4, F1 18.8, **IoU 10.7**. The paper does not report what cues participants used. — [arXiv 2603.20193, Table 11](https://arxiv.org/html/2603.20193v1)

### Inferences
- Human IoU of 10.7% is on par with automatic VLM localizers on PIXAR (IoU 8–12%), so neither humans nor current detectors localize modern MLLM edits well.
- The finding that detectability scales with pixel-structure disruption matches the self-consistency premise. Edits that do not disrupt local statistics, such as regenerated or harmonized ones, defeat both humans and low-level detectors.

### Gaps
- No primary study was retrieved within budget that catalogues which cues people use (lighting or shadow inconsistency, sharpness or texture mismatch, noise grain) on AI edits. Farid's physics-based lighting and shadow work and the "shadows/reflections" perception studies (e.g. Farid & Bravo, "Image forensic analyses that elude the human visual system", SPIE 2010) are relevant **[UNVERIFIED, prior knowledge]**.
- Human-localization studies specific to diffusion inpainting, beyond PIXAR, were not found.
