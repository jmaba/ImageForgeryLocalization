# Critiques of forgery-localization evaluation, and reference-based edit localization (2016-2026)

Verification convention: an arXiv id is listed only when it was retrieved and its title and authors checked through the arXiv API in this session. Venue information marked "(venue unverified)" comes from memory and was NOT checked. Treat it as a lead.

## Q1. Shortcut and bias analyses of forgery benchmarks; any matched-null / transplant design?

### Takeaway
There is solid prior work on format, JPEG and size confounds for *image-level* AIGC detection (Grommelt et al. 2024, B-Free 2024/25) and on the CASIA v2 format confound. For *localization*, the closest prior is TGIF2 (2026). It uses random non-semantic masks to expose object bias, which is the nearest relative of idea (i). I found no paper that scores an authentic image with the mask of its own edit as a matched null (expected AUC 0.5). I also found none that uses a colour/texture-statistics control scorer as a shortcut baseline for localization, and none that uses interior-only (boundary-excluded) scoring.

### Cited Findings
- Grommelt, Weiss, Pfreundt, Keuper, "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets", arXiv:2403.17608 (2024). They show that GenImage carries JPEG-compression and image-size biases, and that detectors learn these factors. Removing the biases improves JPEG robustness and raises cross-generator accuracy by more than 11 points for ResNet50/Swin-T. This is image-level detection, not localization. Venue unverified (possibly ICPR 2024). — [arXiv](https://arxiv.org/abs/2403.17608)
- Guillaro, Zingarini, Usman, Sud, Cozzolino, Verdoliva, "A Bias-Free Training Paradigm for More General AI-generated Image Detection" (B-Free), arXiv:2412.17671 (2024). They state that performance "can be strongly impacted by spurious correlations such as content, format, or resolution". Their fix is to create fakes from reals through SD conditioning, so the two are semantically aligned. This is a *training-side* matched-pair design for detection, not an evaluation null for localization. Venue unverified (believed CVPR 2025). — [arXiv](https://arxiv.org/abs/2412.17671)
- Mandala, "Format-Controlled Multi-Scale JPEG Compression Response Analysis for Image-Level Forgery Screening", arXiv:2607.06615 (2026, single-author preprint). In CASIA v2.0, about 60% of tampered images are TIFF while authentic images are JPEG/BMP only, so a trivial `is_tiff` classifier reaches 0.80 AUC. The paper uses JPEG-only and source-aware splits as controls. This is image-level only. — [arXiv](https://arxiv.org/abs/2607.06615)
- Kwon et al., "Learning JPEG Compression Artifacts for Image Manipulation Detection and Localization" (CAT-Net), arXiv:2108.12947. It notes that CASIA v2 sources were JPEG-compressed before being saved in other formats, with diverse quantization tables. That is a compression-history confound. — [arXiv](https://arxiv.org/abs/2108.12947)
- Mareen, Karageorgiou, Giakoumoglou, Lambert, Papadopoulos, Van Wallendael, "TGIF2: Extended Text-Guided Inpainting Forgery Dataset & Benchmark", arXiv:2603.28613 (2026). It adds edits on "random non-semantic masks", and that evaluation "reveals object bias" in IFL methods fine-tuned on fully regenerated images. — [arXiv](https://arxiv.org/abs/2603.28613). A secondary summary of the full text says the random-mask results suggest that MMFusion and TruFor rely on object/semantic cues and boundaries. This was not verified against the full text. — [arXiv PDF](https://arxiv.org/pdf/2603.28613)
- Kong, Luo, Wang, Li, Rocha, Kot, "Pixel-Inconsistency Modeling for Image Manipulation Localization", arXiv:2310.00234 (2023). It uses random patch shuffling to suppress semantics and test whether localizers depend on semantic cues. Per a secondary summary, MVSS-Net++ degrades under this test. That is a semantic-prior probe, but not a matched null. — [arXiv](https://arxiv.org/abs/2310.00234)
- Ma et al., "IMDL-BenCo: A Comprehensive Benchmark and Codebase for Image Manipulation Detection & Localization", arXiv:2406.10580 (2024). It argues that the lack of a benchmark leads to "insufficient and misleading model evaluations". Venue unverified (believed NeurIPS 2024 Datasets & Benchmarks). — [arXiv](https://arxiv.org/abs/2406.10580)
- Guan et al., "MFC Datasets: Large-Scale Benchmark Datasets for Media Forensic Challenge Evaluation", WACV Workshops 2019, pp. 63-72. It covers NIST/DARPA MediFor, about 176k high-provenance images and over 100k manipulated images, across six tasks including provenance. — [NIST PDF](https://mig.nist.gov/MFC/Web/Papers/MFC_Dataset_WACV2019.pdf); [mlanthology](https://mlanthology.org/wacvw/2019/guan2019wacvw-mfc)

### Inferences
- The "matched transplant null" (i) appears novel. TGIF2's random masks change *where* the edit is, while the proposed null keeps the mask and removes the edit. The two are complementary: TGIF2 tests whether a model needs objects, and the null tests whether a model fires on mask-shaped content priors with no edit evidence at all.
- B-Free's aligned real/fake pairs are the closest philosophical precedent ("differences should stem solely from generation artifacts"). Cite it as motivation.
- (ii) colour/texture control scorer and (iii) interior-only scoring: nothing found. They may exist in passing (for example as boundary-band metrics), but no dedicated paper surfaced.

### Gaps
- I could not find a pixel-level localization paper that quantifies format or size confounds; the CASIA format confound is shown only at image level.
- NIST MFC scoring notes, which include no-score zones around mask boundaries, could be precedent for (iii). I did not retrieve the MFC scoring documentation to confirm this. Check the MFC evaluation plan.
- I did not search Google Scholar for "boundary-excluded" / "no-score region" metrics.

## Q2. Robustness benchmarks for localizers (JPEG/WebP/resize/OSN, generative SR, regeneration)

### Takeaway
OSN robustness is established, led by Wu et al. CVPR 2022. TGIF (2024) benchmarks JPEG and WebP and finds both IFL and SID fail under stronger compression, and fail more under WebP. TGIF2 (2026) is the first that I found to show that *generative super-resolution* strongly weakens forensic traces for localization. SD-autoencoder "laundering" is analysed for detection and camera-model ID, not for localization.

### Cited Findings
- Wu, Zhou, Tian, Liu, "Robust Image Forgery Detection over Online Social Network Shared Images", CVPR 2022, pp. 13440-13449. It models predictable and unseen OSN noise and releases a Facebook/Weibo/WeChat dataset. — [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Robust_Image_Forgery_Detection_Over_Online_Social_Network_Shared_Images_CVPR_2022_paper.html)
- Mareen, Karageorgiou, Van Wallendael, Lambert, Papadopoulos, "TGIF: Text-Guided Inpainting Forgery Dataset", arXiv:2407.11566 (2024). Both IFL and SID "fail when exposed to stronger compression" and are less robust to WebP. Venue unverified (believed IEEE WIFS 2024). — [arXiv](https://arxiv.org/abs/2407.11566)
- TGIF2, arXiv:2603.28613: "generative super-resolution significantly weakens forensic traces". — [arXiv](https://arxiv.org/abs/2603.28613)
- Mandelli, Bestagini, Tubaro, "When Synthetic Traces Hide Real Content: Analysis of Stable Diffusion Image Laundering", arXiv:2407.10736 (2024). Passing an image through SD autoencoders yields a lookalike synthetic copy that hides real-content traces and masks camera-model artifacts. — [arXiv](https://arxiv.org/abs/2407.10736)
- Karageorgiou et al., "Evolution of Detection Performance throughout the Online Lifespan of Synthetic Images", arXiv:2408.11541 (2024). This is detection under real online re-sharing. — [arXiv](https://arxiv.org/abs/2408.11541)
- Du et al., "ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization", arXiv:2505.11003 (2025). Its robustness protocol details were not checked. — [arXiv](https://arxiv.org/abs/2505.11003)
- Fang & Stamm, "Attacking Image Splicing Detection and Localization Algorithms Using Synthetic Traces", arXiv:2211.12314. It is an anti-forensic attack on Siamese trace-consistency localizers. — [arXiv](https://arxiv.org/abs/2211.12314)

### Inferences
- A *systematic* sweep for localizers (JPEG, WebP, resize, generative SR, VAE roundtrip, all combined with a matched null) would extend TGIF/TGIF2 rather than be fully new. The novelty lies in combining it with the nulls.

### Gaps
- I did not verify the exact robustness operations in IMDL-BenCo or ForensicHub; IMDL-BenCo is believed to include JPEG, Gaussian blur and noise.

## Q3. Reference-based / provenance-based localization

### Takeaway
The DARPA MediFor provenance line (Notre Dame / Unicamp, 2017-2019) provides retrieval (provenance filtering), graph construction, and spliced-donor retrieval. That is the "find the original" infrastructure. Pixel-level change localization against a retrieved original for *instruction / LLM edits* is barely addressed. The nearest recent work, SIGMA (2026), uses (original, edited) semantic-feature differencing to *annotate* training masks, not as a deployed forensic localizer. It also states explicitly that pixel differencing is "overwhelmed by diffusion-induced perturbations across all pixels". I found no paper that frames reference retrieval plus comparison as *the* solution for LLM instruction edits.

### Cited Findings
- Pinto, Moreira, Bharati, Brogan, Bowyer, Flynn, Scheirer, Rocha, "Provenance Filtering for Multimedia Phylogeny", arXiv:1706.00447 (2017). Venue unverified (believed IEEE ICIP 2017). — [arXiv](https://arxiv.org/abs/1706.00447)
- Bharati, Moreira, Pinto, Brogan, Bowyer, Flynn, Scheirer, Rocha, "U-Phylogeny: Undirected Provenance Graph Construction in the Wild", arXiv:1705.11187 (2017). Venue unverified (believed ICIP 2017). — [arXiv](https://arxiv.org/abs/1705.11187)
- Moreira, Bharati, Brogan, Pinto, Parowski, Bowyer, Flynn, Rocha, Scheirer, "Image Provenance Analysis at Scale", arXiv:1801.06510 (2018). Venue unverified (believed IEEE TIP 2018). — [arXiv](https://arxiv.org/abs/1801.06510)
- Bharati et al., "Beyond Pixels: Image Provenance Analysis Leveraging Metadata", arXiv:1807.03376 (2018). Venue unverified (believed WACV 2019). — [arXiv](https://arxiv.org/abs/1807.03376)
- Chen, Wu, Davis, Lim, "Efficient Object Embedding for Spliced Image Retrieval", arXiv:1905.11903. It explicitly replaces low-level artifact detection with retrieval of the source images of a spliced query. Venue unverified. — [arXiv](https://arxiv.org/abs/1905.11903)
- Wu, AbdAlmageed, Natarajan, "Deep Matching and Validation Network: An End-to-End Solution to Constrained Image Splicing Localization and Detection", arXiv:1705.09765 (2017). It does "constrained" localization given a donor/host pair, which is a reference-based localization precedent. — [arXiv](https://arxiv.org/abs/1705.09765)
- Zhuang et al., "SIGMA: Semantic-Difference Instruction-Grounding Mask Annotator for Text-Driven Image Manipulation Localization", arXiv:2605.27924 (2026). It differences (original, edited) pairs in a foundation-model feature space plus an instruction prior to produce about 1.1M pseudo-masks. — [arXiv](https://arxiv.org/abs/2605.27924)
- Recent edit-localization datasets ship paired (source, edited) images but evaluate *blind* localizers: X-Edit (Bazyleva, Bonettini, Bharaj, CVPRW 2025, arXiv:2505.11753), EditSleuth (2026, triplets of edited/source/mask, arXiv:2605.08695, title only from search), and LocateEdit-Bench (Wu et al., arXiv:2602.05577, 231K instruction-edited images from four editors). — [X-Edit CVF](https://openaccess.thecvf.com/content/CVPR2025W/XAI4CV/html/Bazyleva_X-Edit_Detecting_and_Localizing_Edits_in_Images_Altered_by_Text-Guided_CVPRW_2025_paper.html); [LocateEdit-Bench](https://arxiv.org/abs/2602.05577); [EditSleuth](https://arxiv.org/html/2605.08695)

### Inferences
- A reference-based pipeline (retrieve, align, then robust semantic change detection) for instruction edits is the direct descendant of MediFor provenance work. SIGMA shows that feature-space differencing works where pixel differencing fails, which supports the approach. A protocol that evaluates reference-based localization against blind localizers *on the same nulls* looks open.
- C2PA / Content Credentials is a natural complement. I did not search it in this pass.

### Gaps
- MediFor provenance *localization* metrics (NIST MFC provenance task scoring) were not retrieved.
- I found no "Find the original" paper by that title. I did not search the RAW-provenance or C2PA literature.

## Q4. Evidence that full regeneration removes local trace contrast

### Takeaway
Yes, empirically. TGIF (2024) and TGIF2 (2026) explicitly show that traditional IFL methods localize spliced inpainting but fail on fully regenerated (FR) images, and that SID detects FR images but cannot localize. GPT-ImgEval reports that GPT-4o regenerates the entire image even when the brush tool is used. I found no formal (theoretical) argument that blind trace-based localization is ill-posed for FR images.

### Cited Findings
- TGIF abstract: "diffusion model-based approaches could either splice the inpainted region into the original image, or regenerate the entire image. In the latter case, traditional image forgery localization (IFL) methods typically fail." — [arXiv:2407.11566](https://arxiv.org/abs/2407.11566)
- TGIF2: the "open problem of localization in FR images" remains. Fine-tuning on FR improves localization, but random masks reveal that the gain comes partly from object bias. — [arXiv:2603.28613](https://arxiv.org/abs/2603.28613)
- GPT-ImgEval (arXiv:2504.02782): GPT-4o's localized editing "still involves regenerating the entire image", and its outputs are easily *detected* at image level by forensic models. Seen through a search summary only; I did not fetch the full text. — [arXiv](https://arxiv.org/html/2504.02782v1)
- SIGMA notes that diffusion editing perturbs all pixels, so naive differencing fails. — [arXiv:2605.27924](https://arxiv.org/abs/2605.27924)
- Mandelli et al. 2024: VAE roundtrip launders real content into lookalike synthetic content and masks camera traces. — [arXiv:2407.10736](https://arxiv.org/abs/2407.10736)

### Inferences
- TGIF2's finding (fine-tuning on FR "works" but partly through object bias) is direct support for idea (i). A matched null would quantify how much FR-localization performance is prior-driven. This is the strongest novelty angle.

### Gaps
- I found no forensic-localization study that targets GPT-Image / Nano-Banana / Qwen-Image-Edit outputs with matched nulls.

## Q5. Dataset pitfalls: pseudo-labelled masks, format/size confounds

### Takeaway
Format/size/JPEG confounds are documented (Grommelt 2024, B-Free, the CASIA format confound). Pseudo-labelled masks are now an explicit *design choice*: SIGMA auto-annotates about 1.1M masks, and TGIF-style datasets use inpainting masks rather than true changed-pixel masks. I found no paper auditing *mislabelled or segmentation-predicted masks* in IML benchmarks as a pitfall.

### Cited Findings
- See Grommelt (arXiv:2403.17608), B-Free (arXiv:2412.17671), and Mandala (arXiv:2607.06615) above.
- SIGMA reports that instruction-only grounding "misses unintended editor side-effects". That implies masks derived from prompts or segmentation under-cover the true changed region. — [arXiv:2605.27924](https://arxiv.org/abs/2605.27924)
- CASIA v2 near-duplicate leakage (398 near-duplicate pairs) is reported only in a GitHub project, which is not peer-reviewed. — [GitHub](https://github.com/pramodpptrv/cea-convnext-resnest-image-forgery)

### Gaps
- I did not verify which public instruction-edit IML datasets derive masks from SAM, Grounded-SAM or difference thresholds. Each dataset paper (for example LocateEdit-Bench) would need to be checked.

## What remains open / novel (summary)
- (i) A matched transplant null for localization: not found. Closest: TGIF2 random masks (2026), B-Free aligned pairs (detection), and Kong et al. patch shuffling.
- (ii) A colour/texture-statistics control scorer: not found for localization. The image-level analogue is the trivial `is_tiff` classifier (Mandala 2026).
- (iii) Interior-only scoring: not found in the papers searched. Check NIST MFC no-score zones.
- (iv) Robustness sweep including generative SR: partly done (TGIF/TGIF2, Wu CVPR 2022). Combining it with nulls is new.
- (v) Pseudo-label mask audit: open. SIGMA shows the trend toward auto-masks.
- (vi) Reference-based localization for LLM/instruction edits: open as a framed solution. Infrastructure exists from MediFor provenance (2017-2019) and DMVN; SIGMA shows feature differencing works.
- (vii) Empirical evidence of FR trace loss exists (TGIF/TGIF2). A formal ill-posedness argument was not found.
