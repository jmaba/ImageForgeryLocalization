# Physics-based (lighting / shading / shadow) forgery detection and generative relighting for forensics: novelty check (2016-2026)

Search scope and method: arXiv API (fielded abstract queries, cs.CV, 2016-2026; about 10 query variants), plus web searches (Google-style) for the Farid lab, Rocha/Carvalho, intrinsic-decomposition AIGC detection, and relighting-for-forensics. Search date: 2026-11-04. Every arXiv ID below was retrieved and its abstract read through the arXiv API, unless marked "(web only)". Limits: Google Scholar, IEEE Xplore and Semantic Scholar were not queried directly, so non-arXiv journal or conference papers (IEEE TIFS, WIFS, IH&MMSec) may be under-covered. Abstracts were read but no full texts, so most quantitative numbers are missing (see Gaps).

Context only (already known, not re-researched): Johnson & Farid (2005-2007), Kee & Farid (2010-2013, including "Exposing Digital Forgeries From 3-D Lighting Environments", WIFS 2010 — [PDF](https://hfarid.org/downloads/publications/wifs10.pdf)), Riess & Angelopoulou (2010), de Carvalho et al. (2013, illuminant maps — [PDF](https://faui1-files.cs.fau.de/public/publications/mmsec/2013-de-Carvalho-EDI.pdf)).

## Q1. Which papers since 2016 automate lighting, shadow or illuminant forensics with deep or foundation models? Do any localize edits per region?

### Takeaway
There is a thin but real line of work. It covers learned illuminant-map CNNs (2018), learned face-lighting regressors (2017), geometric classifiers for generated images (Shadows Don't Lie, CVPR 2024), Farid's SH-lighting analysis of DALL-E 2 (2022), face specular/Phong-based deepfake detectors (2026), and a 2026 depth+normals-aided localizer (GFrame). Almost all of it does **whole-image or whole-face classification**. The only per-pixel localizer that uses geometry (GFrame, 2026) learns from depth and normals as fused features and fits no explicit lighting model. A 2026 benchmark (Esfahani et al.) frames per-region "physical plausibility diagnosis", but only on Mitsuba-rendered scenes and with VLM baselines.

### Cited Findings
**Classical-to-deep illuminant and lighting estimators (2016-2019)**
- Pomari, Ruppert, Rezende, Rocha, Carvalho, "Image Splicing Detection Through Illumination Inconsistencies and Deep Learning", IEEE ICIP 2018. Illuminant maps (from de Carvalho et al.) feed a CNN with transfer learning; trained on the DSO-1 / DSI-1 datasets; aimed at splicing detection. The poster also mentions localization. Accuracy figures were not retrieved — [FAPESP record](https://bv.fapesp.br/en/publicacao/221244), [SigPort poster](https://sigport.org/node/3588) (web only, no arXiv).
- Zhou, Sun, Yacoob, Jacobs, "Label Denoising Adversarial Network (LDAN) for Inverse Lighting of Face Images", arXiv:1709.01993 (2017). A CNN regresses SH lighting from a single face. It lists "image forgery detection" only as a motivating application and runs no forensic experiments — [arXiv](https://arxiv.org/abs/1709.01993).
- Hadwiger & Riess, "Deep Metric Color Embeddings for Splicing Localization in Severely Degraded Images", arXiv:2206.10737 (2022). A colour or illuminant-related learned embedding, made for splicing **localization** under strong compression, where high-frequency cues fail. It is the closest Riess-lab deep successor to illuminant-colour forensics — [arXiv](https://arxiv.org/abs/2206.10737).
- A 2020 face paper ("Face Forgery Detection by 3D Decomposition", arXiv:2011.09737) decomposes faces into 3D shape, common texture, identity texture, ambient light and direct light. It notes that prior work found forgery evidence in "inconsistency among 2D illuminant maps", and argues that manipulation methods simulate direct light poorly — [arXiv HTML](https://arxiv.org/html/2011.09737) (snippet seen via web search; abstract not separately verified).

**Physics and geometry cues against generative images (2022-2024)**
- Farid, "Lighting (In)consistency of Paint by Text", arXiv:2207.13744 (2022). An SH-lighting consistency analysis of DALL-E 2 images. Global illumination is "relatively consistent with natural images", but lighting **within** an image "can be highly variable". This is an exploratory analysis, not an automated localizer — [arXiv](https://arxiv.org/abs/2207.13744); summary in [web result](https://deepai.org/publication/lighting-in-consistency-of-paint-by-text).
- Farid, "Perspective (In)consistency of Paint by Text", arXiv:2206.14617 (2022). DALL-E 2 images show geometric inconsistencies in vanishing points, cast shadows and reflections. The analysis is manual or semi-manual — [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.14617).
- Farid, Content Authenticity Initiative blog series (2025) on shadows, reflections, vanishing points and lighting environments. It argues that diffusion models without explicit 3D models "struggle to create images that are physically consistent", and warns this advantage may be temporary — [CAI lighting blog](https://contentauthenticity.org/blog/photo-forensics-from-lighting-environments), [CAI shadows/reflections](https://contentauthenticity.org/blog/photo-forensics-from-lighting-shadows-and-reflections), [PetaPixel 2025](https://petapixel.com/2025/07/23/three-ways-to-spot-a-fake-ai-generated-photo/).
- Sarkar, Mai, Mahapatra, Lazebnik, Forsyth, Bhattad, "Shadows Don't Lie and Lines Can't Bend! Generative Models don't know Projective Geometry...for now", arXiv:2311.17138 (CVPR 2024). Generated images are "prequalified" to fool signal-based detectors. Three classifiers then see only derived geometry, never pixels: (1) a perspective field, (2) detected lines, (3) object–shadow relations. These detect generated images "more reliably than SOTA local signal based detectors" across several generators. Saliency maps localize the geometric problems, but the task is **image-level** real-vs-generated, not edit localization — [arXiv](https://arxiv.org/abs/2311.17138).
- Ciamarra, Caldelli, Becattini, Seidenari, Del Bimbo, "Deepfake detection by exploiting surface anomalies: the SurFake approach", arXiv:2310.20621 (2023). Builds a descriptor of the depicted surfaces (scene geometry and illumination) and trains a CNN for deepfake classification on FF++. It improves accuracy when fused with RGB — [arXiv](https://arxiv.org/abs/2310.20621).

**Recent (2025-2026) physics-guided detectors and localizers**
- Fei, Jia, Huang, Zhang, Zhou, "Exploring Specular Reflection Inconsistency for Generalizable Face Forgery Detection" (SRI-Net), arXiv:2602.06452 (2026). Estimates face texture with Retinex and isolates ambient versus direct light with spherical-harmonic modelling. It then separates the Phong specular component, and two-stage cross-attention relates specular, texture and direct light. Reported as superior on traditional and diffusion deepfake datasets. Face-only, image-level classification — [arXiv](https://arxiv.org/abs/2602.06452), [HTML](https://arxiv.org/html/2602.06452).
- Yu, Guo, Ma, Yang, "When 2D Cues Fail: Improving Image Manipulation Localization with Reliable 3D Geometry" (GFrame), arXiv:2607.18040 (July 2026). Monocular reconstruction gives depth and surface normals for manipulated images. The method **estimates their reliability** (geometry is noisy on manipulated images), fuses the reliable cues with RGB features across scales, and outputs **pixel-level IML**. It claims "excellent performance under limited budget constraints". The abstract mentions no lighting, shading, intrinsics or SH model. Code not yet released — [arXiv](https://arxiv.org/abs/2607.18040).
- Esfahani, Salem, Alser, Calhoun, "Do Vision Models Learn Physical Constraints or Rendering Shortcuts? A Counterfactual Benchmark for Grounded Physical Consistency", arXiv:2610.09205 (Oct 2026). 5,500 Mitsuba-3 images in 500 scene families, each with one clean image and ten violations (shadow, reflection, support, surface response, occlusion). Labels include masks and boxes. VLM baselines (LLaVA-1.5-7B, Qwen2.5-VL-7B, InternVL3.5-8B) reach 64.0–67.8% category macro-F1. LLaVA drops from 64.0% to 40.8% under intervention shift, so the models partly learn rendering shortcuts — [arXiv](https://arxiv.org/abs/2610.09205).
- Tan et al., "Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images", arXiv:2510.10231 (2025). Targets semantic anomalies, including "violations of physical laws", in AIGC with MLLM reasoning. Not a lighting-model approach — [arXiv](https://arxiv.org/abs/2510.10231).
- Near-miss (the name says "physics", the method does not): Qiu, Zhao, Qu, "Beyond Semantics: Uncovering the Physics of Fakes via Universal Physical Descriptors…", arXiv:2604.04608 (2026). Its "physical features" are low-level statistics (Laplacian variance, Sobel stats, residual noise variance) injected into CLIP, giving 99.8% on GenImage subsets (Wukong, SDv1.4). It uses no lighting or geometry — [arXiv](https://arxiv.org/abs/2604.04608).
- Near-miss (optics, not lighting): Jeon & Woo, "Seeing Through the Blur: Unlocking Defocus Maps for Deepfake Detection", arXiv:2509.23289 (2025) — [arXiv](https://arxiv.org/abs/2509.23289).

### Inferences
- Per-region **lighting-model** localization (fit a global model, then score each region's residual) has not been automated with modern estimators in any paper found. The per-region localizers found (Hadwiger & Riess 2022, GFrame 2026) do not fit an explicit lighting model.
- The face line (2011.09737, SRI-Net 2602.06452) is the only place where SH or Phong lighting models are fitted inside a learned detector, and it stays face-specific and image-level.

### Gaps
- Quantitative numbers (AUC, F1, IoU) for Pomari 2018, Shadows Don't Lie, SRI-Net and GFrame were not extracted; that needs a full-text read.
- Non-arXiv TIFS, WIFS and IH&MMSec work on deep illuminant or shadow forensics in 2016-2022 (for example Mazumdar & Bora's shadow/lighting work, or Kumar et al.) was not verified and may exist.
- I found no 2024-2026 Farid-lab peer-reviewed paper that automates shadow or lighting analysis of diffusion images. The 2025 material is blog posts and interviews.

## Q2. Has anyone combined monocular normals + intrinsic decomposition + SH lighting fits for forgery localization? Which work is closest, and what exactly differs?

### Takeaway
I found no paper that does this. Searches for intrinsic decomposition / albedo / shading combined with AIGC detection, forgery or localization returned nothing relevant. The closest works are **GFrame (arXiv:2607.18040)**, which uses monocular depth and normals for IML with no lighting model and no intrinsics, and **SRI-Net (arXiv:2602.06452)**, which uses Retinex texture plus an SH/Phong decomposition but only for faces and at image level. **Farid 2022 (arXiv:2207.13744)** is the closest conceptually (SH lighting consistency within DALL-E 2 images), but it is not automated at scale, uses no foundation models, and does no edit localization.

### Cited Findings
- No hit on arXiv for (intrinsic image OR albedo OR surface normal OR shading) AND (AI-generated OR forgery OR deepfake OR manipulation OR inpainting) AND (detection OR localization). Only GFrame was relevant among the 6 results — [arXiv GFrame](https://arxiv.org/abs/2607.18040).
- A targeted web search for intrinsic decomposition in AIGC detection found none. The tools themselves (Careaga & Aksoy ordinal shading, TOG 2023; colorful diffuse intrinsics, TOG 2024) exist, but no forensic use of them turned up — [compphoto/Intrinsic](https://github.com/compphoto/Intrinsic), [Ordinal shading](https://yaksoy.github.io/intrinsic/).
- A targeted web search for "lighting inconsistency deep learning forgery localization spherical harmonics normals 2023-2025" found no deep-network pixel-level localizer using SH and normals. The nearest hits were SRI-Net and the 2012 EUSIPCO shape-from-shading forgery paper — [EUSIPCO 2012 PDF](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2012/Conference/papers/1569582665.pdf), [SRI-Net HTML](https://arxiv.org/html/2602.06452).
- GFrame's differences from Idea 1: it uses **depth and normals only, as learned features fused with RGB**, plus a reliability estimate. It has no albedo/shading split, no global SH or light-direction fit, no illuminant-colour check, and no training-free residual scoring — [arXiv](https://arxiv.org/abs/2607.18040).
- Sarkar et al. use derived geometry (perspective fields, lines, object–shadow pairs) but for image-level classification of whole generations, with no SH lighting fit — [arXiv](https://arxiv.org/abs/2311.17138).

### Inferences
- Idea 1 looks **unpublished** in its specific form: foundation normals + foundation intrinsics → global SH or light fit → per-region residual map, evaluated on inpainting and instruction-edit datasets. GFrame (July 2026) now holds "monocular normals help IML", so novelty should rest on the explicit image-formation model (shading ≈ SH(n)), intrinsics, illuminant colour, training-free and interpretable residuals, and evaluation on GPT-Image, Nano-Banana and FLUX Kontext edits. GFrame should be cited and used as a baseline.
- Risk: estimator outputs (Marigold, DSINE) may be "self-consistent" on edited regions because they are run on the edited image. GFrame explicitly reports that geometry on manipulated images is noisy, which could be either a signal or a confound.

### Gaps
- GFrame's datasets and numbers, and whether its full text mentions shading or lighting, were not checked (abstract only).
- I could not rule out a non-arXiv conference paper (for example a 2025-2026 WIFS or ICASSP short paper).

## Q3. Has any paper used generative relighting or neural inverse rendering for image forensics (detection or localization)? Near-misses?

### Takeaway
No. I found no paper that uses relighting models (IC-Light, LightLab, DiffusionRenderer, Neural Gaffer, DiLightNet, etc.) or inverse rendering as a forensic **tool**. The nearest are: (a) relighting as a **manipulation type** in a benchmark (BG-REAL, 2026); (b) Farid- and Sarkar-style physics analyses; (c) diffusion-reconstruction detectors (DIRE-like), which use a generator's reconstruction error, not relighting; (d) face-lighting-decomposition deepfake detectors.

### Cited Findings
- arXiv query relighting AND (forensics OR fake detection OR deepfake OR generated image detection OR forgery) returned only 4 hits, none using relighting as a detector. The most relevant was BG-REAL — [arXiv](https://arxiv.org/abs/2607.26232).
- Uluirmak & Kurban, "Image Forgery Detection under Generative and Composited Background Replacement: A Benchmark with Processing-Matched Controls" (BG-REAL), arXiv:2607.26232 (2026). 25,624 images built from 2,485 Open Images photos. Edits include donor compositing, harmonization, generative background inpainting (FLUX.1 Fill among the held-out generators), **foreground-conditioned relighting** and local background edits, each paired with processing-matched authentic controls. Seven 2022-2026 detectors reach **zero-shot AUROC ≤ 0.745**. Localizers mark the preserved subject instead of the replaced background. An AIGC detector flags most authentic images passed through a diffusion autoencoder. Fine-tuned detectors reach about 0.95 AUROC, while background-forgery detectors stay below 0.80 — [arXiv](https://arxiv.org/abs/2607.26232).
- arXiv query (inverse rendering OR neural rendering OR lighting estimation) AND (forensic OR forgery OR fake OR manipulation detection) returned only detectors **of** neural-rendered or 3DGS content, not detectors **using** inverse rendering: Dong et al., arXiv:2411.08642 (2024); Fake3DGS benchmark, arXiv:2604.27590 (2026) — [arXiv 2411.08642](https://arxiv.org/abs/2411.08642), [arXiv 2604.27590](https://arxiv.org/abs/2604.27590).
- A web search for "relighting diffusion model forensics … augmentation" found nothing on relighting as a detector or augmentation. Related diffusion-forensics items: DiffForensics (CVPR 2024, diffusion-pretrained IFDL), "Diffusion models meet image counter-forensics" (WACV 2024, arXiv:2311.13629, where diffusion purification defeats Noiseprint, ManTraNet and Choi), and DiffCoR (reconstruction-difference AIGC detection) — [DiffForensics](https://www.computer.org/csdl/proceedings-article/cvpr/2024/530000m765/20hOHy2oZpu), [arXiv 2311.13629](https://arxiv.org/abs/2311.13629), [DiffCoR](https://www.computer.org/csdl/journal/oj/2025/01/11018794/278Rb188aek).
- Near-miss, face lighting decomposition: 2011.09737 (3D decomposition with ambient and direct light) and SRI-Net 2602.06452 (SH + Phong specular) — [arXiv 2602.06452](https://arxiv.org/abs/2602.06452).
- Near-miss, data augmentation via appearance transport for deepfake training data: Zhu et al., "AOT: Appearance Optimal Transport Based Identity Swapping for Forgery Detection", arXiv:2011.02674 (2020). It generates face swaps with large appearance and lighting variance to improve detectors. It is not a relighting model used for detection — [arXiv](https://arxiv.org/abs/2011.02674).
- Relighting and shadow-generation models are increasingly used as **editing** tools that remove the very cues Idea 1 relies on. Examples: Light-Geometry Interaction maps for joint shadow generation and relighting from monocular depth (arXiv:2602.21820, 2026), PixHt-Lab (arXiv:2303.00137), and MirrorVerse (arXiv:2504.15397), which notes that diffusion editors "often fail to fully adhere to physical laws … shadows, reflections" — [arXiv 2602.21820](https://arxiv.org/abs/2602.21820), [arXiv 2303.00137](https://arxiv.org/abs/2303.00137), [arXiv 2504.15397](https://arxiv.org/abs/2504.15397).

### Inferences
- Idea 2 (relight from several light positions and use cross-relighting inconsistency or residuals to localize edits, or use inverse rendering as a plausibility test) appears **unpublished**. Conceptually it is a relighting analogue of reconstruction-error detectors (DIRE, DiffCoR) and of Farid's lighting-consistency tests.
- BG-REAL shows that relighting is now part of the threat model and that current detectors fail on it. A relighting-aware forensic method would therefore address a documented gap, but it must avoid flagging authentic images that were relit or harmonized as a benign edit.

### Gaps
- I did not verify whether the LightLab (SIGGRAPH 2025), IC-Light (ICLR 2025), DiffusionRenderer (CVPR 2025) or Neural Gaffer (NeurIPS 2024) papers discuss forensic misuse or detection in their ethics or limitations sections; no full texts were read.
- Search engines may miss workshop papers (CVPRW Media Forensics 2025-2026).

## Q4. How do physics cues perform on modern diffusion and LLM edits and under re-sharing (JPEG/resize)? Do generators produce physically inconsistent lighting, shadows or reflections?

### Takeaway
The evidence that generators violate projective, shadow and lighting physics is consistent (Farid 2022 on DALL-E 2; Sarkar et al. CVPR 2024 across several generators; MirrorVerse 2025; the Esfahani 2026 editing benchmark). It comes with explicit "for now" caveats. I found **no quantitative evaluation of physics cues on GPT-Image, Nano-Banana or FLUX Kontext instruction edits**, and no JPEG or resize robustness study of physics-based detectors. The only robustness hints are indirect: Hadwiger & Riess (2022) designed colour embeddings for severely degraded images, and the classical argument holds that geometric and lighting cues survive compression better than high-frequency traces.

### Cited Findings
- DALL-E 2 lighting is highly variable within an image while globally plausible — [Farid arXiv:2207.13744](https://arxiv.org/abs/2207.13744).
- DALL-E 2 has perspective, shadow and reflection inconsistencies — [Farid arXiv:2206.14617](https://ar5iv.labs.arxiv.org/html/2206.14617).
- Pixel-blind geometric classifiers detect prequalified generated images better than SOTA signal detectors, but "…for now" — [Sarkar et al. arXiv:2311.17138](https://arxiv.org/abs/2311.17138).
- Diffusion editors often fail to obey physical laws for shadows and reflections — [MirrorVerse arXiv:2504.15397](https://arxiv.org/abs/2504.15397); editing models may add an object that casts no shadow or a mirror that fails to reflect, and VLM diagnosis reaches only 64-68% macro-F1 and is shortcut-prone — [Esfahani et al. arXiv:2610.09205](https://arxiv.org/abs/2610.09205).
- Under realistic background replacement and relighting edits, current detectors (low-level, localizers, AIGC) reach zero-shot AUROC ≤ 0.745 — [BG-REAL arXiv:2607.26232](https://arxiv.org/abs/2607.26232).
- High-frequency forensic traces are destroyed by diffusion purification, which motivates semantic or physical cues — [arXiv:2311.13629](https://arxiv.org/abs/2311.13629). Colour-based splicing localization was designed for strong compression — [Hadwiger & Riess arXiv:2206.10737](https://arxiv.org/abs/2206.10737).
- Farid warns that generators may eventually learn geometry, so these cues may be temporary — [CAI blog](https://contentauthenticity.org/blog/photo-forensics-from-lighting-shadows-and-reflections).

### Inferences
- Physics residuals should be largely invariant to JPEG and resizing, since normals, intrinsics and SH fits operate at low frequency. This is plausible but **untested** in the literature found.
- Instruction editors such as Kontext, GPT-Image and Nano-Banana regenerate or re-harmonize the whole image. That may make lighting globally consistent (shrinking the signal) while shadow and reflection errors remain. This is an open empirical question.

### Gaps
- No published numbers were found for physics cues on GPT-Image, Nano-Banana or FLUX Kontext edits.
- No published JPEG/resize robustness curves were found for any lighting-based detector from 2016-2026.

## Q5. What remains open or novel?

### Takeaway
Both ideas appear open as of 2026-11-04. The nearest prior art sets the bar: GFrame (geometry-aided IML, 2026), SRI-Net (SH/specular face detection, 2026), Shadows Don't Lie (geometry classifiers, CVPR 2024), Farid 2022 (SH lighting of DALL-E 2) and BG-REAL (relighting as a manipulation, 2026).

### Cited Findings
- No prior work found combines foundation normals with intrinsics and SH fits for localization (Q2 searches) — [GFrame](https://arxiv.org/abs/2607.18040) is the nearest and differs as described above.
- No prior work found uses relighting or inverse rendering as a forensic instrument (Q3 searches) — [BG-REAL](https://arxiv.org/abs/2607.26232) treats relighting only as an attack.
- A per-region physical-violation localization benchmark exists only for synthetic Mitsuba scenes — [arXiv:2610.09205](https://arxiv.org/abs/2610.09205).

### Inferences
Open contributions, ranked by novelty:
1. A training-free (or lightly calibrated) **per-region shading and illuminant-consistency map**. Foundation normals (Marigold-normals, DSINE) plus intrinsic shading (Careaga & Aksoy, Marigold-IID) feed a robust global SH-2 fit, with per-region residuals and chromaticity checks, evaluated on real inpainting and instruction-edit benchmarks with JPEG/resize sweeps. Required baselines: GFrame, TruFor-class localizers, and Shadows Don't Lie features.
2. **Relighting-consistency forensics.** Relight the image under K lights with IC-Light, DiffusionRenderer or Neural Gaffer and measure per-region disagreement between relit outputs and the physically predicted shading change (SH(n) under the new light). Alternatively, measure inverse-rendering reconstruction error per region. Nothing found.
3. A **combined detector** that fuses physics residual maps with signal-level localizers, motivated by BG-REAL's finding that signal localizers mark the wrong region on background or relighting edits.
4. Robustness and adversarial analysis: do relighting or harmonization editors (IC-Light, LightLab) erase physics cues? This is the attack side of Idea 2.

### Gaps
- Novelty claims rest on arXiv plus web search. Before publishing, a Google Scholar or Semantic Scholar sweep and a check of the CVPRW 2025-2026 Media Forensics and WIFS 2024-2025 proceedings are still needed.
- The relighting papers' own texts were not checked for forensic discussion.
