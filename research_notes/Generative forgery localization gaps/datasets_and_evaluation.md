# Datasets, Benchmarks and Evaluation Protocols for Localizing AI-Generated / AI-Edited Image Manipulations (2023 – Oct 2026)

Research date: 2026-10-30. Sources: arXiv abstracts/HTML, GitHub READMEs, Hugging Face cards and file listings.
Facts marked **[unverified-memory]** come from the researcher's background knowledge and were NOT re-checked against a source in this session. The report writer should treat them as tentative or drop them.

---

## Q1. Benchmark inventory: size, generators/editors, edit types, mask method, real sources, formats, licence, release status

### Takeaway
There are now more than 15 localization benchmarks for generative edits. They fall into three generations. (1) **Masked-inpainting sets** (CocoGlide, AutoSplice, TGIF/TGIF2, COCO-Inpaint, GIM, BR-Gen, OpenSDI-local, SID-Set) have masks "by construction". (2) **Mask-free/instruction-editing sets** (DEAL-300K, FragFake, MagicBrush-derived, LocateEdit-Bench, PromptForge-350k, PIXAR, EditStream) have masks that must be *inferred* after editing. (3) **Diagnostic, multi-axis or multi-turn sets** (DiffSeg30k, GIFTBench, Semi-Truths). COCO is the dominant real-image source. Licences and file formats are frequently undocumented, and several 2026 sets are "release upon acceptance".

### Master table

| Benchmark (ID) | Year | Size | Generators / editors | Edit types | GT mask method | Real source | Format / licence / release |
|---|---|---|---|---|---|---|---|
| **CocoGlide** (introduced in TruFor, arXiv 2212.10957) | 2023 | 512 images [unverified-memory] | GLIDE text-guided inpainting [unverified-memory] | Object inpainting in COCO-annotated regions | Inpainting mask = GT (by construction) | COCO | Distributed with TruFor; widely used as an external test set, e.g. by DEAL-300K ([2511.23377](https://arxiv.org/abs/2511.23377)) and GIFTBench ([2610.01778](https://arxiv.org/html/2610.01778)) |
| **AutoSplice** (Jia et al., CVPRW 2023) | 2023 | ~5.9k images (3,621 manipulated) [unverified-memory] | DALL·E 2 [unverified-memory] | Text-prompted local edits | Edit mask | Visual Genome [unverified-memory] | Used as an external test in GIFTBench ([2610.01778](https://arxiv.org/html/2610.01778)). The HF test bundle contains `AutoSplice_tp.tar`, 269 MB ([HF tree](https://huggingface.co/api/datasets/youqiwong/AIGC-Loc-Testsets/tree/main)) |
| **MagicBrush** (arXiv 2306.10012) and MagicBrush-derived test sets | 2023 | ~10k edit triplets [unverified-memory] | DALL·E 2 with human-drawn masks [unverified-memory] | Instruction-guided edits, multi-turn | Human-drawn inpainting mask [unverified-memory] | COCO [unverified-memory] | Editing dataset, re-purposed for forensics. FragFake uses MagicBrush as one of its editors ([2505.15644](https://arxiv.org/abs/2505.15644v2)). The HF bundle has `MagicBrush.tar`, 1.13 GB, plus a `MagicBrush/` folder ([HF tree](https://huggingface.co/api/datasets/youqiwong/AIGC-Loc-Testsets/tree/main)) |
| **TGIF** (arXiv 2407.11566) | 2024 | ≈75k forged images | SD2, SDXL, Adobe Firefly | Text-guided inpainting; two variants: spliced (SP) and fully regenerated (FR) | Inpainting mask (COCO segmentation mask or bounding box) | MS-COCO | Code and data at [github.com/IDLabMedia/tgif-dataset](https://github.com/IDLabMedia/tgif-dataset) ([2407.11566](https://arxiv.org/abs/2407.11566)) |
| **TGIF2** (arXiv 2603.28613) | Mar 2026 | 3,124 authentic images; 271,788 manipulated images across 19 subsets (74,976 inherited from TGIF); ~80/10/10 split | Adobe Firefly (Photoshop 25.4.0, SP only), SD2-inpainting, SDXL-inpainting (FR only), FLUX.1 [schnell], [dev], Fill [dev]. FLUX.1 [pro] excluded | SP vs FR; **semantic masks** (COCO segmentation or bbox) vs **random non-semantic rectangles** (≥64 px, ≤60% of width/height, empty prompt). Also a generative super-resolution attack set | Inpainting mask | MS-COCO val2017 (≤50 images per category; knife and frisbee excluded; both sides ≥512 px; object bbox between 64² px and 512×512) | GitHub (same repo). Dataset licence not stated in the paper; COCO is CC BY 4.0. File format not stated ([2603.28613 HTML](https://arxiv.org/html/2603.28613)) |
| **GIM** (arXiv 2406.16531) | 2024 | >1M AI-manipulated / real pairs | "State-of-the-art generators"; pipeline combines SAM + LLM + generative models (specific models: SD, GLIDE, DDNM [unverified-memory]) | Multiple manipulation tasks (inpainting/removal/editing) | SAM-derived region = inpainting mask | Broad image classes (ImageNet/VOC [unverified-memory]) | Two benchmark settings ([2406.16531](https://arxiv.org/abs/2406.16531)) |
| **Semi-Truths** (arXiv 2411.07472) | Nov 2024 (NeurIPS D&B [unverified-memory]) | 27,600 real images, 223,400 masks, **1,472,700** AI-augmented images | Diverse diffusion models (inpainting and prompt-based editing) | Targeted localized perturbations, with metadata on the magnitude of change | Masks from the augmentation pipeline | Multiple distributions | Pipeline at [github.com/J-Kruk/SemiTruths](https://github.com/J-Kruk/SemiTruths) ([2411.07472](https://arxiv.org/abs/2411.07472)). Primarily a *detection*-robustness set |
| **SID-Set** (SIDA, arXiv 2412.04292, CVPR 2025) | Dec 2024 | 300K images (train 210k / val 30k / test 60k; **test withheld**) | Not specified on the HF card | Real / fully synthetic / tampered (label 0/1/2) | Card says only "binary mask highlighting manipulated regions"; method not documented | OpenImages V7 (+ COCO, Flickr30k material) | Parquet on HF, **CC BY 4.0** ([HF card](https://huggingface.co/datasets/saberzl/SID_Set); [2412.04292](https://arxiv.org/abs/2412.04292)) |
| **OpenSDI / OpenSDID** (arXiv 2503.19653, CVPR 2025) | Mar 2025 | ~300K. Train: 100K real + 100K fake (SD1.5 only). Test: 50K real + 50K fake (10K/10K per generator) | SD1.5, SD2.1, SDXL, SD3, Flux.1; VLMs simulate user prompts | Global synthesis plus local edits (object insertion, background changes) | Not described in the README | megalith-10m | HF datasets; **CC BY-SA 4.0**, academic use ([GitHub](https://github.com/iamwangyabin/OpenSDI); [2503.19653](https://arxiv.org/abs/2503.19653)) |
| **BR-Gen** (arXiv 2504.11922, AAAI 2026) | Apr 2025 | 150,000 locally forged images | LaMa, MAT, SDXL, BrushNet, PowerPaint | **Background/"stuff"** regions (sky, ground, wall, grass, vegetation), to counter object-region bias | Perception-Creation-Evaluation pipeline; quality filtering with BRISQUE, DreamSim, CLIP Score | Not stated in snippet | [github.com/clpbc/BR-Gen](https://github.com/clpbc/BR-Gen) ([2504.11922](https://arxiv.org/html/2504.11922v3)). The HF bundle has `BRGen_Stuff_Val.tar`, 1.40 GB |
| **COCO-Inpaint** (arXiv 2504.18361) | Apr 2025 | 238,302 inpainted images | Six inpainting models | **Four mask-generation strategies**, optional text guidance; designed to emphasise intrinsic inconsistencies "rather than superficial semantic artifacts such as object shapes" | Inpainting mask | COCO | Three-metric protocol ([2504.18361](https://arxiv.org/abs/2504.18361)) |
| **FragFake** (arXiv 2505.15644; v2 retitled "Can VLMs Detect and Localize Fine-Grained AI-Edited Images?") | May 2025 | 20,222 edited images | MagicBrush, GoT, UltraEdit, Gemini-IG | Fine-grained object edits | Automated pipeline | — | Code and HF `Vincent-HKUSTGZ/FragFake`. Metrics are Region Precision and Object Precision (VLM-style, not pixel F1) ([2505.15644v2](https://arxiv.org/abs/2505.15644v2)) |
| **DiffSeg30k** (arXiv 2511.19111) | Nov 2025 | 30k images | 8 diffusion editing models | **Multi-turn (up to 3 sequential edits)**: additions, removals, attribute changes, with VLM-chosen regions | Per-model semantic labels (segmentation by editor identity) | COCO images / prompts | HF `Chaos2629/Diffseg30k` ([2511.19111](https://arxiv.org/abs/2511.19111)) |
| **DEAL-300K** (arXiv 2511.23377) | Nov 2025 | >300,000 annotated images | MLLM-generated instructions + **mask-free** diffusion editor | Instruction edits | **Active-learning change-detection pipeline** produces pixel labels | — | [github.com/ymhzyj/DEAL-300K](https://github.com/ymhzyj/DEAL-300K). Baseline: F1 82.56% in-domain, 80.97% on CocoGlide ([2511.23377](https://arxiv.org/abs/2511.23377)) |
| **NeXT-IMDL** (arXiv 2512.23374) | Dec 2025 | — | 32 editing techniques, including GPT-Image-1 and Gemini-2.0-flash-Image (per search snippet) | — | — | — | Search snippet only; the HTML fetch returned 404. Details unverified ([abs](https://arxiv.org/abs/2512.23374v1)) |
| **ManipBench / ManipShield** (arXiv 2511.14259) | Nov 2025 | — | 25 editing models, including Nano-Banana and GPT-Image (per snippet) | — | — | — | "Released upon publication" (snippet only) ([pdf](https://arxiv.org/pdf/2511.14259v1)) |
| **Omni-273k** (Omni-IML, arXiv 2411.14823, ICLR 2026) | Nov 2024 → ICLR 2026 | ~273k (by name) | Mixed: natural, document, face, scene-text tampering | Generalist IML | Existing masks plus automatic chain-of-thought **text** descriptions of artifacts | Multiple domains | [github.com/qcf-568/OmniIML](https://github.com/qcf-568/OmniIML). Earlier versions said data would come "later"; current availability unverified ([2411.14823](https://arxiv.org/html/2411.14823v2)) |
| **LocateEdit-Bench** (arXiv 2602.05577) | Feb 2026 | 231K edited images per the abstract (HTML renders "231231K", likely a typo); ≥50K per editor; test 6K per editor (24K) | **Qwen-Image-Edit, Flux-Kontext, Step1X-Edit, BAGEL** | Object swap, object addition, attribute modification | **ChatGPT names the edited object → SAM3 segments it**; masks kept only if SAM3 confidence > 0.5. No pixel-difference check and no quantitative mask validation | OmniEdit subset (LAION and OpenImages), resized to ~1024²; InternVL2 filtering | "Open-sourced upon acceptance"; format not given ([2602.05577](https://arxiv.org/abs/2602.05577); [HTML](https://arxiv.org/html/2602.05577)) |
| **PIXAR** (arXiv 2603.20193, "From Masks to Pixels and Meaning") | Mar 2026 | >420K training pairs; balanced 40K test set | Training: Qwen-Image. Test adds FLUX.2, Gemini 2.5, Gemini 3, GPT-image-1.5, Seedream 4.5 | 8 single-edit types (intra-/inter-class replacement, removal, addition, colour, motion, material, background change) plus a multi-edit subset (2–3 edits) | **Pixel difference** D=\|I_orig−I_gen\|, **threshold τ=0.05** (ablations at 0.1 and 0.2); **ORB + RANSAC homography** alignment when resolution or geometry changes (falls back to unaligned if alignment fails); filters: ≥0.2 overlap with input mask, 80% grid-cell concentration, Qwen3 fidelity ≥9/10, human realism ≥4/5 | COCO | [github.com/VILA-Lab/PIXAR](https://github.com/VILA-Lab/PIXAR). Paper CC BY 4.0; dataset licence not stated ([2603.20193 HTML](https://arxiv.org/html/2603.20193)) |
| **PromptForge-350k** (arXiv 2603.29386) | Mar 2026 | ~350k | Four prompt-based editors; a secondary source says they include Nano-Banana, but the abstract does not name them | Prompt-based edits | **Automated: keypoint alignment + semantic-space similarity** | — | Availability not stated on the abstract page. ICL-Net: 62.5% IoU in-domain, 41.5% average IoU on unseen editors ([abs](https://arxiv.org/abs/2603.29386); [emergentmind snippet](https://api.emergentmind.com/papers/2603.29386)) |
| **EditStream** (in FLAME, arXiv 2606.02178) | Jun 2026 | Not stated in the abstract | Instruction-based editors (continuous pipeline) | Instruction edits | Automated (details not in the abstract) | — | Code at [github.com/phoenixnir/FLAME](https://github.com/phoenixnir/FLAME). Pitched as a *continuous* data-synthesis pipeline to keep up with new generators ([2606.02178](https://arxiv.org/abs/2606.02178)) |
| **GIFTBench** (arXiv 2610.01778) | Oct 2026 | 115,013 manipulated images (103,319 train / 11,694 test) + 20,000 real | Classical compositing, traditional and learned inpainting, diffusion editors; VLMs (Qwen-VL, InternVL) plan and verify | 4 sources (Splicing, Copy-Move, Removal, AI-Edit) × 3 semantic targets (Object/Component/Background) × 3 operations (Add/Remove/Replace) × 3 compositions (Single/Fusion/Multi-Source) | Changed pixels relative to source; binary 0/255 masks; samples with manipulated area <0.5% discarded | ADE20K (not redistributed) | Showcase page only ([2610.01778](https://arxiv.org/abs/2610.01778)) |
| **"AIGC-Loc-Testsets"** (HF `youqiwong`) | 2026 | 87 GB | Folders and tars: `GPT-Image-2/`, `NanoBanana2/`, `COCO-Inpaint/`, `DiffSeg30k/`, `MagicBrush/`, `PIXAR_{flux2, gemini, gemini3, gpt, qwen}.tar` (6.6–11.2 GB each), `SID_Set_tp.tar` (2.28 GB), `glide_tp.tar` (=CocoGlide, 62 MB), `AutoSplice_tp.tar`, `BRGen_Stuff_Val.tar`, `retrain_pred_masks_archives/` | Aggregated test sets ("_tp" = tampered positives only?) | Not documented | — | **No dataset card**, no licence, no paper ID; WebDataset tars (viewer fails) ([HF page](https://huggingface.co/datasets/youqiwong/AIGC-Loc-Testsets); [HF tree](https://huggingface.co/api/datasets/youqiwong/AIGC-Loc-Testsets/tree/main)) |
| **IMDL-BenCo** (arXiv 2406.10580, NeurIPS 2024 D&B [unverified-memory]) | 2024 | Codebase and benchmark (not a new dataset) | 8 SOTA IMDL models | — | — | — | 2 training/evaluation protocols, 15 GPU metrics, 3 robustness evaluations ([2406.10580](https://arxiv.org/abs/2406.10580)) |
| Related: **INP-X** (Inpainting Exchange, arXiv 2602.00192) | Jan 2026 | ~90K test images (real, inpainted, exchanged) | Inpainting models | Restores original pixels outside the mask | Exact mask by construction | — | Diagnostic set ([2602.00192](https://arxiv.org/abs/2602.00192)) |

### Cited Findings
- TGIF2 extends TGIF with FLUX.1 edits and random non-semantic masks, and adds a generative super-resolution attack evaluation — [arXiv 2603.28613](https://arxiv.org/abs/2603.28613)
- TGIF2 counts: 3,124 authentic COCO val2017 images; 18,744 manipulated images per SP or FR subset and 9,372 per Random subset; 271,788 in total; SD2-FR outputs at 512×512 and SDXL/FLUX up to 1024 px — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- LocateEdit-Bench masks: ChatGPT identifies the edited object and SAM3 segments it (confidence > 0.5); no pixel-difference verification — [LocateEdit HTML](https://arxiv.org/html/2602.05577)
- PIXAR labels come from thresholded pixel difference (τ=0.05) after ORB/RANSAC homography alignment — [PIXAR HTML](https://arxiv.org/html/2603.20193)
- PromptForge-350k masks come from "keypoint alignment and semantic space similarity" — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)
- DEAL-300K uses an active-learning change-detection pipeline over mask-free diffusion edits — [arXiv 2511.23377](https://arxiv.org/abs/2511.23377)
- The SID-Set test split (60k) is withheld "to prevent leakage"; licence CC BY 4.0 — [HF SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
- OpenSDID trains on SD1.5 only, tests on 5 generators, real images from megalith-10m, CC BY-SA 4.0 — [OpenSDI GitHub](https://github.com/iamwangyabin/OpenSDI)
- The youqiwong HF repository has no dataset card but contains GPT-Image-2 and NanoBanana2 folders plus re-packaged PIXAR, SID-Set, CocoGlide, AutoSplice, BR-Gen, MagicBrush, COCO-Inpaint, and DiffSeg30k test sets — [HF tree API](https://huggingface.co/api/datasets/youqiwong/AIGC-Loc-Testsets/tree/main)

### Inferences
- COCO is the real-image source for CocoGlide, TGIF/TGIF2, COCO-Inpaint, PIXAR, DiffSeg30k, MagicBrush (and partly SID-Set). A model trained on any of these and tested on another sees overlapping scene/content statistics, and possibly the *same* source images. TGIF2 uses COCO **val2017**, which other sets may also draw from; no paper checked cross-dataset image-ID overlap.
- The youqiwong bundle looks like the evaluation suite of an unpublished paper (it includes `retrain_pred_masks_archives`). Its GPT-Image-2 and NanoBanana2 sets are the only ones found that target those 2026 closed editors, but their mask provenance is undocumented. Any use should state that.
- Release status is a real gap. LocateEdit-Bench ("upon acceptance"), ManipBench ("upon publication"), the GIFTBench showcase page only, PromptForge (unstated), and the SID-Set withheld test set mean that most 2026 instruction-editing benchmarks cannot yet be independently reproduced.

### Gaps
- Could not verify exact CocoGlide/AutoSplice/MagicBrush/GIM details in this session (marked [unverified-memory]).
- PromptForge-350k editor names, EditStream size, NeXT-IMDL details (HTML 404), and OmniIML release status were not confirmed.
- File formats (PNG vs JPEG) are **not stated** in the TGIF2, LocateEdit, PIXAR, or GIFTBench texts read. A dataset-by-dataset format audit was not possible without downloading.
- No 2026 benchmark found specifically for MLLM-native editors (GPT-Image-2, Nano Banana 2) other than the undocumented HF folders and PIXAR's GPT-image-1.5 / Gemini 3 test splits.

---

## Q2. Ground-truth masks for instruction-based / fully regenerated edits: methods, reliability, and the meaning of "authentic region"

### Takeaway
Four incompatible GT paradigms coexist:
- (a) the **inpainting mask** (TGIF/TGIF2, COCO-Inpaint, BR-Gen, CocoGlide);
- (b) **semantic segmentation of the named edited object** (LocateEdit-Bench: ChatGPT + SAM3);
- (c) **thresholded pixel difference after geometric alignment** (PIXAR: τ=0.05, ORB+RANSAC);
- (d) **learned/structured change detection** (PromptForge keypoint + semantic similarity; DEAL-300K active-learning change detection).

None of these papers reports a quantitative validation of mask accuracy. For fully regenerated (FR) images, the mask labels intent, not the set of changed pixels. Pixels outside the mask are also changed (VAE re-encoding), so "authentic region" is a convention.

### Cited Findings
- In FR inpainting, "the non-masked area is subtly altered as well". In one TGIF2 example a random-box edit "made notable changes to the sign" outside the mask. Localization in FR images remains "an open problem" — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- Original TGIF: diffusion approaches either splice the inpainted region back or "regenerate the entire image", and in the latter case traditional IFL "typically fail[s]" — [arXiv 2407.11566](https://arxiv.org/abs/2407.11566)
- PIXAR argues that mask-based benchmarks show "widespread misalignment between the human-defined groundtruth and true pixel-level tamper". It replaces masks with D=|I_orig−I_gen|>τ (τ=0.05 default, tested at 0.1 and 0.2). Misalignment is shown only qualitatively (Fig. 1), with no numeric measure. Scaling of D before thresholding is unspecified. If homography fails, the unaligned image is used — [PIXAR HTML](https://arxiv.org/html/2603.20193)
- PIXAR treats untouched pixels as "real" and frames the open question as where "the boundary between the 'real' and the 'generated'" lies — [PIXAR HTML](https://arxiv.org/html/2603.20193)
- LocateEdit-Bench GT is object-semantic (SAM3 segmentation of the ChatGPT-identified object, confidence > 0.5), with no further reliability analysis — [LocateEdit HTML](https://arxiv.org/html/2602.05577)
- PromptForge-350k uses automated keypoint alignment plus semantic-space similarity — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)
- DEAL-300K uses an active-learning change-detection pipeline over mask-free editor outputs — [arXiv 2511.23377](https://arxiv.org/abs/2511.23377)
- GIFTBench defines masks as pixels "changed relative to the original source" and drops samples under 0.5% area — [GIFTBench HTML](https://arxiv.org/html/2610.01778)
- INP-X: VAE reconstruction causes a "subtle, widespread spectral shift" that also affects unedited regions. Restoring the original pixels outside the edit drops detector accuracy from 91% to 55%, so detectors key on global VAE artifacts rather than the local edit — [arXiv 2602.00192](https://arxiv.org/abs/2602.00192)

### Inferences
- **Paradigm conflict:** for an attribute change such as "make the car red", a SAM3 object mask (LocateEdit) labels the whole car. A pixel-difference mask (PIXAR) labels only pixels above τ, possibly speckled and including background drift, and a keypoint/semantic method gives something in between. Models trained on one paradigm are penalised on another for *label-definition* reasons, not detection failure. No paper quantifies this.
- **Threshold sensitivity:** for whole-image regenerators (GPT-Image, Gemini), global colour/tone shifts can exceed τ=0.05 everywhere. Mask area then depends strongly on τ and alignment success, and silent alignment failure produces near-full-image masks. A sensitivity study (IoU of GT masks across τ, and against human annotation on a subset) is missing and would be a clear contribution.
- **A defensible definition:** report two labels for FR images, a *semantic-edit mask* (intent) and a *regenerated-pixel map* (|Δ|>τ after alignment). Evaluate either a three-class output (authentic / regenerated-but-unchanged / semantically edited) or graded "edit magnitude" maps. Semi-Truths' magnitude-of-change metadata ([2411.07472](https://arxiv.org/abs/2411.07472)) is a precedent for graded labels.

### Gaps
- No paper found that measures inter-annotator agreement or human-vs-automatic mask IoU for instruction-edit GT.
- PromptForge and EditStream mask-pipeline details (thresholds, failure rates) were not readable from the abstracts.

---

## Q3. Known dataset biases

### Takeaway
The documented biases are:
- **object/semantic bias:** models learn "where an object is" rather than "where pixels are generated", shown by TGIF2 random masks and motivating BR-Gen and COCO-Inpaint;
- **format bias:** real JPEG vs fake PNG;
- **global-VAE-artifact shortcut:** shown by INP-X;
- **resolution bias:** e.g. SD2-FR at 512² vs others at 1024²;
- **COCO source concentration**;
- **source/axis confounds in cross-dataset evaluation**, shown by GIFTBench.

Mask-size distributions are rarely reported. GIFTBench simply discards masks under 0.5% of the image area.

### Cited Findings
- **Object bias (TGIF2):** TruFor fine-tuned on SD2-FR semantic masks scores F1 0.50 on SD2 Sem but **0.23 on SD2 Rand**. On spliced images, TruFor averages Sem 0.77 vs Rand 0.72 and MMFusion 0.68 vs 0.59 — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- Training on Sem+Rand gives TruFor F1 Sem 0.72 / Rand 0.78 / All 0.75 on FR, versus 0.22 / 0.08 / 0.14 for the off-the-shelf model — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- **Region bias:** BR-Gen argues prior sets concentrate on salient objects or synthetic rectangles and targets stuff/background regions. TruFor (Protocol-MVSS) gets F1 .865 on Columbia but **.085 on Br-Gen** — [BR-Gen](https://arxiv.org/html/2504.11922v3); [GIFTBench HTML](https://arxiv.org/html/2610.01778)
- COCO-Inpaint uses four mask strategies explicitly to avoid "superficial semantic artifacts such as object shapes" — [arXiv 2504.18361](https://arxiv.org/abs/2504.18361)
- **Format bias:** real photos are usually JPEG while generated images are often PNG, creating a "trivial shortcut". The INP-X study converts everything to JPEG. So-Fake found authentic OOD images mostly JPEG and generated/tampered mostly PNG; re-encoding to a common format and resolution lowered every method's accuracy by about 1.7–4.5 points — [search summary of 2602.00192 and So-Fake 2505.18660](https://arxiv.org/pdf/2505.18660) (secondary snippet; not verified in full text)
- **Global-artifact shortcut:** detector accuracy falls from 91% to 55% under Inpainting Exchange — [arXiv 2602.00192](https://arxiv.org/abs/2602.00192)
- **Resolution heterogeneity:** TGIF2 SD2-FR images are 512×512 while SDXL/FLUX reach 1024 px. LocateEdit resizes everything to ~1024² — [TGIF2 HTML](https://arxiv.org/html/2603.28613); [LocateEdit HTML](https://arxiv.org/html/2602.05577)
- **Mask-size / region-statistic confound:** in GIFTBench, background edits "tend to occupy larger regions" with weaker boundary gradients, so semantic-axis gaps coincide with extent and texture changes. Its axes are "associated transfer patterns rather than strict one-variable-at-a-time interventions" — [GIFTBench HTML](https://arxiv.org/html/2610.01778)
- **Generator-only train split:** OpenSDID trains on SD1.5 only, with 5-generator testing — [OpenSDI GitHub](https://github.com/iamwangyabin/OpenSDI)
- **Leakage handling:** GIFTBench excludes IMD2020 from Protocol-CAT external evaluation because it was in that training set. TGIF2 states its split avoids "category overlap and image leakage" — [GIFTBench HTML](https://arxiv.org/html/2610.01778); [TGIF2 HTML](https://arxiv.org/html/2603.28613)

### Inferences
- The FR + random-mask result shows that a localizer can "succeed" by segmenting the prompt-named object. Any instruction-editing benchmark whose GT is a SAM object mask (LocateEdit) **structurally rewards object bias**. Strong numbers there (e.g. SegFormer F1 87.0) may partly reflect semantic segmentation of the edited object.
- Format and resolution normalisation is not a standard protocol for generative-localization benchmarks. A paper that re-encodes all real and fake images through an identical pipeline and reports the delta would fill a gap. So-Fake's 1.7–4.5 point shift is for detection, not localization.
- No benchmark found that controls for COCO image-ID overlap across CocoGlide / TGIF / COCO-Inpaint / PIXAR / DiffSeg30k when used for cross-dataset claims.

### Gaps
- No published mask-area histograms were found in the sources read. Mask-size distributions per dataset remain unverified.
- Format statistics (proportion JPEG vs PNG) per localization dataset are undocumented.

---

## Q4. Metric and protocol issues

### Takeaway
The de-facto protocol is per-image pixel F1 at a **fixed 0.5 threshold** (TGIF2, GIFTBench, LocateEdit for most models). Exceptions such as an EER threshold for ObjectFormer make comparisons inconsistent. Pixel AUC and accuracy are inflated by the dominant authentic class (LocateEdit: AUC ~95–97 vs F1 75–87, Dice/IoU much lower). Real images are often excluded from localization metrics, so false-positive behaviour on pristine images is unmeasured. No benchmark read reports calibration (ECE/reliability). Robustness protocols are shallow: JPEG plus blur, or resize. Exceptions are TGIF2's generative super-resolution and DiffSeg30k's multi-turn edits.

### Cited Findings
- TGIF2 uses pixel F1 at fixed threshold 0.5, IoU in the appendix, and AUC for SID "because some methods overpredict the fake class". No optimal-threshold F1 — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- GIFTBench computes F1/IoU per manipulated image at fixed 0.5 and macro-averages across datasets. **Real images are excluded from localization metrics.** Image AUC appears only on splits with both classes. Its "Avg" columns cover different dataset sets across tables (SparseViT .6029 vs .6273) — [GIFTBench HTML](https://arxiv.org/html/2610.01778)
- GIFTBench says off-diagonal transfer failures are "dominated by false negatives while false positives remain low" and that OOD performance "cannot be adequately characterized by a single aggregate score" — [GIFTBench HTML](https://arxiv.org/html/2610.01778)
- LocateEdit-Bench uses 0.5 for most models but an EER-derived threshold for ObjectFormer. **Inflation example:** SegFormer ACC 96.0 / AUC 97.5 / F1 87.0 / Dice 82.6 / mIoU 77.1; PSCC-Net AUC 95.1 vs F1 75.5 / mIoU 61.9. Robustness covers only JPEG (quality 0.75) and Gaussian blur (σ=1.0) — [LocateEdit HTML](https://arxiv.org/html/2602.05577)
- PIXAR reports both g-IoU (image-level mean) and dataset-level IoU, i.e. two averaging schemes. Its text and tables disagree (SIDA-7B IoU "6.9%→14.9%" in the text vs 11.8 in the table) — [PIXAR HTML](https://arxiv.org/html/2603.20193)
- TGIF2 also has internal inconsistencies: the Table 5 average row conflicts with per-subset values; an MMFusion ×4 Real-ESRGAN entry reads "0.51 (−0.51)" against a 0.69 original; the text says bicubic costs "0.02–0.11" F1 but the table shows −0.14 — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- IMDL-BenCo standardises 15 GPU metrics, 2 protocols, and 3 robustness evaluations, explicitly because inconsistent protocols cause "misleading model evaluations" — [arXiv 2406.10580](https://arxiv.org/abs/2406.10580)
- FragFake scores localization with Region Precision / Object Precision (VLM text output), which is not comparable to pixel F1 — [arXiv 2505.15644](https://arxiv.org/abs/2505.15644v2)
- **Generative super-resolution (TGIF2, F1):** CAT-Net 0.87 → 0.06 (Real-ESRGAN ×2) vs 0.76 with bicubic ×2; TruFor 0.78 → 0.21; MMFusion 0.69 → 0.18. SID AUC: PatchCraft 0.95 → 0.53/0.48. SPAI-ITW *rises* 0.65 → 0.83 — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- TGIF (2024): both IFL and SID fail under stronger compression and are less robust to WEBP — [arXiv 2407.11566](https://arxiv.org/abs/2407.11566)
- Multi-turn editing (up to 3 sequential edits) is built into DiffSeg30k, which found robustness to distortions a major challenge — [arXiv 2511.19111](https://arxiv.org/abs/2511.19111)
- PromptForge-350k claims under 1% IoU loss under "common degradations" — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)

### Inferences
- **Threshold:** a fixed 0.5 punishes miscalibrated but well-ranked models, while optimal-threshold F1 (per image or per dataset) leaks test labels. Reporting both, plus a threshold chosen on a held-out validation set from a *different* generator, is rarely done and would be a contribution.
- **Averaging:** per-image vs per-dataset IoU (PIXAR's g-IoU vs IoU) can diverge substantially when mask sizes vary. With tiny masks, per-image F1 is noisy and pixel-AUC is dominated by easy background. Area-stratified reporting (e.g. <1%, 1–10%, >10% of the image) is absent from every benchmark read.
- **Real-image false positives:** because GIFTBench and others exclude real images from pixel metrics, a model that paints every image "fake" is not penalised. A pristine-image false-positive-area metric is needed.
- **Calibration:** no benchmark reports ECE, reliability diagrams, or selective-prediction (risk–coverage) for localization maps. This gap is unaddressed in the sources read; CBDiff's ([2510.19597](https://arxiv.org/abs/2510.19597)) multiple-hypothesis maps are the nearest work on uncertainty.
- **Robustness:** the protocols that matter in deployment are mostly absent from instruction-editing benchmarks. These include social-media re-compression chains, generative SR/enhancement (only TGIF2), re-editing by a second MLLM editor (only DiffSeg30k-style multi-turn with diffusion editors), and screenshotting or resizing to the platform's maximum.

### Gaps
- Did not find a 2023–2026 paper systematically quantifying fixed-0.5 vs optimal-F1 gaps on generative-localization sets (IMDL-BenCo discusses evaluation standards, but specific numbers were not retrieved).
- No social-media-transmission protocol (e.g. upload to real platforms) was found for AI-edit localization.

---

## Q5. Largest cross-dataset / cross-generator generalization gaps (numbers)

### Takeaway
The largest documented gaps fall into five groups:
- **Cross-source:** Mesorch, trained on Copy-Move, scores F1 .84 in-domain vs .12 on AI-Edit (GIFTBench).
- **Off-the-shelf IFL on FR images:** TruFor F1 0.14, CAT-Net 0.02, vs 0.75–0.89 on spliced images (TGIF2).
- **Generative SR:** CAT-Net 0.87 → 0.06.
- **Region/background shift:** TruFor .865 → .085 on BR-Gen.
- **Unseen-editor drops on instruction editing:** PromptForge 62.5% → 41.5% IoU. PIXAR baselines sit at ~8–19% IoU on modern editors.

### Cited Findings
- GIFTBench Protocol I: Mesorch trained on Copy-Move, in-domain F1 .8406 vs AI-Edit .1201 (gap .7205, the largest in its matrices). Protocol II: trained on Background .9052 vs Object .3690. Protocol III: trained on Add .8551 vs Remove .3901. Multi-Source composition: TruFor .817 → .670 — [GIFTBench HTML](https://arxiv.org/html/2610.01778)
- GIFTBench external: TruFor .837 in-domain vs .164 on NIST16; SparseViT .862 vs .200; TruFor (MVSS protocol) Columbia .865 vs Br-Gen .085. Released checkpoints: SparseViT Splicing .640 vs Removal .249; IML-ViT .622 vs .236 — [GIFTBench HTML](https://arxiv.org/html/2610.01778)
- TGIF2 SP vs FR (off-the-shelf F1): CAT-Net 0.89 (SP) vs 0.02 (FR); TruFor 0.75 vs 0.14; MMFusion 0.64 vs 0.13. SID: B-Free AUC 1.00 on SD2/SDXL vs 0.85–0.88 on FLUX; of 11 methods strong on SD2/SDXL, only 4 keep AUC > 0.8 on all FLUX subsets — [TGIF2 HTML](https://arxiv.org/html/2603.28613)
- PromptForge-350k: ICL-Net 62.5% IoU in-domain vs 41.5% average IoU on unseen editors. A secondary source reports lower performance on closed-source Nano-Banana — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386); [emergentmind](https://api.emergentmind.com/papers/2603.29386)
- PIXAR: on its multi-editor test set, LISA-7B 8.3, SIDA-7B 11.8, FakeShield 9.3, PIXAR-7B 18.1, PIXAR-13B 19.3 IoU (%) — [PIXAR HTML](https://arxiv.org/html/2603.20193)
- LocateEdit Protocol 2 (cross-editor): all models degrade on unseen editors; exact numbers are only in Fig. 7. BAGEL is the hardest editor and Flux-Kontext the easiest — [LocateEdit HTML](https://arxiv.org/html/2602.05577)
- DEAL-300K: 82.56% F1 in-domain vs 80.97% on CocoGlide (small gap, but CocoGlide is old GLIDE inpainting) — [arXiv 2511.23377](https://arxiv.org/abs/2511.23377)
- OpenSDI MaskCLIP: +14.23% relative IoU over the second-best model on the SD1.5→multi-generator protocol (absolute numbers not retrieved) — [arXiv 2503.19653](https://arxiv.org/abs/2503.19653)
- INP-X: detector accuracy 91% → 55% when only the local edit remains — [arXiv 2602.00192](https://arxiv.org/abs/2602.00192)

### Inferences
- Gaps are largest when (i) the edit regime switches from splice to full regeneration, (ii) the semantic target switches between object and background, or (iii) a generative post-process is applied. Within a single regime, cross-generator gaps are moderate (e.g. PromptForge −21 IoU points).
- Closed MLLM editors (GPT-Image-x, Gemini/Nano Banana, Seedream) are the least-covered and appear hardest (PIXAR absolute IoU under 20%). Yet the only public test folders for GPT-Image-2 / NanoBanana2 (youqiwong) lack documentation. A documented, licence-clear benchmark for 2026 closed editors, with dual GT (semantic + pixel-diff) and a format-normalised protocol, is an open opportunity.

### Gaps
- LocateEdit cross-editor numbers (figure only), OpenSDI absolute IoU per generator, and the GPT-Image-2 / NanoBanana2 test results were not retrieved.
- No head-to-head of the same model across LocateEdit vs PIXAR vs PromptForge was found, so the mask-paradigm effect on cross-benchmark transfer cannot be quantified from the literature.
