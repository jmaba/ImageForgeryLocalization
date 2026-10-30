# Re-analysis of previous work, 2025-26 state of the art, and direction

_Written 2026-10-30. arXiv IDs were checked against arXiv on that date. Dataset facts come from the Hugging Face
dataset cards/files the sampler actually reads._

## 1. What the earlier projects established

| Repo | What it did | What we keep | What limited it |
|---|---|---|---|
| `NaturalVsComputerGeneratedImageAnalysis` | CIFAKE (32 px), SVM on 73 handcrafted features, ResNet ablation, DIRE-simple | ablation discipline | 32 px images, detection only, a single old generator (SD1.4) |
| `RealVsFake` | TGIF; classical forensics, DIRE-like DDPM reconstruction, PRNU, spectral, DINO; Streamlit app | TGIF experience; reconstruction signal | image-level only; no masks used |
| `featureAnalysis` | 12+ handcrafted / probe-model feature families, statistics layer, `localization/` with ~15 exotic model ideas | the **ROADMAP critique** (below) | per-image scalars; resizing destroyed CFA/PRNU/JPEG traces; no mask-aware evaluation; several "diffusion" cues were compressibility proxies |
| `testingMethods` | runnable TruFor, FOCAL, ManTra-Net, PSCC-Net, IID-Net + weights | **baselines** for phase 2 | — |
| `JepaApproach` (FoJEPA) | proposal + skeleton: JEPA prediction error on a forensic trace stream as a self-supervised anomaly map | the core idea of "**inconsistency with own context**" | never trained; needs a large pretraining budget |

Lessons we carry over, as rules:

1. **Mask-aware, patch-level evaluation from day one.** An image-level AUC says nothing about localization.
2. **Measure the shortcuts.** JPEG-vs-PNG, resolution, dataset identity and semantics ("the edited thing is
   an object") can all explain a high score. Every feature now ships with confound probes and a no-knowledge
   `pixels` control.
3. **Generalization is the benchmark.** Train on some editors, test on unseen ones (leave-one-dataset-out,
   transfer matrices).
4. **Low-level streams must see native pixels.** Foundation models need resizing, but residual and
   reconstruction streams do not get it.

## 2. Field in 2025-26: the "frozen latent + simple head" shift

* **Detection is now largely solved by linear probes on frozen vision foundation models.** *Simplicity
  Prevails* (arXiv 2602.01738) shows a linear classifier on frozen Perception Encoder, MetaCLIP-2 or DINOv3
  features beats specialised detectors in the wild by more than 30 % accuracy. **The same paper reports these
  models are blind to VAE reconstruction and to localized editing.** That gap is exactly our problem.
* **Self-supervised forensic latents trained on real images only.** *Forensic Twins* (arXiv 2609.31514) does
  redundancy-reduction SSL on two pixel-disjoint crops of a frozen residual extractor's output, so the shared
  signal is the acquisition fingerprint. A GMM on real embeddings then gives 97.99 % zero-shot AUC over 27
  unseen generators. *Self-supervised AIGI detection from camera metadata* (arXiv 2512.05651) follows the same
  philosophy.
* **Latent forensic knowledge inside VFMs.** *Forensic Reserve / RGE* (arXiv 2610.08639) finds sparse,
  origin-sensitive directions inside frozen encoders and adapts less than 0.2 % of parameters with 500 labels.
  So intermediate layers matter, which is why the extractor supports `:lK`.
* **Localization methods for AI edits.** FLAME (arXiv 2606.02178) argues diffusion suppresses local
  high-frequency variance (an "energy gap") and uses a SAM adapter. ICL-Net / PromptForge-350k
  (arXiv 2603.29386) uses intra-image contrastive learning; it reaches 62.5 % IoU in-domain but only 41.5 %
  on unseen editors. ForenScope / GIFTBench (arXiv 2610.01778) fuses multi-depth VFM features and reports
  asymmetric cross-source transfer with recall-dominated failures. CBDiff (arXiv 2510.19597) and SDiFL
  (arXiv 2508.20182) are diffusion-based localizers.
* **Open problem flagged by TGIF2 (arXiv 2603.28613): fully regenerated (FR) images.** When an editor
  re-renders the whole image, as GPT-Image-2 and Nano-Banana-2 do, the "authentic" area is also synthetic.
  Noise-inconsistency localizers fail, and models fine-tuned on FR show object bias on random masks.

## 3. Datasets sampled (`configs/datasets.yaml`)

| key | content | labels in sample |
|---|---|---|
| `opensdi` | OpenSDI test (`nebula/OpenSDI_test`): SD1.5/2.1/XL/3, Flux.1, full generation + mask inpainting | real, full, local |
| `sid_set` | SID-Set validation (`saberzl/SID_Set`): real / FLUX full synthetic / FLUX tampered | real, full, local |
| `diffseg30k` | DiffSeg30k validation (`Chaos2629/Diffseg30k`): multi-turn local edits by 8 diffusion editors | local (+ real if mask empty) |
| `gpt_image_2` | GPT-Image-2 instruction edits of COCO (from `youqiwong/AIGC-Loc-Testsets`) - FR-style | local |
| `nanobanana2` | Nano-Banana-2 (Gemini 3.1 Flash Image) edits, same source prompts | local |
| `cocoglide` | CocoGLIDE (TruFor benchmark) | local |
| `autosplice` | AutoSplice, DALL-E 2 text-prompted local edits | local |

**Provenance caveat:** `youqiwong/AIGC-Loc-Testsets` (source of the last four rows) has no dataset card,
licence or paper. It is fine for exploration, but for publication take CocoGlide and AutoSplice from their
original releases and treat the GPT-Image-2 and Nano-Banana-2 masks as unverified (their annotation method
is undocumented).

Not sampled yet (too large to sample remotely, or not released): TGIF2 (GitHub), PromptForge-350k,
LocateEdit-Bench (release pending), GIFTBench, BR-Gen (multi-part zip), PIXAR (7-11 GB tars), FragFake
(image-level only, no masks).

## 4. Direction

**Stage A: latents (frozen or self-supervised, no masks).**
A1 semantic VFM patch tokens (DINOv2/v3, SigLIP2, CLIP; last and intermediate layers);
A2 low-level native-resolution streams (SRM residual statistics, SD-VAE reconstruction error = AEROBLADE);
A3 (next) our own **real-only SSL residual encoder** in the spirit of Forensic Twins, but with a
*dense* (patch) objective. This revives FoJEPA's idea cheaply: predict a patch's residual embedding from its
context, then use the prediction error as the anomaly map.

**Stage B: simple discriminators.**
B1 linear / small-MLP patch probe (supervised);
B2 one-class Gaussian/GMM on real patches (no fake labels; generalizes by construction);
B3 training-free intra-image consistency (kNN / prototype distance in latent space).

**Phase 0 gate (this commit).** A latent stream moves to Stage B only if, on held-out datasets
(transfer off-diagonal), it beats the `pixels` control by a clear margin, and the margin is not explained by
format or dataset confounds. Results are in `outputs/reports/index.html` and are summarised in
`docs/02_phase0_results.md`.
