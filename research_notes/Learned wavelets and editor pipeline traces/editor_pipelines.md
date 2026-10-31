# Internal pipelines of modern image editors: what they share, as a basis for forensic traces

Notation for sources:
- **[V]**: verified in this session by fetching the cited page.
- **[R]**: taken from the researcher's prior knowledge of the cited primary source, not re-fetched this session. The URL is the primary source, but the report writer should treat [R] items as "high confidence, needs spot-check".
- **[C]**: community reverse-engineering or third-party summary. Marked as such, and lower confidence.

Research date: 2026-10-31.

---

## Q1. Latent diffusion / flow-matching inpainting and editing (SD1.5/2/XL, SD3/3.5, FLUX.1 Fill/Kontext, Qwen-Image-Edit, Step1X-Edit, HiDream-E1, OmniGen/OmniGen2, BrushNet/PowerPaint)

### Takeaway
Every model in this family encodes the source image with a convolutional KL-VAE at 8x spatial compression. Older models (SD1.5/2/XL, OmniGen v1) use 4 latent channels; newer ones (SD3, FLUX, Qwen-Image, BAGEL, OmniGen2, Step1X-Edit) use 16. All of them decode every output pixel through that VAE's decoder. They differ mainly in how the source is injected:
- **channel concatenation**: SD-inpaint, FLUX Fill, BrushNet branch;
- **sequence or in-context token concatenation**: Kontext, Qwen-Image-Edit, BAGEL, OmniGen2;
- **per-step latent replacement**: non-inpaint checkpoints in diffusers.

Pixel-space paste-back of unmasked pixels is optional and pipeline-dependent: diffusers `apply_overlay`/`padding_mask_crop`, and BrushNet's blurred blend. Instruction editors (Kontext, Qwen-Image-Edit, BAGEL, etc.) have no mask, so they regenerate the whole frame.

### Cited Findings

**SD 1.5 / 2 / XL inpainting (UNet, 4-channel VAE)**
- LDM/SD autoencoder: downsampling factor f=8 with 4 latent channels in the SD release (KL-regularised, perceptual + adversarial loss). [R] — [Rombach et al., LDM, arXiv 2112.10752](https://arxiv.org/abs/2112.10752)
- SD-Inpainting checkpoints add "5 additional input channels (4 for the encoded masked-image and 1 for the mask itself)" to the UNet, giving 9 input channels. [R] — [HF model card stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)
- SDXL keeps a 4-channel f=8 VAE (retrained with a larger batch) and uses a native ~1024² resolution with aspect-ratio bucketing. [R] — [Podell et al., SDXL, arXiv 2307.01952](https://arxiv.org/abs/2307.01952)
- SD1.5-Inpainting was "finetuned on 512x512 images". [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
- Non-inpaint checkpoints used for inpainting "generally tend to preserve the mask area (that is why you can see the mask outline)". Inpaint-specific checkpoints "are more likely to change your unmasked area". [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
  - Mechanism, from diffusers source: with a 4-channel UNet, the pipeline re-noises the original latent at each step and blends `latents = (1 - mask) * init_latents_proper + mask * latents`. The latent-resolution mask is used. [R] — [diffusers pipeline_stable_diffusion_inpaint.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py)
- Pixel paste-back is opt-in. "you can use the VaeImageProcessor.apply_overlay() method to force the unmasked area of an image to remain the same at the expense of some more unnatural transitions". [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
- Mask blur: `blur_factor` softens "the transition between the original image and inpaint area". [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
- `padding_mask_crop`: the pipeline crops the masked area with padding, upscales the crop and mask "to a higher resolution for inpainting, and then overlaid on the original image". This is an internal crop, resize and paste-back. [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
- The docs recommend `output_type="latent"` when chaining pipelines, "to avoid an unnecessary decode-encode step". This implicitly acknowledges that each VAE round-trip alters the image. [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
- Input H and W are resized or snapped to multiples of `vae_scale_factor` (8) by `VaeImageProcessor`. [R] — [diffusers VaeImageProcessor docs](https://huggingface.co/docs/diffusers/api/image_processor)

**BrushNet / PowerPaint (SD1.5-based plug-ins)**
- BrushNet: "we use a VAE encoder instead of randomly initialized convolution layers to process the masked image". The mask is downsampled with cubic interpolation. The noisy latent, masked-image latent and mask are concatenated as branch input. Base model: "All models use Stable Diffusion V1.5 as base model". [V] — [BrushNet, arXiv 2403.06976](https://arxiv.org/html/2403.06976)
- BrushNet paste-back: "the generated image and masked image are blended with a blurred mask". The authors justify this because "VAE encoding and decoding operations have inherent limitations and may not ensure complete image reconstruction". [V] — [BrushNet, arXiv 2403.06976](https://arxiv.org/html/2403.06976)
- PowerPaint builds on SD1.5 inpainting, adding learnable task prompts (context-aware fill, object insertion, removal, outpainting) and the same 9-channel input design. [R] — [PowerPaint, arXiv 2312.03594](https://arxiv.org/abs/2312.03594)

**SD3 / 3.5 (MMDiT, rectified flow, 16-channel VAE)**
- SD3 uses a 16-channel f=8 VAE, rectified-flow training, and an MMDiT on 2x2 latent patches, so the effective token grid is 16 px. [R] — [Esser et al., SD3, arXiv 2403.03206](https://arxiv.org/abs/2403.03206)

**FLUX.1 Fill / Kontext (12B rectified-flow transformer, 16-channel VAE)**
- FLUX autoencoder: "16 latent channels", trained "with an adversarial objective from scratch". On 256² ImageNet it reaches PSNR 31.1 / SSIM 0.896, better than the SD3-VAE and SD VAEs. This sets a finite ceiling on round-trip fidelity. [V] — [FLUX.1 Kontext, arXiv 2506.15742](https://arxiv.org/html/2506.15742v2)
- Kontext context conditioning: context images are encoded by the frozen FLUX AE and their tokens are appended to the target tokens ("sequence concatenation"). Channel-wise concatenation performed worse. 3D RoPE gives target tokens time index 0 and context image *i* time index *i* (a "virtual time step"). The design "supports different input/output resolutions and aspect ratios". [V] — [arXiv 2506.15742](https://arxiv.org/html/2506.15742v2)
- Distillation: Kontext [pro] uses LADD (latent adversarial diffusion distillation). [dev] uses guidance distillation into a 12B transformer. Inference takes ~3-5 s at 1024². [V] — [arXiv 2506.15742](https://arxiv.org/html/2506.15742v2)
- FLUX.1 Fill concatenates the packed masked-image latent and mask with the noisy latent along the channel axis. [R] — [HF FLUX.1-Fill-dev card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev); [diffusers FluxFillPipeline](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- The diffusers Kontext pipeline resizes the input to one of a list of "preferred Kontext resolutions" (~1 MP, sides multiples of 16). The output is therefore resampled relative to the input unless the user undoes it. [R] — [diffusers pipeline_flux_kontext.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_kontext.py)

**Qwen-Image / Qwen-Image-Edit (20B MMDiT, Wan-2.1-based VAE)**
- VAE: "single-encoder, dual-decoder", based on Wan-2.1-VAE with 8x8 compression and 16 channels. The encoder is frozen; the image decoder is fine-tuned on text-rich images. [V] — [Qwen-Image, arXiv 2508.02324](https://arxiv.org/html/2508.02324v1)
- MMDiT: 60 layers, ~20B parameters. MSRoPE positions are centred on the image, with an extra frame dimension that separates input and output images in editing. [V] — [arXiv 2508.02324](https://arxiv.org/html/2508.02324v1)
- Edit conditioning is a dual encoding. The input image goes to Qwen2.5-VL (semantic features), and its VAE latent is concatenated with the noised latent "along the sequence dimension". [V] — [arXiv 2508.02324](https://arxiv.org/html/2508.02324v1)
- Training resolutions: 256 → 640 → 1328 px, with aspect ratios from 1:3 to 3:1. [V] — [arXiv 2508.02324](https://arxiv.org/html/2508.02324v1)
- The diffusers Qwen-Image-Edit pipeline rescales the input to roughly 1024² total area, with sides snapped to multiples of 32 or 16. [R] — [diffusers QwenImageEditPipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage)

**BAGEL (unified MoT, uses the FLUX VAE)**
- Generation uses a frozen FLUX VAE with "downsample ratio of 8 and a latent channel of 16", plus a 2x2 patch embedding (so 16 px per token). Understanding uses SigLIP2-so400m/14 with NaViT, up to 980². [V] — [BAGEL, arXiv 2505.14683](https://arxiv.org/html/2505.14683)
- Images are generated by rectified flow on VAE tokens; text by next-token prediction. For editing, later tokens attend to the clean VAE tokens and ViT tokens of preceding images. Generation resolution is (512, 1024) for short/long side in the later stages. [V] — [arXiv 2505.14683](https://arxiv.org/html/2505.14683)

**OmniGen / OmniGen2 / Step1X-Edit / HiDream-E1**
- OmniGen2 decouples an autoregressive Qwen2.5-VL-3B (semantic conditions from its hidden states at the `<|img|>` token) from a ~4B diffusion transformer trained from scratch. Reference images enter through the FLUX VAE ("low-level visual details (crucial for editing)"). [V, via search snippets of the paper] — [OmniGen2, arXiv 2506.18871](https://arxiv.org/pdf/2506.18871)
- OmniGen v1 is a single transformer (Phi-3 initialised) that consumes SDXL-VAE latents (4 channels) of input images in-context. [R] — [OmniGen, arXiv 2409.11340](https://arxiv.org/abs/2409.11340)
- Step1X-Edit pairs a Qwen2.5-VL MLLM with a DiT image decoder; the source image is VAE-encoded and concatenated as tokens. [R; search returned nothing] — [Step1X-Edit, arXiv 2504.17761](https://arxiv.org/abs/2504.17761)
- HiDream-E1: instruction editor built on HiDream-I1. [R, low confidence] — [HiDream-E1 HF card](https://huggingface.co/HiDream-ai/HiDream-E1-Full)

### Inferences
- In mask-free editors there is no "original pixel" path to the output. Kontext, Qwen-Image-Edit, BAGEL, OmniGen2 and Step1X-Edit inject the source only as latent or tokens, so the whole frame is a VAE decode.
  - Even "unchanged" regions are at best a VAE reconstruction (bounded by roughly 31 dB PSNR for the FLUX AE at 256²) with added generative drift.
  - This is consistent with PIXAR's observation of "widespread background speckles" (see Q2).
- Masked pipelines create three distinct forensic regimes:
  - (a) pure latent replacement: the unmasked region equals a VAE reconstruction of the original;
  - (b) pixel paste-back with a hard or blurred mask: the unmasked region is bit-exact original, with a feathered transition band;
  - (c) crop-upscale-inpaint-downscale-overlay (`padding_mask_crop`): the edit region carries resampling traces at a different scale.
- Latent masks are downsampled 8x, so latent-blending edits have 8-px-quantised boundaries before decode. Patchified transformers (SD3, FLUX, Qwen, BAGEL) act on 16-px tokens.
- Two VAE "dialects" exist (4-channel SD-family vs 16-channel FLUX/SD3/Wan-family), so decoder fingerprints should cluster into these families. Qwen's decoder is fine-tuned on top of Wan, which makes it a sub-dialect.

### Gaps
- Exact default resizing code in each official repo (Step1X-Edit, HiDream-E1, OmniGen2) was not verified this session.
- FLUX.1 Fill's exact mask packing (it is reported that the mask is pixel-unshuffled 8x8 into 64 channels) is not verified.
- HiDream-E1 architecture details are unverified. No primary technical report was found.

---

## Q2. Autoregressive / unified MLLM editors (GPT-Image, Gemini Flash Image / Nano Banana, BAGEL, Janus-Pro, Emu3/3.5, Show-o, Lumina-mGPT, VAR/Infinity, MAR)

### Takeaway
"Autoregressive" editors fall into three groups:
- **Discrete-token systems**: Janus-Pro, Emu3/3.5, Show-o, Lumina-mGPT/Chameleon, VAR/Infinity. They use VQ, IBQ, multi-scale residual or bitwise quantisers with 16x downsampling, and decode with a learned conv decoder, sometimes a diffusion decoder (Emu3.5).
- **Continuous-token systems**: MAR (diffusion head), BAGEL (rectified flow on FLUX-VAE tokens).
- **Closed products**: GPT-Image and Nano Banana. Their internals are undocumented, but they behave as full regenerations, with resolution snapping, mask-as-guidance, and provenance signals (C2PA + SynthID).

### Cited Findings

**Closed products**
- OpenAI image API, current docs as of 2026-10:
  - Models: GPT Image 2.5 "Sunburst"/"Flare" are recommended; `gpt-image-2` and `gpt-image-1.5` are legacy; "`gpt-image-1` was shut down on October 23, 2026".
  - Sizes: width and height must be multiples of 16, aspect ratio between 1:3 and 3:1, max edge 3840, total pixels 655,360-8,294,400. Defaults are 1024², 1536x1024 and 1024x1536.
  - Formats: png (default), jpeg, webp, with `output_compression` 0-100 for jpeg/webp.
  - Masking is "entirely prompt-based"; the model treats the mask as guidance and "may not follow its exact shape with complete precision".
  - `input_fidelity` is not settable on gpt-image-2, which "always processes image inputs at high fidelity". [V] — [OpenAI image generation guide](https://developers.openai.com/api/docs/guides/image-generation)
- OpenAI provenance: images from ChatGPT, Codex and the API include both C2PA metadata and SynthID watermarks. Metadata is easily stripped (social media, screenshots). [V, via search snippet of the help article] — [OpenAI Help: C2PA in images](https://help.openai.com/en/articles/8912793-c2pa-in-images)
- GPT-4o image generation was described by OpenAI as natively autoregressive inside GPT-4o; the exact tokenizer and decoder are undisclosed. [R] — [OpenAI GPT-4o image generation system card addendum](https://openai.com/index/gpt-4o-image-generation-system-card-addendum/)
  - [C] Users widely reported a warm/yellow colour cast and slight geometric shifts or re-framing on "edits". This is community observation; no primary source was found this session.
- Gemini image models (2026-10 docs): `gemini-nano-banana-2.1`, `gemini-3.1-flash-image` (Nano Banana 2), `gemini-3-pro-image` (Nano Banana Pro), and `gemini-2.5-flash-image` (legacy).
  - Sizes are 1K/2K/4K, plus 512px on 3.1 Flash, with fixed aspect ratio sets (1:1 … 21:9). Output is requested as image/png or image/jpeg.
  - "All generated images include a SynthID watermark." [V] — [Gemini API image generation docs](https://ai.google.dev/gemini-api/docs/image-generation)
- PIXAR (forensics) observes:
  - Some models, "e.g., Gemini", "do not support a given target resolution", which "may result in pixel-space misalignment" and makes difference maps unreliable. The authors therefore align with ORB+RANSAC homography before differencing.
  - Failure modes include "large image regions are repainted beyond the target area", "widespread background speckles", and spill-over ("relighting halos, color bleeding, deblurring, seam removal") outside the mask.
  - Labels come from |I_orig − I_gen| > τ, with τ = 0.05 by default. [V] — [PIXAR, arXiv 2603.20193](https://arxiv.org/html/2603.20193)

**Open models**
- PIXAR's generators include Qwen-Image, Flux.2, Gemini, GPT-image-1.5 and Seedream 4.5, with 420K+ pairs. [V, search summary] — [arXiv 2603.20193](https://arxiv.org/pdf/2603.20193)
- BAGEL: see Q1. It is a continuous VAE-token model (FLUX VAE, rectified flow), not discrete VQ. [V] — [arXiv 2505.14683](https://arxiv.org/html/2505.14683)
- Emu3.5 tokenizer and decoders:
  - IBQ tokenizer with "downsampling factor of 16", codebook 131,072 x 256-dim, ~455M parameters, with SigLIP distillation.
  - Optional diffusion decoder "generates images at twice the resolution of the vanilla decoder", distilled from ~50 to 4 steps; outputs up to 2K.
  - DiDA converts autoregressive decoding into parallel bidirectional prediction (~20x faster).
  - Pre-training caps images at 1,024 tokens (512²); SFT runs at 768-1024 px. [V] — [Emu3.5, arXiv 2510.26583](https://arxiv.org/html/2510.26583)
- Emu3: single-transformer next-token model with an SBER-MoVQGAN-derived VQ tokenizer (codebook 32,768). [R] — [Emu3, arXiv 2409.18869](https://arxiv.org/abs/2409.18869)
- Janus-Pro: decoupled visual encoding, with SigLIP for understanding and a VQ tokenizer (LlamaGen-style, 16x downsampling, codebook 16,384) for generation, at 384² output. [R] — [Janus-Pro, arXiv 2501.17811](https://arxiv.org/abs/2501.17811)
- Show-o: MAGVIT-v2 lookup-free quantiser (16x, 8,192 codes), masked/discrete diffusion for images inside one transformer. [R] — [Show-o, arXiv 2408.12528](https://arxiv.org/abs/2408.12528)
- Lumina-mGPT: Chameleon-initialised next-token model reusing Chameleon's VQ image tokenizer. [R] — [Lumina-mGPT, arXiv 2408.02657](https://arxiv.org/abs/2408.02657)
- VAR: next-scale prediction with a multi-scale residual VQ tokenizer (16x). [R] — [VAR, arXiv 2404.02905](https://arxiv.org/abs/2404.02905)
- Infinity: bitwise tokenizer with an effectively infinite vocabulary (binary spherical quantisation), plus bitwise self-correction. [R] — [Infinity, arXiv 2412.04431](https://arxiv.org/abs/2412.04431)
- MAR: continuous tokens (KL-16 tokenizer) with a small per-token diffusion MLP head replacing VQ. [R] — [MAR, arXiv 2406.11838](https://arxiv.org/abs/2406.11838)

**Forensics benchmarks that use these editors**
- PromptForge-350k covers four editors, including Nano-banana, BAGEL and Flux.Kontext. It reports 41.5% average IoU on unseen editors. [V, search summary] — [arXiv 2603.29386](https://arxiv.org/abs/2603.29386)
- LocateEdit-Bench includes Qwen-Image-Edit, Flux-Kontext, Step1X-Edit and BAGEL. Sources conflict on scale: 231K images from either 4 or 44 editors. [V, search summary; conflict noted] — [arXiv 2602.05577](https://arxiv.org/html/2602.05577v1)

### Inferences
- For all token-based editors, the input is re-encoded into a token grid of typically 16 px per token. The output is produced entirely by a learned decoder, so there is no pixel copy-through. Edits are therefore full regenerations.
  - Unchanged areas carry decoder or quantiser reconstruction error. For VQ models this is larger and more texture-hallucinatory than for KL-VAEs.
- The closed APIs force output sizes onto discrete grids (multiples of 16, fixed 1K/2K/4K buckets). An input of arbitrary size is therefore resampled, and possibly re-framed. PIXAR's need for homography alignment is direct evidence of geometric drift in at least Gemini.
- Product outputs add a container layer: PNG/JPEG/WebP with user-selected compression, C2PA manifests, and in-pixel SynthID. Each of these alters the pixel or metadata statistics independently of the generator.

### Gaps
- No primary documentation of GPT-Image or Nano Banana internals (tokenizer type, decoder, latent grid) exists. Anything said about them is inference.
- No primary source found for the GPT-4o "yellow tint / shift" claims. They are community-only.
- Whether Gemini outputs carry C2PA was not verified in the fetched docs; only SynthID was confirmed.
- INP-X (arXiv 2602.00192) could not be found by search.

---

## Q3. Post-processing commonly applied (upscalers, face restoration, sharpening, compression)

### Takeaway
Upscaling, face restoration and compression are separate learned or lossy stages bolted onto the editor. Generative super-resolution is documented to significantly weaken forensic traces.

### Cited Findings
- TGIF2 extends TGIF (SD2, SDXL, Adobe Firefly) with FLUX.1 [schnell], [dev] and Fill [dev] inpainting. It finds that IFL/SID detectors degrade on FLUX and that "generative super-resolution significantly weakens forensic traces". [V, search summary] — [TGIF2, arXiv 2603.28613](https://arxiv.org/abs/2603.28613)
- The diffusers docs chain inpaint → SDXL refiner → img2img and mention `StableDiffusionUpscalePipeline` as an alternative to padding-crop upscaling. [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)
- Real-ESRGAN: a blind SR GAN trained with a high-order degradation model. It is the de facto default upscaler in community UIs. [R] — [Real-ESRGAN, arXiv 2107.10833](https://arxiv.org/abs/2107.10833)
- Face restoration: GFPGAN [R] — [arXiv 2101.04061](https://arxiv.org/abs/2101.04061); CodeFormer [R] — [arXiv 2206.11253](https://arxiv.org/abs/2206.11253)
- Product-level compression: OpenAI offers jpeg/webp with `output_compression` 0-100 and suggests JPEG for latency [V] — [OpenAI guide](https://developers.openai.com/api/docs/guides/image-generation). Gemini returns png or jpeg on request [V] — [Gemini docs](https://ai.google.dev/gemini-api/docs/image-generation)
- Emu3.5's optional diffusion decoder is itself a built-in 2x generative upsampler. [V] — [arXiv 2510.26583](https://arxiv.org/html/2510.26583)

### Inferences
- Tile-diffusion upscalers and face restorers re-synthesise high frequencies globally or per face crop. This adds a second learned-decoder fingerprint, often at a different grid (e.g. 4x SR) or at face-crop boundaries.
- A detector trained on raw VAE traces should be tested under a post-SR condition.

### Gaps
- No quantitative source was gathered this session on how often production services apply sharpening or SR silently. This is undocumented for both OpenAI and Google.

---

## Q4. Shared stages vs. differences

### Takeaway
Two stages are shared by every editor examined:
1. **Resample to a model grid.** Sides are snapped to multiples of 8, 16 or 32, or to fixed buckets.
2. **Decode every output pixel through a learned decoder.** This is a conv VAE (f=8, 4 or 16 channels), a VQ/IBQ decoder (f=16), or a diffusion decoder, operating on an 8 or 16 px latent/token lattice and normally in fp16/bf16.

The original pixels reach the output only when an explicit pixel-space paste-back is applied. This happens in some open inpainting pipelines and is never documented for instruction or AR editors.

### Comparison table

| Editor | Input encoder | Latent/token grid | Source conditioning | Mask handling | Output pixels from | Paste-back | Resolution handling | Sources |
|---|---|---|---|---|---|---|---|---|
| SD1.5/2 inpaint | KL-VAE, 4 ch | 8 px | channel-concat (9-ch UNet) | 8x-downsampled mask channel | VAE decoder (whole frame) | optional `apply_overlay`, `padding_mask_crop` overlay | 512² train; ×8 snapping | [diffusers](https://huggingface.co/docs/diffusers/using-diffusers/inpaint) [V]; [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting) [R] |
| SD non-inpaint + mask | KL-VAE, 4 ch | 8 px | re-noised init latent | per-step latent blend | VAE decoder | optional | ×8 | [diffusers src](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py) [R] |
| SDXL inpaint | KL-VAE, 4 ch | 8 px | channel-concat | as above | VAE decoder | optional | ~1024² buckets | [SDXL](https://arxiv.org/abs/2307.01952) [R] |
| BrushNet / PowerPaint | SD1.5 VAE | 8 px | parallel branch (BrushNet), 9-ch (PowerPaint) | cubic-downsampled mask | VAE decoder | **blurred-mask blend (BrushNet default)** | SD1.5 | [BrushNet](https://arxiv.org/html/2403.06976) [V]; [PowerPaint](https://arxiv.org/abs/2312.03594) [R] |
| SD3/3.5 | KL-VAE, 16 ch | 8 px VAE, 16 px token (2x2) | img2img / inpaint variants | latent | VAE decoder | optional | ~1 MP | [SD3](https://arxiv.org/abs/2403.03206) [R] |
| FLUX.1 Fill | FLUX AE, 16 ch | 16 px token | channel-concat masked latent + mask | packed mask | VAE decoder | optional (pipeline) | ×16 | [card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) [R] |
| FLUX.1 Kontext | FLUX AE, 16 ch | 16 px token | sequence concat, 3D-RoPE time offset | none | VAE decoder | none | preferred ~1 MP buckets (diffusers) | [Kontext](https://arxiv.org/html/2506.15742v2) [V] |
| Qwen-Image-Edit | Wan-2.1 VAE (fine-tuned decoder), 16 ch + Qwen2.5-VL | 8 px VAE, 16 px token | sequence concat + VLM tokens | none | VAE decoder | none | 256→1328 train; ~1 MP | [Qwen-Image](https://arxiv.org/html/2508.02324v1) [V] |
| BAGEL | FLUX VAE 16 ch + SigLIP2 | 16 px token | clean VAE + ViT tokens in context | none | VAE decoder | none | (512,1024) short/long | [BAGEL](https://arxiv.org/html/2505.14683) [V] |
| OmniGen2 | FLUX VAE + Qwen2.5-VL-3B | 16 px token (assumed) | VAE ref latents into DiT; MLLM hidden states | none | VAE decoder | none | n/a | [OmniGen2](https://arxiv.org/pdf/2506.18871) [V-snippet] |
| OmniGen v1 | SDXL VAE 4 ch | 8 px VAE | in-context latents | none | VAE decoder | none | n/a | [OmniGen](https://arxiv.org/abs/2409.11340) [R] |
| Step1X-Edit | VAE + Qwen2.5-VL | n/a | token concat | none | VAE decoder | none | n/a | [Step1X-Edit](https://arxiv.org/abs/2504.17761) [R] |
| Janus-Pro | VQ (16x, 16k codes) + SigLIP | 16 px | tokens in context | none | VQ decoder | none | 384² | [Janus-Pro](https://arxiv.org/abs/2501.17811) [R] |
| Emu3.5 | IBQ (16x, 131k codes) | 16 px | tokens in context | none | VQ decoder or diffusion decoder (2x) | none | 512²→1024², up to 2K | [Emu3.5](https://arxiv.org/html/2510.26583) [V] |
| Show-o / Lumina-mGPT / VAR / Infinity | MAGVIT-v2 / Chameleon VQ / multi-scale RVQ / bitwise BSQ | 16 px | tokens | none | quantiser decoder | none | 256-1024 | [Show-o](https://arxiv.org/abs/2408.12528), [Lumina-mGPT](https://arxiv.org/abs/2408.02657), [VAR](https://arxiv.org/abs/2404.02905), [Infinity](https://arxiv.org/abs/2412.04431) [R] |
| MAR | KL-16 continuous | 16 px | n/a (generator) | none | KL decoder | none | 256 | [MAR](https://arxiv.org/abs/2406.11838) [R] |
| GPT-Image (1.5/2/2.5) | undisclosed | inferred ≥16 px (sizes are multiples of 16) | image inputs at "high fidelity" | mask = guidance only | undisclosed learned decoder | none documented | multiples of 16, 1:3-3:1, ≤3840 px | [OpenAI](https://developers.openai.com/api/docs/guides/image-generation) [V] |
| Gemini Flash/Pro Image | undisclosed | undisclosed | multimodal context | no mask API | undisclosed | none documented | 0.5K/1K/2K/4K buckets; misalignment reported | [Gemini](https://ai.google.dev/gemini-api/docs/image-generation) [V]; [PIXAR](https://arxiv.org/html/2603.20193) [V] |

### Cited Findings (supporting the shared-stage claim)
- The VAE round-trip is lossy. BrushNet: VAE encode/decode "may not ensure complete image reconstruction" [V] — [arXiv 2403.06976](https://arxiv.org/html/2403.06976). FLUX AE: PSNR 31.1 dB at 256² [V] — [arXiv 2506.15742](https://arxiv.org/html/2506.15742v2)
- Off-mask changes are pervasive in modern editors: spill-over, global repainting, background speckles, and misalignment needing homography. [V] — [PIXAR, arXiv 2603.20193](https://arxiv.org/html/2603.20193)
- Size quantisation in products: multiples of 16 (OpenAI) [V] — [OpenAI](https://developers.openai.com/api/docs/guides/image-generation); fixed K-buckets (Gemini) [V] — [Gemini](https://ai.google.dev/gemini-api/docs/image-generation)
- fp16 inference is the default in diffusers examples (`dtype=torch.float16`, `variant="fp16"`). [V] — [diffusers inpainting guide](https://huggingface.co/docs/diffusers/using-diffusers/inpaint)

### Inferences
- **Universal traces to target** (inference):
  1. A learned-decoder spectral fingerprint. The decoder's upsampling stack (transposed conv / nearest+conv) generates every output pixel, which produces periodic spectral peaks at the latent period (8 px or 16 px) and its harmonics.
  2. Resampling traces from snapping the input to the model grid. Bicubic or Lanczos interpolation correlations appear when input and output sizes differ.
  3. Absence of the camera pipeline (no CFA/demosaic, PRNU or JPEG-grid continuity) over the whole frame for mask-free editors, or inside the mask for paste-back editors.
- **Where they differ** (relevant for localisation vs detection):
  - Paste-back pipelines leave a sharp or feathered *boundary* between bit-exact camera pixels and decoder pixels. This is localisable from the residual.
  - Mask-free and AR editors produce a *whole-frame* decoder signature in which "edited" and "unedited" regions differ only in the magnitude of generative change. Localisation then requires semantic or difference reasoning (as in PIXAR's |Δ|>τ labels) rather than detecting a boundary.
- **Grid families**: 8 px with 4 channels (SD1.x/2/XL, OmniGen v1, BrushNet/PowerPaint); 8 px VAE with 16 channels behind 16 px transformer tokens (SD3, FLUX, Qwen, BAGEL, OmniGen2); pure 16 px quantiser decoders (Janus, Emu3.5, Show-o, VAR, Infinity, MAR). A multi-scale (wavelet) front end that resolves 8- and 16-px periodicities would cover all three.
- Mixed precision (fp16/bf16 decode) plausibly adds quantisation noise in decoder activations, but no source was found that measures it as a forensic trace.

### Gaps
- No source found that directly measures periodic spectral peaks of each specific VAE decoder (FLUX vs SD vs Wan/Qwen). This needs empirical verification, e.g. by VAE round-tripping real images.
- Internals of the closed systems (GPT-Image, Nano Banana, Seedream) are undocumented, so their grid or decoder family is unknown.
- The effect of mixed-precision decoding on traces is unquantified in the literature found.
