# Result tables (copied from the git-ignored outputs/ folder, 2026-11-06)

| File | What | Source in outputs/ |
|---|---|---|
| benchmark_preregistered.md | pre-registered tests H1-H3 on the controlled benchmark (docs/08) | benchmark/results.md |
| localization_clean.md | cell AUC, matched nulls, all methods incl. FLAME and B-Free, clean | dtrace/localization_dtflamebfree.md |
| localization_{jpeg75,webp80,resize50}.md | the same with FLAME, re-shared (no B-Free map) | dtrace/localization_flame__*.md |
| image_level_{clean,jpeg75,webp80,resize50}.md | image AUROC vs own originals, all detectors | dtrace/image_level_flame_bfree*.md |
| robustness.md | key numbers side by side over the four conditions | dtrace/robustness_flame.md |
| dtrace_{A,B}_validation.json | training-time validation per autoencoder | dtrace/dtrace_*_history.json |
| triage.md | four-outcome image triage, in distribution and leave-one-source-out | triage/results.md |
| generator_consistency_pilot.md | autoencoder / denoising error maps (docs/07) | gencons/results_pilot.md |
| ensemble_{clean,jpeg75}.md | stacked ensemble with physics (docs/07) | ensemble/results_*.md |
| jpeg_control_edits_only_WITHDRAWN.md | re-encoding only the edited images; invalid control, see docs/11 | dtrace/jpeg_control.md |
