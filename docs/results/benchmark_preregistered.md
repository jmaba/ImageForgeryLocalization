# Controlled diffusion-edit benchmark - pre-registered analysis (docs/08)

jobs: {'bench_fluxfill_fr': 120, 'bench_fluxfill_sp': 120, 'bench_kontext_fr': 120, 'bench_kontext_sp': 120, 'bench_orig': 120, 'bench_sdxlinp_fr': 120, 'bench_sdxlinp_sp': 120}

## H1 / H1b / H1c - matched autoencoder LPIPS map, higher error = edited

| test                 | set                          | map           |   n |   auc |   null |   diff |   diff_lo95 | verdict   |   auc_hi95 |
|:---------------------|:-----------------------------|:--------------|----:|------:|-------:|-------:|------------:|:----------|-----------:|
| H1 (primary, pooled) | sdxlinp + fluxfill fr        | matched ae_lp | 120 | 0.501 |  0.435 |  0.066 |       0.044 | False     |    nan     |
| H1                   | sdxlinp fr                   | ae_lp_sdxl    | 120 | 0.496 |  0.427 |  0.068 |       0.041 | False     |    nan     |
| H1                   | fluxfill fr                  | ae_lp_flux    | 120 | 0.506 |  0.442 |  0.064 |       0.036 | False     |    nan     |
| H1b                  | kontext fr (object mask)     | ae_lp_flux    | 120 | 0.429 |  0.442 | -0.014 |      -0.045 | False     |    nan     |
| H1b                  | kontext fr (difference mask) | ae_lp_flux    | 107 | 0.39  |  0.45  | -0.059 |      -0.095 | False     |    nan     |
| H1c                  | sdxlinp sp                   | ae_lp_sdxl    | 120 | 0.114 |  0.427 | -0.313 |     nan     | True      |      0.13  |
| H1c                  | fluxfill sp                  | ae_lp_flux    | 120 | 0.106 |  0.442 | -0.336 |     nan     | True      |      0.123 |
| H1c                  | kontext sp                   | ae_lp_flux    | 120 | 0.078 |  0.442 | -0.365 |     nan     | True      |      0.092 |

## H2 - image level: originals vs edited, image-mean matched autoencoder error

| test   | originals_vs   | feature    |   auroc | verdict     |
|:-------|:---------------|:-----------|--------:|:------------|
| H2     | sdxlinp fr     | ae_lp_sdxl |   0.979 | True        |
| H2     | sdxlinp fr     | ae_l2_sdxl |   0.866 | (secondary) |
| H2     | sdxlinp sp     | ae_lp_sdxl |   0.56  | True        |
| H2     | sdxlinp sp     | ae_l2_sdxl |   0.542 | (secondary) |
| H2     | fluxfill fr    | ae_lp_flux |   0.988 | True        |
| H2     | fluxfill fr    | ae_l2_flux |   0.952 | (secondary) |
| H2     | fluxfill sp    | ae_lp_flux |   0.549 | True        |
| H2     | fluxfill sp    | ae_l2_flux |   0.553 | (secondary) |
| H2     | kontext fr     | ae_lp_flux |   0.996 | True        |
| H2     | kontext fr     | ae_l2_flux |   0.962 | (secondary) |
| H2     | kontext sp     | ae_lp_flux |   0.558 | True        |
| H2     | kontext sp     | ae_l2_flux |   0.555 | (secondary) |

## Baselines and autoencoder maps per variant (AUC paired with the original + same mask)

| set         | scorer     |   auc |   null |   auc_minus_null |   iou_at_0p5 |   fp_area_on_originals |
|:------------|:-----------|------:|-------:|-----------------:|-------------:|-----------------------:|
| sdxlinp_fr  | x_mesorch  | 0.651 |  0.728 |           -0.077 |        0.015 |                  0.007 |
| sdxlinp_fr  | x_trufor   | 0.571 |  0.688 |           -0.117 |        0.04  |                  0.041 |
| sdxlinp_fr  | cmp_z2     | 0.597 |  0.63  |           -0.033 |      nan     |                nan     |
| sdxlinp_fr  | ae_lp_sd2  | 0.467 |  0.438 |            0.029 |      nan     |                nan     |
| sdxlinp_fr  | ae_lp_sdxl | 0.496 |  0.427 |            0.068 |      nan     |                nan     |
| sdxlinp_fr  | ae_lp_sd35 | 0.485 |  0.453 |            0.032 |      nan     |                nan     |
| sdxlinp_fr  | ae_lp_flux | 0.486 |  0.442 |            0.044 |      nan     |                nan     |
| sdxlinp_sp  | x_mesorch  | 0.784 |  0.728 |            0.056 |        0.081 |                  0.007 |
| sdxlinp_sp  | x_trufor   | 0.955 |  0.688 |            0.267 |        0.638 |                  0.041 |
| sdxlinp_sp  | cmp_z2     | 0.749 |  0.63  |            0.119 |      nan     |                nan     |
| sdxlinp_sp  | ae_lp_sd2  | 0.314 |  0.438 |           -0.124 |      nan     |                nan     |
| sdxlinp_sp  | ae_lp_sdxl | 0.114 |  0.427 |           -0.313 |      nan     |                nan     |
| sdxlinp_sp  | ae_lp_sd35 | 0.335 |  0.453 |           -0.118 |      nan     |                nan     |
| sdxlinp_sp  | ae_lp_flux | 0.369 |  0.442 |           -0.073 |      nan     |                nan     |
| fluxfill_fr | x_mesorch  | 0.681 |  0.728 |           -0.047 |        0.022 |                  0.007 |
| fluxfill_fr | x_trufor   | 0.641 |  0.688 |           -0.047 |        0.057 |                  0.041 |
| fluxfill_fr | cmp_z2     | 0.617 |  0.63  |           -0.013 |      nan     |                nan     |
| fluxfill_fr | ae_lp_sd2  | 0.37  |  0.438 |           -0.068 |      nan     |                nan     |
| fluxfill_fr | ae_lp_sdxl | 0.365 |  0.427 |           -0.062 |      nan     |                nan     |
| fluxfill_fr | ae_lp_sd35 | 0.432 |  0.453 |           -0.02  |      nan     |                nan     |
| fluxfill_fr | ae_lp_flux | 0.506 |  0.442 |            0.064 |      nan     |                nan     |
| fluxfill_sp | x_mesorch  | 0.72  |  0.728 |           -0.008 |        0.055 |                  0.007 |
| fluxfill_sp | x_trufor   | 0.927 |  0.688 |            0.239 |        0.558 |                  0.041 |
| fluxfill_sp | cmp_z2     | 0.739 |  0.63  |            0.109 |      nan     |                nan     |
| fluxfill_sp | ae_lp_sd2  | 0.384 |  0.438 |           -0.053 |      nan     |                nan     |
| fluxfill_sp | ae_lp_sdxl | 0.381 |  0.427 |           -0.046 |      nan     |                nan     |
| fluxfill_sp | ae_lp_sd35 | 0.381 |  0.453 |           -0.072 |      nan     |                nan     |
| fluxfill_sp | ae_lp_flux | 0.106 |  0.442 |           -0.336 |      nan     |                nan     |
| kontext_fr  | x_mesorch  | 0.707 |  0.728 |           -0.021 |        0.053 |                  0.007 |
| kontext_fr  | x_trufor   | 0.64  |  0.688 |           -0.048 |        0.102 |                  0.041 |
| kontext_fr  | cmp_z2     | 0.664 |  0.63  |            0.035 |      nan     |                nan     |
| kontext_fr  | ae_lp_sd2  | 0.299 |  0.438 |           -0.139 |      nan     |                nan     |
| kontext_fr  | ae_lp_sdxl | 0.295 |  0.427 |           -0.132 |      nan     |                nan     |
| kontext_fr  | ae_lp_sd35 | 0.347 |  0.453 |           -0.106 |      nan     |                nan     |
| kontext_fr  | ae_lp_flux | 0.429 |  0.442 |           -0.014 |      nan     |                nan     |
| kontext_sp  | x_mesorch  | 0.829 |  0.728 |            0.101 |        0.305 |                  0.007 |
| kontext_sp  | x_trufor   | 0.933 |  0.688 |            0.245 |        0.584 |                  0.041 |
| kontext_sp  | cmp_z2     | 0.794 |  0.63  |            0.164 |      nan     |                nan     |
| kontext_sp  | ae_lp_sd2  | 0.314 |  0.438 |           -0.124 |      nan     |                nan     |
| kontext_sp  | ae_lp_sdxl | 0.309 |  0.427 |           -0.118 |      nan     |                nan     |
| kontext_sp  | ae_lp_sd35 | 0.274 |  0.453 |           -0.179 |      nan     |                nan     |
| kontext_sp  | ae_lp_flux | 0.078 |  0.442 |           -0.365 |      nan     |                nan     |

## H3 (exploratory) - stacker trained on the pilot domains, applied to the benchmark

|             |   stack: mesorch+trufor+comparator |   stack: + autoencoder maps |
|:------------|-----------------------------------:|----------------------------:|
| null        |                              0.701 |                       0.703 |
| sdxlinp_fr  |                              0.604 |                       0.607 |
| sdxlinp_sp  |                              0.94  |                       0.939 |
| fluxfill_fr |                              0.664 |                       0.696 |
| fluxfill_sp |                              0.913 |                       0.917 |
| kontext_fr  |                              0.688 |                       0.711 |
| kontext_sp  |                              0.933 |                       0.932 |
| fr mean     |                              0.652 |                       0.671 |
| sp mean     |                              0.929 |                       0.929 |
