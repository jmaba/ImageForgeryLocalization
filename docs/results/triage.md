# Image-level triage: authentic / fully synthetic / regenerated / spliced

train 1790 images, test 1050, out of distribution 2550

Train: benchmark extension (200 photos x 3 editors) + 2/3 of the synthetic set. Test: the pre-registered benchmark (120 photos x 3 editors, never used for training) + 1/3 synthetic.

## Held-out test (pre-registered benchmark + held-out synthetic)

| features    | model    |   n_test |   balanced_acc |   macro_f1 |   recall authentic |   recall fully synthetic |   recall regenerated |   recall spliced |   auroc_authentic_vs_rest |
|:------------|:---------|---------:|---------------:|-----------:|-------------------:|-------------------------:|---------------------:|-----------------:|--------------------------:|
| autoencoder | boosting |     1050 |          0.77  |      0.77  |              0.475 |                    0.881 |                0.878 |            0.847 |                     0.913 |
| autoencoder | logistic |     1050 |          0.823 |      0.794 |              0.833 |                    0.852 |                0.861 |            0.744 |                     0.947 |
| detectors   | boosting |     1050 |          0.433 |      0.431 |              0.092 |                    0.405 |                0.508 |            0.728 |                     0.618 |
| detectors   | logistic |     1050 |          0.487 |      0.461 |              0.425 |                    0.476 |                0.283 |            0.764 |                     0.674 |
| both        | boosting |     1050 |          0.833 |      0.832 |              0.658 |                    0.895 |                0.886 |            0.892 |                     0.96  |
| both        | logistic |     1050 |          0.856 |      0.833 |              0.875 |                    0.862 |                0.844 |            0.842 |                     0.974 |

## Confusion matrix (both feature sets, boosting), held-out test

|                      |   pred authentic |   pred fully synthetic |   pred regenerated |   pred spliced |
|:---------------------|-----------------:|-----------------------:|-------------------:|---------------:|
| true authentic       |               79 |                      4 |                  1 |             36 |
| true fully synthetic |                0 |                    188 |                 21 |              1 |
| true regenerated     |                2 |                     35 |                319 |              4 |
| true spliced         |               32 |                      3 |                  4 |            321 |

## Most important features (permutation, held-out test)

|              |   drop in balanced accuracy |
|:-------------|----------------------------:|
| lp_contrast  |                       0.198 |
| lp_p10_flux  |                       0.08  |
| tru_sd       |                       0.064 |
| l2_log_flux  |                       0.052 |
| lp_p10_sdxl  |                       0.046 |
| lp_p10_sd35  |                       0.022 |
| tru_max      |                       0.021 |
| lp_mean_flux |                       0.02  |
| lp_p90_flux  |                       0.02  |
| lp_mean_sd2  |                       0.017 |
| lp_p90_sd35  |                       0.011 |
| lp_mean_sdxl |                       0.01  |

## Out of distribution (generators, sizes and sources not seen in training)

| dataset               | features    |   n | true_class                                    |   accuracy |   -> authentic |   -> fully synthetic |   -> regenerated |   -> spliced |
|:----------------------|:------------|----:|:----------------------------------------------|-----------:|---------------:|---------------------:|-----------------:|-------------:|
| coco_orig_gpt_image_2 | autoencoder | 150 | authentic                                     |      0.427 |          0.427 |                0.013 |            0.007 |        0.553 |
| coco_orig_nanobanana2 | autoencoder | 150 | authentic                                     |      0.473 |          0.473 |                0.013 |            0.013 |        0.5   |
| cocoglide             | both        | 150 | other                                         |    nan     |          0.667 |                0.067 |            0.007 |        0.26  |
| gpt_image_2           | both        | 150 | regenerated                                   |      0.013 |          0.7   |                0.013 |            0.013 |        0.273 |
| nanobanana2           | both        | 150 | regenerated                                   |      0.007 |          0.76  |                0.02  |            0.007 |        0.213 |
| opensdi               | both        | 600 | authentic, edited (fr or sp), fully synthetic |      0.406 |          0.38  |                0.132 |            0.037 |        0.452 |
| sid_set               | both        | 300 | authentic, edited (fr or sp), fully synthetic |      0.422 |          0.207 |                0.227 |            0.26  |        0.307 |
| tgif2_flux1dev_fr     | both        | 150 | regenerated                                   |      0.5   |          0.007 |                0.487 |            0.5   |        0.007 |
| tgif2_flux1dev_sp     | both        | 150 | spliced                                       |      0.833 |          0.14  |                0.027 |            0     |        0.833 |
| tgif2_orig            | both        | 150 | authentic                                     |      0.653 |          0.653 |                0.033 |            0.007 |        0.307 |
| tgif2_sd2_fr          | both        | 150 | regenerated                                   |      0     |          0.127 |                0.873 |            0     |        0     |
| tgif2_sd2_sp          | both        | 150 | spliced                                       |      0.78  |          0.207 |                0.013 |            0     |        0.78  |
| tgif2_sdxl_fr         | both        | 150 | regenerated                                   |      0.847 |          0.013 |                0.113 |            0.847 |        0.027 |

## Leave one source out (boosting on autoencoder + detector features; train on all other sources, the pre-registered benchmark never in training; last row: all sources -> benchmark)

| held_out      |    n | classes                         |   recall authentic |   recall regenerated |   auroc authentic vs rest |   recall spliced |   recall fully synthetic |   edited (fr/sp) called edited |
|:--------------|-----:|:--------------------------------|-------------------:|---------------------:|--------------------------:|-----------------:|-------------------------:|-------------------------------:|
| GPT-Image-2   |  300 | authentic, regenerated          |              1     |                0.247 |                     1     |          nan     |                  nan     |                        nan     |
| Nano-Banana-2 |  300 | authentic, regenerated          |              1     |                0.293 |                     1     |          nan     |                  nan     |                        nan     |
| TGIF2         |  900 | authentic, regenerated, spliced |              0.713 |                0.442 |                     0.888 |            0.423 |                  nan     |                        nan     |
| cocoglide     |  150 |                                 |            nan     |              nan     |                   nan     |          nan     |                  nan     |                        nan     |
| extension     | 1400 | authentic, regenerated, spliced |              0.785 |                0.637 |                     0.872 |            0.563 |                  nan     |                        nan     |
| opensdi       |  600 | authentic, fully synthetic      |              0.451 |              nan     |                     0.644 |          nan     |                    0.244 |                          0.506 |
| sid_set       |  300 | authentic, fully synthetic      |              0.56  |              nan     |                     0.794 |          nan     |                    0.483 |                          0.483 |
| synthetic     |  600 | fully synthetic                 |            nan     |              nan     |                   nan     |          nan     |                    0.023 |                        nan     |
| benchmark     |  840 | authentic, regenerated, spliced |              0.742 |                0.875 |                     0.938 |            0.828 |                  nan     |                        nan     |
