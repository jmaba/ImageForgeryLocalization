# Statistical and Mathematical Framework for Within-Image Region Inconsistency Detection

Scope: tools for deciding, *inside one image*, which regions are statistically inconsistent with the rest. The tools are mapped to forensic use, with assumptions, guarantees, failure modes and recipes. Conventions:
- **[own derivation]** marks reasoning or formulas I derived or assembled myself; they are not quoted from a source.
- **[from memory, verify]** marks standard textbook facts I am confident in but did not re-fetch in this session. The DOI or URL is given so they can be checked.
- Everything else was confirmed against a search result or page fetched in this session (2026-10-31).

General setting used throughout [own derivation]: the image is split into regions or patches indexed by `i = 1..m`. Each patch gives a forensic feature vector `x_i ∈ R^d`, for example noise residual statistics, JPEG DCT statistics, CFA residuals, or a learned embedding. H0 is "all patches come from one generating process P" (one camera, one processing history). The alternative is "a subset S has a different law Q ≠ P". The candidate S can be fixed in advance (a test of one region against the rest), searched over (a scan), or estimated (segmentation).

---

## 1. Two-sample / distribution-comparison tests (region vs. rest, region vs. region)

### Takeaway
Kernel MMD and energy distance are consistent, nonparametric two-sample tests, and they are two forms of the same family. Their permutation nulls are exact **only under exchangeability**. Overlapping or neighbouring patches violate exchangeability, so the permutation null must be replaced by a block or wild bootstrap, or by thinning. For scalar features, KS and Anderson–Darling have distribution-free nulls under i.i.d. Hotelling T² and Box's M depend on Gaussianity, and Box's M is notoriously non-robust. High-dimensional covariance tests (Ledoit–Wolf, Li–Chen) exist for d comparable to n.

### Cited Findings
**MMD (Gretton et al. 2012)**
- Definition [from memory, verify, JMLR 13:723–773, https://jmlr.org/papers/v13/gretton12a.html]: `MMD²(P,Q) = E k(X,X') + E k(Y,Y') − 2E k(X,Y)`. With a characteristic kernel (e.g. Gaussian), MMD = 0 iff P = Q. The unbiased U-statistic estimator is `MMD²_u = 1/(m(m−1)) Σ_{i≠j} k(x_i,x_j) + 1/(n(n−1)) Σ_{i≠j} k(y_i,y_j) − 2/(mn) Σ_{i,j} k(x_i,y_j)`.
- Null distribution [from memory, verify, same JMLR paper]: under H0 the statistic is degenerate. `m·MMD²_u` converges to an infinite weighted sum of centred χ² variables, `Σ_l λ_l (z_l² − 2)`, where λ_l are eigenvalues of the centred kernel. In practice the threshold comes from permutation, a Gamma approximation, or a spectral approximation. The paper also gives distribution-free large-deviation (McDiarmid/Hoeffding-type) bounds, which are valid but conservative. Under H1 the statistic is asymptotically normal, with variance `σ²_{H1}` given in the paper.
- Power and kernel choice [from memory, verify]: test power is approximately `Φ(√n·MMD²/σ_{H1} − c_α/(n σ_{H1}))`. This motivates choosing the kernel to maximise `MMD²/σ_{H1}` on a held-out split (Gretton et al., NeurIPS 2012, "Optimal kernel choice for large-scale two-sample tests"; Liu et al. 2020, deep-kernel MMD, arXiv:2002.09116). Data splitting is needed, otherwise the permutation test is invalidated.
- High-dimensional power loss: kernel and distance tests lose power as d grows unless the signal grows [from memory, verify; Ramdas et al. 2015, AAAI, arXiv:1406.2083].
- **Dependence:** the standard permutation bootstrap "breaks the temporal dependence" and returns "an elevated number of false positives" on time series. The wild bootstrap (multiplying the V-statistic summands by an auxiliary dependent random process) gives consistent MMD and HSIC tests for dependent data — [Chwialkowski, Sejdinovic, Gretton, NeurIPS 2014, arXiv:1408.5404](https://arxiv.org/abs/1408.5404); [NeurIPS page](https://papers.nips.cc/paper_files/paper/2014/hash/4e382cb49370f64415df2672b19fb1f2-Abstract.html). The paper's theory covers time series (1-D mixing). Extending it to 2-D random fields is **[own inference]**: the argument needs mixing plus a dependent multiplier field with matching correlation range.

**Energy distance**
- `E(P,Q) = 2E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖ ≥ 0`, with equality iff P = Q (finite first moments). Energy distance equals MMD with the distance-induced kernel [from memory, verify; Székely & Rizzo 2013, JSPI, doi:10.1016/j.jspi.2013.03.018; Sejdinovic et al. 2013, Ann. Stat., doi:10.1214/13-AOS1140]. Its null and calibration therefore behave like MMD's: degenerate, calibrated by permutation, and needing the same dependence fixes.

**Wasserstein / sliced Wasserstein**
- Empirical W_p converges at rate `n^{-1/d}` in dimension d (curse of dimensionality) [from memory, verify; Fournier & Guillin 2015, PTRF, doi:10.1007/s00440-014-0583-7]. Sliced Wasserstein averages 1-D W_p over random projections. Each 1-D W_p is computed from sorted samples, and the sample complexity does not depend on d [from memory, verify; Nadjahi et al. 2020, arXiv:2003.05783]. No closed-form null exists for either, so calibrate with a permutation or block bootstrap. Note that 1-D W_1 between empirical CDFs equals `∫|F_n − G_m|`, a close relative of KS and Cramér–von Mises.

**KS / Anderson–Darling on scalar statistics**
- Two-sample KS: `D = sup_t |F_m(t) − G_n(t)|`. Under H0 with continuous F and i.i.d. samples, `√(mn/(m+n))·D` converges to the Kolmogorov distribution, which does not depend on F [from memory, verify; standard; e.g., Lehmann & Romano, *Testing Statistical Hypotheses*]. Anderson–Darling weights the tails by `1/(F(1−F))` and is more powerful for tail differences. A k-sample version exists (Scholz & Stephens 1987, JASA, doi:10.1080/01621459.1987.10478517) [from memory, verify].
- **Failure mode** [own derivation]: positive serial or spatial correlation inflates the variance of the empirical CDF. The effective sample size drops, so tabulated KS critical values give anti-conservative p-values. Ties from quantised features (e.g. JPEG DCT counts) also break the continuity assumption.

**Hotelling T² and covariance equality**
- Hotelling: `T² = (mn/(m+n)) (x̄−ȳ)ᵀ S_pooled^{-1} (x̄−ȳ)`. Under Gaussian i.i.d. data with equal covariance, `(m+n−d−1)/(d(m+n−2))·T²` follows `F(d, m+n−d−1)` exactly. This is a closed-form finite-sample null [from memory, verify; Anderson, *An Introduction to Multivariate Statistical Analysis*]. It needs n > d and fails or loses power as d/n → 1. Bai & Saranadasa (1996) and Chen & Qin (2010, Ann. Stat., doi:10.1214/09-AOS716) provide high-dimensional mean tests [from memory, verify].
- Box's M (Box 1949, Biometrika, doi:10.1093/biomet/36.3-4.317): `M = (N−k) ln|S_p| − Σ(n_i−1) ln|S_i|`, with a χ² or F approximation. It is very sensitive to non-normality, so heavy-tailed residuals cause false rejections [from memory, verify]. High-dimensional covariance tests: Ledoit & Wolf 2002, Ann. Stat., doi:10.1214/aos/1031689018; Schott 2007; Li & Chen 2012, Ann. Stat., doi:10.1214/12-AOS993 [from memory, verify].

**Likelihood-ratio tests under Gaussian scale mixtures (GSM)**
- Natural-image wavelet coefficients are well modelled as `x = √z·u`, with u Gaussian and z a positive hidden multiplier (Wainwright & Simoncelli, NIPS 1999; Portilla et al. 2003, IEEE TIP, doi:10.1109/TIP.2003.818640) [from memory, verify].
- [own derivation] Conditional on local z (estimated from neighbours), a region-vs-rest LRT for a change of noise covariance `Σ_0 → Σ_1` is `Λ = Σ_{i∈S} [ log|Σ_0|/|Σ_1| − x_iᵀ(Σ_1^{-1}−Σ_0^{-1})x_i / z_i ]`. Wilks' theorem gives `2 log Λ ~ χ²_q` asymptotically, but only for a fixed S chosen in advance, with i.i.d. data and the null in the interior of the parameter space. Once S is searched over (a scan), Wilks no longer holds, and calibration needs scan-statistic theory or Monte Carlo (Section 3).

### Inferences
- [own derivation] Recipe for "region R vs. rest": (1) compute features on **non-overlapping** patches, or subsample with stride ≥ the correlation length. (2) Use MMD with the kernel and bandwidth chosen on a disjoint split, or use energy distance. (3) Calibrate with a **spatial block permutation**: permute whole blocks of side ≥ the correlation range between R and its complement, or use a wild bootstrap with a spatially smooth multiplier field. (4) Report the p-value together with the effective sample size.
- [own derivation] Comparing "region vs. rest" mixes the tested region into the reference when R is large. Use a leave-region-out reference, and pair it with robust estimation (Section 5) so that a forgery covering more than 50% of the image does not become the reference.
- For 1-D summary statistics (e.g. local noise variance), KS or AD with block-corrected effective n is cheap and interpretable. For multi-dimensional learned embeddings, MMD or energy distance is the default.

### Gaps
- I found no paper that rigorously establishes a 2-D random-field version of the wild-bootstrap MMD test for image patches. The extension is inferred.
- I did not find forensic papers that report calibrated MMD p-values per region. Most forensic pipelines use heuristic scores.

---

## 2. A-contrario framework (NFA) and its forensic use

### Takeaway
An a-contrario detector declares a structure meaningful when its probability under a background ("naive") model is so small that the **expected number of false detections over all tests performed**, the NFA, is below ε (usually 1). The guarantee is `E[#false detections] ≤ ε`. It comes from linearity of expectation, so it holds **under any dependence** between tests, which makes it a per-family error-rate (PFER) bound in multiple-testing terms. Its weak point is the background model: the guarantee holds only if the per-test tail probabilities are valid under the real null. In practice that requires independence *within* a test, for example between the pixel or block votes summed in a binomial tail. Several IPOL forensic tools use NFA: ZERO (JPEG grid), Noisesniffer (noise), and the CFA "intermediate values" and double-demosaicing methods.

### Cited Findings
- Core definition [from memory, verify; Desolneux, Moisan & Morel, *From Gestalt Theory to Image Analysis: A Probabilistic Approach*, Springer 2008, doi:10.1007/978-0-387-74378-3; also Grompone von Gioi et al., LSD, IPOL 2012, doi:10.5201/ipol.2012.gjmr-lsd]: for a candidate e with observed statistic k(e), `NFA(e) = N_tests · P_{H0}[K(e) ≥ k(e)]`. e is ε-meaningful if NFA(e) ≤ ε. **Proposition:** E[# ε-meaningful events under H0] ≤ ε. Proof sketch: `E[Σ_e 1{N·P(K_e≥k_e) ≤ ε}] = Σ_e P(p_e ≤ ε/N) ≤ N·ε/N = ε`, which uses only super-uniformity of each p-value and linearity of expectation. No independence across tests is needed [standard; restated as own derivation].
- Typical H0 tail is binomial [from memory, verify]: if under H0 each of n elements independently "agrees" with probability p, then `P(K ≥ k) = B(n,k,p) = Σ_{j≥k} C(n,j) p^j (1−p)^{n−j}`. Independence *among the n elements* is needed for this tail to be valid.
- **ZERO** (Nikoukhah, Anger, Colom, Morel, Grompone von Gioi, IPOL 2021): it "identifies the presence of a JPEG grid when a significant number of DCT zeros is observed for a given grid origin". It follows "Desolneux, Moisan and Morel's a contrario theory", which "associates a number of false alarms (NFA) with each tampering detection". "Detections are obtained by a threshold of the NFA", which gives automatic operation with false-alarm control. It is applied locally to find "misaligned or missing JPEG grids" and outputs two masks (foreign grid, missing grid) — [IPOL 390](https://ipol.im/pub/art/2021/390); [PDF](https://www.ipol.im/pub/art/2021/390/article.pdf). In its predecessor (blocking-artifact grid detector, IPOL 2020), low-NFA discrepant grids are a strong forgery cue — [IPOL 283](https://www.ipol.im/pub/art/2020/283/).
  - [from memory, verify] In ZERO, each 8×8 block votes for the grid origin (out of 64) that maximises its DCT-zero count. Under H0, votes are uniform with p = 1/64, and the NFA uses a binomial tail over a region multiplied by the number of tested regions and origins. Check the exact N_tests in the PDF.
- **Noisesniffer** (Gardella, Musé, Morel, Colom; IWBF 2021; refined in IPOL 2024): builds a background noise model and detects local deviations. The 2024 version adds a region-growing step. The method uses "an a contrario validation step to control the expected number of false detections" — [IPOL 2024/462](https://ipol.im/pub/art/2024/462); [Gardella thesis page](https://centreborelli.ens-paris-saclay.fr/en/node/3174); [IPOL announce](https://tools.ipol.im/mailman/archive/announce/2024-April/000232.html).
- **A-contrario CFA:** Bammey, Grompone von Gioi & Morel, "Image Forgeries Detection through Mosaic Analysis: the Intermediate Values Algorithm", IPOL 2021, vol. 11, pp. 317–343, Bayer CFA only — [IPOL 2021/355](https://ipol.im/pub/art/2021/355). The earlier MIPR 2018 work used the a-contrario paradigm to give detections a very low NFA — [IP Paris record](https://researchportal.ip-paris.fr/en/publications/automatic-detection-of-demosaicing-image-artifacts-and-its-use-in/). The 2022 WIFS double-demosaicing paper uses an a-contrario scheme to set thresholds with false-alarm guarantees; code at github.com/qbammey/ddem — [IP Paris record](https://researchportal.ip-paris.fr/en/publications/demosaicing-to-detect-demosaicing-and-image-forgeries/).

### Inferences
- [own derivation] **How NFA handles multiple testing:** NFA is Bonferroni stated in expectation. `NFA ≤ ε ⇔ p ≤ ε/N`, and the guarantee is PFER ≤ ε, which also implies FWER ≤ ε. It is therefore valid under arbitrary dependence across tests, including overlapping candidate regions, multiple scales, and multiple grid origins, *provided N counts every test actually considered*, including all region shapes, scales and parameters scanned. Its cost is conservativeness when tests are strongly positively correlated, because the effective number of tests is far below N.
- [own derivation] **Where dependence breaks it:** the dependence that matters is *within* a test, because it decides whether `P_{H0}[K≥k]` is correct. If neighbouring blocks' votes are positively correlated (overlapping windows, textured areas with structured DCT content, resampling), the binomial tail underestimates the true tail and the NFA becomes anti-conservative. Remedies: use disjoint blocks; build H0 tails from per-image empirical background distributions, as Noisesniffer does with its background model; or check calibration by running the detector on authentic images and counting detections (expected ≤ ε).
- [own derivation] Recipe: choose a statistic whose H0 law is known or estimable per image. Enumerate the candidate family (rectangles or regions at dyadic scales, giving N_tests). Compute NFA = N_tests × tail probability. Threshold at ε = 1, or smaller for a stricter PFER. Then apply a maximality/exclusion step (keep the most meaningful region and remove nested duplicates), as in LSD-style pipelines.

### Gaps
- I could not fetch the ZERO PDF (over 10 MB), so the exact N_tests and tail formula are marked [from memory, verify].
- I found no published systematic calibration study of forensic NFA detectors on large authentic-image corpora (empirical PFER vs. ε).

---

## 3. Multiple comparisons over many regions

### Takeaway
Pick the error criterion first. FWER and PFER (Bonferroni or NFA) are valid under any dependence but conservative. Benjamini–Hochberg FDR is valid under independence or PRDS, which positively correlated neighbouring-patch tests often satisfy plausibly but not provably. Benjamini–Yekutieli is valid under arbitrary dependence at a log(m) cost. Cluster-level inference from fMRI (random field theory, permutation of max statistics) and scan statistics (Kulldorff) control error at the region level rather than the pixel level. fMRI experience shows that *parametric* RFT cluster p-values can be badly anti-conservative when the smoothness assumptions fail, while permutation stays nominal.

### Cited Findings
- **Bonferroni:** reject if p_i ≤ α/m, so FWER ≤ α under any dependence [standard].
- **Benjamini–Hochberg (1995):** sort p-values, find the largest k with `p_(k) ≤ kα/m`, reject the first k. Then FDR ≤ (m_0/m)α under independence [from memory, verify; JRSS-B, doi:10.1111/j.2517-6161.1995.tb02031.x].
- **Benjamini–Yekutieli (2001):** BH remains valid under positive regression dependence on a subset (PRDS). Under arbitrary dependence, replace α with `α / Σ_{i=1}^m 1/i ≈ α / (ln m + 0.577)` [from memory, verify; Ann. Stat. 29(4), doi:10.1214/aos/1013699998].
- **Conformal p-values + BH:** conformal p-values for testing whether new points come from the reference distribution are marginally valid but mutually dependent. Bates et al. show they are positively dependent, so BH controls FDR exactly, though only in a relatively weak marginal sense. They also give a calibration-conditional variant with stronger guarantees — [Bates, Candès, Lei, Romano, Sesia, Ann. Stat. 51(1):149–178, 2023, doi:10.1214/22-AOS2244](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-1/Testing-for-outliers-with-conformal-p-values/10.1214/22-AOS2244.full); [arXiv:2104.08279](https://arxiv.org/abs/2104.08279).
- **Random field theory / cluster inference:** Worsley et al. 1996, "A unified statistical approach for determining significant signals in images of cerebral activation", Hum. Brain Mapp. 4:58–73, doi:10.1002/(SICI)1097-0193(1996)4:1<58::AID-HBM4>3.0.CO;2-O [from memory, verify]. The FWER threshold uses the expected Euler characteristic: `P(max Z > u) ≈ E[χ(A_u)] = Σ_d R_d(S) ρ_d(u)`, with resels R_d set by the smoothness (FWHM) and Euler-characteristic densities ρ_d. For a 2-D Gaussian field, `ρ_2(u) = (4 ln 2)(2π)^{-3/2} u e^{-u²/2}` per resel [from memory, verify]. It assumes a smooth, stationary, sufficiently smooth Gaussian field and a high threshold.
- **Permutation max-statistic:** Nichols & Holmes 2002, Hum. Brain Mapp., doi:10.1002/hbm.1058 [from memory, verify]. It controls FWER exactly under exchangeability by using the permutation distribution of the image-wide maximum or cluster size.
- **Cluster failure:** for a nominal 5% FWE, parametric methods were "conservative for voxelwise inference and invalid for clusterwise inference", with worst-case FWE up to 70%. The nonparametric permutation test was nominal for both. The failure was attributed largely to non-Gaussian spatial autocorrelation functions. A cluster-defining threshold of p = 0.001 was much better than p = 0.01 — [Eklund, Nichols, Knutsson, PNAS 2016, doi:10.1073/pnas.1602413113](https://pmc.ncbi.nlm.nih.gov/articles/PMC4948312). Caveat: a 2018 follow-up reported nominal rates with two-sided tests plus ICA-FIX cleaning, and noted the original paper's implication about the share of literature affected was overstated. AFNI's developer disputed the framing — [AFNI slides](https://afni.nimh.nih.gov/pub/dist/edu/2019_04_milwaukee/afni07_advanced/Clusters_2017.pdf); [PMC follow-up record via search](https://pmc.ncbi.nlm.nih.gov/articles/mid/NIHMS992425).
- **Scan statistics (Kulldorff 1997):** maximise a likelihood ratio over a family of windows, usually circles or ellipses, and calibrate the maximum by Monte Carlo replication under H0. This controls FWER for the most likely cluster [from memory, verify; Comm. Stat. Theory Methods 26(6):1481–1496, doi:10.1080/03610929708831995].

### Inferences
- [own derivation] **Mapping to forensics:**
  - Pixel-wise or patch-wise heatmap thresholding: use BH if the per-patch p-values are positively dependent (smoothed residual statistics plausibly are). Use BY when the dependence structure is unknown.
  - "Is there any forged region, and where?": use the scan statistic or NFA over a family of candidate regions. The FWER/PFER guarantee is image-level, and it answers the detection question directly.
  - Cluster-extent inference (threshold the map, then test connected-component size against the null distribution of maximum cluster size) suits blob-like splices. Calibrate the null by spatial-block permutation or by simulating the residual field, **not** by Gaussian RFT formulas. The Eklund et al. result is the cautionary precedent.
- [own derivation] The forensic analogue of fMRI "subjects" is lacking. One image has one realisation, so there is nothing to permute across subjects. The usable within-image nulls are: (a) spatial-block permutation, which assumes stationarity of the authentic background; (b) a parametric background model fitted robustly, as in the a-contrario approach; (c) an external null learned from a corpus of authentic images from similar sources, which gives conformal-style calibration.

### Gaps
- I found no forensic paper that applies RFT/Euler-characteristic thresholds to forgery heatmaps. The mapping is inferred.
- How PRDS applies to forensic heatmap p-values has not been verified anywhere I found.

---

## 4. Spatial dependence: autocorrelation, effective sample size, block resampling

### Takeaway
Neighbouring patch statistics are positively correlated, and overlapping windows share pixels. So the effective number of independent observations is far smaller than the number of patches, naive i.i.d. tests are anti-conservative, and permutation nulls that shuffle individual patches are invalid. Fixes: estimate the correlation and use an effective n, use block or moving-block bootstrap or block permutation, use the wild bootstrap, or subsample at a stride beyond the correlation length.

### Cited Findings
- Moran's I [from memory, verify; Moran 1950, Biometrika 37:17–23, doi:10.2307/2332142]: `I = (n/W) Σ_ij w_ij (x_i−x̄)(x_j−x̄) / Σ_i (x_i−x̄)²`, with E[I] = −1/(n−1) under no autocorrelation. Normal approximation or permutation give its null.
- Moving-block bootstrap for stationary weakly dependent data: resample overlapping blocks of length ℓ, with ℓ → ∞ and ℓ/n → 0 for consistency [from memory, verify; Künsch 1989, Ann. Stat. 17(3):1217–1241, doi:10.1214/aos/1176347265]. Spatial block bootstrap and subsampling extend this to random fields (Lahiri, *Resampling Methods for Dependent Data*, Springer 2003, doi:10.1007/978-1-4757-3803-2) [from memory, verify].
- Breaking dependence in permutation causes inflated false positives in kernel tests, and the wild bootstrap fixes this — [Chwialkowski et al. 2014, arXiv:1408.5404](https://arxiv.org/abs/1408.5404).
- fMRI evidence that non-Gaussian spatial autocorrelation breaks parametric cluster inference while permutation survives — [Eklund et al. 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4948312).

### Inferences
- [own derivation] **Effective sample size:** for the mean of n stationary observations with correlation function ρ(h), `Var(x̄) = (σ²/n)·[1 + Σ_{h≠0} (1−|h|/n) ρ(h)]`, so `n_eff ≈ n / (1 + Σ_{h≠0} ρ(h))`. On a 2-D grid with separable AR(1) correlation of parameter ρ in each direction, `n_eff ≈ n·((1−ρ)/(1+ρ))²`. Example: ρ = 0.5 gives n_eff ≈ n/9. For sliding windows of side w and stride s < w, adjacent windows share a fraction (1 − s/w) of their pixels in each direction. Statistics computed from them are correlated by roughly that overlap even when the pixels are i.i.d., so n_eff ≈ number of *disjoint* windows, not the number of overlapping ones.
- [own derivation] **Why overlapping windows invalidate naive tests:** (i) the i.i.d. variance formula underestimates the variance by the factor above, so z/t/KS statistics are inflated. (ii) Patch-level permutation assumes exchangeability, but patches near a boundary share pixels with both sides. (iii) Multiple-testing corrections that assume independence (Šidák, BH under independence) count tests wrongly. Bonferroni and NFA stay valid, but they become very conservative.
- [own derivation] **Practical recipe:**
  1. Compute the residual feature field.
  2. Estimate its autocorrelation (variogram or FFT) on the robustly estimated "authentic" part, and get the correlation length L.
  3. Use non-overlapping blocks of side ≥ L for the tests, or a block permutation with block side ≈ 2L.
  4. Report Moran's I of the residual statistic as a diagnostic. Under the null after detrending, it should be near zero at lags ≥ L.
  5. If heatmaps come from overlapping windows, keep them for display, but do inference on a thinned grid.
- Moran's I computed on the forensic score map has two roles: as a **test statistic**, positive autocorrelation of scores signals a coherent anomalous blob (local Moran's I, Anselin 1995 LISA, doi:10.1111/j.1538-4632.1995.tb00338.x [from memory, verify]), and as a **diagnostic** of whether the null is mis-specified.

### Gaps
- I found no forensic-specific study of the correlation lengths of common residuals (SRM/noiseprint/DCT statistics). They must be measured per feature.

---

## 5. Change-point / segmentation formulations, robust statistics, conformal, spectral

### Takeaway
Localisation can be posed as estimating a partition. Potts or Mumford–Shah energies with statistical data terms, solved by graph cuts, give MAP partitions. MDL gives a penalised-likelihood rule for whether to split at all. Spectral clustering or community detection on a patch-affinity graph (Forensic Similarity Graphs) gives partitions with Cheeger-type guarantees on cut quality. Robust estimators (median/MAD, Huber, MCD) encode the "majority authentic" assumption, and their breakdown point (at most 50%) is exactly the limit: when the edit covers more than 50% of the image, "authentic" and "forged" are not identifiable from one image without external priors. Conformal anomaly detection gives finite-sample-valid p-values under exchangeability with a calibration set.

### Cited Findings
- **Forensic Similarity Graphs** (Mayer & Stamm, IEEE JSTSP 14(5):1049–1064, 2020; arXiv:1912.02861): patches are vertices and edges are weighted by learned forensic similarity. Community detection by modularity optimisation, spectral clustering, or normalised spectral clustering detects forgery (more than one community exists) and localises it. In the reported comparison spectral clustering generally did best (Columbia 0.86 vs. 0.83 for modularity and 0.74 for Splicebuster). Normalised spectral clustering was best on Carvalho (0.97) — [arXiv:1912.02861](https://arxiv.org/abs/1912.02861); [dblp](https://dblp.dagstuhl.de/rec/journals/jstsp/MayerS20.html). The numbers came from search snippets, not a full read of the paper.
- **Spectral clustering / Cheeger:** for a graph Laplacian with second eigenvalue λ_2 (normalised), Cheeger's inequality is `λ_2/2 ≤ h(G) ≤ √(2λ_2)`, where h is the conductance [from memory, verify; Chung, *Spectral Graph Theory*, AMS 1997; von Luxburg 2007 tutorial, arXiv:0711.0189, doi:10.1007/s11222-007-9033-z]. A small λ_2 certifies that a sparse cut exists (two weakly connected communities). The sweep cut on the Fiedler vector finds a cut with conductance ≤ √(2λ_2).
- **Potts / Mumford–Shah / graph cuts:** the Mumford–Shah functional is from Mumford & Shah 1989, CPAM 42:577–685, doi:10.1002/cpa.3160420503. For binary labels with submodular Potts pairwise terms, min-cut gives the exact global MAP; α-expansion gives a factor-2 approximation for multi-label Potts [from memory, verify; Boykov, Veksler, Zabih 2001, IEEE TPAMI, doi:10.1109/34.969114]. Chan–Vese two-phase piecewise-constant models (IEEE TIP 2001, doi:10.1109/83.902291) are the 2-region special case [from memory, verify].
- **MDL:** choose the model (one region vs. two regions plus a boundary) that minimises total code length `L(model) + L(data | model)` [from memory, verify; Rissanen 1978, Automatica, doi:10.1016/0005-1098(78)90005-5; Leclerc 1989, IJCV, "Constructing simple stable descriptions for image partitioning", doi:10.1007/BF00133569].
- **Conformal p-values:** with calibration scores from authentic data exchangeable with a test score, `p = (1 + #{j: s_j ≥ s_test})/(n+1)` is super-uniform in finite samples. The p-values are positively dependent across test points, and BH controls FDR — [Bates et al. 2023](https://arxiv.org/abs/2104.08279).
- **Robust statistics:** Huber's M-estimator (Ann. Math. Stat. 1964, doi:10.1214/aoms/1177703732) [from memory, verify]. Median and MAD have breakdown point 50%, the maximum for affine-equivariant location and scale. Rousseeuw's LMS/LTS and the MCD for multivariate data have breakdown up to about 50% (Rousseeuw 1984, JASA, doi:10.1080/01621459.1984.10477105) [from memory, verify]. The Huber M-estimator of location with fixed scale has breakdown 50% for location but needs a robust scale (MAD) to keep it [from memory, verify].

### Inferences
- [own derivation] **Two-region MDL / penalised LR as a change-point test:** for Gaussian features with region-specific parameters, `ΔL = n·log σ̂²_all − n_R log σ̂²_R − n_{R^c} log σ̂²_{R^c} − λ·(k_params·log n + c·|∂R|)`, and you split if ΔL > 0. The boundary-length penalty `c·|∂R|` is the Mumford–Shah / Potts perimeter term, so MDL, Potts-MAP and penalised-likelihood scans are the same objective with different penalty constants. Graph cut solves it exactly for binary labels when the data term is per-pixel (with parameters alternated, as in GrabCut-style EM). Only the penalty calibrates false alarms, so the false-alarm rate should be checked by Monte Carlo on authentic images, or the penalty replaced by an NFA-style term (log N_tests).
- [own derivation] **Breakdown when the edit covers more than 50%:** within a single image, H0 ("one source") vs. H1 ("two sources") is symmetric in the labels. Statistics alone can say *that* two populations exist and *where the boundary is*, but not *which* one is authentic. Labelling needs an asymmetric prior: a camera-model fingerprint (PRNU/noiseprint match), JPEG grid alignment with the image origin (ZERO's "foreign grid" vs. aligned grid), EXIF consistency, or a semantic prior (the edited object is usually a foreground object). Recipe: (1) detect inconsistency symmetrically (2-community test, Section 5 spectral, MMD between communities); (2) assign "authentic" using an external reference when one exists; (3) otherwise report both segments as "inconsistent pair" without assigning authenticity. A median/MAD reference silently assumes that the forged area is under 50%. Above that, it flags the authentic remainder.
- [own derivation] **Spectral approach as a test:** under H0 (one community), λ_2 of the normalised Laplacian of a similarity graph with i.i.d. edge noise concentrates away from 0. Under a planted 2-community model, λ_2 is small. A Monte-Carlo null for λ_2 can be built by block-permuting patch features and recomputing the graph. Mayer & Stamm use a threshold on the community structure. I found no closed-form null for λ_2 under their learned similarity.
- [own derivation] **Conformal recipe:** score each patch with a nonconformity score, for example a negative forensic-similarity to an image-level robust centroid, or a learned anomaly score. Calibrate against scores from (a) a corpus of authentic images from comparable sources, which gives exchangeability across images, not within one image, or (b) a held-out spatial split of the same image, where within-image dependence breaks exact exchangeability. Then apply BH. The guarantee holds only if calibration and test scores are exchangeable, so camera or processing mismatch between the calibration corpus and the test image breaks it.

### Gaps
- No closed-form null distribution of Forensic Similarity Graph spectral statistics was found.
- I did not verify the exact clustering threshold or decision rule used by Mayer & Stamm (I did not read the full text).

---

## 6. Random-matrix theory: Marchenko–Pastur (MP) for patch covariance

### Takeaway
For n patches of dimension p of pure i.i.d. noise with variance σ², the eigenvalues of the sample covariance follow the Marchenko–Pastur law with support `[σ²(1−√γ)², σ²(1+√γ)²]`, where γ = p/n. Structure (image content, low-rank) produces eigenvalues outside the upper edge (spikes, the BBP transition). The bulk thus identifies noise, which allows estimation of σ² per region and a test of whether the regional noise levels are consistent, a direct noise-inconsistency forensic cue. Patch-eigenvalue noise estimation is established (Chen, Zhu, Heng, ICCV 2015). The edge eigenvalue has a Tracy–Widom null, which allows formal tests.

### Cited Findings
- MP law [from memory, verify; Marchenko & Pastur 1967, Math. USSR-Sbornik 1:457–483, doi:10.1070/SM1967v001n04ABEH001994]: density `f(λ) = √((λ_+−λ)(λ−λ_−)) / (2π σ² γ λ)` on `[λ_−, λ_+]`, with `λ_± = σ²(1±√γ)²` (plus a point mass at 0 if γ > 1).
- Largest-eigenvalue fluctuations under the white null: centred and scaled, they converge to Tracy–Widom TW_1 (real case) [from memory, verify; Johnstone 2001, Ann. Stat. 29(2):295–327, doi:10.1214/aos/1009210544].
- Spiked model / BBP phase transition [from memory, verify; Baik, Ben Arous, Péché 2005, Ann. Probab., doi:10.1214/009117905000000233; Baik & Silverstein 2006]: a population spike ℓ produces an outlier sample eigenvalue only if `ℓ > σ²(1+√γ)`. Below that it is absorbed in the bulk, which sets a fundamental detectability limit.
- Optimal singular-value hard threshold for denoising under MP noise: `(4/√3)·√n·σ` for square matrices, with a median-based version when σ is unknown [from memory, verify; Gavish & Donoho 2014, IEEE TIT, doi:10.1109/TIT.2014.2323359, arXiv:1305.5870].
- Patch-covariance eigenvalue noise estimation: Chen, Zhu & Heng, "An Efficient Statistical Method for Image Noise Level Estimation", ICCV 2015, pp. 477–485, doi:10.1109/ICCV.2015.62. It relates noise variance to patch-covariance eigenvalues, exploits the fact that clean patches lie near a low-dimensional subspace, and argues that many methods underestimate noise — [CVF open access](https://openaccess.thecvf.com/content_iccv_2015/html/Chen_An_Efficient_Statistical_ICCV_2015_paper.html). The search snippets did not explicitly confirm that the paper names MP, so treat the MP framing as [from memory, verify].

### Inferences
- [own derivation] **Forensic recipe (region noise consistency via MP):**
  1. In each region R (say 128×128), extract n non-overlapping (or lightly overlapping) p = 7×7 patches of a high-pass residual. Here γ = 49/n.
  2. Compute the eigenvalues of the patch covariance and estimate σ²_R by fitting MP to the lower bulk. Discard the top r eigenvalues iteratively until the remainder is consistent with MP, e.g. compare the empirical CDF of the bulk to the MP CDF (KS), or match the mean and median of the bulk (Chen et al.-style).
  3. Get a per-region σ̂²_R with a standard error. The bulk mean is approximately normal with variance ≈ 2σ⁴/(np) for pure noise [own derivation from the χ² variance of the trace; verify for overlapping patches].
  4. Test the homogeneity of σ̂²_R across regions against the image-wide robust model σ²(intensity) (signal-dependent noise: Poisson–Gaussian, so condition on mean intensity), using robust z-scores and BY or NFA for multiplicity.
- [own derivation] **Caveats:**
  - Overlapping patches violate MP's i.i.d.-rows assumption. The bulk shape changes (rows are dependent), so use non-overlapping patches or recalibrate the edges by simulation.
  - Demosaicing, JPEG and denoising make the noise colored, i.e. Σ ≠ σ²I. MP then generalises to the Silverstein equation for general population covariance and the edges move. Calibrate by simulating with the image's estimated noise PSD.
  - Signal-dependent noise must be stratified by intensity.
  - Low-texture regions give clean estimates. Textured regions push structure into the bulk when the spike strength is near the BBP threshold, which biases σ̂² upward.

### Gaps
- I did not find a forensic paper that explicitly uses MP/Tracy–Widom tests for splicing detection, though noise-level-inconsistency methods (e.g. Noisesniffer) are closely related.

---

## 7. Closed-form nulls, simulation checks and proof-assistant verifiability

### Takeaway
Exact or closed-form nulls exist for: KS (Kolmogorov distribution, asymptotic, distribution-free), Hotelling T² (exact F under Gaussian i.i.d.), the binomial tails used by NFA (exact under independent Bernoulli votes), conformal p-values (exact finite-sample super-uniformity under exchangeability), permutation tests (exact under exchangeability), BH/BY bounds, MP/Tracy–Widom (asymptotic), and RFT Euler-characteristic approximations (asymptotic, high threshold). Degenerate kernel statistics (MMD, energy) have only an implicit null (weighted χ² sums), so they need permutation or bootstrap. Scan statistics, segmentation energies and spectral statistics need Monte Carlo. The combinatorial/measure-theoretic guarantees (NFA ≤ ε, Bonferroni, BH under independence, conformal super-uniformity, permutation exactness) are short proofs that suit Lean/mathlib. Asymptotic laws (MP, TW, KS limit, Wilks, RFT) are far harder to formalise and are better checked by Monte Carlo.

### Cited Findings
- Lean/mathlib status for the CLT: a standalone project (Degenne, `clt`) reports a full proof of the CLT for real random variables *assuming Prokhorov's theorem*, last updated June 2025 — [Reservoir: RemyDegenne/clt](https://reservoir.lean-lang.org/@RemyDegenne/clt). A mathlib Zulip thread notes the CLT was missing from mathlib and that Riesz–Markov–Kakutani was being merged as an ingredient toward Prokhorov — [Zulip archive](https://leanprover-community.github.io/archive/stream/287929-mathlib4/topic/Status.20of.20the.20Central.20limit.20theorem.html). The UW Math AI Lab ran a Winter 2025 CLT formalisation project — [UW AI Math](https://ai.math.uw.edu/projects/winter-2025). A 2026 arXiv paper reports formalising a probability textbook in Lean — [arXiv:2607.27298](https://arxiv.org/abs/2607.27298). I could not confirm whether the full CLT is merged into mathlib as of Oct 2026.
- Mathlib infrastructure reported as present: Ionescu-Tulcea theorem (Marion, 2025), i.e. probability measures on Markov chain trajectories — [arXiv:2506.18616](https://arxiv.org/abs/2506.18616).
- Closed-form nulls (sources in the sections above): Hotelling F, KS/Kolmogorov, binomial NFA, conformal, BH/BY, MP, TW, RFT EC densities.

### Inferences
- [own derivation] **Classification table:**

| Tool | Null distribution | Exact / asymptotic | Assumptions that matter in images | Verify by |
|---|---|---|---|---|
| NFA / Bonferroni PFER bound | none needed (union/expectation bound) | exact inequality | per-test tails must be valid; across-test dependence irrelevant | **Lean** (linearity of expectation + super-uniformity); MC for tail validity |
| Binomial tail of block votes | Binomial(n,p) | exact | independent votes (disjoint blocks) | Lean (Mathlib has binomial distributions as `PMF.binomial`; verify name) + MC |
| Permutation test | permutation distribution | exact (super-uniform, with +1 correction) | exchangeability → fails under spatial dependence unless block-permuted | **Lean** (group-invariance argument, finite) |
| Conformal p-value | uniform on {1/(n+1),…,1} | exact, finite-sample | exchangeability of calibration and test scores | **Lean** (finite exchangeability, rank uniformity) |
| BH under independence/PRDS | FDR ≤ α m₀/m | exact inequality | independence/PRDS | Lean feasible (finite combinatorics; independence case is short) |
| BY | FDR ≤ α | exact inequality | none | Lean feasible |
| KS two-sample | Kolmogorov (asym.), exact finite tables | exact finite (combinatorial) + asymptotic | i.i.d., continuous | finite-n exact via combinatorics; asymptotic by MC |
| Hotelling T² | F(d, n−d−1) | exact under Gaussian | Gaussian, i.i.d., n>d | MC (Lean would need Wishart theory; heavy) |
| Box's M | χ²/F approx. | asymptotic | Gaussian (very non-robust) | MC |
| MMD / energy | weighted χ² sum (implicit) | asymptotic | i.i.d. | permutation/wild bootstrap; MC |
| Sliced/W_p | none closed form | — | — | permutation/bootstrap |
| Wilks LRT (fixed region) | χ²_q | asymptotic | regularity, i.i.d., fixed region | MC |
| Scan statistic (max LR) | none closed form (EVT approx.) | — | — | MC (Kulldorff-style) |
| RFT cluster/peak | EC heuristic | asymptotic, high threshold | smooth stationary Gaussian field | MC / permutation (Eklund lesson) |
| MP bulk / TW edge | MP, TW₁ | asymptotic (p,n→∞) | i.i.d. rows, white noise | MC at the actual (p,n) |
| Cheeger inequality | deterministic inequality | exact | none | **Lean** (linear algebra; feasible, nontrivial) |
| Graph-cut optimality (binary submodular) | deterministic | exact | submodular pairwise | Lean feasible (max-flow/min-cut exists in some libraries; check) |
| Breakdown point of median = 50% | deterministic | exact | — | **Lean** (finite order statistics) |

- [own derivation] **Recommended split:** formalise the *finite, distribution-free* guarantees in Lean: the NFA expectation bound, permutation/conformal super-uniformity, Bonferroni, BH and BY, the median breakdown point, and possibly the Cheeger inequality. They are the logical backbone of a "sound" localisation claim, and their proofs need only finite probability spaces or basic measure theory. Validate the *distributional modelling assumptions* by Monte Carlo on authentic images and on synthetic random fields with matched autocorrelation: the binomial-vote independence, the MP edges at finite (p,n), KS calibration under dependence, and the RFT/scan thresholds. A proof assistant cannot certify that real camera noise satisfies the null. Only empirical calibration (observed false-alarm counts ≤ ε on authentic corpora) can.
- [own derivation] **Calibration check for any detector:** run it on N_auth authentic images. For NFA detectors the mean number of detections per image should be ≤ ε. For p-value detectors, the per-image minimum p-value multiplied by the number of tests should be roughly super-uniform. Plot an empirical-vs-nominal FWER/FDR curve. This is the forensic counterpart of Eklund et al.'s null-data check.

### Gaps
- I could not confirm the exact current mathlib names and coverage of binomial distributions, exchangeability, max-flow/min-cut, or the Cheeger inequality as of Oct 2026. Treat the Lean feasibility claims as [own inference].
- I found no existing Lean formalisation of the a-contrario NFA bound, conformal prediction, or the BH procedure (not searched exhaustively).
