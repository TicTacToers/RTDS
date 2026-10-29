# PI-RADS 3 csPCa prediction: summary of experiments

This document is self-contained. The file names it mentions (scripts, CSVs) are for provenance only. Every number needed for the review is in this document, and Appendix A contains the full weight tables that would otherwise be in separate files.

## 1. Task and data

- **Question:** does MRI radiomics improve on PSA density (PSAD) for predicting clinically significant prostate cancer (csPCa, ISUP ≥ 2) in PI-RADS 3 patients, i.e. for deciding whether to biopsy?
- **Data:** PI-CAI public training set, the only dataset used. It has 1500 studies from 1476 patients across 3 centers (RUMC, PCNN, ZGT).
  - Images: T2W and ADC, `.mha`, Zenodo record 6624726 v2.0.
  - Labels and masks: `picai_labels` repository.
- **Cohort:** 1500 studies → 151 with highest lesion PI-RADS = 3 → 151 with biopsy or surgery result → **149 with PSAD and outcome; 18 csPCa (12.1%)**.
  - By center: RUMC 85 patients / 14 csPCa; PCNN 47 / 2; ZGT 17 / 2.
  - Missing PSAD (399 marksheet rows) was filled in as PSA ÷ prostate volume. Without this the cohort would be 100 patients with 16 csPCa.
- **Evaluation used throughout, unless stated otherwise:**
  - **Cross-validation:** stratified 5-fold, seed 42.
  - **Threshold:** each model's 90%-sensitivity cut-off is chosen on the training folds only, from inner 3-fold out-of-fold predictions.
  - **Confidence intervals:** AUC 95% CI from 2000 bootstrap resamples of one split's out-of-fold predictions. ΔAUC versus PSAD uses a paired bootstrap.
  - **Stability check:** the same pipeline was repeated over 20 further random splits (seeds 0–19), reporting mean and range.
  - **Leakage control:** scaling, feature selection, PCA and hyperparameter tuning are all fitted inside the training folds.
- **Environment:** Python 3.13 venv. PyRadiomics was installed from GitHub (commit `8ed5793`) because the PyPI source packages fail to build on 3.13.

## 2. Baseline: whole-gland radiomics

**Method:**
- **Features:** PyRadiomics first-order and GLCM features, 42 from T2W and 42 from ADC (84 in total).
- **Region:** the AI whole-gland mask (Bosma22b), resampled to each image grid with nearest-neighbour interpolation.
- **Exclusions:** no shape features, since gland size is already in PSAD.
- **Settings:** T2W normalised (scale 100, bin width 5); ADC not normalised (bin width 50).
- **Models:**
  - PSAD rule
  - logistic regression on PSAD + age
  - Lasso (`LogisticRegressionCV`, L1, 10 Cs, inner 3-fold, AUC scoring)
  - Lasso on radiomics + PSAD
  - random forest (300 trees, min leaf 5, balanced class weights)

| Model | AUC (95% CI) | Mean AUC over 20 splits (range) | Missed csPCa / biopsies avoided at threshold |
|---|---|---|---|
| PSAD rule | **0.618** (0.47–0.76) | 0.618 (no fitting, so constant) | 3 / 19.5% |
| Clinical (PSAD + age) | 0.507 (0.36–0.66) | 0.569 (0.52–0.62) | 1 / 4.7% |
| Radiomics Lasso (84) | 0.466 (0.32–0.61) | 0.471 (0.38–0.56) | 1 / 12.1% |
| Radiomics + PSAD Lasso | 0.471 (0.33–0.62) | 0.504 (0.43–0.58) | 0 / 8.1% |
| Random forest (radiomics + PSAD) | 0.480 (0.32–0.65) | 0.486 (0.41–0.56) | 2 / 12.8% |

Further threshold metrics for the baseline (at the 90%-sensitivity cut-off learned on training folds):

| Model | Sensitivity | Specificity | NPV |
|---|---|---|---|
| PSAD rule | 0.833 | 0.198 | 0.897 |
| Clinical (PSAD + age) | 0.944 | 0.046 | 0.857 |
| Radiomics Lasso (84) | 0.944 | 0.130 | 0.944 |
| Radiomics + PSAD Lasso | 1.000 | 0.092 | 1.000 |
| Random forest | 0.889 | 0.130 | 0.895 |

**Diagnostics on the baseline:**
- Only 2 of the 84 features had a single-feature AUC outside 0.35–0.65.
- Mean gland ADC was 1442 ± 175 with csPCa versus 1473 ± 286 without.
- The strongest features were energy and total-energy features. These scale with the number of voxels and correlate with prostate volume; volume alone gives AUC 0.357, i.e. smaller glands in csPCa. So gland size leaked back in despite the shape features being excluded.

| Feature | Single-feature AUC | Spearman ρ with prostate volume |
|---|---|---|
| adc_firstorder_Energy | 0.33 | +0.56 |
| adc_firstorder_TotalEnergy | 0.35 | +0.75 |
| t2w_firstorder_TotalEnergy | 0.37 | +0.54 |
| t2w_glcm_Imc1 | 0.35 | +0.24 |
| t2w_glcm_Imc2 | 0.64 | −0.48 |
| adc_firstorder_Minimum | 0.63 | +0.10 |
| Prostate volume (reference) | 0.36 | – |

(AUC below 0.5 means the value is lower in csPCa.)

- **Center confounding:**

| Center | Patients | csPCa | csPCa rate | Mean gland ADC (×10⁻⁶ mm²/s) | Mean PSAD (ng/mL/cm³) |
|---|---|---|---|---|---|
| RUMC | 85 | 14 | 16% | 1394 | 0.19 |
| PCNN | 47 | 2 | 4% | 1644 | 0.19 |
| ZGT | 17 | 2 | 12% | 1363 | 0.21 |

  A logistic regression on the 84 features predicts the center with 77% cross-validated accuracy, against 57% for always guessing the largest center.

**Conclusion:** whole-gland radiomics performed at chance level, and adding it to PSAD lowered the AUC.

## 3. Change 1: label-free zonal features, a small feature set, Lasso vs Ridge

**What changed:**
- **Region:** AI zonal masks (Yuan23: label 1 = peripheral zone PZ, label 2 = transition zone TZ). These are anatomy masks drawn for every patient, so they carry no outcome information.
- **Lesion masks rejected as leaky:**
  - Human csPCa outlines exist only for positive cases.
  - The AI lesion masks (Bosma22a) come from a model trained on these same patients' labels.
- **12 pre-specified features:**
  - ADC 10th percentile, median and interquartile range in the PZ and in the TZ (6 features: `adc_pz_p10`, `adc_pz_median`, `adc_pz_iqr`, `adc_tz_p10`, `adc_tz_median`, `adc_tz_iqr`)
  - median T2W in the PZ and in the TZ, as z-scores within the gland (2 features: `t2w_pz_median`, `t2w_tz_median`)
  - "dark blob": the largest connected component of voxels with ADC < 1000 inside the gland, after a 1-voxel in-plane erosion. Features: its volume in mL (`dark_blob_ml`), its mean ADC (`dark_blob_adc`), its mean T2W z-score (`dark_blob_t2w`), plus the fraction of gland voxels below ADC 1000 (`adc_low_fraction`) (4 features)
- **No gland-volume features.**
- **Data-quality fix:** voxels with ADC ≤ 0 are excluded. These are failed fits, present in 95 of 149 ADC maps and covering up to 24% of the gland in one patient. **The baseline whole-gland features did not get this fix.**
- **Models:**
  - Lasso, unchanged from the baseline
  - Ridge (L2, otherwise the same settings)
  - each with and without PSAD
- **Interpretation:** standardised weights over 100 refits (20 splits × 5 folds), with how often Lasso kept each feature and how often its sign agreed (Appendix A1).

| Model | AUC (95% CI) | ΔAUC vs PSAD (95% CI) | Mean over 20 splits (range) | Missed / avoided |
|---|---|---|---|---|
| Region Lasso (12) | 0.583 (0.46–0.71) | −0.036 (−0.21 to +0.16) | 0.599 (0.53–0.66) | 0 / 6.0% |
| Region Ridge (12) | 0.553 (0.40–0.70) | −0.065 (−0.29 to +0.17) | 0.577 (0.52–0.66) | 2 / 15.4% |
| **Region Lasso + PSAD** | **0.635** (0.50–0.76) | +0.017 (−0.13 to +0.17) | **0.645** (0.52–0.72) | 0 / 4.0% |
| Region Ridge + PSAD | 0.570 (0.42–0.72) | −0.049 (−0.20 to +0.09) | 0.608 (0.55–0.69) | 3 / 16.1% |
| Check: PSAD + prostate volume (plain logistic) | 0.598 | – | 0.600 (0.56–0.63) | – |

**Improvements over the baseline:**
- Imaging-only AUC rose from 0.47 to 0.58–0.60, and stayed above 0.5 on all 20 splits (the baseline went as low as 0.33–0.38).
- Region Lasso + PSAD had the best mean over repeated splits of any model tried (0.645).
- The small gain is not explained by gland volume alone: PSAD + volume scores 0.60.

**What the models learned:**
- **`t2w_tz_median`** was the most consistently used feature: kept in 97% of Lasso refits, always with a negative sign, single-feature AUC 0.31. It is a gland-size proxy, though:
  - it correlates with volume (ρ +0.56);
  - its direction is inconsistent across centers (AUC 0.24 RUMC, 0.39 PCNN, 0.83 ZGT).
- **`dark_blob_t2w`** is the most credible imaging signal: a dark ADC spot that is also dark on T2W means more csPCa.
  - kept in 83–89% of refits; sign agreement 84–100%
  - consistent across centers (AUC 0.27 / 0.37 / 0.33)
  - independent of volume (ρ 0.02) and of PSAD (ρ −0.01)
  - **It was identified after seeing the results**, so it is not pre-registered.
- **`adc_pz_median` and `adc_pz_p10`** have large opposite-signed weights, are kept in only about half the refits, and are strongly correlated (ρ 0.71). This is a collinearity artifact and carries no interpretable meaning.

**Shortcomings:**
- No model significantly beats PSAD; every ΔAUC CI includes 0.
- The thresholds behave poorly. The Lasso models hit 100% sensitivity but avoid only 4–6% of biopsies, against PSAD's 19.5%.
- Volume still leaks in through `t2w_tz_median`.
- The absolute ADC threshold of 1000 ignores the scale differences between centers.

## 4. Change 2: PCA before the penalised logistic model

**What changed:**
- **Pipeline:** standardise → PCA on the 12 region features → logistic regression (L1 or L2).
- **PSAD** goes in as a separate, scaled input and is not mixed into the components.
- **Tuning:** the number of components (2, 3, 4, 6 or 8) and C (6 values from 10⁻³ to 10²) are tuned together by an inner 3-fold `GridSearchCV` inside every training fold.
- **Interpretation:** the component weights were projected back onto the original features (loadings transposed × coefficients; Appendix A2).

| Model | AUC (95% CI) | ΔAUC vs PSAD (95% CI) | Mean over 20 splits (range) | Missed / avoided |
|---|---|---|---|---|
| PCA + Ridge (12) | 0.632 (0.49–0.77) | +0.014 (−0.19 to +0.22) | 0.593 (0.54–0.65) | 1 / 7.4% |
| PCA + Lasso (12) | 0.620 (0.48–0.75) | +0.002 (−0.21 to +0.21) | 0.585 (0.53–0.66) | 2 / 18.8% |
| PCA + Ridge + PSAD | **0.672** (0.54–0.80) | +0.054 (−0.11 to +0.22) | 0.633 (0.56–0.71) | 0 / 0.7% |
| PCA + Lasso + PSAD | 0.637 (0.50–0.75) | +0.018 (−0.13 to +0.17) | 0.629 (0.57–0.67) | 2 / 19.5% |

Number of components chosen by the inner CV over 100 refits:

| Model | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| PCA + Ridge | 11 | 7 | 2 | 31 | 49 |
| PCA + Lasso | 19 | 12 | 4 | 33 | 32 |
| PCA + Ridge + PSAD | 22 | 7 | 2 | 34 | 35 |
| PCA + Lasso + PSAD | 27 | 16 | 4 | 30 | 23 |

**Improvements:**
- The weights became stable and interpretable. The cancelling PZ pair disappears (`adc_pz_median` weight about −0.05), and the top six features keep the same sign in 92–100% of refits.
- PCA + Lasso + PSAD has a narrower range over splits (0.57–0.67 against 0.52–0.72) and matches PSAD's biopsies avoided (19.5%) with 2 missed cancers against PSAD's 3.

**Shortcomings:**
- **The 0.672 is a single-split artifact.** Its mean over 20 splits (0.633) is below Region Lasso + PSAD (0.645).
- Over repeated splits, PCA changes AUC by only −0.02 to +0.03 compared with the same models without PCA.
- The inner CV cannot settle on a number of components: it picks 2 in about 20% of refits and 6–8 in about 65%.
- Several learned directions are clinically counterintuitive: a larger dark blob, a higher PZ ADC 10th percentile and more PZ ADC spread all point toward *less* csPCa. This suggests gland makeup (benign enlargement, inflammation) rather than lesion signal.
- The PCA + Ridge + PSAD threshold collapsed: 0.7% of biopsies avoided.

## 5. Change 3: bagging and boosting

**What changed:** all three use the 12 region features + PSAD, with hyperparameters fixed in advance rather than tuned.
- **Bagged Lasso:** 25 Lasso models, averaged. It uses a custom **stratified** bootstrap that resamples csPCa and non-csPCa patients separately. Plain `BaggingClassifier` crashed because some resamples, or their inner folds, contained no csPCa at all.
- **Random forest:** settings as in the baseline.
- **Gradient boosting:** `HistGradientBoosting`, depth 2, learning rate 0.05, 100 iterations, min leaf 10, L2 penalty 1.0, balanced class weights.

| Model | AUC (95% CI) | ΔAUC vs PSAD (95% CI) | Mean over 20 splits (range) | Missed / avoided |
|---|---|---|---|---|
| Region Lasso + PSAD (reference) | 0.635 (0.50–0.76) | +0.017 | 0.645 (0.52–0.72) | 0 / 4.0% |
| Bagged Lasso + PSAD | 0.636 (0.49–0.77) | +0.018 (−0.14 to +0.19) | 0.627 (**0.59–0.67**) | 1 / 18.1% |
| Random forest + PSAD | 0.509 (0.35–0.66) | −0.109 (−0.33 to +0.11) | 0.568 (0.52–0.61) | 2 / 6.7% |
| Gradient boosting + PSAD | 0.512 (0.35–0.66) | −0.106 (−0.33 to +0.12) | 0.554 (0.49–0.61) | 2 / 6.7% |

**Improvements:**
- Bagging roughly halved the spread across splits.
- It gave the most clinically balanced threshold result of the cross-validated models: 1 missed, 18.1% avoided, against PSAD's 3 missed, 19.5%.

**Shortcomings:**
- Bagging did not improve mean AUC.
- Both tree methods were clearly worse than the linear models, consistent with too few events (18) for non-linear learning.
- None beats PSAD.
- The bagged Lasso run took about 49 minutes.

## 6. Change 4: pre-train on the rest of PI-CAI, test once on PI-RADS 3

**What changed:**
- **Training set:** all studies with highest PI-RADS ≠ 3, minus 8 studies from patients who also have a PI-RADS 3 study, leaving 1341 studies with 405 csPCa (30%).
  - PI-RADS 1–2 or no lesion: 611 studies, 8 csPCa
  - PI-RADS 4–5: 730 studies, 397 csPCa
  - PSAD was missing for 50, so the "+ PSAD" models train on 1291 studies.
- **Test set:** the 149 PI-RADS 3 patients, scored once. No patient appears on both sides; the script asserts this.
- **Models fixed before scoring the test set:**
  - Lasso + PSAD as the declared primary model
  - Lasso imaging-only, Ridge + PSAD and Ridge imaging-only as secondary models
  - PSAD as the reference
  - Lasso and Ridge were seeded (`random_state = 42`).
- **Thresholds:** every model's 90%-sensitivity threshold, PSAD's included, was learned on the training set.

| Model | CV AUC within training set | **Test AUC (95% CI)** | ΔAUC vs PSAD (95% CI) | Test sensitivity | Test specificity | Test NPV | Missed / avoided |
|---|---|---|---|---|---|---|---|
| **Lasso + PSAD (primary)** | 0.747 | **0.604** (0.45–0.74) | −0.014 (−0.18 to +0.17) | 0.833 | 0.221 | 0.906 | 3 / 21.5% |
| Lasso, imaging only | 0.735 | 0.577 (0.43–0.72) | −0.042 (−0.22 to +0.16) | 0.833 | 0.237 | 0.912 | 3 / 22.8% |
| Ridge + PSAD | 0.746 | 0.606 (0.46–0.75) | −0.012 (−0.18 to +0.17) | 0.833 | 0.221 | 0.906 | 3 / 21.5% |
| Ridge, imaging only | 0.734 | 0.581 (0.43–0.72) | −0.038 (−0.22 to +0.16) | 0.833 | 0.237 | 0.912 | 3 / 22.8% |
| PSAD alone | 0.740 | **0.618** (0.47–0.76) | – | 0.889 | 0.244 | 0.941 | 2 / 22.8% |

**Improvements:**
- This is the cleanest design used: one pre-declared primary model, no cross-validation noise on the test set, and thresholds learned externally to it. Thresholds transferred reasonably (test sensitivity 0.83–0.89 against the 0.90 target).
- It confirms the image features carry csPCa signal in the general PI-CAI population: imaging-only CV AUC is 0.735.
- The learned weights are clinically sensible (Appendix A3):
  - fraction of gland with ADC < 1000: +0.88
  - PZ median ADC: −0.57
  - dark-blob ADC: −0.40
  - dark-blob T2W: −0.16
  - PSAD: +0.32
- It explains the negative result: those same features have close-to-chance single-feature AUCs within PI-RADS 3 (`adc_low_fraction` 0.48; `dark_blob_adc` 0.56, the wrong direction). The radiologist's PI-RADS 3 score already reflects these ADC/T2W appearances, so within that group they add little.

**Shortcomings:**
- Nine times more training data did not improve test performance; it matches the cross-validated results.
- **Population shift:** obvious PI-RADS 4–5 lesions differ from equivocal PI-RADS 3 lesions.
- **Label shift:** PI-RADS 1–2 negatives were mostly confirmed by MRI only, while the test cohort is biopsy-confirmed.
- `t2w_tz_median` (the volume proxy) is still heavily weighted (−0.54).
- Lasso kept all 12 features, so Lasso and Ridge are effectively identical at this sample size.

## 7. Overall result

| Approach | Best imaging + PSAD AUC (mean over splits, or test) | Beats PSAD (0.618)? |
|---|---|---|
| Baseline whole-gland Lasso | 0.47–0.50 | No, worse |
| Zonal features, Lasso | 0.645 | No; ΔAUC CI includes 0 |
| + PCA | 0.629–0.633 | No |
| + Bagging | 0.627 | No |
| Boosting / random forest | 0.55–0.57 | No, worse |
| Pre-train on 1341 studies, test on PI-RADS 3 | 0.604 (test) | No |

The negative result is robust: **in this cohort, hand-crafted whole-gland and zonal MRI features do not add meaningful discrimination over PSAD for csPCa in PI-RADS 3.** The zonal redesign was the only change that substantially improved on the baseline radiomics, moving it from chance to weak signal. Every other change affected stability or interpretability, not discrimination.

## 8. Methodological caveats for the review

1. **Small number of events.** With 18 csPCa cases every CI is wide (about ±0.15 AUC), and one cancer equals 5.6 percentage points of sensitivity. Missed-cancer and biopsies-avoided comparisons rest on 0–3 patients.
2. **Multiplicity and analyst degrees of freedom.** About 20 model variants were evaluated on the same 149 patients.
   - The zonal feature set was designed *after* the baseline diagnostics on this cohort (energy features as volume proxies, ADC by center).
   - `dark_blob_t2w` was singled out *after* seeing the results.
   - Only the pre-training step had a declared primary model, and even then the analyst had already seen this cohort's cross-validated results.
3. **Unequal preprocessing.** Zero-valued ADC voxels were excluded for the region features but **not** for the baseline whole-gland features, which may disadvantage the baseline (e.g. the ADC minimum and 10th percentile).
4. **Non-determinism.** The baseline Lasso and the Change 1 Ridge use liblinear without `random_state`, so results drift slightly between runs: whole-gland Lasso AUC was 0.462 in one run and 0.467 in another, and Lasso weights varied a little. Only the pre-training models (Change 4) are seeded.
5. **The CIs ignore training variability.** The bootstrap is on one split's out-of-fold predictions. The 20-split ranges partly address this, but there is no nested bootstrap.
6. **Center confounding is not corrected.** There is no ADC harmonisation (e.g. ComBat) or per-center analysis, and the ADC < 1000 threshold is absolute despite the scale differences between centers.
7. **The masks were not quality-checked.** The AI gland and zonal masks were never visually inspected. The `picai_labels` README itself notes faulty segmentations, e.g. case `11050_1001070`.
8. **The PSAD imputation affects the cohort.** PSA ÷ volume replaced missing PSAD, growing the cohort from 100 to 149 patients. It's justified by the definition of PSAD, but it is an analysis choice.
9. **A matched-sensitivity metric was excluded.** The pipeline also computes "biopsies avoided at matched sensitivity", but it sets its threshold on the evaluation data, so it is optimistic and is not reported here.
10. **No external validation.** An external dataset (AI4AR) is planned but not yet available, so all results are internal to PI-CAI.
11. **Fixed ensemble hyperparameters.** The random forest and gradient boosting settings were fixed without tuning, by design given the small number of events, which may slightly understate them.

## 9. Provenance (not accessible to the reader)

These are the source names behind each part, listed for reference only.

**Scripts:**
- `psad_baseline.py`: cohort and PSAD.
- `extract_radiomics.py`, `compare_models.py`, `radiomics_pipeline.ipynb`: Section 2.
- `region_features.py`, `improved_models.py`: Section 3.
- `pca_models.py`: Section 4.
- `ensemble_models.py`: Section 5.
- `pretrain_test.py`: Section 6.

**Result files:** `results_internal_notebook.csv`, `results_improved.csv`, `coefficients_improved.csv`, `results_pca.csv`, `coefficients_pca.csv`, `results_ensemble.csv`, `results_pretrain_test.csv`, `coefficients_pretrain_test.csv`.

## Appendix A: weight tables

Weights are on standardised features; positive means a higher value predicts csPCa. "Single-feature AUC" is computed on the 149 PI-RADS 3 patients; below 0.5 means the value is lower in csPCa.

### A1. Change 1: region models, over 100 refits (20 splits × 5 folds)

"Kept %" = share of refits with a non-zero weight. "Same sign %" = among refits that kept the feature, the share agreeing on its direction.

**Region Lasso (12)**

| Feature | Mean weight | Kept % | Same sign % | Single-feature AUC |
|---|---|---|---|---|
| adc_pz_median | −1.695 | 43 | 100 | 0.518 |
| adc_pz_p10 | +1.627 | 52 | 100 | 0.588 |
| adc_tz_p10 | −0.595 | 44 | 98 | 0.452 |
| t2w_tz_median | −0.509 | 97 | 100 | 0.312 |
| adc_pz_iqr | +0.399 | 83 | 72 | 0.440 |
| adc_tz_iqr | +0.357 | 40 | 95 | 0.469 |
| dark_blob_ml | −0.277 | 38 | 87 | 0.450 |
| adc_low_fraction | −0.179 | 38 | 89 | 0.481 |
| t2w_pz_median | −0.137 | 49 | 84 | 0.522 |
| dark_blob_t2w | −0.118 | 89 | 84 | 0.374 |
| adc_tz_median | +0.095 | 25 | 56 | 0.433 |
| dark_blob_adc | −0.009 | 42 | 57 | 0.559 |

**Region Ridge (12)** (Ridge keeps every feature, so "kept" is always 100%)

| Feature | Mean weight | Same sign % | Single-feature AUC |
|---|---|---|---|
| adc_pz_median | −1.005 | 95 | 0.518 |
| adc_pz_p10 | +0.985 | 100 | 0.588 |
| adc_tz_p10 | −0.424 | 99 | 0.452 |
| t2w_tz_median | −0.326 | 100 | 0.312 |
| adc_tz_iqr | +0.249 | 64 | 0.469 |
| dark_blob_ml | −0.228 | 94 | 0.450 |
| adc_pz_iqr | +0.215 | 84 | 0.440 |
| t2w_pz_median | −0.145 | 78 | 0.522 |
| adc_low_fraction | −0.141 | 90 | 0.481 |
| dark_blob_t2w | −0.103 | 92 | 0.374 |
| dark_blob_adc | −0.009 | 58 | 0.559 |
| adc_tz_median | +0.004 | 93 | 0.433 |

**Region Lasso + PSAD**

| Feature | Mean weight | Kept % | Same sign % |
|---|---|---|---|
| adc_pz_p10 | +0.941 | 36 | 100 |
| adc_pz_median | −0.938 | 36 | 100 |
| t2w_tz_median | −0.501 | 98 | 100 |
| adc_tz_p10 | −0.335 | 35 | 97 |
| psad | +0.317 | 98 | 100 |
| adc_tz_iqr | +0.206 | 30 | 97 |
| adc_low_fraction | −0.146 | 33 | 97 |
| dark_blob_t2w | −0.132 | 83 | 93 |
| t2w_pz_median | −0.130 | 37 | 92 |
| adc_pz_iqr | +0.125 | 83 | 84 |
| dark_blob_ml | −0.122 | 26 | 81 |
| adc_tz_median | +0.010 | 17 | 59 |
| dark_blob_adc | −0.010 | 34 | 68 |

**Region Ridge + PSAD**

| Feature | Mean weight | Same sign % |
|---|---|---|
| t2w_tz_median | −0.222 | 100 |
| adc_pz_p10 | +0.186 | 100 |
| psad | +0.174 | 100 |
| adc_pz_median | −0.161 | 95 |
| dark_blob_t2w | −0.104 | 100 |
| adc_tz_p10 | −0.082 | 99 |
| t2w_pz_median | −0.081 | 78 |
| adc_tz_median | −0.065 | 99 |
| adc_tz_iqr | +0.063 | 56 |
| dark_blob_ml | −0.058 | 99 |
| adc_pz_iqr | −0.056 | 99 |
| adc_low_fraction | −0.056 | 93 |
| dark_blob_adc | −0.022 | 55 |

### A2. Change 2: PCA models, weights projected back onto the features (100 refits)

| Feature | PCA + Ridge | PCA + Lasso | PCA + Ridge + PSAD | PCA + Lasso + PSAD |
|---|---|---|---|---|
| psad | – | – | +0.286 (100%) | +0.462 (100%) |
| t2w_tz_median | −0.385 (100%) | −0.469 (100%) | −0.305 (100%) | −0.429 (100%) |
| adc_pz_iqr | −0.366 (100%) | −0.423 (100%) | −0.246 (100%) | −0.362 (100%) |
| dark_blob_ml | −0.436 (99%) | −0.547 (93%) | −0.185 (98%) | −0.331 (97%) |
| dark_blob_t2w | −0.190 (96%) | −0.258 (98%) | −0.216 (100%) | −0.305 (99%) |
| adc_pz_p10 | +0.305 (100%) | +0.359 (96%) | +0.197 (99%) | +0.290 (97%) |
| adc_tz_p10 | −0.249 (92%) | −0.307 (86%) | −0.115 (83%) | −0.174 (74%) |
| t2w_pz_median | −0.153 (67%) | −0.198 (65%) | −0.087 (59%) | −0.136 (53%) |
| adc_tz_median | −0.116 (98%) | −0.151 (98%) | −0.082 (99%) | −0.130 (99%) |
| dark_blob_adc | +0.024 (72%) | +0.090 (74%) | +0.045 (78%) | +0.098 (78%) |
| adc_pz_median | −0.042 (78%) | −0.055 (77%) | −0.024 (79%) | −0.046 (75%) |
| adc_tz_iqr | +0.156 (53%) | +0.174 (60%) | +0.026 (65%) | +0.025 (59%) |
| adc_low_fraction | +0.063 (53%) | +0.061 (54%) | +0.020 (58%) | −0.001 (58%) |

Values are mean weight (same-sign %).

### A3. Change 4: pre-trained models, fitted once on the 1341 training studies

| Feature | Lasso + PSAD (primary) | Lasso, imaging only | Ridge + PSAD | Ridge, imaging only | Single-feature AUC on PI-RADS 3 test |
|---|---|---|---|---|---|
| adc_low_fraction | +0.878 | +0.981 | +0.889 | +1.005 | 0.481 |
| adc_pz_median | −0.573 | −0.654 | −0.608 | −0.708 | 0.518 |
| t2w_tz_median | −0.540 | −0.534 | −0.544 | −0.541 | 0.312 |
| adc_tz_median | +0.413 | +0.508 | +0.432 | +0.547 | 0.433 |
| dark_blob_adc | −0.399 | −0.370 | −0.404 | −0.376 | 0.559 |
| psad | +0.320 | – | +0.322 | – | 0.618 |
| adc_tz_p10 | +0.188 | +0.131 | +0.183 | +0.121 | 0.452 |
| t2w_pz_median | −0.164 | −0.173 | −0.165 | −0.172 | 0.522 |
| dark_blob_t2w | −0.160 | −0.158 | −0.162 | −0.160 | 0.374 |
| dark_blob_ml | −0.116 | −0.163 | −0.124 | −0.177 | 0.450 |
| adc_pz_p10 | +0.070 | +0.103 | +0.093 | +0.135 | 0.588 |
| adc_tz_iqr | −0.060 | −0.067 | −0.071 | −0.084 | 0.469 |
| adc_pz_iqr | +0.003 | +0.074 | +0.023 | +0.098 | 0.440 |
