"""
MODEL COMPARISON: does MRI radiomics beat PSA density for PI-RADS 3?
====================================================================

What this script does (in plain words):
  1. Loads the clinical data (via psad_baseline.py, same filters) and the
     radiomics table (from extract_radiomics.py), and joins them per patient.
  2. Compares five ways of deciding "biopsy or not":
       1. PSAD rule              - the baseline to beat
       2. Clinical model         - logistic regression on PSAD + age
       3. Radiomics only         - LASSO logistic regression on MRI features
       4. Radiomics + PSAD       - LASSO logistic regression on both
       5. Random forest          - radiomics + PSAD, a non-linear alternative
  3. Every rule is tuned to the SAME safety level (e.g. 90% sensitivity),
     then we compare how many biopsies each one avoids.
  4. Internal results: 5-fold cross-validation on PI-CAI
     (every patient is tested once, by a model that never saw them).
  5. External results: train on ALL of PI-CAI, test on AI4AR.
  6. Saves the result tables as CSV files for the report and poster.

Important rules followed inside the code (good practice, worth mentioning
in the methods section):
  - The cut-off for 90% sensitivity is chosen on TRAINING data only.
  - LASSO picks features on TRAINING data only (never on the test patients).
  - The scaling of features is learned on TRAINING data only.

How to run:
  - Test with fake data:   python compare_models.py --demo
  - Real data:             python compare_models.py
    (needs radiomics_picai.csv, and radiomics_ai4ar.csv for external testing)

Packages needed:  pip install pandas numpy scikit-learn
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import psad_baseline as pb

# ===========================================================================
# SETTINGS
# ===========================================================================
RADIOMICS_MAIN = "radiomics_picai.csv"
RADIOMICS_EXTERNAL = "radiomics_ai4ar.csv"   # set to None if not available yet
N_FOLDS = 5
TARGET_SENSITIVITY = pb.TARGET_SENSITIVITY   # same as the baseline script
SEED = pb.RANDOM_SEED


# ===========================================================================
# The models
# ===========================================================================
def lasso():
    """Logistic regression that keeps only the most useful features.
    It tries 10 strengths of 'feature pruning' and picks the best one
    using an inner 3-fold cross-validation on the training data."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegressionCV(Cs=10, cv=3, penalty="l1", solver="liblinear",
                                       scoring="roc_auc", max_iter=5000)),
    ])


def plain_logistic():
    return Pipeline([("scale", StandardScaler()), ("model", LogisticRegression())])


def random_forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  class_weight="balanced", random_state=SEED)


def model_list(radiomic_cols, clinical_cols):
    """name -> (columns used, model). None = use PSAD directly as the score."""
    return {
        "1. PSAD rule": (["psad"], None),
        f"2. Clinical ({' + '.join(clinical_cols)})": (clinical_cols, plain_logistic()),
        "3. Radiomics only": (radiomic_cols, lasso()),
        "4. Radiomics + PSAD": (radiomic_cols + ["psad"], lasso()),
        "5. Random forest (radiomics + PSAD)": (radiomic_cols + ["psad"], random_forest()),
    }


# ===========================================================================
# Fit a model + choose its safe cut-off, using TRAINING data only
# ===========================================================================
def fit(model, X, y):
    if model is None:   # PSAD rule: nothing to learn except the cut-off
        return None, pb.cutoff_for_sensitivity(X["psad"], y, TARGET_SENSITIVITY)

    # Honest training scores: each training patient is scored by a model
    # that did not see them (inner cross-validation). Using normal training
    # scores would make the cut-off too optimistic.
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    honest = cross_val_predict(clone(model), X, y, cv=inner, method="predict_proba")[:, 1]
    cut = pb.cutoff_for_sensitivity(honest, y, TARGET_SENSITIVITY)
    fitted = clone(model).fit(X, y)
    return fitted, cut


def score(fitted, X):
    return X["psad"].to_numpy() if fitted is None else fitted.predict_proba(X)[:, 1]


def summarise(y, prob, biopsy):
    metrics = pb.clinical_metrics(y, biopsy)
    metrics["AUC"] = roc_auc_score(y, prob)
    # Matched comparison: models rarely land at EXACTLY the target sensitivity
    # on new patients, which makes 'biopsies_avoided_%' unfair to compare.
    # Here the cut-off is set on the evaluation patients themselves so every
    # model sits at the same sensitivity. Slightly optimistic, so use it to
    # COMPARE models, and report the columns above as the honest result.
    cut = pb.cutoff_for_sensitivity(prob, y, TARGET_SENSITIVITY)
    metrics["avoided_%_at_matched_sens"] = np.mean(prob < cut) * 100
    return metrics


# ===========================================================================
# Internal validation: k-fold cross-validation on the main dataset
# ===========================================================================
def cross_validate(data, models):
    y = data["label"].to_numpy()
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    results = {}
    for name, (cols, model) in models.items():
        prob = np.zeros(len(data))
        biopsy = np.zeros(len(data), dtype=int)
        for train_idx, test_idx in folds.split(data, y):
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            fitted, cut = fit(model, train[cols], y[train_idx])
            prob[test_idx] = score(fitted, test[cols])
            biopsy[test_idx] = prob[test_idx] >= cut
        results[name] = summarise(y, prob, biopsy)
        print(f"  done: {name}")
    return pd.DataFrame(results).T


# ===========================================================================
# External validation: train on all of main, test on external
# ===========================================================================
def external_validate(main, external, models):
    results, fitted_models = {}, {}
    for name, (cols, model) in models.items():
        fitted, cut = fit(model, main[cols], main["label"].to_numpy())
        prob = score(fitted, external[cols])
        results[name] = summarise(external["label"].to_numpy(), prob, prob >= cut)
        fitted_models[name] = fitted
    return pd.DataFrame(results).T, fitted_models


def show_selected_features(fitted_models, radiomic_cols):
    """Which radiomics features did LASSO keep? (useful for the report)"""
    for name, fitted in fitted_models.items():
        if fitted is None or not hasattr(fitted, "named_steps"):
            continue
        coefs = fitted.named_steps["model"].coef_[0]
        cols = radiomic_cols + (["psad"] if "PSAD" in name else [])
        if len(coefs) != len(cols):
            continue
        kept = pd.Series(coefs, index=cols)
        kept = kept[kept != 0].sort_values(key=abs, ascending=False)
        print(f"\n{name}: LASSO kept {len(kept)} of {len(cols)} features. Top 5:")
        print(kept.head(5).round(3).to_string())


# ===========================================================================
# Loading and joining the data
# ===========================================================================
def join(clinical, radiomics, name):
    radiomics = radiomics.astype({"patient_id": str, "study_id": str})
    if "study_id" not in radiomics or radiomics["study_id"].eq("none").all():
        keys = ["patient_id"]
    else:
        keys = ["patient_id", "study_id"]
    merged = clinical.merge(radiomics, on=keys, how="inner")
    print(f"[{name}] patients with clinical data AND radiomics: {len(merged)} "
          f"(of {len(clinical)}; csPCa: {merged['label'].sum()})")
    return merged.reset_index(drop=True)


def fake_radiomics(clinical, seed, scanner_shift=0.0):
    """Made-up radiomics: 84 features, of which 4 carry a weak cancer signal.
    scanner_shift mimics a different hospital/scanner."""
    rng = np.random.default_rng(seed)
    n, y = len(clinical), clinical["label"].to_numpy()
    feats = rng.normal(scanner_shift, 1, (n, 84))
    feats[:, :4] -= 0.7 * y[:, None]      # e.g. lower ADC with cancer
    cols = [f"adc_feat{i}" for i in range(42)] + [f"t2w_feat{i}" for i in range(42)]
    df = pd.DataFrame(feats, columns=cols)
    df.insert(0, "patient_id", clinical["patient_id"].to_numpy())
    df.insert(1, "study_id", clinical["study_id"].to_numpy())
    return df


# ===========================================================================
# MAIN
# ===========================================================================
def main(demo):
    if demo:
        print("*** DEMO MODE: made-up data, the numbers mean nothing ***\n")
        clin_main = pb.standardise(pb.make_fake_data(1500, 0), pb.MAIN_DATASET)
        ext_cfg = dict(pb.MAIN_DATASET, name="Fake external")
        clin_ext = pb.standardise(pb.make_fake_data(500, 1, shift=1.0), ext_cfg)
        rad_main = fake_radiomics(clin_main, 0)
        rad_ext = fake_radiomics(clin_ext, 1, scanner_shift=0.3)
        ext_name = "Fake external"
    else:
        clin_main = pb.load_dataset(pb.MAIN_DATASET)
        rad_main = pd.read_csv(RADIOMICS_MAIN)
        has_ext = pb.EXTERNAL_DATASET is not None and RADIOMICS_EXTERNAL is not None
        clin_ext = pb.load_dataset(pb.EXTERNAL_DATASET) if has_ext else None
        rad_ext = pd.read_csv(RADIOMICS_EXTERNAL) if has_ext else None
        ext_name = pb.EXTERNAL_DATASET["name"] if has_ext else None

    main_data = join(clin_main, rad_main, pb.MAIN_DATASET["name"])
    ext_data = join(clin_ext, rad_ext, ext_name) if clin_ext is not None else None

    radiomic_cols = [c for c in rad_main.columns if c.startswith(("t2w_", "adc_"))]
    if ext_data is not None:   # only use features present in both datasets
        radiomic_cols = [c for c in radiomic_cols if c in ext_data.columns]

    # Age only if every patient has it (in both datasets)
    has_age = main_data["age"].notna().all() and (
        ext_data is None or ext_data["age"].notna().all())
    clinical_cols = ["psad", "age"] if has_age else ["psad"]
    models = model_list(radiomic_cols, clinical_cols)

    pd.set_option("display.width", 160)
    print(f"\nTarget sensitivity for all models: {TARGET_SENSITIVITY:.0%}")
    print(f"Radiomics features available: {len(radiomic_cols)}\n")

    print(f"INTERNAL: {N_FOLDS}-fold cross-validation on {pb.MAIN_DATASET['name']}")
    internal = cross_validate(main_data, models)
    print("=" * 110)
    print(internal.round(3).to_string())
    internal.to_csv("results_internal.csv")

    if ext_data is not None:
        print(f"\nEXTERNAL: train on all {pb.MAIN_DATASET['name']}, test on {ext_name}")
        external, fitted = external_validate(main_data, ext_data, models)
        print("=" * 110)
        print(external.round(3).to_string())
        external.to_csv("results_external.csv")
        show_selected_features(fitted, radiomic_cols)

    print("\nSaved: results_internal.csv" + (", results_external.csv" if ext_data is not None else ""))
    print("How to read it:")
    print(f"  - 'avoided_%_at_matched_sens': biopsies avoided with every model at exactly")
    print(f"    {TARGET_SENSITIVITY:.0%} sensitivity -> the fair head-to-head comparison.")
    print("  - The other columns: what happens with the cut-off chosen on training data")
    print("    -> the honest real-world result. If sensitivity drops well below the")
    print("    target here, the cut-off does not transfer well: report that.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run with fake data")
    main(parser.parse_args().demo)
