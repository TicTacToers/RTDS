"""
IMPROVED MODELS: small label-free feature set, LASSO vs RIDGE
=============================================================

What this script does (in plain words):
  1. Loads the PI-RADS 3 cohort, the 12 zonal "region features"
     (region_features.py) and the old 84 whole-gland features (baseline).
  2. Compares, with the SAME 5-fold cross-validation as compare_models.py:
       - the baseline: PSAD rule, and LASSO on the 84 whole-gland features
       - LASSO and RIDGE on the 12 region features, with and without PSAD
     LASSO (L1) may switch features off completely; RIDGE (L2) keeps every
     feature but shrinks its weight. Both pick their strength by an inner
     cross-validation on the training patients only.
  3. Reports AUC with a bootstrap 95% CI, the AUC difference versus PSAD
     (paired bootstrap), and how stable the AUC is over 20 different splits.
  4. Shows what each model learned: the weight per feature (features are
     standardised, so weights are comparable), how often LASSO kept it, and
     how often the sign was the same, over 100 refits (20 splits x 5 folds).

How to run (after region_features.py):
  python improved_models.py
"""

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import compare_models as cm
import psad_baseline as pb
import region_features as rf

warnings.filterwarnings("ignore", category=FutureWarning)

N_REPEATS = 20        # extra random splits to check how stable the AUC is
N_BOOTSTRAP = 2000    # resamples for the confidence intervals


def ridge():
    """Like cm.lasso(), but L2: every feature keeps a (shrunken) weight."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegressionCV(Cs=10, cv=3, penalty="l2", solver="liblinear",
                                       scoring="roc_auc", max_iter=5000)),
    ])


def load_data():
    with contextlib.redirect_stdout(io.StringIO()):
        clinical = pb.load_dataset(pb.MAIN_DATASET)
    ids = {"patient_id": str, "study_id": str}
    whole = pd.read_csv(cm.RADIOMICS_MAIN, dtype=ids)
    region = pd.read_csv(rf.OUTPUT, dtype=ids)
    data = clinical.merge(whole, on=["patient_id", "study_id"]).merge(
        region, on=["patient_id", "study_id"])
    whole_cols = [c for c in whole.columns if c.startswith(("t2w_", "adc_"))]
    print(f"Patients: {len(data)} (csPCa: {data['label'].sum()})")
    return data.reset_index(drop=True), whole_cols


def model_list(whole_cols):
    region = rf.FEATURES
    return {
        "Baseline: PSAD rule": (["psad"], None),
        "Baseline: whole-gland LASSO (84)": (whole_cols, cm.lasso()),
        "Region LASSO (12)": (region, cm.lasso()),
        "Region RIDGE (12)": (region, ridge()),
        "Region LASSO + PSAD": (region + ["psad"], cm.lasso()),
        "Region RIDGE + PSAD": (region + ["psad"], ridge()),
    }


def out_of_fold(data, cols, model, seed):
    """Every patient scored by a model that never saw them (as cm.cross_validate)."""
    y = data["label"].to_numpy()
    prob = np.zeros(len(data))
    biopsy = np.zeros(len(data), dtype=int)
    for train_idx, test_idx in StratifiedKFold(cm.N_FOLDS, shuffle=True,
                                               random_state=seed).split(data, y):
        fitted, cut = cm.fit(model, data.iloc[train_idx][cols], y[train_idx])
        prob[test_idx] = cm.score(fitted, data.iloc[test_idx][cols])
        biopsy[test_idx] = prob[test_idx] >= cut
    return prob, biopsy


def compare(data, models):
    y = data["label"].to_numpy()
    rng = np.random.default_rng(0)
    boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(2 * N_BOOTSTRAP))
             if 0 < y[i].sum() < len(i)][:N_BOOTSTRAP]
    psad_prob = data["psad"].to_numpy()

    rows = {}
    for name, (cols, model) in models.items():
        prob, biopsy = out_of_fold(data, cols, model, cm.SEED)
        row = cm.summarise(y, prob, biopsy)
        auc_boot = [roc_auc_score(y[i], prob[i]) for i in boots]
        diff_boot = [a - roc_auc_score(y[i], psad_prob[i]) for a, i in zip(auc_boot, boots)]
        repeats = [roc_auc_score(y, out_of_fold(data, cols, model, s)[0])
                   for s in range(N_REPEATS)]
        row.update({
            "AUC_CI_low": np.percentile(auc_boot, 2.5),
            "AUC_CI_high": np.percentile(auc_boot, 97.5),
            "dAUC_vs_PSAD": row["AUC"] - roc_auc_score(y, psad_prob),
            "dAUC_CI_low": np.percentile(diff_boot, 2.5),
            "dAUC_CI_high": np.percentile(diff_boot, 97.5),
            "AUC_20splits_mean": np.mean(repeats),
            "AUC_20splits_min": np.min(repeats),
            "AUC_20splits_max": np.max(repeats),
        })
        rows[name] = row
        print(f"  done: {name}")
    return pd.DataFrame(rows).T


def what_was_learned(data, models):
    """Weights of the region LASSO/RIDGE models over 100 training sets."""
    y = data["label"].to_numpy()
    univariate = {c: roc_auc_score(y, data[c]) for c in rf.FEATURES + ["psad"]}
    tables = {}
    for name, (cols, model) in models.items():
        if not name.startswith("Region"):
            continue
        coefs = []
        for seed in range(N_REPEATS):
            for train_idx, _ in StratifiedKFold(cm.N_FOLDS, shuffle=True,
                                                random_state=seed).split(data, y):
                fitted = clone(model).fit(data.iloc[train_idx][cols], y[train_idx])
                coefs.append(fitted.named_steps["model"].coef_[0])
        coefs = pd.DataFrame(coefs, columns=cols)
        median_sign = np.sign(coefs.median())
        tables[name] = pd.DataFrame({
            "mean_weight": coefs.mean(),
            "kept_%": (coefs != 0).mean() * 100,
            "same_sign_%": (np.sign(coefs).eq(median_sign) & (coefs != 0)).mean() * 100,
            "single_feature_AUC": pd.Series(univariate)[cols],
        }).sort_values("mean_weight", key=abs, ascending=False)
    return tables


def main():
    data, whole_cols = load_data()
    models = model_list(whole_cols)
    print("\nComparing models (this takes a few minutes)...")
    results = compare(data, models)
    results.to_csv("results_improved.csv")

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("AUC (5-fold CV, seed as baseline) with bootstrap 95% CI, and versus PSAD")
    print("=" * 100)
    print(results[["AUC", "AUC_CI_low", "AUC_CI_high", "dAUC_vs_PSAD", "dAUC_CI_low",
                   "dAUC_CI_high", "AUC_20splits_mean", "AUC_20splits_min",
                   "AUC_20splits_max"]].astype(float).round(3).to_string())
    print("\nAt the 90%-sensitivity threshold learned on training folds:")
    print(results[["sensitivity", "specificity", "NPV", "missed_csPCa",
                   "biopsies_avoided_%"]].astype(float).round(3).to_string())

    print("\n" + "=" * 100)
    print("What the models learned (standardised weights over 100 refits;")
    print("positive = higher value -> more likely csPCa)")
    print("=" * 100)
    learned = what_was_learned(data, models)
    for name, table in learned.items():
        print(f"\n{name}:")
        print(table.round(3).to_string())
    pd.concat(learned, names=["model", "feature"]).to_csv("coefficients_improved.csv")
    print("\nSaved results_improved.csv and coefficients_improved.csv")


if __name__ == "__main__":
    main()
