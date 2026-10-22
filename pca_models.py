"""
PCA MODELS: does removing feature correlation help?
===================================================

What this script does (in plain words):
  1. Uses the same patients, region features and cross-validation as
     improved_models.py.
  2. Before the logistic regression, the 12 region features are turned into
     uncorrelated "principal components" (PCA). PSAD is NOT mixed into the
     components; it stays a separate input.
  3. How many components to keep (2-8) and how strongly to shrink the weights
     are both chosen by an inner cross-validation on the TRAINING patients
     only. PCA itself is also learned on training patients only.
  4. Compares the PCA models with the earlier results (baseline + region
     LASSO/RIDGE), and shows what was learned: the model's weights are
     translated back from components to the original 12 features.

How to run (after region_features.py):
  python pca_models.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import compare_models as cm
import improved_models as im
import region_features as rf

warnings.filterwarnings("ignore", category=FutureWarning)

N_COMPONENTS = [2, 3, 4, 6, 8]
C_VALUES = np.logspace(-3, 2, 6)


def pca_model(penalty, with_psad):
    """Scale -> PCA on the region features (+ PSAD passed alongside) -> logistic.
    Number of components and penalty strength are tuned by inner 3-fold CV."""
    region = Pipeline([("scale", StandardScaler()), ("pca", PCA(random_state=cm.SEED))])
    parts = [("region", region, rf.FEATURES)]
    if with_psad:
        parts.append(("psad", StandardScaler(), ["psad"]))
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    pipe = Pipeline([
        ("prep", ColumnTransformer(parts)),
        ("model", LogisticRegression(penalty=penalty, solver=solver, max_iter=5000,
                                     random_state=cm.SEED)),
    ])
    grid = {"prep__region__pca__n_components": N_COMPONENTS, "model__C": C_VALUES}
    inner = StratifiedKFold(3, shuffle=True, random_state=cm.SEED)
    return GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc")


def model_list():
    region_psad = rf.FEATURES + ["psad"]
    return {
        "PCA + RIDGE (12)": (rf.FEATURES, pca_model("l2", False)),
        "PCA + LASSO (12)": (rf.FEATURES, pca_model("l1", False)),
        "PCA + RIDGE + PSAD": (region_psad, pca_model("l2", True)),
        "PCA + LASSO + PSAD": (region_psad, pca_model("l1", True)),
    }


def weights_on_features(fitted, with_psad):
    """Translates the component weights back to the original (standardised)
    features: weight_feature = sum over components of loading x weight."""
    best = fitted.best_estimator_
    pca = best.named_steps["prep"].named_transformers_["region"].named_steps["pca"]
    coef = best.named_steps["model"].coef_[0]
    n = pca.n_components_
    weights = pd.Series(pca.components_.T @ coef[:n], index=rf.FEATURES)
    if with_psad:
        weights["psad"] = coef[n]
    return weights, n


def what_was_learned(data, models):
    y = data["label"].to_numpy()
    tables = {}
    for name, (cols, model) in models.items():
        weights, chosen = [], []
        for seed in range(im.N_REPEATS):
            for train_idx, _ in StratifiedKFold(cm.N_FOLDS, shuffle=True,
                                                random_state=seed).split(data, y):
                fitted = clone(model).fit(data.iloc[train_idx][cols], y[train_idx])
                w, n = weights_on_features(fitted, "PSAD" in name)
                weights.append(w)
                chosen.append(n)
        weights = pd.DataFrame(weights)
        positive = (weights > 0).sum()
        nonzero = (weights != 0).sum()
        tables[name] = pd.DataFrame({
            "mean_weight": weights.mean(),
            "same_sign_%": np.maximum(positive, nonzero - positive)
                           / nonzero.replace(0, np.nan) * 100,
        }).sort_values("mean_weight", key=abs, ascending=False)
        counts = pd.Series(chosen).value_counts().sort_index()
        print(f"  {name}: components chosen over 100 refits -> "
              + ", ".join(f"{k}: {v}x" for k, v in counts.items()))
    return tables


def main():
    data, whole_cols = im.load_data()
    models = model_list()
    print("\nComparing PCA models (this takes several minutes)...")
    results = im.compare(data, models)

    earlier = pd.read_csv("results_improved.csv", index_col=0)
    combined = pd.concat([earlier, results])
    combined.to_csv("results_pca.csv")

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("AUC with bootstrap 95% CI, versus PSAD, and over 20 splits (earlier models included)")
    print("=" * 100)
    print(combined[["AUC", "AUC_CI_low", "AUC_CI_high", "dAUC_vs_PSAD", "dAUC_CI_low",
                    "dAUC_CI_high", "AUC_20splits_mean", "AUC_20splits_min",
                    "AUC_20splits_max"]].astype(float).round(3).to_string())
    print("\nAt the 90%-sensitivity threshold learned on training folds:")
    print(combined[["sensitivity", "specificity", "NPV", "missed_csPCa",
                    "biopsies_avoided_%"]].astype(float).round(3).to_string())

    print("\n" + "=" * 100)
    print("What the PCA models learned (weights translated back to the original features)")
    print("=" * 100)
    learned = what_was_learned(data, models)
    for name, table in learned.items():
        print(f"\n{name}:")
        print(table.round(3).to_string())
    pd.concat(learned, names=["model", "feature"]).to_csv("coefficients_pca.csv")
    print("\nSaved results_pca.csv and coefficients_pca.csv")


if __name__ == "__main__":
    main()
