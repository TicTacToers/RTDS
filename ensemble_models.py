"""
ENSEMBLE MODELS: does bagging or boosting help?
===============================================

What this script does (in plain words):
  1. Uses the same patients, 12 region features + PSAD, and the same
     cross-validation as improved_models.py.
  2. Tries three ensembles. Their settings are fixed in advance (not tuned),
     because tuning on 18 csPCa cases would mostly tune to noise:
       - Bagged LASSO: 25 LASSO models, each trained on a bootstrap resample
         of the training patients, averaged (reduces variance). The resample
         is stratified (csPCa and non-csPCa drawn separately): a plain
         bootstrap of ~95 training patients can draw too few csPCa cases.
       - Random forest: bagged decision trees (as in compare_models.py)
       - Gradient boosting: 100 shallow trees (depth 2), each fixing the
         previous ones' errors, with a small learning rate (reduces bias)
  3. Compares them with PSAD and the plain Region LASSO + PSAD model.

How to run (after improved_models.py):
  python ensemble_models.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier

import compare_models as cm
import improved_models as im
import region_features as rf

warnings.filterwarnings("ignore", category=FutureWarning)


class StratifiedBagging(ClassifierMixin, BaseEstimator):
    """Averages n_estimators copies of `estimator`, each fitted on a bootstrap
    resample drawn separately within each class (so every resample keeps
    the same number of csPCa cases as the training data)."""

    def __init__(self, estimator=None, n_estimators=25, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y)
        groups = [np.flatnonzero(y == c) for c in self.classes_]
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
            self.estimators_.append(clone(self.estimator).fit(X.iloc[idx], y[idx]))
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def model_list():
    cols = rf.FEATURES + ["psad"]
    return {
        "Bagged LASSO + PSAD": (cols, StratifiedBagging(
            estimator=cm.lasso(), n_estimators=25, random_state=cm.SEED)),
        "Random forest + PSAD": (cols, cm.random_forest()),
        "Gradient boosting + PSAD": (cols, HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=100, min_samples_leaf=10,
            l2_regularization=1.0, class_weight="balanced", random_state=cm.SEED)),
    }


def main():
    data, _ = im.load_data()
    print("\nComparing ensembles (this takes several minutes)...")
    results = im.compare(data, model_list())

    earlier = pd.read_csv("results_improved.csv", index_col=0)
    keep = ["Baseline: PSAD rule", "Region LASSO + PSAD"]
    combined = pd.concat([earlier.loc[keep], results])
    combined.to_csv("results_ensemble.csv")

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print(combined[["AUC", "AUC_CI_low", "AUC_CI_high", "dAUC_vs_PSAD", "dAUC_CI_low",
                    "dAUC_CI_high", "AUC_20splits_mean", "AUC_20splits_min",
                    "AUC_20splits_max"]].astype(float).round(3).to_string())
    print("\nAt the 90%-sensitivity threshold learned on training folds:")
    print(combined[["sensitivity", "specificity", "NPV", "missed_csPCa",
                    "biopsies_avoided_%"]].astype(float).round(3).to_string())
    print("\nSaved results_ensemble.csv")


if __name__ == "__main__":
    main()
