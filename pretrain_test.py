"""
PRE-TRAIN, THEN TEST: learn from the rest of PI-CAI, test once on PI-RADS 3
===========================================================================

What this script does (in plain words):
  1. TRAINING set: every PI-CAI study whose highest PI-RADS is NOT 3
     (PI-RADS 1-2/none and 4-5). Studies of patients who also have a
     PI-RADS 3 study are left out, so no patient is on both sides.
  2. TEST set: the 149 PI-RADS 3 patients from psad_baseline.py. They are
     never used for training, tuning or choosing a threshold.
  3. Models are fixed here, BEFORE the test set is scored:
       primary   - LASSO on the 12 region features + PSAD
       secondary - LASSO on region features only, RIDGE versions of both
       reference - PSAD alone
     The 90%-sensitivity threshold of every model (PSAD included) is learned
     on the training set, then applied unchanged to the test set.
  4. Reports test AUC with a bootstrap 95% CI, the AUC difference versus PSAD
     (paired bootstrap), the clinical numbers at the learned threshold, and
     the pre-trained model's weights.

Caveats worth reporting:
  - PI-RADS 4-5 lesions are obvious and PI-RADS 3 lesions are equivocal by
    definition, so what is learned may not transfer.
  - Most PI-RADS 1-2 negatives were confirmed by MRI only (no biopsy).

How to run (after region_features.py):
  python pretrain_test.py
"""

import contextlib
import io
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import compare_models as cm
import improved_models as im
import psad_baseline as pb
import region_features as rf

warnings.filterwarnings("ignore", category=FutureWarning)

TRAIN_FEATURES = "region_features_pretrain.csv"
N_BOOTSTRAP = 2000


def penalised(penalty):
    """Same as cm.lasso() / im.ridge(), with a fixed seed so runs repeat exactly."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegressionCV(Cs=10, cv=3, penalty=penalty, solver="liblinear",
                                       scoring="roc_auc", max_iter=5000,
                                       random_state=cm.SEED)),
    ])


# Fixed before the test set is scored. The first model is the primary one.
MODELS = {
    "Pre-trained LASSO + PSAD (primary)": (rf.FEATURES + ["psad"], penalised("l1")),
    "Pre-trained LASSO, imaging only": (rf.FEATURES, penalised("l1")),
    "Pre-trained RIDGE + PSAD": (rf.FEATURES + ["psad"], penalised("l2")),
    "Pre-trained RIDGE, imaging only": (rf.FEATURES, penalised("l2")),
    "PSAD alone (reference)": (["psad"], None),
}


def training_studies(test_patient_ids):
    ms = pd.read_csv(pb.MAIN_DATASET["file"], dtype={"patient_id": str, "study_id": str})
    ms["pirads_max"] = ms["lesion_PIRADS"].apply(pb.highest_number)
    pirads3_patients = set(ms.loc[ms["pirads_max"] == 3, "patient_id"]) | test_patient_ids
    train = ms[(ms["pirads_max"] != 3) & ~ms["patient_id"].isin(pirads3_patients)].copy()
    print(f"Training studies (PI-RADS not 3): {(ms['pirads_max'] != 3).sum()}, "
          f"after removing patients who also have a PI-RADS 3 study: {len(train)}")
    train["label"] = train["case_csPCa"].astype(str).str.upper().eq("YES").astype(int)
    psa = pd.to_numeric(train["psa"], errors="coerce")
    vol = pd.to_numeric(train["prostate_volume"], errors="coerce")
    train["psad"] = pd.to_numeric(train["psad"], errors="coerce").fillna(psa / vol)
    train["pirads_group"] = np.where(train["pirads_max"] >= 4, "PI-RADS 4-5",
                                     "PI-RADS 1-2/none")
    return train


def load_sets():
    with contextlib.redirect_stdout(io.StringIO()):
        test = pb.load_dataset(pb.MAIN_DATASET)
    ids = {"patient_id": str, "study_id": str}
    test = test.merge(pd.read_csv(rf.OUTPUT, dtype=ids), on=["patient_id", "study_id"])

    train = training_studies(set(test["patient_id"]))
    if not os.path.exists(TRAIN_FEATURES):
        print(f"Extracting region features for {len(train)} training studies...")
        rf.extract_all(train, output=TRAIN_FEATURES)
    train = train.merge(pd.read_csv(TRAIN_FEATURES, dtype=ids), on=["patient_id", "study_id"])

    overlap = set(train["patient_id"]) & set(test["patient_id"])
    assert not overlap, f"patients on both sides: {overlap}"
    print(f"\nTRAINING: {len(train)} studies with features, csPCa {train['label'].sum()} "
          f"({train['label'].mean():.0%}); PSAD missing for {train['psad'].isna().sum()}")
    print(train.groupby("pirads_group")["label"].agg(studies="size", csPCa="sum").to_string())
    print(f"TEST:     {len(test)} PI-RADS 3 patients, csPCa {test['label'].sum()}")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(train, test):
    y = test["label"].to_numpy()
    rng = np.random.default_rng(0)
    boots = [i for i in (rng.integers(0, len(y), len(y)) for _ in range(2 * N_BOOTSTRAP))
             if 0 < y[i].sum() < len(i)][:N_BOOTSTRAP]
    psad = test["psad"].to_numpy()

    rows, fitted_models = {}, {}
    for name, (cols, model) in MODELS.items():
        tr = train.dropna(subset=cols)
        # Sanity check: does the model learn anything within the training set?
        train_cv_auc = roc_auc_score(tr["label"], im.out_of_fold(tr, cols, model, cm.SEED)[0])
        fitted, cut = cm.fit(model, tr[cols], tr["label"].to_numpy())
        prob = cm.score(fitted, test[cols])
        row = cm.summarise(y, prob, (prob >= cut).astype(int))
        auc_boot = [roc_auc_score(y[i], prob[i]) for i in boots]
        diff_boot = [a - roc_auc_score(y[i], psad[i]) for a, i in zip(auc_boot, boots)]
        row.update({
            "train_n": len(tr), "train_CV_AUC": train_cv_auc,
            "AUC_CI_low": np.percentile(auc_boot, 2.5),
            "AUC_CI_high": np.percentile(auc_boot, 97.5),
            "dAUC_vs_PSAD": row["AUC"] - roc_auc_score(y, psad),
            "dAUC_CI_low": np.percentile(diff_boot, 2.5),
            "dAUC_CI_high": np.percentile(diff_boot, 97.5),
        })
        rows[name] = row
        fitted_models[name] = (cols, fitted)
        print(f"  done: {name}")
    return pd.DataFrame(rows).T, fitted_models


def weights(fitted_models, test):
    y = test["label"]
    tables = {}
    for name, (cols, fitted) in fitted_models.items():
        if fitted is None:
            continue
        coef = pd.Series(fitted.named_steps["model"].coef_[0], index=cols)
        tables[name] = pd.DataFrame({
            "weight": coef,
            "single_feature_AUC_test": [roc_auc_score(y, test[c]) for c in cols],
        }).sort_values("weight", key=abs, ascending=False)
    return tables


def main():
    train, test = load_sets()
    print("\nTraining on the non-PI-RADS-3 studies, testing once on PI-RADS 3...")
    results, fitted_models = evaluate(train, test)
    results.to_csv("results_pretrain_test.csv")

    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("TEST on 149 PI-RADS 3 patients (never used for training)")
    print("=" * 100)
    print(results[["train_n", "train_CV_AUC", "AUC", "AUC_CI_low", "AUC_CI_high",
                   "dAUC_vs_PSAD", "dAUC_CI_low", "dAUC_CI_high"]].astype(float)
          .round(3).to_string())
    print("\nAt the 90%-sensitivity threshold learned on the TRAINING set:")
    print(results[["sensitivity", "specificity", "NPV", "missed_csPCa",
                   "biopsies_avoided_%"]].astype(float).round(3).to_string())

    print("\n" + "=" * 100)
    print("What the pre-trained models learned (standardised weights;")
    print("positive = higher value -> more likely csPCa)")
    print("=" * 100)
    tables = weights(fitted_models, test)
    for name, table in tables.items():
        print(f"\n{name}:")
        print(table.round(3).to_string())
    pd.concat(tables, names=["model", "feature"]).to_csv("coefficients_pretrain_test.csv")
    print("\nSaved results_pretrain_test.csv and coefficients_pretrain_test.csv")


if __name__ == "__main__":
    main()
