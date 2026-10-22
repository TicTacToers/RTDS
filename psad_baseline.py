"""
PSAD BASELINE MODEL for PI-RADS 3 lesions  (version 2: two datasets)
=====================================================================

What this script does (in plain words):
  1. Loads a MAIN dataset (e.g. PI-CAI) and, optionally, an EXTERNAL dataset
     (e.g. AI4AR from another country).
  2. Renames each dataset's columns to the same standard names, so the rest
     of the script does not care where the data came from.
  3. Keeps only PI-RADS 3 patients (and, if you want, only biopsy-confirmed).
  4. Splits the MAIN dataset into TRAINING (70%) and TEST (30%).
  5. Builds the baselines the AI model must beat:
       A. Biopsy everyone
       B. PSAD with fixed cut-offs (0.10, 0.15, 0.20 ng/mL/cm3)
       C. PSAD cut-off chosen on TRAINING data to reach a safe sensitivity
       D. Clinical model (logistic regression on PSAD + age)
  6. Reports results on:
       - the internal TEST set (same hospitals as training)
       - the EXTERNAL dataset (different hospital/country) = external validation

How to run:
  - Test with fake data first:     python psad_baseline.py --demo
  - With real data: fill in the DATASET SETTINGS below, then:
                                   python psad_baseline.py

Packages needed:  pip install pandas numpy scikit-learn
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ===========================================================================
# DATASET SETTINGS - this is the only part you normally need to change
# ===========================================================================
# For each dataset, write the name of YOUR column on the right-hand side.
# Use None if the dataset does not have that column.
#
#   pirads     : PI-RADS score. Can be "3" or a list like "3,4" (one per lesion);
#                the script uses the HIGHEST score per patient.
#   psa        : PSA (ng/mL)
#   volume     : prostate volume (mL)
#   psad       : PSA density. If None, it is calculated as psa / volume.
#   age        : age in years
#   label      : csPCa yes/no (YES/NO, 1/0, True/False)
#   isup       : ISUP grade(s). Only used if label is None: csPCa = ISUP >= 2
#   histopath  : biopsy/surgery type. Used to keep only biopsy-confirmed cases.
#   center     : hospital name/code (optional, just reported)

MAIN_DATASET = {
    "name": "PI-CAI",
    "file": "picai_labels/clinical_information/marksheet.csv",
                                      # from github.com/DIAGNijmegen/picai_labels
    "columns": {
        "patient_id": "patient_id",   # used to match radiomics features
        "study_id": "study_id",       # (set to None if the dataset has no study id)
        "pirads": "lesion_PIRADS",
        "psa": "psa",
        "volume": "prostate_volume",
        "psad": "psad",
        "age": "patient_age",
        "label": "case_csPCa",        # CHECK: if this column is missing, set to None
        "isup": "case_ISUP",          # ...and the script will use ISUP instead
        "histopath": "histopath_type",
        "center": "center",
    },
}

# Set EXTERNAL_DATASET to the real AI4AR configuration when that data is
# available. Keeping it disabled prevents the PI-CAI-only pipeline from
# trying to load a placeholder file.
EXTERNAL_DATASET = None

ONLY_BIOPSY_CONFIRMED = True   # drop patients without a histopathology result
TARGET_SENSITIVITY = 0.90      # the "safe" sensitivity the group agrees on
FIXED_CUTOFFS = [0.10, 0.15, 0.20]
TEST_SIZE = 0.30               # share of MAIN patients kept aside for testing
RANDOM_SEED = 42               # makes the split the same every run


# ===========================================================================
# STEP 1: Load each dataset and give it standard column names
# ===========================================================================
def highest_number(value):
    """Turns '3,4' or '3' or 3.0 into the highest number (4, 3, 3).
    Returns NaN if there is no number (e.g. empty cell)."""
    if pd.isna(value):
        return np.nan
    numbers = []
    for part in str(value).replace(";", ",").split(","):
        try:
            numbers.append(float(part.strip()))
        except ValueError:
            pass  # skip text like "N/A"
    return max(numbers) if numbers else np.nan


def standardise(df, dataset):
    """Renames columns to standard names and cleans the data."""
    cols = dataset["columns"]
    name = dataset["name"]
    out = pd.DataFrame(index=df.index)

    # Patient identifiers (kept as text so "10001" and 10001 match)
    for id_col in ["patient_id", "study_id"]:
        real = cols.get(id_col)
        out[id_col] = df[real].astype(str) if real is not None else "none"

    # Check that every column we were told about really exists
    for std_name, real_name in cols.items():
        if real_name is not None and real_name not in df.columns:
            raise KeyError(
                f"[{name}] column '{real_name}' (for '{std_name}') not found.\n"
                f"Columns in the file are: {list(df.columns)}"
            )

    # PI-RADS: highest lesion score per patient
    if cols["pirads"] is not None:
        out["pirads"] = df[cols["pirads"]].apply(highest_number)
    else:
        out["pirads"] = 3  # assume the file is already PI-RADS 3 only

    # PSA density: use the column, and calculate psa / volume where it is empty
    # (PI-CAI leaves psad blank for ~400 patients that do have PSA and volume)
    out["psad"] = (pd.to_numeric(df[cols["psad"]], errors="coerce")
                   if cols["psad"] is not None else np.nan)
    if cols["psa"] is not None and cols["volume"] is not None:
        psa = pd.to_numeric(df[cols["psa"]], errors="coerce")
        vol = pd.to_numeric(df[cols["volume"]], errors="coerce")
        out["psad"] = out["psad"].fillna(psa / vol)

    out["age"] = (pd.to_numeric(df[cols["age"]], errors="coerce")
                  if cols["age"] is not None else np.nan)

    # Outcome: csPCa = 1, no csPCa = 0
    if cols["label"] is not None:
        text = df[cols["label"]].astype(str).str.strip().str.upper()
        out["label"] = text.isin(["1", "1.0", "YES", "TRUE"]).astype(int)
        out.loc[df[cols["label"]].isna(), "label"] = np.nan
    elif cols["isup"] is not None:
        isup = df[cols["isup"]].apply(highest_number)
        out["label"] = (isup >= 2).astype(float)
        out.loc[isup.isna(), "label"] = np.nan
    else:
        raise ValueError(f"[{name}] needs either a 'label' or an 'isup' column.")

    out["histopath"] = (df[cols["histopath"]]
                        if cols["histopath"] is not None else "unknown")
    out["center"] = df[cols["center"]] if cols["center"] is not None else name

    # --- Filtering, with a short report of what was removed ---
    print(f"[{name}] patients in file: {len(out)}")
    out = out[out["pirads"] == 3]
    print(f"[{name}] PI-RADS 3 (highest lesion = 3): {len(out)}")

    if ONLY_BIOPSY_CONFIRMED and cols["histopath"] is not None:
        has_histo = out["histopath"].notna() & (out["histopath"].astype(str).str.strip() != "")
        out = out[has_histo]
        print(f"[{name}] ...with biopsy/surgery result: {len(out)}")

    out = out.dropna(subset=["psad", "label"])
    out["label"] = out["label"].astype(int)
    print(f"[{name}] ...with PSAD and outcome available: {len(out)} "
          f"(csPCa: {out['label'].sum()}, {out['label'].mean():.0%})\n")
    return out


def load_dataset(dataset):
    df = pd.read_csv(dataset["file"])
    return standardise(df, dataset)


# ===========================================================================
# STEP 2: The clinical numbers we care about
# ===========================================================================
def clinical_metrics(y_true, biopsy):
    """y_true: 1 = has csPCa.  biopsy: 1 = rule says 'do a biopsy'."""
    y_true, biopsy = np.asarray(y_true), np.asarray(biopsy)
    tp = np.sum((biopsy == 1) & (y_true == 1))  # cancer found
    fn = np.sum((biopsy == 0) & (y_true == 1))  # cancer MISSED
    tn = np.sum((biopsy == 0) & (y_true == 0))  # biopsy correctly avoided
    fp = np.sum((biopsy == 1) & (y_true == 0))  # unnecessary biopsy

    def safe_div(a, b):
        return a / b if b > 0 else float("nan")

    return {
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "NPV": safe_div(tn, tn + fn),
        "biopsies_avoided_%": safe_div(tn + fn, len(y_true)) * 100,
        "missed_csPCa": int(fn),
        "biopsy_yield": safe_div(tp, tp + fp),
    }


def cutoff_for_sensitivity(scores, y_true, target):
    """Highest cut-off that still catches at least `target` of the cancers."""
    cancer_scores = np.sort(np.asarray(scores)[np.asarray(y_true) == 1])
    allowed_misses = int(np.floor(len(cancer_scores) * (1 - target)))
    return cancer_scores[allowed_misses]


# ===========================================================================
# STEP 3: Fit on TRAINING data, then evaluate on any dataset
# ===========================================================================
def fit_baselines(train):
    """Learns everything that needs learning, using TRAINING data only."""
    fitted = {}
    fitted["psad_cut"] = cutoff_for_sensitivity(
        train["psad"], train["label"], TARGET_SENSITIVITY)

    fitted["features"] = ["psad"] + (["age"] if train["age"].notna().all() else [])
    model = LogisticRegression()
    model.fit(train[fitted["features"]], train["label"])
    fitted["model"] = model
    train_prob = model.predict_proba(train[fitted["features"]])[:, 1]
    fitted["prob_cut"] = cutoff_for_sensitivity(
        train_prob, train["label"], TARGET_SENSITIVITY)
    return fitted


def evaluate(data, fitted, title):
    """Applies the fitted baselines to a dataset and prints a results table."""
    y = data["label"]
    results = {"A. Biopsy everyone": clinical_metrics(y, np.ones(len(data)))}

    for c in FIXED_CUTOFFS:
        results[f"B. PSAD >= {c:.2f}"] = clinical_metrics(y, data["psad"] >= c)

    c = fitted["psad_cut"]
    results[f"C. PSAD >= {c:.3f} (tuned)"] = clinical_metrics(y, data["psad"] >= c)

    feats = fitted["features"]
    usable = data[feats].notna().all(axis=1)
    prob = fitted["model"].predict_proba(data.loc[usable, feats])[:, 1]
    results[f"D. Clinical model ({' + '.join(feats)})"] = clinical_metrics(
        y[usable], prob >= fitted["prob_cut"])

    print("=" * 100)
    print(f"{title}: {len(data)} patients, {y.sum()} with csPCa")
    print("=" * 100)
    print(pd.DataFrame(results).T.round(3).to_string())
    print(f"\nAUC  PSAD alone: {roc_auc_score(y, data['psad']):.3f}   "
          f"Clinical model: {roc_auc_score(y[usable], prob):.3f}\n")


# ===========================================================================
# DEMO DATA (made up, only to check the script runs)
# ===========================================================================
def make_fake_data(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.20
    pirads = rng.choice(["3", "3,4", "2", "4,5", "3"], n)
    volume = rng.normal(50, 15, n).clip(15, 120)
    psa = np.where(label, rng.normal(9 + shift, 3, n),
                   rng.normal(6 + shift, 2.5, n)).clip(0.5, 30)
    return pd.DataFrame({
        "patient_id": [f"{seed}{i:05d}" for i in range(n)],
        "study_id": [f"{seed}{i:07d}" for i in range(n)],
        "lesion_PIRADS": pirads,
        "psa": psa.round(2),
        "prostate_volume": volume.round(1),
        "psad": (psa / volume).round(3),
        "patient_age": rng.normal(66, 7, n).round(),
        "case_csPCa": np.where(label, "YES", "NO"),
        "case_ISUP": np.where(label, 2, 0),
        "histopath_type": rng.choice(["MRBx", "SysBx+MRBx", None], n, p=[.5, .3, .2]),
        "center": rng.choice(["RUMC", "ZGT", "PCNN"], n),
    })


# ===========================================================================
# MAIN
# ===========================================================================
def main(demo):
    if demo:
        print("*** DEMO MODE: made-up data, the numbers mean nothing ***\n")
        main_data = standardise(make_fake_data(1500, 0), MAIN_DATASET)
        demo_external = dict(MAIN_DATASET, name="Fake external")
        external = standardise(make_fake_data(500, 1, shift=1.0), demo_external)
    else:
        main_data = load_dataset(MAIN_DATASET)
        external = load_dataset(EXTERNAL_DATASET) if EXTERNAL_DATASET else None

    train, test = train_test_split(
        main_data, test_size=TEST_SIZE, stratify=main_data["label"],
        random_state=RANDOM_SEED)
    print(f"Training: {len(train)} patients | Internal test: {len(test)} patients")
    print(f"Target sensitivity for tuned rules (C, D): {TARGET_SENSITIVITY:.0%}\n")

    fitted = fit_baselines(train)
    evaluate(test, fitted, f"INTERNAL TEST ({MAIN_DATASET['name']})")
    if external is not None:
        name = "Fake external" if demo else EXTERNAL_DATASET["name"]
        evaluate(external, fitted, f"EXTERNAL VALIDATION ({name})")

    print("Note: cut-offs C and D were chosen on TRAINING data only, then applied")
    print("unchanged to the test and external data. If sensitivity drops below the")
    print("target on external data, that is a real finding worth reporting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run with fake data")
    main(parser.parse_args().demo)
