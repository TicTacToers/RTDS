"""
RADIOMICS EXTRACTION (whole prostate gland)
===========================================

What this script does (in plain words):
  1. Gets the list of PI-RADS 3 patients from psad_baseline.py
     (same filters as the baseline, so everyone works on the same patients).
  2. For each patient, opens the T2W and ADC MRI images and the WHOLE-GLAND
     mask (an outline of the prostate).
  3. Lines up the mask with each image (the ADC image has a different
     resolution, so the mask is "resampled" to fit it).
  4. Uses PyRadiomics to measure ~100 numbers per image inside the prostate:
     brightness statistics (e.g. mean ADC) and texture (how uniform/patchy).
  5. Saves one row per patient to a CSV file:
        patient_id, study_id, t2w_..., adc_...

Why the WHOLE GLAND and not the lesion?
  In PI-CAI only cancer (csPCa) lesions are outlined. Benign PI-RADS 3 lesions
  have no outline. Using lesion outlines would let the model "see" the answer
  (data leakage). The whole-gland mask exists for every patient, so every
  patient is measured the same way.

How to run:
  - Test with fake images first:   python extract_radiomics.py --demo
  - PI-CAI:                        python extract_radiomics.py --dataset main
  - AI4AR (external):              python extract_radiomics.py --dataset external

Packages needed:
  pip install SimpleITK
  pip install git+https://github.com/AIM-Harvard/pyradiomics.git
  (the normal "pip install pyradiomics" fails on new Python versions;
   installing from GitHub works, tested on Python 3.13)
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

import psad_baseline as pb

# ===========================================================================
# PATH SETTINGS - where the images and masks are on YOUR computer
# ===========================================================================
# {patient_id} and {study_id} are filled in automatically for each patient.
PATHS = {
    "main": {   # PI-CAI
        "dataset": pb.MAIN_DATASET,
        "images": {
            "t2w": "picai_public_images_fold/{patient_id}/{patient_id}_{study_id}_t2w.mha",
            "adc": "picai_public_images_fold/{patient_id}/{patient_id}_{study_id}_adc.mha",
        },
        "mask": "picai_labels/anatomical_delineations/whole_gland/AI/Bosma22b/"
                "{patient_id}_{study_id}.nii.gz",
        "output": "radiomics_picai.csv",
    },
    "external": {   # AI4AR - CHANGE these patterns after downloading
        "dataset": pb.EXTERNAL_DATASET,
        "images": {
            "t2w": "ai4ar/{patient_id}/t2w.nii.gz",
            "adc": "ai4ar/{patient_id}/adc.nii.gz",
        },
        "mask": "ai4ar/{patient_id}/prostate_gland.nii.gz",
        "output": "radiomics_ai4ar.csv",
    },
}

# ===========================================================================
# RADIOMICS SETTINGS - keep identical for both datasets!
# ===========================================================================
# T2W intensities have no fixed unit (they differ between scanners), so we
# normalise them. ADC values are physical measurements, so we keep them as-is.
T2W_SETTINGS = {"normalize": True, "normalizeScale": 100, "binWidth": 5}
ADC_SETTINGS = {"normalize": False, "binWidth": 50}
FEATURE_CLASSES = ["firstorder", "glcm"]   # brightness + texture
# Shape features describe the outline itself (= prostate size). We already
# have prostate volume in PSAD, so shape is left out to avoid double counting.

logging.getLogger("radiomics").setLevel(logging.ERROR)  # hide chatty messages


# ===========================================================================
# Helper functions
# ===========================================================================
def make_extractor(settings):
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    for feature_class in FEATURE_CLASSES:
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def mask_on_image_grid(mask, image):
    """Puts the mask on the same grid (size, spacing, position) as the image.
    'Nearest neighbour' keeps the mask as 0/1 instead of blurring it."""
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)   # any label > 0 = prostate
    return sitk.Resample(mask, image, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def extract_one_patient(row, paths, extractors):
    ids = {"patient_id": row["patient_id"], "study_id": row["study_id"]}
    mask = sitk.ReadImage(paths["mask"].format(**ids))
    features = dict(ids)

    for seq, pattern in paths["images"].items():
        image = sitk.ReadImage(pattern.format(**ids), sitk.sitkFloat32)
        seq_mask = mask_on_image_grid(mask, image)
        if sitk.GetArrayViewFromImage(seq_mask).sum() < 10:
            raise ValueError(f"mask almost empty on {seq} image")
        result = extractors[seq].execute(image, seq_mask, label=1)
        for name, value in result.items():
            if not name.startswith("diagnostics_"):   # skip technical info
                features[f"{seq}_{name.replace('original_', '')}"] = float(value)
    return features


def extract_all(patients, paths):
    extractors = {"t2w": make_extractor(T2W_SETTINGS),
                  "adc": make_extractor(ADC_SETTINGS)}
    rows, failed = [], []
    for i, (_, row) in enumerate(patients.iterrows(), start=1):
        try:
            rows.append(extract_one_patient(row, paths, extractors))
        except Exception as error:   # missing file, empty mask, ...
            failed.append((row["patient_id"], str(error)))
        if i % 10 == 0 or i == len(patients):
            print(f"  processed {i}/{len(patients)} patients")

    table = pd.DataFrame(rows)
    table.to_csv(paths["output"], index=False)
    n_feat = table.shape[1] - 2 if len(table) else 0
    print(f"\nSaved {len(table)} patients x {n_feat} features -> {paths['output']}")
    if failed:
        print(f"{len(failed)} patients FAILED (first 5 shown):")
        for pid, err in failed[:5]:
            print(f"  {pid}: {err}")
    return table


# ===========================================================================
# DEMO: fake images, to check the pipeline runs before real data arrives
# ===========================================================================
def make_fake_images(patients, folder="demo_images", seed=0):
    """Creates small fake MRI volumes. The 'prostate' is a sphere; patients
    with csPCa get a darker spot on ADC (low ADC = suspicious)."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    z, y, x = np.mgrid[0:16, 0:48, 0:48]
    gland = ((z - 8) ** 2 / 36 + (y - 24) ** 2 / 225 + (x - 24) ** 2 / 225) <= 1
    spot = ((z - 8) ** 2 + (y - 20) ** 2 + (x - 28) ** 2) <= 16

    for _, row in patients.iterrows():
        pid, sid = row["patient_id"], row["study_id"]
        t2w = rng.normal(300, 40, gland.shape) + gland * 100
        adc = rng.normal(1500, 150, gland.shape)
        if row["label"] == 1:
            adc[spot] -= rng.uniform(300, 700)
        t2w_img = sitk.GetImageFromArray(t2w.astype(np.float32))
        t2w_img.SetSpacing((0.5, 0.5, 3.0))
        # ADC: half the resolution -> tests the mask resampling
        adc_img = sitk.GetImageFromArray(adc[:, ::2, ::2].astype(np.float32))
        adc_img.SetSpacing((1.0, 1.0, 3.0))
        mask_img = sitk.GetImageFromArray(gland.astype(np.uint8))
        mask_img.SetSpacing((0.5, 0.5, 3.0))
        os.makedirs(f"{folder}/{pid}", exist_ok=True)
        sitk.WriteImage(t2w_img, f"{folder}/{pid}/{pid}_{sid}_t2w.mha")
        sitk.WriteImage(adc_img, f"{folder}/{pid}/{pid}_{sid}_adc.mha")
        sitk.WriteImage(mask_img, f"{folder}/{pid}/{pid}_{sid}_gland.nii.gz")

    return {
        "images": {"t2w": folder + "/{patient_id}/{patient_id}_{study_id}_t2w.mha",
                   "adc": folder + "/{patient_id}/{patient_id}_{study_id}_adc.mha"},
        "mask": folder + "/{patient_id}/{patient_id}_{study_id}_gland.nii.gz",
        "output": "radiomics_demo.csv",
    }


# ===========================================================================
# MAIN
# ===========================================================================
def main(dataset_key, demo):
    if demo:
        print("*** DEMO MODE: fake images ***\n")
        patients = pb.standardise(pb.make_fake_data(300, 0), pb.MAIN_DATASET).head(40)
        paths = make_fake_images(patients)
    else:
        paths = PATHS[dataset_key]
        patients = pb.load_dataset(paths["dataset"])

    print(f"Extracting radiomics for {len(patients)} patients...")
    table = extract_all(patients, paths)

    if demo and len(table):
        check = table.merge(patients[["patient_id", "label"]], on="patient_id")
        print("\nSanity check - mean ADC inside prostate (should be lower with csPCa):")
        print(check.groupby("label")["adc_firstorder_Mean"].mean().round(0)
              .rename({0: "no csPCa", 1: "csPCa"}).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["main", "external"], default="main")
    parser.add_argument("--demo", action="store_true", help="run with fake images")
    args = parser.parse_args()
    main(args.dataset, args.demo)
