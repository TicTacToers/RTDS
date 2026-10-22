"""
REGION FEATURES: a small, label-free feature set per prostate zone
==================================================================

What this script does (in plain words):
  1. For each PI-RADS 3 patient (same cohort as psad_baseline.py), opens the
     T2W and ADC scans, the whole-gland mask and the ZONAL mask
     (peripheral zone = PZ, transition zone = TZ).
  2. Measures a short, pre-specified list of 12 numbers that a radiologist
     would recognise:
       - ADC 10th percentile, median and spread (IQR) in the PZ and in the TZ
       - median T2W brightness in the PZ and in the TZ
       - the "darkest ADC spot": the largest connected blob of gland tissue
         with ADC below 1000 (the image itself points at suspicious tissue)
  3. Saves one row per patient to region_features_picai.csv.

Why this does NOT leak the outcome:
  Every mask used here (whole gland, PZ/TZ) is anatomy drawn by an AI for
  EVERY patient, whatever the biopsy result. The csPCa lesion outlines are
  never used: they exist only for cancers (human) or come from an AI trained
  on these same patients' labels (Bosma22a). The "darkest spot" is found from
  the ADC image alone.

Why no volume features:
  Whole-gland "Energy" features mostly measured prostate size, which PSAD
  already contains. Every feature here is a level (percentile, median), a
  spread, or the size of the dark blob - never the size of the gland.

How to run:
  python region_features.py
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk

import extract_radiomics as er
import psad_baseline as pb

# ===========================================================================
# SETTINGS
# ===========================================================================
ZONES = "picai_labels/anatomical_delineations/zonal_pz_tz/AI/Yuan23/{patient_id}_{study_id}.nii.gz"
PZ_LABEL, TZ_LABEL = 1, 2           # as documented in the Yuan23 README
LOW_ADC = 1000                      # ADC below this (x10^-6 mm2/s) is suspicious
OUTPUT = "region_features_picai.csv"

FEATURES = [
    "adc_pz_p10", "adc_pz_median", "adc_pz_iqr",
    "adc_tz_p10", "adc_tz_median", "adc_tz_iqr",
    "t2w_pz_median", "t2w_tz_median",
    "adc_low_fraction", "dark_blob_ml", "dark_blob_adc", "dark_blob_t2w",
]


# ===========================================================================
# Helper functions
# ===========================================================================
def label_on_grid(mask, image):
    """Puts a label mask on the image grid without blurring the labels."""
    return sitk.Resample(mask, image, sitk.Transform(),
                         sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def zone_stats(values, prefix):
    p10, p25, p50, p75 = np.percentile(values, [10, 25, 50, 75])
    return {f"{prefix}_p10": p10, f"{prefix}_median": p50, f"{prefix}_iqr": p75 - p25}


def dark_blob(adc_img, gland_on_adc):
    """Largest connected group of gland voxels with ADC < LOW_ADC.
    The gland edge is shaved off by one voxel in-plane first, because the
    border mixes in fat/noise that can look falsely dark. ADC = 0 voxels are
    failed fits, not tissue, so they never count as dark."""
    core = sitk.BinaryErode(gland_on_adc, [1, 1, 0])
    low = sitk.Cast((adc_img < LOW_ADC) & (adc_img > 0), sitk.sitkUInt8) * core
    blobs = sitk.RelabelComponent(sitk.ConnectedComponent(low, True))  # 1 = largest
    return sitk.GetArrayFromImage(blobs) == 1, sitk.GetArrayFromImage(core) > 0


def features_one_patient(row):
    ids = {"patient_id": row["patient_id"], "study_id": row["study_id"]}
    paths = er.PATHS["main"]
    t2w = sitk.ReadImage(paths["images"]["t2w"].format(**ids), sitk.sitkFloat32)
    adc = sitk.ReadImage(paths["images"]["adc"].format(**ids), sitk.sitkFloat32)
    gland = sitk.Cast(sitk.ReadImage(paths["mask"].format(**ids)) > 0, sitk.sitkUInt8)
    zones = sitk.Cast(sitk.ReadImage(ZONES.format(**ids)), sitk.sitkUInt8)

    adc_arr = sitk.GetArrayFromImage(adc)
    adc_ok = adc_arr > 0   # ADC <= 0 = failed fit (in 95 of 149 patients), not tissue
    t2w_arr = sitk.GetArrayFromImage(t2w)
    zones_adc = sitk.GetArrayFromImage(label_on_grid(zones, adc))
    zones_t2w = sitk.GetArrayFromImage(label_on_grid(zones, t2w))
    gland_t2w = sitk.GetArrayFromImage(label_on_grid(gland, t2w)) > 0
    gland_adc_img = label_on_grid(gland, adc)

    # T2W has no fixed unit: express it as z-scores within this patient's gland
    gland_vals = t2w_arr[gland_t2w]
    t2w_z = (t2w_arr - gland_vals.mean()) / gland_vals.std()
    # T2W resampled onto the ADC grid, to read T2W inside the dark ADC blob
    t2w_z_img = sitk.GetImageFromArray(t2w_z.astype(np.float32))
    t2w_z_img.CopyInformation(t2w)
    t2w_z_on_adc = sitk.GetArrayFromImage(
        sitk.Resample(t2w_z_img, adc, sitk.Transform(), sitk.sitkLinear, 0.0))

    out = dict(ids)
    for name, label in [("pz", PZ_LABEL), ("tz", TZ_LABEL)]:
        adc_zone = adc_arr[(zones_adc == label) & adc_ok]
        t2w_zone = t2w_z[zones_t2w == label]
        if len(adc_zone) < 10 or len(t2w_zone) < 10:
            raise ValueError(f"{name.upper()} mask almost empty")
        out.update(zone_stats(adc_zone, f"adc_{name}"))
        out[f"t2w_{name}_median"] = np.median(t2w_zone)

    blob, core = dark_blob(adc, gland_adc_img)
    voxel_ml = np.prod(adc.GetSpacing()) / 1000
    core_ok = core & adc_ok
    out["adc_low_fraction"] = np.mean(adc_arr[core_ok] < LOW_ADC)
    out["dark_blob_ml"] = blob.sum() * voxel_ml
    if blob.any():
        out["dark_blob_adc"] = adc_arr[blob].mean()
        out["dark_blob_t2w"] = t2w_z_on_adc[blob].mean()
    else:
        # No tissue below the threshold: use the darkest 1% of the gland core
        darkest = adc_arr[core_ok] <= np.percentile(adc_arr[core_ok], 1)
        out["dark_blob_adc"] = adc_arr[core_ok][darkest].mean()
        out["dark_blob_t2w"] = t2w_z_on_adc[core_ok][darkest].mean()
    return out


def extract_all(patients):
    rows, failed = [], []
    for i, (_, row) in enumerate(patients.iterrows(), start=1):
        try:
            rows.append(features_one_patient(row))
        except Exception as error:   # missing file, empty zone, ...
            failed.append((row["patient_id"], str(error)))
        if i % 25 == 0 or i == len(patients):
            print(f"  processed {i}/{len(patients)} patients")
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(table)} patients x {len(FEATURES)} features -> {OUTPUT}")
    if failed:
        print(f"{len(failed)} patients FAILED:")
        for pid, err in failed:
            print(f"  {pid}: {err}")
    return table


if __name__ == "__main__":
    patients = pb.load_dataset(pb.MAIN_DATASET)
    extract_all(patients)
