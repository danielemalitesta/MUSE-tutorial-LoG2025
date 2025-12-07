import os
import sys

src_path = os.path.abspath('../..')
print(src_path)
sys.path.append(src_path)

from src.utils import create_directory, dump_pickle, raw_data_path, processed_data_path, set_seed

import pandas as pd

set_seed(seed=42)

input_path = os.path.join(raw_data_path, "eicu")
output_path = os.path.join(processed_data_path, "eicu")
create_directory(output_path)

# lab
lab_raw = pd.read_csv(os.path.join(input_path, 'lab.csv'))
print(lab_raw.shape)
print(lab_raw.head())

print(lab_raw.labname.nunique())

print(lab_raw.groupby("labname").labmeasurenamesystem.nunique().unique())

print(lab_raw.groupby("labname").labmeasurenameinterface.nunique())

print(lab_raw[lab_raw.labname == "urinary osmolality"].labmeasurenameinterface.unique())

print(lab_raw[lab_raw.labname == "urinary osmolality"].labmeasurenamesystem.unique())

print(lab_raw[lab_raw.labname == "urinary osmolality"][lab_raw.labmeasurenameinterface == "mOsm/kg"])

print(lab_raw[lab_raw.labname == "urinary osmolality"][lab_raw.labmeasurenameinterface == "mOs/kH2O"])

print(lab_raw.labresult.isna().sum())

lab_raw = lab_raw.dropna(subset=["labresult"])
lab_raw["normalized_labresult"] = lab_raw.groupby('labname')['labresult'].transform(lambda x: (x - x.mean()) / x.std())
print(lab_raw)

print(output_path)

lab_raw.to_csv(os.path.join(output_path, 'lab_tmp.csv'), index=False)

# apacheapsvar
apacheapsvar = pd.read_csv(os.path.join(input_path, 'apacheApsVar.csv'))
print(apacheapsvar.shape)
print(apacheapsvar.head())

print(apacheapsvar.isna().sum())

apacheapsvar = apacheapsvar.replace(-1, float("nan"))
print(apacheapsvar)

apacheapsvar = apacheapsvar
apacheapsvar['nan_rate'] = apacheapsvar.isna().sum(axis=1) / apacheapsvar.shape[1]

print(apacheapsvar[apacheapsvar.nan_rate > 0.50])

apacheapsvar = apacheapsvar[apacheapsvar.nan_rate <= 0.50]
print(apacheapsvar)

print(apacheapsvar.intubated.unique())

print(apacheapsvar.vent.unique())

print(apacheapsvar.dialysis.unique())

print(apacheapsvar.eyes.unique())

print(apacheapsvar.motor.unique())

print(apacheapsvar.verbal.unique())

print(apacheapsvar.meds.unique())

category = [
    "intubated",
    "vent",
    "dialysis",
    "eyes",
    "motor",
    "verbal",
    "meds",
]

continuous = [
    "urine",
    "wbc",
    "temperature",
    "respiratoryrate",
    "sodium",
    "heartrate",
    "meanbp",
    "ph",
    "hematocrit",
    "creatinine",
    "albumin",
    "pao2",
    "pco2",
    "bun",
    "glucose",
    "bilirubin",
    "fio2",
]


def process_df(df, category_col, continuous_col):
    # Fill NaNs in category columns with the mode (most frequent value)
    for col in category_col:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Normalize continuous columns
    for col in continuous_col:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
        df[col].fillna(0.0, inplace=True)

    return df

apacheapsvar = process_df(apacheapsvar, category, continuous)
print(apacheapsvar)

apacheapsvar = apacheapsvar.drop(columns=["nan_rate"])
print(apacheapsvar)

apacheapsvar.to_csv(os.path.join(output_path, 'apacheapsvar_tmp.csv'), index=False)
