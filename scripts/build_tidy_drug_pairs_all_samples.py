""" 
Build a dataset for binary classification with multimodal features.
The dataset contains all the available samples.

Here we treat every family/specimen-treatment as a single data sample.
"""
# print(__name__)
import os
import sys
assert sys.version_info >= (3, 5)

from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np


fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
from src import load_data
from src.load_data import PDX_SAMPLE_COLS

# Seed
np.random.seed(cfg.seed)

DATASET_NAME = "tidy_drug_pairs_all_samples"
outdir = cfg.DATA_PROCESSED_DIR/DATASET_NAME
os.makedirs(outdir, exist_ok=True)

target = "Response"

# Load data
data = load_data.load_tidy_dataset_rsp(single_drug=False, add_type_labels=True)

# ----------------------
# Data summary
sm = {}

# Number of patients and specimens
df = data.groupby("patient_id").agg({"specimen_id": "nunique"}).reset_index()
sm["Patients"] = df.patient_id.nunique()
sm["Primary tumor specimens"] = df.specimen_id.sum()

# Single drug treatments
df = data[data["Drug1"] == data["Drug2"]]
sm["Single-drug treatments"] = df.trt.nunique()

# Drug-pair treatments
df = data[data["Drug1"] != data["Drug2"]]
df = df[df["pair_aug"] == False]
sm["Drug-pair treatments"] = df.trt.nunique()

# Group treatments
sm["Treatment groups"] = data.Group.nunique()

# Gene expression profiles
ge = data[[c for c in data.columns if c.startswith("ge_")]]
ge = ge.drop_duplicates()
# print(ge.shape[0] - ge.duplicated().sum())
# print(ge.shape)
sm["Gene expression profiles"] = ge.shape[0]

# Hisgology slides
sm["Histology slides"] = data.image_id.nunique()

# Histology tiles
tfr_dir = cfg.DATADIR/"PDX_FIXED_RSP_DRUG_PAIR"/"299px_302um"
tile_cnts = pd.read_csv(tfr_dir/"tile_counts_per_slide.csv")
df = data.drop_duplicates(subset=["image_id"])
df = df[["smp", "image_id", "slide", "Response"]].astype({"image_id": int, "slide": int})
df = df.merge(tile_cnts, on=["smp", "slide", "Response"], how="inner")
sm["Histology tiles"] = df["max_tiles"].sum()

# Single-drug response samples in the ML dataset
sm["Single-drug response samples"] = data[data.single==True].shape[0]

# Single-drug response samples in the ML dataset
sm["Drug-pair response samples"] = data[data.single==False].shape[0]

# Drug response samples in the ML dataset
sm["Drug response samples"] = data.shape[0]

pd.Series(sm)
summary = pd.Series(sm)
print(summary)
del df
# ----------------------

df = data; del data
pprint(df.groupby(["ctype", "Response"]).agg({"Group": "nunique", "smp": "nunique"}).reset_index().rename(
    columns={"Group": "Group_unq", "smp": "smp_unq"}))
pprint(df[target].value_counts())

# Save annotations file
df.to_csv(outdir/cfg.ANNOTATIONS_FILENAME, index=False)
print("\nFinal dataframe", df.shape)

# add slideflow required columns (sbumitter_id, slide) and save annotations file
print("\nCreate and save annotations file for slideflow.")
df_sf = df.reset_index(drop=True)
df_sf.insert(loc=1, column="submitter_id", value=df_sf["image_id"].values, allow_duplicates=False)
if "slide" not in df_sf.columns:
    df_sf.insert(loc=2, column="slide", value=df_sf["image_id"].values, allow_duplicates=False)
df_sf.to_csv(outdir/cfg.SF_ANNOTATIONS_FILENAME, index=False)

print("\nDone.")
