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

# import pdb; pdb.set_trace()
# import ipdb; ipdb.set_trace()

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
from src import load_data
from src.load_data import PDX_SAMPLE_COLS

# Seed
seed = 42
np.random.seed(seed)

# PRJNAME = 'bin_rsp_balance_03'
# outdir = cfg.MAIN_PRJDIR/PRJNAME
# os.makedirs(outdir, exist_ok=True)

DATASET_NAME = "tidy_single_drug_all_samples"
outdir = cfg.DATA_PROCESSED_DIR/DATASET_NAME
os.makedirs(outdir, exist_ok=True)

target = "Response"

# Load data
rsp = load_data.load_rsp(single_drug=True)
rna = load_data.load_rna()
dd = load_data.load_dd()
cref = load_data.load_crossref()
pdx = load_data.load_pdx_meta2()

# Merge rsp with rna
print("\nMerge rsp and rna")
print(rsp.shape)
print(rna.shape)
rsp_rna = rsp.merge(rna, on="Sample", how="inner")
print(rsp_rna.shape)

# Merge with dd
print("Merge with descriptors")
print(rsp_rna.shape)
print(dd.shape)
# rsp_rna_dd = rsp_rna.merge(dd, left_on="Drug1", right_on="ID", how="inner")
dd1 = dd.copy()
dd1 = dd1.rename(columns={"ID": "Drug1"})
fea_id0 = 1
fea_pfx = "dd_"
dd1 = dd1.rename(columns={c: "dd1_" + c.split(fea_pfx)[1] for c in dd1.columns[fea_id0:] if ~c.startswith(fea_pfx)})
rsp_rna_dd = rsp_rna.merge(dd1, left_on="Drug1", right_on="Drug1", how="inner")
print(rsp_rna_dd.shape)

# Merge with pdx meta
print("Merge with pdx meta")
print(pdx.shape)
print(rsp_rna_dd.shape)
rsp_rna_dd_pdx = pdx.merge(rsp_rna_dd, on=["patient_id", "specimen_id"], how="inner")
print(rsp_rna_dd_pdx.shape)

# Merge cref
print("Merge with cref")
# (we loose some samples because we filter the bad slides)
print(cref.shape)
print(rsp_rna_dd_pdx.shape)
data = cref.merge(rsp_rna_dd_pdx, on=PDX_SAMPLE_COLS, how="inner").reset_index(drop=True)
print(data.shape)

# Add 'slide' column
data.insert(loc=5, column="slide", value=data["image_id"], allow_duplicates=True)

if "index" in data.columns:
    cols = data.columns.tolist()
    cols.remove("index")
    data = data[["index"] + cols]

# import ipdb; ipdb.set_trace()

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