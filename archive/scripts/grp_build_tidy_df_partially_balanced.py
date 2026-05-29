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

DATASET_NAME = "tidy_partially_balanced"
outdir = cfg.DATA_PROCESSED_DIR/DATASET_NAME
os.makedirs(outdir, exist_ok=True)

target = "Response"

# Load data
rsp = load_data.load_rsp()
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
rsp_rna_dd = rsp_rna.merge(dd, left_on="Drug1", right_on="ID", how="inner") #.reset_index(drop=True)
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
data = cref.merge(rsp_rna_dd_pdx, on=PDX_SAMPLE_COLS, how="inner")
print(data.shape)

# Add 'slide' column
data.insert(loc=5, column="slide", value=data["image_id"], allow_duplicates=True)

if "index" in data.columns:
    cols = data.columns.tolist()
    cols.remove("index")
    data = data[["index"] + cols]

# -----------------------------------------------------------------------------
# Subsample data to create balanced dataset in terms of drug response and ctype
# -----------------------------------------------------------------------------
print('\nSubsample master dataframe to create balanced dataset.')
r0 = data[data[target] == 0]  # non-responders
r1 = data[data[target] == 1]  # responders
# xx = r1[ r1.columns[:5].tolist() + ['trt', 'Group', 'grp_name', 'Response'] ]
# xx.to_csv("r1.csv", index=False)

# Aggregate non-responders to balance the responders
def sub_non_responders(ref_df, src_df):
    hh = ref_df.groupby("ctype").agg({"grp_name": "nunique"}).reset_index()
    dfs = []
    for ctype, cnt in zip(hh.ctype, hh.grp_name):
        aa = src_df[src_df.ctype == ctype]
        unique_groups = aa.grp_name.unique()
        groups = np.random.choice(unique_groups, size=cnt, replace=False)
        aa = aa[aa.grp_name.isin(groups)]
        dfs.append(aa)
    src_df_sub = pd.concat(dfs, axis=0)
    # pprint(ref_df.groupby(["ctype", "Response"]).agg({"grp_name": "nunique"}).reset_index())
    # pprint(src_df_sub.groupby(["ctype", "Response"]).agg({"grp_name": "nunique"}).reset_index())
    return src_df_sub

r0_sub = sub_non_responders(ref_df=r1, src_df=r0)
# pprint(r1.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
# pprint(r0_sub.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())

# Concat non-responders and responders
# xx = r0_sub[ r0_sub.columns[:5].tolist() + ['trt', 'Group', 'grp_name', 'Response'] ]
# xx.to_csv("r0.csv", index=False)
df_balanced = pd.concat([r0_sub, r1], axis=0)
df_balanced = df_balanced.sort_values(["grp_name", "smp"], ascending=True).reset_index(drop=True)
# xx = df_balanced[ df_balanced.columns[:5].tolist() + ['trt', 'Group', 'grp_name', 'Response'] ]
# xx.to_csv("df_balanced.csv", index=False)
pprint(df_balanced.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
del r0_sub, r0, r1

# --------------------------------------------------------------
# import ipdb; ipdb.set_trace()

r0_rest = data[~data["smp"].isin(df_balanced["smp"].values)]
r0_rest = r0_rest[r0_rest["ctype"].isin(df_balanced["ctype"].unique())]

r0_sub = sub_non_responders(ref_df=df_balanced, src_df=r0_rest)
# pprint(df_balanced.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
# pprint(r0_sub.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())

df = pd.concat([r0_sub, df_balanced], axis=0)
df = df.sort_values(["grp_name", "smp"], ascending=True).reset_index(drop=True)
pprint(df.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
del r0_sub, r0_rest
# --------------------------------------------------------------

# import ipdb; ipdb.set_trace()

# df = df.reset_index()
# pprint(df.reset_index().groupby(["ctype", target]).agg({"index": "nunique"}).reset_index().rename(columns={"index": "samples"}))
# pprint(df[target].value_counts())

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
