"""
Aggregate predictions from multiple training splits.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
# from src.models import (build_model_rsp, build_model_rsp_baseline, keras_callbacks, load_best_model,
#                         calc_tf_preds, calc_smp_preds)
# from src.ml.scale import get_scaler
# from src.ml.evals import calc_scores, save_confusion_matrix
# from src.ml.keras_utils import plot_prfrm_metrics
# from src.utils.classlogger import Logger
# from src.utils.utils import (cast_list, create_outdir, create_outdir_2, dump_dict, fea_types_to_str_name,
#                              get_print_func, read_lines, Params, Timer)
# from src.datasets.tidy import split_data_and_extract_fea, extract_fea, TidyData
# from src.tf_utils import get_tfr_files
# from src.sf_utils import (create_manifest, create_tf_data, calc_class_weights,
#                           parse_tfrec_fn_rsp, parse_tfrec_fn_rna)
# from src.sf_utils import bold, green, blue, yellow, cyan, red

# ------------------------------------
# Multimodal
# ------------------------------------
dataname = "tidy_drug_pairs_all_samples"
prjname = "bin_rsp_drug_pairs_all_samples"

datadir = fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd"
splits_dir_list = sorted(datadir.glob("split_*"))

# import ipdb; ipdb.set_trace()
all_te_scores = []
missing_scores = []
for split_dir in splits_dir_list:
    split_id = str(split_dir.name).split("split_")[1].split("_")[0]
    if (split_dir/"test_scores.csv").exists():
        te_scores = pd.read_csv(split_dir/"test_scores.csv")
        te_scores["split"] = split_id
        all_te_scores.append(te_scores)
    else:
        missing_scores.append(split_id)

print(f"Test scores were found for these splits: {missing_scores}")
mm = pd.concat(all_te_scores, axis=0)
mm = mm.rename(columns={"pred_for": "metric"})

# ------------------------------------
# LGBM
# ------------------------------------
lgb_datadir = fdir/"../data/PDX_Transfer_Learning_Classification/Results_MultiModal_Learning/1.0_True_False_100_31"
splits_dir_list = sorted(lgb_datadir.glob("cv_*"))

# import ipdb; ipdb.set_trace()
all_te_scores = []
missing_scores = []
for split_dir in splits_dir_list:
    split_id = str(split_dir.name).split("cv_")[1].split("_")[0]
    if (split_dir/"te_scores.csv").exists():
        te_scores = pd.read_csv(split_dir/"te_scores.csv")
        te_scores["split"] = split_id
        all_te_scores.append(te_scores)
    else:
        missing_scores.append(split_id)

print(f"Test scores were found for these splits: {missing_scores}")
lgb = pd.concat(all_te_scores, axis=0)

def agg_metrics(df):
    df = df.groupby("metric").agg(smp_mean=("smp", "mean"), smp_std=("smp", "std"),
                                  grp_mean=("Group", "mean"), grp_std=("Group", "std")
                                  ).reset_index()
    return df

mm_res = agg_metrics(mm)
lgb_res = agg_metrics(lgb)

# import ipdb; ipdb.set_trace()
commonn_metrics = set(mm_res["metric"].values).intersection(set(lgb_res["metric"].values))
mm_res = mm_res[mm_res["metric"].isin(commonn_metrics)]
lgb_res = lgb_res[lgb_res["metric"].isin(commonn_metrics)]

print("\nMulti-modal")
print(mm_res)
print("\nLGBM")
print(lgb_res)
print("Done.")
