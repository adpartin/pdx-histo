"""
A batch prcoessing that calls trn_multimodal.py with the same set of parameters
but different split ids.
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
from pathlib import Path

fdir = Path(__file__).resolve().parent
sys.path.append(str(fdir/".."))
import src
from src.config import cfg

import src.trn_multimodal as trn_multimodal
from src.utils.utils import Timer

timer = Timer()

dataname = "tidy_drug_pairs_all_samples"
prjname = "bin_rsp_drug_pairs_all_samples"

tfr_dir_name = "PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles"
pred_tfr_dir_name = "PDX_FIXED_RSP_DRUG_PAIR"

splitdir = cfg.DATADIR/"PDX_Transfer_Learning_Classification/Processed_Data/Data_For_MultiModal_Learning/Data_Partition"
splits_dir_list = sorted(splitdir.glob("cv_*"))

# Input args:
# --train --eval --use_tile
args = sys.argv[1:]

# --------
# Baseline
# --------
datadir = fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_ge_dd"
base_args_list = ["--tfr_dir_name", tfr_dir_name,
                  "--pred_tfr_dir_name", pred_tfr_dir_name,
                  "--dataname", dataname,
                  "--prjname", prjname,
                  "--target", "Response",
                  "--id_name", "smp",
                  "--split_on", "Group",
                  "--use_ge", "--use_dd1", "--use_dd2"
]

# # -----------
# # Multi-modal
# # -----------
# datadir = fdir/"../projects/bin_rsp_drug_pairs_all_samples/runs_tile_ge_dd"
# splits_dir_list = sorted(datadir.glob("split_*"))
# base_args_list = ["--tfr_dir_name", tfr_dir_name,
#                   "--pred_tfr_dir_name", pred_tfr_dir_name,
#                   "--dataname", dataname,
#                   "--prjname", prjname,
#                   "--target", "Response",
#                   "--id_name", "smp",
#                   "--split_on", "Group",
#                   "--use_tile", "--use_ge", "--use_dd1", "--use_dd2"
# ]

# import ipdb; ipdb.set_trace()
for split_dir in splits_dir_list:
    print(f"split_dir: {split_dir.name}")
    split_id = str(split_dir.name).split("cv_")[1]
    args_list = base_args_list.copy()
    args_list.extend(["--split_id", split_id])
    args_list.extend(args)
    trn_multimodal.main(args_list)

timer.display_timer()
print("Done.")
