"""
TODO: problems repdocucing the same data batches for every run!

https://github.com/tensorflow/tensorflow/issues/13932
https://github.com/NVIDIA/framework-determinism
"""
import os
import sys
assert sys.version_info >= (3, 5)

# https://www.codegrepper.com/code-examples/python/suppres+tensorflow+warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from collections import OrderedDict
import glob
from pathlib import Path
from pprint import pprint, pformat
import shutil
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# import ipdb; ipdb.set_trace()

import tensorflow as tf
assert tf.__version__ >= "2.0"

fdir = Path(__file__).resolve().parent
# from config import cfg
sys.path.append(str(fdir/".."))
sys.path.append(str(fdir/"../src"))
import src
from src.config import cfg
from src.models import build_model_rsp, build_model_rsp_baseline, keras_callbacks
from src.ml.scale import get_scaler
from src.ml.evals import calc_scores, save_confusion_matrix
from src.ml.keras_utils import plot_prfrm_metrics
from src.utils.classlogger import Logger
from src.utils.utils import (cast_list, create_outdir, create_outdir_2, dump_dict, get_print_func,
                             read_lines, Params, Timer)
from src.datasets.tidy import split_data_and_extract_fea, extract_fea, TidyData
from src.tf_utils import get_tfr_files
from src.sf_utils import (create_manifest, create_tf_data, calc_class_weights,
                          parse_tfrec_fn_rsp, parse_tfrec_fn_rna)

# Seed
import random
random.seed(cfg.seed)
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)


parser = argparse.ArgumentParser("Train NN.")
parser.add_argument("-t", "--target",
                    type=str,
                    nargs='+',
                    default=["Response"],
                    choices=["Response", "ctype", "csite"],
                    help="Name of target output.")
parser.add_argument("--id_name",
                    type=str,
                    default="smp",
                    choices=["slide", "smp"],
                    help="Column name of the ID.")
parser.add_argument("--split_on",
                    type=str,
                    default="Group",
                    choices=["Sample", "slide", "Group"],
                    help="Specify the hard split variable/column (default: None).")
parser.add_argument("--prjname",
                    type=str,
                    help="Project name (folder that contains the annotations.csv dataframe).")
parser.add_argument("--dataname",
                    type=str,
                    default="tidy_drug_pairs_all_samples",
                    help="Project name (folder that contains the annotations.csv dataframe).")
parser.add_argument("--trn_phase",
                    type=str,
                    default="train",
                    choices=["train", "evaluate"],
                    help="Project name (folder that contains the annotations.csv dataframe).")
parser.add_argument("--n_samples",
                    type=int,
                    default=-1,
                    help="Total samples from tr_id to process.")
parser.add_argument("--nn_arch",
                    type=str,
                    default="multimodal",
                    choices=["baseline", "multimodal", "lgbm"],
                    help="NN architecture (default: multimodal).")

args, other_args = parser.parse_known_args()
pprint(args)

split_on = "none" if args.split_on is (None or "none") else args.split_on


from datetime import datetime
t = datetime.now()
t = [t.year, "-", str(t.month).zfill(2), "-", str(t.day).zfill(2),
     "_", "h", str(t.hour).zfill(2), "-", "m", str(t.minute).zfill(2)]
t = "".join([str(i) for i in t])
outdir = fdir/f"../tests.out/test_seed/{t}"
os.makedirs(outdir)


# Create outdir (using the loaded hyperparamters)
prm_file_path = outdir/f"params_{args.nn_arch}.json"
shutil.copy(fdir/f"../default_params/default_params_{args.nn_arch}.json", prm_file_path)
params = Params(prm_file_path)


# Save hyper-parameters
params.save(outdir/"params.json")


# Logger
lg = Logger(outdir/"logger.log")
print_fn = get_print_func(lg.logger)
print_fn(f"File path: {fdir}")
print_fn(f"\n{pformat(vars(args))}")


# Load dataframe (annotations)
annotations_file = cfg.DATA_PROCESSED_DIR/args.dataname/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
data = data.astype({"image_id": str, "slide": str})
print_fn(data.shape)


print("\nFull dataset:")
if args.target[0] == "Response":
    print_fn(data.groupby(["ctype", "Response"]).agg({split_on: "nunique", "smp": "nunique"}).reset_index().rename(
        columns={split_on: f"{split_on}_unq", "smp": "smp_unq"}))
else:
    print_fn(data[args.target[0]].value_counts())


# Determine tfr_dir (where TFRecords are stored)
if args.target[0] == "Response":
    if params.single_drug:
        tfr_dir = cfg.SF_TFR_DIR_RSP
    else:
        tfr_dir = cfg.SF_TFR_DIR_RSP_DRUG_PAIR_10percent  # TODO: required to change
elif args.target[0] == "ctype":
    tfr_dir = cfg.SF_TFR_DIR_RNA_NEW

label = f"{params.tile_px}px_{params.tile_um}um"
tfr_dir = tfr_dir/label


# Scalers for each feature set
ge_scaler, dd1_scaler, dd2_scaler = None, None, None

ge_cols  = [c for c in data.columns if c.startswith("ge_")]
dd1_cols = [c for c in data.columns if c.startswith("dd1_")]
dd2_cols = [c for c in data.columns if c.startswith("dd2_")]

if params.use_ge and len(ge_cols) > 0:
    ge_scaler = get_scaler(data[ge_cols])

if params.use_dd1 and len(dd1_cols) > 0:
    dd1_scaler = get_scaler(data[dd1_cols])

if params.use_dd2 and len(dd2_cols) > 0:
    dd2_scaler = get_scaler(data[dd2_cols])


# -----------------------------------------------
# Data splits
# -----------------------------------------------
splitdir = cfg.DATADIR/"PDX_Transfer_Learning_Classification/Processed_Data/Data_For_MultiModal_Learning/Data_Partition"
split_id = 81

tr_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"TrainList.txt")), int)
vl_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"ValList.txt")), int)
te_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"TestList.txt")), int)

# Update ids
index_col_name = "index"
tr_id = sorted(set(data[index_col_name]).intersection(set(tr_id)))
vl_id = sorted(set(data[index_col_name]).intersection(set(vl_id)))
te_id = sorted(set(data[index_col_name]).intersection(set(te_id)))

# Subsample train samples
if (args.n_samples > 0) and (args.n_samples < len(tr_id)):
    tr_id = tr_id[:args.n_samples]


# --------------
# w/o TidyData
# --------------
kwargs = {"ge_cols": ge_cols,
          "dd1_cols": dd1_cols,
          "dd2_cols": dd2_cols,
          "ge_scaler": ge_scaler,
          "dd1_scaler": dd1_scaler,
          "dd2_scaler": dd2_scaler,
          "ge_dtype": cfg.GE_DTYPE,
          "dd_dtype": cfg.DD_DTYPE,
          "index_col_name": index_col_name,
          "split_on": split_on
          }
tr_ge, tr_dd1, tr_dd2, tr_meta = split_data_and_extract_fea(data, ids=tr_id, **kwargs)
vl_ge, vl_dd1, vl_dd2, vl_meta = split_data_and_extract_fea(data, ids=vl_id, **kwargs)
te_ge, te_dd1, te_dd2, te_meta = split_data_and_extract_fea(data, ids=te_id, **kwargs)

tr_meta.to_csv(outdir/"tr_meta.csv", index=False)
vl_meta.to_csv(outdir/"vl_meta.csv", index=False)
te_meta.to_csv(outdir/"te_meta.csv", index=False)

ge_shape = (tr_ge.shape[1],)
dd_shape = (tr_dd1.shape[1],)

# Variables (dict/dataframes/arrays) that are passed as features to the NN
xtr = {"ge_data": tr_ge.values, "dd1_data": tr_dd1.values, "dd2_data": tr_dd2.values}
xvl = {"ge_data": vl_ge.values, "dd1_data": vl_dd1.values, "dd2_data": vl_dd2.values}
xte = {"ge_data": te_ge.values, "dd1_data": te_dd1.values, "dd2_data": te_dd2.values}

# import ipdb; ipdb.set_trace()
print_fn("\nTrain:")
print_fn(tr_meta.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
print_fn("\nValidation:")
print_fn(vl_meta.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
print_fn("\nTest:")
print_fn(te_meta.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())

# Make sure indices do not overlap
assert len( set(tr_id).intersection(set(vl_id)) ) == 0, "Overlapping indices btw tr and vl"
assert len( set(tr_id).intersection(set(te_id)) ) == 0, "Overlapping indices btw tr and te"
assert len( set(vl_id).intersection(set(te_id)) ) == 0, "Overlapping indices btw tr and vl"

# Print split ratios
print_fn("")
print_fn("Train samples {} ({:.2f}%)".format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
print_fn("Val   samples {} ({:.2f}%)".format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
print_fn("Test  samples {} ({:.2f}%)".format( len(te_id), 100*len(te_id)/data.shape[0] ))

tr_grp_unq = set(tr_meta[split_on].values)
vl_grp_unq = set(vl_meta[split_on].values)
te_grp_unq = set(te_meta[split_on].values)
print_fn("")
print_fn(f"Total intersects on {split_on} btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}")
print_fn(f"Total intersects on {split_on} btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}")
print_fn(f"Total intersects on {split_on} btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}")
print_fn(f"Unique {split_on} in tr: {len(tr_grp_unq)}")
print_fn(f"Unique {split_on} in vl: {len(vl_grp_unq)}")
print_fn(f"Unique {split_on} in te: {len(te_grp_unq)}")


# --------------------------
# Obtain T/V/E tfr filenames
# --------------------------
# List of sample names for T/V/E
tr_smp_names = list(tr_meta[args.id_name].values)
vl_smp_names = list(vl_meta[args.id_name].values)
te_smp_names = list(te_meta[args.id_name].values)

# TFRecords filenames
train_tfr_files = get_tfr_files(tfr_dir, tr_smp_names)
val_tfr_files = get_tfr_files(tfr_dir, vl_smp_names)
test_tfr_files = get_tfr_files(tfr_dir, te_smp_names)
print("Total samples {}".format(len(train_tfr_files) + len(val_tfr_files) + len(test_tfr_files)))

# Missing tfrecords
print("\nThese samples miss a tfrecord ...\n")
df_miss = data.loc[~data[args.id_name].isin(tr_smp_names + vl_smp_names + te_smp_names), ["smp", "image_id"]]
print(df_miss)

assert sorted(tr_smp_names) == sorted(tr_meta[args.id_name].values.tolist()), "Sample names in the tr_smp_names and tr_meta don't match."
assert sorted(vl_smp_names) == sorted(vl_meta[args.id_name].values.tolist()), "Sample names in the vl_smp_names and vl_meta don't match."
assert sorted(te_smp_names) == sorted(te_meta[args.id_name].values.tolist()), "Sample names in the te_smp_names and te_meta don't match."

# -------------------------------
# Class weight
# -------------------------------
if args.nn_arch == "baseline":
    class_weights_method = "BY_SAMPLE"
elif args.nn_arch == "multimodal":
    class_weights_method = "BY_TILE"
else:
    class_weights_method = "NONE"

# import ipdb; ipdb.set_trace()
tile_cnts = pd.read_csv(tfr_dir/"tile_counts_per_slide.csv")
cat = tile_cnts[tile_cnts["tfr_fname"].isin(train_tfr_files)]
cat = cat.groupby(args.target[0]).agg({"smp": "nunique", "max_tiles": "sum", "n_tiles": "sum", "slide": "nunique"}).reset_index()
categories = {}
for i, row_data in cat.iterrows():
    dct = {"num_samples": row_data["smp"], "num_tiles": row_data["n_tiles"]}
    categories[row_data[args.target[0]]] = dct

class_weight = calc_class_weights(train_tfr_files,
                                  class_weights_method=class_weights_method,
                                  categories=categories)


if args.nn_arch == "multimodal":
    # -------------------------------
    # Parsing funcs
    # -------------------------------

    # import ipdb; ipdb.set_trace()

    if args.target[0] == "Response":
        # Response
        parse_fn = parse_tfrec_fn_rsp
        parse_fn_train_kwargs = {
            "use_tile": params.use_tile,
            "use_ge": params.use_ge,
            "use_dd1": params.use_dd1,
            "use_dd2": params.use_dd2,
            "ge_scaler": ge_scaler,
            "dd1_scaler": dd1_scaler,
            "dd2_scaler": dd2_scaler,
            "id_name": args.id_name,
            "augment": params.augment,
        }
    else:
        # Ctype
        parse_fn = parse_tfrec_fn_rna
        parse_fn_train_kwargs = {
            'use_tile': params.use_tile,
            'use_ge': params.use_ge,
            'ge_scaler': ge_scaler,
            'id_name': args.id_name,
            'MODEL_TYPE': params.model_type,
            'AUGMENT': params.augment,
        }

    parse_fn_non_train_kwargs = parse_fn_train_kwargs.copy()
    parse_fn_non_train_kwargs["augment"] = False


    # ----------------------------------------
    # Number of tiles/examples in each dataset
    # ----------------------------------------
    # import ipdb; ipdb.set_trace()
    total_train_tiles = tile_cnts[tile_cnts[args.id_name].isin(tr_smp_names)]["n_tiles"].sum()
    total_val_tiles = tile_cnts[tile_cnts[args.id_name].isin(vl_smp_names)]["n_tiles"].sum()
    total_test_tiles = tile_cnts[tile_cnts[args.id_name].isin(te_smp_names)]["n_tiles"].sum()

    eval_batch_size = 8 * params.batch_size
    train_steps = total_train_tiles // params.batch_size
    validation_steps = total_val_tiles // eval_batch_size
    test_steps = total_test_tiles // eval_batch_size


    # -------------------------------
    # Create TF datasets
    # -------------------------------
    print("\nCreate TF datasets ...")

    # Training
    import ipdb; ipdb.set_trace()
    train_data = create_tf_data(
        deterministic=True,
        batch_size=params.batch_size,
        # include_meta=False,
        include_meta=True,
        # interleave=True,
        interleave=False,
        n_concurrent_shards=params.n_concurrent_shards,
        parse_fn=parse_fn,
        # prefetch=2,
        prefetch=1,
        # repeat=True,
        repeat=False,
        seed=cfg.seed,
        # shuffle_files=True,
        shuffle_files=False,
        shuffle_size=params.shuffle_size,
        tfrecords=train_tfr_files,
        **parse_fn_train_kwargs)

    # # Determine feature shapes from data
    # bb = next(train_data.__iter__())

    # # Infer dims of features from the data
    # # import ipdb; ipdb.set_trace()
    # if params.use_ge:
    #     ge_shape = bb[0]["ge_data"].numpy().shape[1:]
    # else:
    #     ge_shape = None
    # if params.use_dd1:
    #     dd_shape = bb[0]["dd1_data"].numpy().shape[1:]
    # else:
    #     dd_shape = None
    # for i, item in enumerate(bb):
    #     print(f"\nItem {i}")
    #     if isinstance(item, dict):
    #         for k in item.keys():
    #             print(f"\t{k}: {item[k].numpy().shape}")
    # for i, rec in enumerate(train_data.take(2)):
    #     tf.print(rec[1])

    # Evaluation (val, test, train)
    create_tf_data_eval_kwargs = {
        "deterministic": True,
        "batch_size": eval_batch_size,
        "include_meta": False,
        "interleave": False,
        "parse_fn": parse_fn,
        "prefetch": 2,
        "repeat": False,
        "seed": None,
        "shuffle_files": False,
        "shuffle_size": None,
    }

    # import ipdb; ipdb.set_trace()
    create_tf_data_eval_kwargs.update({"tfrecords": val_tfr_files, "include_meta": False})
    val_data = create_tf_data(
        **create_tf_data_eval_kwargs,
        **parse_fn_non_train_kwargs
    )

    create_tf_data_eval_kwargs.update({"tfrecords": test_tfr_files, "include_meta": True})
    test_data = create_tf_data(
        **create_tf_data_eval_kwargs,
        **parse_fn_non_train_kwargs
    )

    create_tf_data_eval_kwargs.update({"tfrecords": val_tfr_files, "include_meta": True})
    eval_val_data = create_tf_data(
        **create_tf_data_eval_kwargs,
        **parse_fn_non_train_kwargs
    )

    create_tf_data_eval_kwargs.update({"tfrecords": train_tfr_files, "include_meta": True, "prefetch": 1})
    eval_train_data = create_tf_data(
        **create_tf_data_eval_kwargs,
        **parse_fn_non_train_kwargs
    )


# @tf.function
def get_df_meta(data_with_meta):
    # meta_keys = ["smp", "Group", "grp_name", "Response"]
    meta_keys = ["smp", "image_id", "tile_id"]
    meta_agg = {k: None for k in meta_keys}
    y_true, y_pred_prob, y_pred_label = [], [], []

    for i, batch in enumerate(data_with_meta):
        if (i+1) % 50 == 0:
            print(f"\rbatch {i+1}", end="")

        fea = batch[0]
        label = batch[1]
        meta = batch[2]

        # # True labels
        y_true.extend( label.numpy().tolist() )  # when batch[1] is array

        # Meta
        for k in meta_keys:
            vv = [val_bytes.decode("utf-8") for val_bytes in meta[k].numpy().tolist()]
            if meta_agg[k] is None:
                meta_agg[k] = vv
            else:
                meta_agg[k].extend(vv)

        del batch, fea, label, meta

    df_meta = pd.DataFrame(meta_agg)
    df_meta["Response"] = y_true
    return df_meta

# import ipdb; ipdb.set_trace()
# df_meta1 = get_df_meta(train_data)
# df_meta2 = get_df_meta(train_data)
# df_meta1.equals(df_meta2)

df_meta = get_df_meta(train_data)
df_meta.to_csv(outdir/"df_meta.csv", index=False)
