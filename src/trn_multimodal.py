"""
Prediction of drug response with TFRecords.
"""
import os
import sys
assert sys.version_info >= (3, 5)

# https://www.codegrepper.com/code-examples/python/suppres+tensorflow+warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from collections import OrderedDict
import csv
import glob
from pathlib import Path
from pprint import pprint, pformat
import shutil
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
assert tf.__version__ >= "2.0"

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

fdir = Path(__file__).resolve().parent
# from config import cfg
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
from src.models import (build_model_rsp, build_model_rsp_baseline, keras_callbacks, load_best_model,
                        calc_tf_preds, calc_smp_preds)
from src.ml.scale import get_scaler
from src.ml.evals import calc_scores, save_confusion_matrix
from src.ml.keras_utils import plot_prfrm_metrics
from src.utils.classlogger import Logger
from src.utils.utils import (cast_list, create_outdir, create_outdir_2, dump_dict, fea_types_to_str_name,
                             get_print_func, read_lines, Params, Timer)
from src.datasets.tidy import split_data_and_extract_fea, extract_fea, TidyData
from src.tf_utils import get_tfr_files
from src.sf_utils import (create_manifest, create_tf_data, calc_class_weights,
                          parse_tfrec_fn_rsp, parse_tfrec_fn_rna)
from src.sf_utils import bold, green, blue, yellow, cyan, red

# Seed
np.random.seed(cfg.seed)
tf.random.set_seed(cfg.seed)


def print_groupby_stat(df, print_fn=print):
    print_fn(df.groupby(["ctype", "Response"]).agg({split_on: "nunique", "smp": "nunique"}).reset_index().rename(
        columns={split_on: f"{split_on}_unq", "smp": "smp_unq"}))


parser = argparse.ArgumentParser("Train NN.")
parser.add_argument("-t", "--target",
                    type=str,
                    nargs='+',
                    default=["Response"],
                    choices=["Response", "ctype", "csite"],
                    help="Name of target to be precited.")
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
parser.add_argument("--split_id",
                    type=int,
                    default=81,
                    help="Split id of the unique split for T/V/E sets (default: 0).")
parser.add_argument("--prjname",
                    type=str,
                    help="Project name (to the store the restuls.")
parser.add_argument("--dataname",
                    type=str,
                    default="tidy_all",
                    help="Data name (folder that contains the annotations.csv dataframe).")
parser.add_argument("--rundir",
                    type=str,
                    default=None,
                    help="Dir that contains the saved models (used for evaluation mode).")
parser.add_argument("--train",
                    action="store_true",
                    help="Train mode.")
parser.add_argument("--eval",
                    action="store_true",
                    help="Evaluate mode.")
parser.add_argument("--n_samples",
                    type=int,
                    default=-1,
                    help="Total samples from tr_id to process.")
parser.add_argument("--tfr_dir_name",
                    type=str,
                    default="PDX_FIXED_RSP_DRUG_PAIR_0.1_of_tiles",
                    help="Dir name that contains TFRecords that are used for training.")
parser.add_argument("--pred_tfr_dir_name",
                    type=str,
                    default="PDX_FIXED_RSP_DRUG_PAIR",
                    help="Dir name that contains TFRecords that are used for prediction/evaluation.")
parser.add_argument("--scale_fea",
                    action="store_true",
                    help="Scale features.")
parser.add_argument("--use_tile",
                    action="store_true",
                    help="Use tiles.")
parser.add_argument("--use_ge",
                    action="store_true",
                    help="Use gene expression.")
parser.add_argument("--use_dd1",
                    action="store_true",
                    help="Use drug descriptors for drug 1.")
parser.add_argument("--use_dd2",
                    action="store_true",
                    help="Use drug descriptors for drug 2.")

args, other_args = parser.parse_known_args()
pprint(args)

split_on = "none" if args.split_on is (None or "none") else args.split_on


# Create project dir (if it doesn't exist)
# import ipdb; ipdb.set_trace()
prjdir = cfg.MAIN_PRJDIR/args.prjname
os.makedirs(prjdir, exist_ok=True)


# Create outdir (using the loaded hyperparamters) or
# use content (model) from an existing run
fea_strs = ["use_tile", "use_ge", "use_dd1", "use_dd2"]
args_dict = vars(args)
fea_names = "_".join([k.split("use_")[-1] for k in fea_strs if args_dict[k] is True])
prm_file_path = prjdir/f"params_{fea_names}.json"
if prm_file_path.exists() is False:
    shutil.copy(fdir/f"../default_params/default_params_{fea_names}.json", prm_file_path)
params = Params(prm_file_path)

if args.rundir is not None:
    outdir = Path(args.rundir).resolve()
    assert outdir.exists(), f"The {outdir} doen't exist."
    print_fn = print
else:
    outdir = create_outdir_2(prjdir, args)

    # Save hyper-parameters
    params.save(outdir/"params.json")

    # Logger
    lg = Logger(outdir/"logger.log")
    print_fn = get_print_func(lg.logger)
    print_fn(f"File path: {fdir}")
    print_fn(f"\n{pformat(vars(args))}")


# Load dataframe (annotations)
annotations_file = cfg.DATA_PROCESSED_DIR/args.dataname/cfg.SF_ANNOTATIONS_FILENAME
dtype = {"image_id": str, "slide": str}
data = pd.read_csv(annotations_file, dtype=dtype, engine="c", na_values=["na", "NaN"])
# data = data.astype({"image_id": str, "slide": str})
print_fn(data.shape)


print_fn("\nFull dataset:")
if args.target[0] == "Response":
    print_groupby_stat(data, print_fn=print_fn)
else:
    print_fn(data[args.target[0]].value_counts())


# Determine tfr_dir (the path to TFRecords)
tfr_dir = (cfg.DATADIR/args.tfr_dir_name).resolve()
pred_tfr_dir = (cfg.DATADIR/args.pred_tfr_dir_name).resolve()
label = f"{params.tile_px}px_{params.tile_um}um"
tfr_dir = tfr_dir/label
pred_tfr_dir = pred_tfr_dir/label

# Create outcomes (for drug response)
# outcomes = {}
# unique_outcomes = list(set(data[args.target[0]].values))
# unique_outcomes.sort()
# for smp, o in zip(data[args.id_name], data[args.target[0]]):
#     outcomes[smp] = {"outcome": unique_outcomes.index(o)}


# Scalers for each feature set
# import ipdb; ipdb.set_trace()
ge_scaler, dd1_scaler, dd2_scaler = None, None, None

ge_cols  = [c for c in data.columns if c.startswith("ge_")]
dd1_cols = [c for c in data.columns if c.startswith("dd1_")]
dd2_cols = [c for c in data.columns if c.startswith("dd2_")]

if args.scale_fea:
    if args.use_ge and len(ge_cols) > 0:
        ge_scaler = get_scaler(data[ge_cols])
    if args.use_dd1 and len(dd1_cols) > 0:
        dd1_scaler = get_scaler(data[dd1_cols])
    if args.use_dd2 and len(dd2_cols) > 0:
        dd2_scaler = get_scaler(data[dd2_cols])


# Create manifest
# print_fn("\nCreate/load manifest ...")
# timer = Timer()
# manifest = create_manifest(directory=tfr_dir, n_files=None)
# timer.display_timer(print_fn)


# -----------------------------------------------
# Data splits
# -----------------------------------------------

# --------------
# Yitan's splits
# --------------
# import ipdb; ipdb.set_trace()
if args.use_dd1 is False and args.use_dd2 is False:
    splitdir = cfg.DATADIR/"PDX_Transfer_Learning_Classification/Processed_Data/Data_For_MultiModal_Learning/Data_Partition_Drug_Specific"
    splitdir = splitdir/params.drug_specific
else:
    splitdir = cfg.DATADIR/"PDX_Transfer_Learning_Classification/Processed_Data/Data_For_MultiModal_Learning/Data_Partition"

tr_id = cast_list(read_lines(str(splitdir/f"cv_{args.split_id}"/"TrainList.txt")), int)
vl_id = cast_list(read_lines(str(splitdir/f"cv_{args.split_id}"/"ValList.txt")), int)
te_id = cast_list(read_lines(str(splitdir/f"cv_{args.split_id}"/"TestList.txt")), int)

# Update ids
index_col_name = "index"
tr_id = sorted(set(data[index_col_name]).intersection(set(tr_id)))
vl_id = sorted(set(data[index_col_name]).intersection(set(vl_id)))
te_id = sorted(set(data[index_col_name]).intersection(set(te_id)))

# Subsample train samples
if args.n_samples > 0:
    if args.n_samples < len(tr_id):
        tr_id = tr_id[:args.n_samples]
    if args.n_samples < len(vl_id):
        vl_id = vl_id[:args.n_samples]
    if args.n_samples < len(te_id):
        te_id = te_id[:args.n_samples]


# --------------
# TidyData
# --------------
# TODO: finish and test this class
# td = TidyData(data,
#               ge_prfx="ge_",
#               dd1_prfx="dd1_",
#               dd2_prfx="dd2_",
#               index_col_name="index",
#               split_ids={"tr_id": tr_id, "vl_id": vl_id, "te_id": te_id}
# )
# ge_scaler = td.ge_scaler
# dd1_scaler = td.dd1_scaler
# dd2_scaler = td.dd2_scaler

# tr_meta = td.tr_meta
# vl_meta = td.vl_meta
# te_meta = td.te_meta
# tr_meta.to_csv(outdir/"tr_meta.csv", index=False)
# vl_meta.to_csv(outdir/"vl_meta.csv", index=False)
# te_meta.to_csv(outdir/"te_meta.csv", index=False)

# # Variables (dict/dataframes/arrays) that are passed as features to the NN
# xtr = {"ge_data": td.tr_ge.values, "dd1_data": td.tr_dd1.values, "dd2_data": td.tr_dd2.values}
# xvl = {"ge_data": td.vl_ge.values, "dd1_data": td.vl_dd1.values, "dd2_data": td.vl_dd2.values}
# xte = {"ge_data": td.te_ge.values, "dd1_data": td.te_dd1.values, "dd2_data": td.te_dd2.values}

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

print_fn("\nTrain:")
print_groupby_stat(tr_meta, print_fn=print_fn)
print_fn("\nValidation:")
print_groupby_stat(vl_meta, print_fn=print_fn)
print_fn("\nTest:")
print_groupby_stat(te_meta, print_fn=print_fn)

# Make sure indices do not overlap
assert len( set(tr_id).intersection(set(vl_id)) ) == 0, "Overlapping indices btw tr and vl"
assert len( set(tr_id).intersection(set(te_id)) ) == 0, "Overlapping indices btw tr and te"
assert len( set(vl_id).intersection(set(te_id)) ) == 0, "Overlapping indices btw vl and te"

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
# test_tfr_files = get_tfr_files(tfr_dir, te_smp_names)
test_tfr_files = get_tfr_files(pred_tfr_dir, te_smp_names)
print("Total samples {}".format(len(train_tfr_files) + len(val_tfr_files) + len(test_tfr_files)))

# Missing tfrecords
print("\nThese samples miss a tfrecord:")
df_miss = data.loc[~data[args.id_name].isin(tr_smp_names + vl_smp_names + te_smp_names), ["smp", "image_id"]]
print(df_miss)

assert sorted(tr_smp_names) == sorted(tr_meta[args.id_name].values.tolist()), "Sample names in the tr_smp_names and tr_meta don't match."
assert sorted(vl_smp_names) == sorted(vl_meta[args.id_name].values.tolist()), "Sample names in the vl_smp_names and vl_meta don't match."
assert sorted(te_smp_names) == sorted(te_meta[args.id_name].values.tolist()), "Sample names in the te_smp_names and te_meta don't match."


# -------------------------------
# Class weight
# -------------------------------
# import ipdb; ipdb.set_trace()
tile_cnts = pd.read_csv(tfr_dir/"tile_counts_per_slide.csv")
tile_cnts.insert(loc=0, column="tfr_abs_fname", value=tile_cnts["tfr_fname"].map(lambda s: str(tfr_dir/s)))
cat = tile_cnts[tile_cnts["tfr_abs_fname"].isin(train_tfr_files)]
cat = cat.groupby(args.target[0]).agg({"smp": "nunique", "max_tiles": "sum", "n_tiles": "sum", "slide": "nunique"}).reset_index()
categories = {}
for i, row_data in cat.iterrows():
    dct = {"num_samples": row_data["smp"], "num_tiles": row_data["n_tiles"]}
    categories[row_data[args.target[0]]] = dct

class_weight = calc_class_weights(train_tfr_files,
                                  class_weights_method=params.class_weights_method,
                                  categories=categories)
# class_weight = {"Response": class_weight}


# --------------------------
# Build tf.data objects
# --------------------------
# import ipdb; ipdb.set_trace()
if args.use_tile:

    # -------------------------------
    # Parsing funcs
    # -------------------------------
    # import ipdb; ipdb.set_trace()
    if args.target[0] == "Response":
        # Response
        parse_fn = parse_tfrec_fn_rsp
        parse_fn_train_kwargs = {
            "use_tile": args.use_tile,
            "use_ge": args.use_ge,
            "use_dd1": args.use_dd1,
            "use_dd2": args.use_dd2,
            "ge_scaler": ge_scaler,
            "dd1_scaler": dd1_scaler,
            "dd2_scaler": dd2_scaler,
            "id_name": args.id_name,
            "augment": params.augment,
            "application": params.base_image_model,
            # "application": None,
        }
    else:
        # Ctype
        parse_fn = parse_tfrec_fn_rna
        parse_fn_train_kwargs = {
            'use_tile': args.use_tile,
            'use_ge': args.use_ge,
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
    tr_tiles = tile_cnts[tile_cnts[args.id_name].isin(tr_smp_names)]["n_tiles"].sum()
    vl_tiles = tile_cnts[tile_cnts[args.id_name].isin(vl_smp_names)]["n_tiles"].sum()
    te_tiles = tile_cnts[tile_cnts[args.id_name].isin(te_smp_names)]["n_tiles"].sum()

    eval_batch_size = 4 * params.batch_size
    tr_steps = tr_tiles // params.batch_size
    vl_steps = vl_tiles // eval_batch_size
    # te_steps = te_tiles // eval_batch_size

    # -------------------------------
    # Create TF datasets
    # -------------------------------
    print("\nCreating TF datasets.")

    # Training
    # import ipdb; ipdb.set_trace()
    train_data = create_tf_data(
        batch_size=params.batch_size,
        deterministic=False,
        include_meta=False,
        interleave=True,
        n_concurrent_shards=params.n_concurrent_shards,  # 32, 64
        parse_fn=parse_fn,
        prefetch=1,  # 2
        repeat=True,
        seed=None,  # cfg.seed,
        shuffle_files=True,
        shuffle_size=params.shuffle_size,  # 8192
        tfrecords=train_tfr_files,
        **parse_fn_train_kwargs)

    # Determine feature shapes from data
    bb = next(train_data.__iter__())

    # Infer dims of features from the data
    # import ipdb; ipdb.set_trace()
    if args.use_ge:
        ge_shape = bb[0]["ge_data"].numpy().shape[1:]
    else:
        ge_shape = None

    if args.use_dd1:
        dd_shape = bb[0]["dd1_data"].numpy().shape[1:]
    else:
        dd_shape = None

    # Print keys and dims
    for i, item in enumerate(bb):
        print(f"\nItem {i}")
        if isinstance(item, dict):
            for k in item.keys():
                print(f"\t{k}: {item[k].numpy().shape}")
        elif isinstance(item.numpy(), np.ndarray):
            print(item)

    # for i, rec in enumerate(train_data.take(2)):
    #     tf.print(rec[1])

    # Evaluation (val, test, train)
    create_tf_data_eval_kwargs = {
        "batch_size": eval_batch_size,
        "include_meta": False,
        "interleave": False,
        "parse_fn": parse_fn,
        "prefetch": None,  # 2
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


# ----------------------
# Prep for training
# ----------------------
# import ipdb; ipdb.set_trace()

# Loss and target
if args.use_tile:
    loss = losses.BinaryCrossentropy(label_smoothing=params.label_smoothing)
else:
    if params.y_encoding == "onehot":
        if index_col_name in data.columns:
            # Using Yitan's T/V/E splits
            # print(te_meta[["index", "Group", "grp_name", "Response"]])
            ytr = pd.get_dummies(tr_meta[args.target[0]].values)
            yvl = pd.get_dummies(vl_meta[args.target[0]].values)
            yte = pd.get_dummies(te_meta[args.target[0]].values)
        else:
            ytr = y_onehot.iloc[tr_id, :].reset_index(drop=True)
            yvl = y_onehot.iloc[vl_id, :].reset_index(drop=True)
            yte = y_onehot.iloc[te_id, :].reset_index(drop=True)

        loss = losses.CategoricalCrossentropy()

    elif params.y_encoding == "label":
        if index_col_name in data.columns:
            # Using Yitan's T/V/E splits
            ytr = tr_meta[args.target[0]].values
            yvl = vl_meta[args.target[0]].values
            yte = te_meta[args.target[0]].values
            loss = losses.BinaryCrossentropy(label_smoothing=params.label_smoothing)
        else:
            ytr = ydata_label[tr_id]
            yvl = ydata_label[vl_id]
            yte = ydata_label[te_id]
            loss = losses.SparseCategoricalCrossentropy()
    else:
        raise ValueError(f"Unknown value for y_encoding ({params.y_encoding}).")


# -------------
# Train model
# -------------
model = None

# import ipdb; ipdb.set_trace()
if args.train is True:

    # ----------------------
    # Callbacks
    # ----------------------
    class BatchCSVLogger(tf.keras.callbacks.Callback):
        """ Write training logs on every batch. """
        def __init__(self,
                     filename,
                     # val_filename=None,
                     validate_on_batch=None,
                     validation_data=None,
                     validation_steps=None):
            """ ... """
            super(BatchCSVLogger, self).__init__()
            self.filename = filename
            # self.val_filename = val_filename
            self.validate_on_batch = validate_on_batch
            self.validation_data = validation_data
            self.validation_steps = validation_steps

        def on_train_begin(self, logs=None):
            self.csv_file = open(self.filename, "w")
            # self.val_csv_file = open(self.val_filename, "w")
            self.epoch = 0
            self.step = 0  # global batch
            self.results = []
            
        def on_epoch_begin(self, epoch, logs=None):
            keys = list(logs.keys())
            self.epoch = epoch + 1

        def on_train_batch_begin(self, batch, logs=None):
            keys = list(logs.keys())
            self.step += 1

        # def on_test_batch_end(self, batch, logs=None):
        #     keys = list(logs.keys())
        #     batch = batch + 1
        #     res = OrderedDict({"step": self.step, "epoch": self.epoch, "batch": batch})
        #     res.update(logs)  # logs contains the metrics

        #     if self.step == 1:
        #         # keys = list(logs.keys())
        #         # val_keys = ["val_"+str(k) for k in keys]
        #         # fieldnames = ["step", "epoch", "batch"] + keys + val_keys + ["lr"]
        #         fieldnames = ["step", "epoch", "batch"] + val_keys + ["lr"]
        #         self.fieldnames = fieldnames

        #         # self.csv_file = open(self.filename, "w")
        #         self.val_writer = csv.DictWriter(self.val_csv_file, fieldnames=self.fieldnames)
        #         self.val_writer.writeheader()
        #         self.val_csv_file.flush()

        #     # Get the current learning rate from model's optimizer
        #     lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        #     res.update({"lr": lr})
        #     self.val_writer.writerow(res)
        #     self.val_csv_file.flush()

        # def on_train_batch_end(self, batch, logs=None):
        #     keys = list(logs.keys())
        #     batch = batch + 1
        #     # self.step += 1
        #     res = OrderedDict({"step": self.step, "epoch": self.epoch, "batch": batch})
        #     res.update(logs)  # logs contains the metrics for the training set

        #     # if (self.validate_on_batch is not None) and (batch % self.validate_on_batch == 0):
        #     #     evals = self.model.evaluate(self.validation_data, verbose=0, steps=self.validation_steps)
        #     #     val_logs = {"val_"+str(k): v for k, v in zip(keys, evals)}
        #     #     res.update(val_logs)
        #     # else:
        #     #     val_logs = {"val_"+str(k): np.nan for k in keys}
        #     #     res.update(val_logs)

        #     if self.step == 1:
        #         # keys = list(logs.keys())
        #         # val_keys = ["val_"+str(k) for k in keys]
        #         # fieldnames = ["step", "epoch", "batch"] + keys + val_keys + ["lr"]
        #         fieldnames = ["step", "epoch", "batch"] + keys + ["lr"]
        #         self.fieldnames = fieldnames

        #         # self.csv_file = open(self.filename, "w")
        #         self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        #         self.writer.writeheader()
        #         self.csv_file.flush()

        #     # Get the current learning rate from model's optimizer
        #     lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        #     res.update({"lr": lr})
        #     self.writer.writerow(res)
        #     self.csv_file.flush()

        def on_train_batch_end(self, batch, logs={}):
            keys = list(logs.keys())
            batch = batch + 1
            # self.step += 1
            res = OrderedDict({"step": self.step, "epoch": self.epoch, "batch": batch})
            res.update(logs)  # logs contains the metrics for the training set

            if (self.validate_on_batch is not None) and (batch % self.validate_on_batch == 0):
                evals = self.model.evaluate(self.validation_data, verbose=0, steps=self.validation_steps)
                val_logs = {"val_"+str(k): v for k, v in zip(keys, evals)}
                res.update(val_logs)
            else:
                val_logs = {"val_"+str(k): np.nan for k in keys}
                res.update(val_logs)

            if self.step == 1:
                # keys = list(logs.keys())
                val_keys = ["val_"+str(k) for k in keys]
                fieldnames = ["step", "epoch", "batch"] + keys + val_keys + ["lr"]
                self.fieldnames = fieldnames

                self.csv_file = open(self.filename, "w")
                self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
                self.writer.writeheader()
                self.csv_file.flush()

            # Get the current learning rate from model's optimizer
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            res.update({"lr": lr})
            self.writer.writerow(res)
            self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()
            # self.val_csv_file.close()

    class BatchEarlyStopping(tf.keras.callbacks.Callback):
        """
        https://stackoverflow.com/questions/57618220
        BatchEarlyStopping(monitor="loss", batch_patience=20, validate_on_batch=10)
        """
        def __init__(self,
                     validation_data,
                     validation_steps=None,
                     validate_on_batch=100,
                     monitor="loss",
                     batch_patience=0,
                     print_fn=print):
            """ ... """
            super(BatchEarlyStopping, self).__init__()
            self.batch_patience = batch_patience
            self.best_weights = None
            # self.monitor = monitor  # ap
            self.validate_on_batch = validate_on_batch
            self.validation_data = validation_data
            self.validation_steps = validation_steps
            self.print_fn = print_fn

        def on_train_begin(self, logs=None):
            # self.wait = 0           # number of batches it has waited when loss is no longer minimum
            self.stopped_epoch = 0  # epoch the training stops at
            self.stopped_batch = 0  # epoch the training stops at
            self.best = np.Inf      # init the best as infinity
            self.val_loss = np.Inf
            self.epoch = 0
            self.step_id = 0  # global batch
            # self.print_fn("\n{}.".format(yellow("Start training")))

        def on_epoch_begin(self, epoch, logs=None):
            keys = list(logs.keys())
            self.epoch = epoch + 1
            self.wait = 0  # number of batches it has waited when loss is no longer minimum
            # if self.epoch == 1:
            #     self.wait = 0  # number of batches it has waited when loss is no longer minimum
            # self.print_fn("\n{} {}.\n".format( yellow("Start epoch"), yellow(epoch)) )

        def on_epoch_end(self, epoch, logs=None):
            keys = list(logs.keys())
            # outpath = str(outdir/f"model_at_epoch_{self.epoch}")
            # self.model.save(outpath)
            self.print_fn("")

            # In case there are remaining batches at the of epoch that were not evaluated in self.validate_on_batch
            evals = self.model.evaluate(self.validation_data, verbose=0, steps=self.validation_steps)
            current = evals[0]

            if np.less(current, self.best):
                self.best = current
                self.wait = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1

        def on_train_batch_end(self, batch, logs=None):
            keys = list(logs.keys())  # metrics/logs names
            batch = batch + 1

            self.step_id += 1
            res = OrderedDict({"step_id": self.step_id, "epoch": self.epoch, "batch": batch})
            res.update(logs)

            color = yellow if self.epoch == 1 else red

            if batch % self.validate_on_batch == 0:
                # Log metrics before evaluation
                self.print_fn("\repoch: {}, batch: {}/{}, loss: {:.4f}, val_loss: {:.4f}, best_val_loss: {:.4f} (wait: {})".format(
                    self.epoch, batch, tr_steps, logs["loss"], self.val_loss, self.best, color(self.wait)), end="\r")

                evals = self.model.evaluate(self.validation_data, verbose=0, steps=self.validation_steps)
                val_logs = {"val_"+str(k): v for k, v in zip(keys, evals)}
                res.update(val_logs)

                self.val_loss = evals[0]
                # current = logs.get("loss")
                # current = logs.get(self.monitor)
                current = self.val_loss

                if np.less(current, self.best):
                    self.best = current
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1

                # Log metrics after evaluation
                self.print_fn("\repoch: {}, batch: {}/{}, loss: {:.4f}, val_loss: {:.4f}, best_val_loss: {:.4f} (wait: {})".format(
                    self.epoch, batch, tr_steps, logs["loss"], self.val_loss, self.best, color(self.wait)), end="\r")

                # Don't terminate on the first epoch
                if (self.wait >= self.batch_patience) and (self.epoch > 1):
                    self.stopped_epoch = self.epoch
                    self.stopped_batch = batch
                    self.model.stop_training = True
                    self.print_fn("\n{}".format(red("Terminate training")))
                    self.print_fn("Restores model weights from the best epoch-batch set.")
                    self.model.set_weights(self.best_weights)

            else:
                self.print_fn("\repoch: {}, batch: {}/{}, loss: {:.4f}, val_loss: {:.4f}, best_val_loss: {:.4f} (wait: {})".format(
                    self.epoch, batch, tr_steps, logs["loss"], self.val_loss, self.best, color(self.wait)), end="\r")
                val_logs = {"val_"+str(k): np.nan for k in keys}
                res.update(val_logs)

            # Get the current learning rate from model's optimizer.
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            res.update({"lr": lr})
            results.append(res)

        def on_train_end(self, logs=None):
            if self.stopped_batch > 0:
                self.print_fn("Early stop on epoch {} and batch {}.".format(self.stopped_epoch, self.stopped_batch))

    def get_lr_callback(batch_size=8):
        """
        https://www.kaggle.com/cdeotte/triple-stratified-kfold-with-tfrecords
        """
        lr_start   = 0.000005
        lr_max     = 0.00000125 * batch_size
        lr_min     = 0.000001
        lr_ramp_ep = 5
        lr_sus_ep  = 0
        lr_decay   = 0.8

        def lrfn(epoch):
            if epoch < lr_ramp_ep:
                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

            elif epoch < lr_ramp_ep + lr_sus_ep:
                lr = lr_max

            else:
                lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

            return lr

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
        return lr_callback

    # Callbacks list
    monitor = "val_loss"
    # monitor = "val_pr-auc"
    callbacks = keras_callbacks(outdir, monitor=monitor, patience=params.patience)
    # callbacks = keras_callbacks(outdir, monitor="auc", patience=params.patience)

    # Learning rate schedule
    # callbacks.append(get_lr_callback(batch_size=params.batch_size))

    if args.use_tile:
        # callbacks = []
        results = []

        # callbacks.append(BatchEarlyStopping(validation_data=val_data,
        #                                   validate_on_batch=params.validate_on_batch,
        #                                   batch_patience=params.batch_patience,
        #                                   print_fn=print)); fit_verbose=0

        # callbacks = keras_callbacks(outdir, monitor="val_loss"); fit_verbose=1
        fit_verbose = 1

        # callbacks.append(BatchCSVLogger(filename=outdir/"batch_training.log", 
        #                                 # val_filename=outdir/"val_batch_training.log", 
        #                                 validate_on_batch=params.validate_on_batch,
        #                                 validation_data=val_data))
    else:
        fit_verbose = 1

    # Mixed precision
    if params.use_fp16:
        print_fn("\nTrain with mixed precision")
        if int(tf.keras.__version__.split(".")[1]) == 4:  # TF 2.4
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_global_policy(policy)
        elif int(tf.keras.__version__.split(".")[1]) == 3:  # TF 2.3
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        print_fn("Compute dtype: %s" % policy.compute_dtype)
        print_fn("Variable dtype: %s" % policy.variable_dtype)

    # ----------------------
    # Output bias
    # ----------------------
    # import ipdb; ipdb.set_trace()
    print_fn("\nCompute the bias of the NN output.")

    # Calc output bias
    if args.use_tile:
        # from sf_utils import get_categories_from_manifest
        # categories = get_categories_from_manifest(train_tfr_files, manifest, outcomes)
        neg = categories[0]["num_tiles"]
        pos = categories[1]["num_tiles"]
    else:
        neg, pos = np.bincount(tr_meta[args.target[0]].values)

    total = neg + pos
    print_fn("Samples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(total, pos, 100 * pos / total))
    output_bias = np.log([pos/neg])
    print_fn(f"Output bias: {output_bias}")
    # output_bias = None

    # ----------------------
    # Define model
    # ----------------------
    # import ipdb; ipdb.set_trace()
    if args.target[0] == "Response":
        build_model_kwargs = {"base_image_model": params.base_image_model,
                              "dense1_dd1": params.dense1_dd1,
                              "dense1_dd2": params.dense1_dd2,
                              "dense1_ge": params.dense1_ge,
                              "dense1_img": params.dense1_img,
                              "dense2_img": params.dense2_img,
                              "dense1_top": params.dense1_top,
                              "dd_shape": dd_shape,
                              "ge_shape": ge_shape,
                              "learning_rate": params.learning_rate,
                              "loss": loss,
                              "optimizer": params.optimizer,
                              "output_bias": output_bias,
                              "dropout1_top": params.dropout1_top,
                              "use_dd1": args.use_dd1,
                              "use_dd2": args.use_dd2,
                              "use_ge": args.use_ge,
                              "use_tile": args.use_tile,
                              # "model_type": params.model_type,
        }
        model = build_model_rsp(**build_model_kwargs)
    else:
        raise NotImplementedError("Need to check this method")
        model = build_model_rna()

    # import ipdb; ipdb.set_trace()
    print_fn("")
    model.summary(print_fn=print_fn)

    # import ipdb; ipdb.set_trace()
    # steps = 40
    # res = model.evaluate(train_data, steps=tr_steps, verbose=1)
    # print("Loss: {:0.4f}".format(res[0]))


    # Initial model
    print_fn("")
    # initial_model_path = prjdir/f"base_{args.nn_arch}_model"
    fea_types_str = fea_types_to_str_name(args)
    initial_model_path = prjdir/f"initial_model_{fea_types_str}"
    if initial_model_path.exists():
        print_fn("Loading initial model (from the global dir).")
        model = tf.keras.models.load_model(initial_model_path)
    else:
        print_fn("Saving initial model (to the global dir).")
        model.save(initial_model_path)

    # # Save initial model to the outdir for reference
    # print_fn("Saving initial model (to the current run dir).")
    # model.save(outdir/initial_model_path.name)

    # # Test: confirm that the loaded model has the same performance
    # initial_model_path = prjdir/"base_multimodal_model"
    # model.save(initial_model_path)
    # model1 = tf.keras.models.load_model(initial_model_path)
    # res = model.evaluate(val_data, steps=vl_steps, verbose=1)
    # res1 = model.evaluate(val_data, steps=vl_steps, verbose=1)

    # # Base model weights
    # weights_path = prjdir/"base_multimodal_weights"
    # initial_weights = weights_path/"initial_weights"
    # if weights_path.exists():
    #     model.load_weights(initial_weights)
    # else:
    #     model.save_weights(initial_weights)

    # # Dump/log performance measures of the base model
    # aa = model.evaluate(val_data, steps=vl_steps, verbose=1)
    # print_fn("Base model val_loss: {}".format(aa[0]))

    timer = Timer()

    if args.use_tile is True:
        print_fn(f"Train steps:      {tr_steps}")
        print_fn(f"Validation steps: {vl_steps}")

        history = model.fit(x=train_data,
                            validation_data=val_data,
                            steps_per_epoch=tr_steps,
                            validation_steps=vl_steps,
                            class_weight=class_weight,
                            epochs=params.epochs,
                            verbose=fit_verbose,
                            callbacks=callbacks)
        del train_data, val_data

        # import ipdb; ipdb.set_trace()
        # res = pd.DataFrame(results)
        # res.to_csv(outdir/"results.csv", index=False)
    else:
        xtr = {"ge_data": tr_ge.values, "dd1_data": tr_dd1.values, "dd2_data": tr_dd2.values}
        xvl = {"ge_data": vl_ge.values, "dd1_data": vl_dd2.values, "dd2_data": vl_dd2.values}
        xte = {"ge_data": te_ge.values, "dd1_data": te_dd1.values, "dd2_data": te_dd2.values}
        # ytr = {"Response": ytr}
        # yvl = {"Response": yvl}
        # yte = {"Response": yte}

        history = model.fit(x=xtr,
                            y=ytr,
                            validation_data=(xvl, yvl),
                            class_weight=class_weight,
                            epochs=params.epochs,
                            shuffle=True,
                            verbose=1,
                            callbacks=callbacks)
    print_fn("")
    timer.display_timer(print_fn)

    # Save final model
    final_model_fpath = outdir/f"final_model.ckpt"
    model.save(final_model_fpath)

if args.eval is True:
    if model is None:
        model = load_best_model(outdir)

    # import ipdb; ipdb.set_trace()
    if args.use_tile:
        create_tf_data_eval_kwargs.update({"tfrecords": test_tfr_files, "include_meta": True})
        test_data = create_tf_data(
            **create_tf_data_eval_kwargs,
            **parse_fn_non_train_kwargs
        )

        # create_tf_data_eval_kwargs.update({"tfrecords": val_tfr_files, "include_meta": True})
        # eval_val_data = create_tf_data(
        #     **create_tf_data_eval_kwargs,
        #     **parse_fn_non_train_kwargs
        # )

        # create_tf_data_eval_kwargs.update({"tfrecords": train_tfr_files, "include_meta": True})
        # eval_train_data = create_tf_data(
        #     **create_tf_data_eval_kwargs,
        #     **parse_fn_non_train_kwargs
        # )

        timer = Timer()

        print_fn("\n{}".format(bold("Test set predictions.")))
        calc_tf_preds(test_data, te_meta, model, outdir, args, name="test", print_fn=print_fn)
        del test_data

        # print_fn("\n{}".format(bold("Validation set predictions.")))
        # calc_tf_preds(eval_val_data, vl_meta, model, outdir, args, name="val", print_fn=print_fn)
        # del eval_val_data

        # print_fn("\n{}".format(bold("Train set predictions.")))
        # calc_tf_preds(eval_train_data, tr_meta, model, outdir, args, name="train", print_fn=print_fn)
        # del eval_train_data

        timer.display_timer(print_fn)

    else:
        # import ipdb; ipdb.set_trace()
        print_fn("\n{}".format(bold("Keras NN.")))
        print_fn("Test:")
        calc_smp_preds(xdata=xte, meta=te_meta, model=model, outdir=outdir, name="test_keras", print_fn=print_fn)
        print_fn("\nVal:")
        calc_smp_preds(xdata=xvl, meta=vl_meta, model=model, outdir=outdir, name="val_keras", print_fn=print_fn)
        print_fn("\nTrain:")
        calc_smp_preds(xdata=xtr, meta=tr_meta, model=model, outdir=outdir, name="train_keras", print_fn=print_fn)


# --------------------------
# Lightgbm baseline
# --------------------------
if args.use_tile is False:
    # import ipdb; ipdb.set_trace()
    import lightgbm as lgb
    # from joblib import Parallel, delayed
    lgb_cls_init_kwargs = {"boosting_type": "gbdt",
                           "num_leaves": 31,
                           "max_depth": -1,
                           "learning_rate": 0.1,
                           "n_estimators": 5000,
                           "n_jobs": 8,
                           "objective": "binary",
                           "class_weight": "balanced"}
    lgb_cls = lgb.LGBMClassifier(**lgb_cls_init_kwargs)
    xtr_lgb = pd.concat([tr_ge, tr_dd1, tr_dd2], axis=1)
    xvl_lgb = pd.concat([vl_ge, vl_dd1, vl_dd2], axis=1)
    xte_lgb = pd.concat([te_ge, te_dd1, te_dd2], axis=1)
    # https://stackoverflow.com/questions/60582050
    lgb_cls.fit(xtr_lgb.values, ytr,
                eval_set=(xvl_lgb.values, yvl),
                early_stopping_rounds=200)

    print_fn("\n{}".format(bold("LightGBM Classifier.")))
    print_fn("Test:")
    calc_smp_preds(xdata=xte_lgb, meta=te_meta, model=lgb_cls, outdir=outdir, name="test_lgb", print_fn=print_fn)
    print_fn("\nVal:")
    calc_smp_preds(xdata=xvl_lgb, meta=vl_meta, model=lgb_cls, outdir=outdir, name="val_lgb", print_fn=print_fn)
    print_fn("\nTrain:")
    calc_smp_preds(xdata=xtr_lgb, meta=tr_meta, model=lgb_cls, outdir=outdir, name="train_lgb", print_fn=print_fn)


print_fn("\nDone.")
