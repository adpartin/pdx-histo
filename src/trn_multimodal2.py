"""
Prediction of drug response with TFRecords.
"""
import os
import sys
assert sys.version_info >= (3, 5)

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
from src.models import build_model_rsp, build_model_rsp_baseline, keras_callbacks
from src.ml.scale import get_scaler
from src.ml.evals import calc_scores, calc_preds, dump_preds, save_confusion_matrix
from src.ml.keras_utils import plot_prfrm_metrics
from src.utils.classlogger import Logger
from src.utils.utils import (cast_list, create_outdir, create_outdir_2, dump_dict, get_print_func,
                             read_lines, Params, Timer)
from src.datasets.tidy import split_data_and_extract_fea, extract_fea
from src.tf_utils import get_tfr_files, calc_records_in_tfr_files, count_data_items
from src.sf_utils import (green, interleave_tfrecords, create_tf_data, calc_class_weights,
                          parse_tfrec_fn_rsp, parse_tfrec_fn_rna,
                          create_manifest)
from src.sf_utils import bold, green, blue, yellow, cyan, red

# Seed
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
                    default=None,
                    choices=["Sample", "slide", "Group"],
                    help="Specify the hard split variable/column (default: None).")
parser.add_argument("--prjname",
                    type=str,
                    help="Project name (folder that contains the annotations.csv dataframe).")
parser.add_argument("--dataname",
                    type=str,
                    default="tidy_all",
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
                    choices=["baseline", "multimodal"],
                    help="NN architecture (default: multimodal).")

args, other_args = parser.parse_known_args()
pprint(args)


# Create outdir
import ipdb; ipdb.set_trace()
prjdir = cfg.MAIN_PRJDIR/args.prjname
split_on = "none" if args.split_on is (None or "none") else args.split_on
# version 1
# outdir = prjdir/f"multimodal/split_on_{split_on}"
# os.makedirs(outdir, exist_ok=True)
# version 2
# base_outdir = prjdir/f"multimodal"
# outdir = create_outdir(base_outdir)
# version 3
# prm_file_path = prjdir/"params.json"
# if prm_file_path.exists() is False:
#     shutil.copy(fdir/"../default_params/default_params_multimodal.json", prm_file_path)
os.makedirs(prjdir, exist_ok=True)
prm_file_path = prjdir/f"params_{args.nn_arch}.json"
if prm_file_path.exists() is False:
    shutil.copy(fdir/f"../default_params/default_params_{args.nn_arch}.json", prm_file_path)
params = Params(prm_file_path)
outdir = create_outdir_2(prjdir, params)


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


# Import hyper-parameters
# import ipdb; ipdb.set_trace()
# prm_file_path = prjdir/"multimodal/params.json"
# if prm_file_path.exists() is False:
#     shutil.copy(fdir/"../default_params/default_params_multimodal.json", prm_file_path)
# params = Params(prm_file_path)
params.save(outdir/"params.json")


# import ipdb; ipdb.set_trace()
print("\nFull dataset:")
if args.target[0] == "Response":
    print_fn(data.groupby(["ctype", "Response"]).agg({split_on: "nunique", "smp": "nunique"}).reset_index().rename(
        columns={split_on: f"{split_on}_unq", "smp": "smp_unq"}))
else:
    print_fn(data[args.target[0]].value_counts())


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


# Determine tfr_dir and the data parsing funcs
if args.target[0] == "Response":
    if params.single_drug:
        tfr_dir = cfg.SF_TFR_DIR_RSP
    else:
        tfr_dir = cfg.SF_TFR_DIR_RSP_DRUG_PAIR

elif args.target[0] == "ctype":
    tfr_dir = cfg.SF_TFR_DIR_RNA_NEW

label = f"{params.tile_px}px_{params.tile_um}um"
tfr_dir = tfr_dir/label


# Create outcomes (for drug response)
outcomes = {}
unique_outcomes = list(set(data[args.target[0]].values))
unique_outcomes.sort()

for smp, o in zip(data[args.id_name], data[args.target[0]]):
    outcomes[smp] = {"outcome": unique_outcomes.index(o)}


# Create manifest
print_fn("\nCreate/load manifest ...")
timer = Timer()
manifest = create_manifest(directory=tfr_dir, n_files=None)
timer.display_timer(print_fn)


# -----------------------------------------------
# Data splits
# -----------------------------------------------

# --------------
# My (ap) splits
# --------------
# splitdir = cfg.DATA_PROCESSED_DIR/args.dataname/f'annotations.splits/split_on_{split_on}'
# split_id = 0

# split_pattern = f'1fold_s{split_id}_*_id.txt'
# single_split_files = glob.glob(str(splitdir/split_pattern))

# assert len(single_split_files) >= 2, f'Split {split_id} contains only one file.'
# for id_file in single_split_files:
#     if 'tr_id' in id_file:
#         tr_id = cast_list(read_lines(id_file), int)
#     elif 'vl_id' in id_file:
#         vl_id = cast_list(read_lines(id_file), int)
#     elif 'te_id' in id_file:
#         te_id = cast_list(read_lines(id_file), int)
# ------------------------------------------------------------

# --------------
# Yitan's splits
# --------------
# import ipdb; ipdb.set_trace()
splitdir = cfg.DATADIR/"PDX_Transfer_Learning_Classification/Processed_Data/Data_For_MultiModal_Learning/Data_Partition"
# split_id = 0
# split_id = 2
split_id = 81

index_col_name = "index"
tr_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"TrainList.txt")), int)
vl_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"ValList.txt")), int)
te_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"TestList.txt")), int)

# Update ids
tr_id = sorted(set(data[index_col_name]).intersection(set(tr_id)))
vl_id = sorted(set(data[index_col_name]).intersection(set(vl_id)))
te_id = sorted(set(data[index_col_name]).intersection(set(te_id)))

if (args.n_samples > 0) and (args.n_samples < len(tr_id)):
    tr_id = tr_id[:args.n_samples]

# import ipdb; ipdb.set_trace()
# vl_id = vl_id[:20]
vl = data[data[index_col_name].isin(vl_id)]
te = data[data[index_col_name].isin(te_id)]
r0 = vl[vl[args.target[0]] == 0]  # non-responders
r1 = vl[vl[args.target[0]] == 1]  # responders

r0 = r0[ r0["ctype"].isin( te["ctype"].unique() ) ]
r0 = r0.sample(n=r1.shape[0])
vl = pd.concat([r0, r1], axis=0)
vl_id = vl["index"].values.tolist()


# ------------------------------------------------------------

# kwargs = {"ge_cols": ge_cols,
#           "dd_cols": dd_cols,
#           "ge_scaler": ge_scaler,
#           "dd_scaler": dd_scaler,
#           "ge_dtype": cfg.GE_DTYPE,
#           "dd_dtype": cfg.DD_DTYPE
# }
# tr_ge, tr_dd = split_data_and_extract_fea(data, ids=tr_id, **kwargs)
# vl_ge, vl_dd = split_data_and_extract_fea(data, ids=vl_id, **kwargs)
# te_ge, te_dd = split_data_and_extract_fea(data, ids=te_id, **kwargs)

# ge_shape = (te_ge.shape[1],)
# dd_shape = (tr_dd.shape[1],)

# # Variables (dict/dataframes/arrays) that are passed as features to the NN
# xtr = {"ge_data": tr_ge.values, "dd_data": tr_dd.values}
# xvl = {"ge_data": vl_ge.values, "dd_data": vl_dd.values}
# xte = {"ge_data": te_ge.values, "dd_data": te_dd.values}

# # Extarct meta for T/V/E
# tr_meta = data.iloc[tr_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
# vl_meta = data.iloc[vl_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
# te_meta = data.iloc[te_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)

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

ge_shape = (tr_ge.shape[1],)
dd_shape = (tr_dd1.shape[1],)

# Variables (dict/dataframes/arrays) that are passed as features to the NN
xtr = {"ge_data": tr_ge.values, "dd1_data": tr_dd1.values, "dd2_data": tr_dd2.values}
xvl = {"ge_data": vl_ge.values, "dd1_data": vl_dd1.values, "dd2_data": vl_dd2.values}
xte = {"ge_data": te_ge.values, "dd1_data": te_dd1.values, "dd2_data": te_dd2.values}

# import ipdb; ipdb.set_trace()
print_fn("\nTrain:")
print_fn(tr_meta.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
print_fn("\nVal:")
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
print(data.loc[~data[args.id_name].isin(tr_smp_names + vl_smp_names + te_smp_names), ["smp", "image_id"]])

ydata = data[args.target[0]].values
ytr = ydata[tr_id]
yvl = ydata[vl_id]
yte = ydata[te_id]

assert sorted(tr_smp_names) == sorted(tr_meta[args.id_name].values.tolist()), "Sample names in the tr_smp_names and tr_meta don't match."
assert sorted(vl_smp_names) == sorted(vl_meta[args.id_name].values.tolist()), "Sample names in the vl_smp_names and vl_meta don't match."
assert sorted(te_smp_names) == sorted(te_meta[args.id_name].values.tolist()), "Sample names in the te_smp_names and te_meta don't match."

# split_outdir = outdir/f'split_{split_id}'
# os.makedirs(split_outdir, exist_ok=True)
# tr_df.to_csv(split_outdir/'dtr.csv', index=False)
# vl_df.to_csv(split_outdir/'dvl.csv', index=False)
# te_df.to_csv(split_outdir/'dte.csv', index=False)

# Number of tiles/examples in each dataset
# import ipdb; ipdb.set_trace()
tile_cnts = pd.read_csv(cfg.SF_TFR_DIR_RSP_DRUG_PAIR/label/"tile_counts_per_slide.csv")
# tr_tile_cnts = tile_cnts.merge(tr_meta[["smp", "Group", "grp_name", "Response"]], on="smp", how="inner")

total_train_tiles = tile_cnts[tile_cnts[args.id_name].isin(tr_smp_names)]["max_tiles"].sum()
total_val_tiles = tile_cnts[tile_cnts[args.id_name].isin(vl_smp_names)]["max_tiles"].sum()
total_test_tiles = tile_cnts[tile_cnts[args.id_name].isin(te_smp_names)]["max_tiles"].sum()

eval_batch_size = 8 * params.batch_size
train_steps = total_train_tiles // params.batch_size
validation_steps = total_val_tiles // eval_batch_size
test_steps = total_test_tiles // eval_batch_size

# import ipdb; ipdb.set_trace()
# print(len(outcomes))
# print(len(manifest))
# print(outcomes[list(outcomes.keys())[3]])
# print(manifest[list(manifest.keys())[3]])


# -------------------------------
# create SlideflowModel model SFM
# -------------------------------
# import ipdb; ipdb.set_trace()

#slide_annotations = outcomes

##SLIDES = list(slide_annotations.keys())
#SAMPLES = list(slide_annotations.keys())

##outcomes_ = [slide_annotations[slide]['outcome'] for slide in SLIDES]
#outcomes_ = [slide_annotations[smp]['outcome'] for smp in SAMPLES]

#if params.model_type == 'categorical':
#    NUM_CLASSES = len(list(set(outcomes_)))  # infer this from other variables

##ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SLIDES, outcomes_), -1)]
#ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SAMPLES, outcomes_), -1)]


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
        # "MODEL_TYPE": params.model_type,
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


# -------------------------------
# Class weight
# -------------------------------
# class_weights_method = "BY_SAMPLE"
class_weights_method = "BY_TILE"
# class_weights_method = "NONE"

# import ipdb; ipdb.set_trace()
class_weight = calc_class_weights(train_tfr_files,
                                  class_weights_method=class_weights_method,
                                  manifest=manifest,
                                  outcomes=outcomes,
                                  MODEL_TYPE=params.model_type)
# class_weight = {"Response": class_weight}


# -------------------------------
# Create TF datasets
# -------------------------------
print("\nCreate TF datasets ...")

# Training
# import ipdb; ipdb.set_trace()
train_data = create_tf_data(
    batch_size=params.batch_size,
    include_meta=False,
    interleave=True,
    # n_concurrent_shards=32,
    n_concurrent_shards=64,
    parse_fn=parse_fn,
    prefetch=2,
    repeat=True,
    seed=None,  # cfg.seed,
    shuffle_files=True,
    shuffle_size=8192,
    tfrecords=train_tfr_files,
    **parse_fn_train_kwargs)

# Determine feature shapes from data
bb = next(train_data.__iter__())

# Infer dims of features from the data
# import ipdb; ipdb.set_trace()
if params.use_ge:
    ge_shape = bb[0]["ge_data"].numpy().shape[1:]
else:
    ge_shape = None

if params.use_dd1:
    dd_shape = bb[0]["dd1_data"].numpy().shape[1:]
else:
    dd_shape = None

for i, item in enumerate(bb):
    print(f"\nItem {i}")
    if isinstance(item, dict):
        for k in item.keys():
            print(f"\t{k}: {item[k].numpy().shape}")

for i, rec in enumerate(train_data.take(2)):
    tf.print(rec[1])

# Evaluation (val, test, train)
# eval_batch_size = 8 * params.batch_size
# eval_batch_size = 16 * params.batch_size
create_tf_data_eval_kwargs = {
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

# val_data = create_tf_data(
#     tfrecords=val_tfr_files,
#     shuffle_files=False,
#     interleave=False,
#     shuffle_size=None,
#     repeat=False,
#     batch_size=eval_batch_size,
#     seed=None,
#     prefetch=2,
#     parse_fn=parse_fn,
#     include_meta=False,
#     **parse_fn_non_train_kwargs)

# test_batch_size = 8 * params.batch_size
# test_data = create_tf_data(
#     tfrecords=test_tfr_files,
#     shuffle_files=False,
#     interleave=False,
#     shuffle_size=None,
#     repeat=False,
#     batch_size=test_batch_size,
#     seed=None,
#     prefetch=2,
#     parse_fn=parse_fn,
#     include_meta=True,
#     **parse_fn_non_train_kwargs)


# ----------------------
# Callbacks
# ----------------------
# import ipdb; ipdb.set_trace()


class EarlyStopOnBatch(tf.keras.callbacks.Callback):
    """
    EarlyStopOnBatch(monitor="loss", batch_patience=20, validate_on_batch=10)
    """
    def __init__(self,
                 validation_data,
                 validation_steps=None,
                 validate_on_batch=100,
                 monitor="loss",
                 batch_patience=0,
                 print_fn=print):
        """ 
        Args:
            validate_on_batch : 
        """
        super(EarlyStopOnBatch, self).__init__()
        self.batch_patience = batch_patience
        self.best_weights = None
        # self.monitor = monitor  # ap
        self.validate_on_batch = validate_on_batch  # ap
        self.validation_data = validation_data
        self.validation_steps = validation_steps
        self.print_fn = print_fn

    def on_train_begin(self, logs=None):
        # self.wait = 0           # number of batches it has waited when loss is no longer minimum
        self.stopped_epoch = 0  # epoch the training stops at
        self.stopped_batch = 0  # epoch the training stops at
        self.best = np.Inf      # init the best as infinity
        self.epoch = None
        self.val_loss = np.Inf
        self.step_id = 0  # global batch
        # self.print_fn("\n{}.".format(yellow("Start training")))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        self.epoch = epoch + 1
        if self.epoch == 1:
            self.wait = 0  # number of batches it has waited when loss is no longer minimum
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
                self.epoch, batch, train_steps, logs["loss"], self.val_loss, self.best, color(self.wait)), end="\r")
            # self.print_fn("\repoch: {}, batch: {}, loss {:.4f}, val_loss {:.4f}, best_val_loss {:.4f} (wait: {}); {}".format(
            #     self.epoch, batch, logs["loss"], self.val_loss, self.best, color(self.wait)))

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
                self.epoch, batch, train_steps, logs["loss"], self.val_loss, self.best, color(self.wait)), end="\r")
            # self.print_fn("\repoch: {}, batch: {}, loss {:.4f}, val_loss {:.4f}, best_val_loss {:.4f} (wait: {})".format(
            #     self.epoch, batch, logs["loss"], self.val_loss, self.best, color(self.wait)))

            # Don't terminate of the first epoch
            if (self.wait >= self.batch_patience) and (self.epoch > 1):
                self.stopped_epoch = self.epoch
                self.stopped_batch = batch
                self.model.stop_training = True
                self.print_fn("\n{}".format(red("Terminate training")))
                self.print_fn("Restores model weights from the best epoch-batch set.")
                self.model.set_weights(self.best_weights)

        else:
            self.print_fn("\repoch: {}, batch: {}/{}, loss: {:.4f}, val_loss: {:.4f}, best_val_loss: {:.4f} (wait: {})".format(
                self.epoch, batch, train_steps, logs["loss"], self.val_loss, self.best, color(self.wait)), end="\r")
            # self.print_fn("\repoch: {}, batch: {}, loss {:.4f}, val_loss {:.4f}, best_val_loss {:.4f} (wait: {})".format(
            #     self.epoch, batch, logs["loss"], self.val_loss, self.best, color(self.wait)))
            val_logs = {"val_"+str(k): np.nan for k in keys}
            res.update(val_logs)

        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        res.update({"lr": lr})
        results.append(res)

    def on_train_end(self, logs=None):
        if self.stopped_batch > 0:
            self.print_fn("Early stop on epoch {} and batch {}.".format(self.stopped_epoch, self.stopped_batch))


# lg_r = Logger(outdir/"logger.log", verbose=False)
mycallback_kwargs = {"validation_data": val_data,
                     "validate_on_batch": params.validate_on_batch,
                     "batch_patience": params.batch_patience,
                     "print_fn": print
                     }
# callbacks = keras_callbacks(outdir, monitor="val_loss", **mycallback_kwargs)
callbacks = keras_callbacks(outdir)

results = []
mycallback = EarlyStopOnBatch(monitor="loss", **mycallback_kwargs)
callbacks.append(mycallback)


# ----------------------
# Prep for training
# ----------------------
# import ipdb; ipdb.set_trace()

# Mixed precision
if params.use_fp16:
    print_fn("Train with mixed precision")
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


# Prep the target
# y_encoding = "onehot"
y_encoding = "label"  # to be used binary cross-entropy

if y_encoding == "onehot":
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

elif y_encoding == "label":
    if index_col_name in data.columns:
        # Using Yitan's T/V/E splits
        ytr = tr_meta[args.target[0]].values
        yvl = vl_meta[args.target[0]].values
        yte = te_meta[args.target[0]].values
        loss = losses.BinaryCrossentropy()
    else:
        ytr = ydata_label[tr_id]
        yvl = ydata_label[vl_id]
        yte = ydata_label[te_id]
        loss = losses.SparseCategoricalCrossentropy()

else:
    raise ValueError(f"Unknown value for y_encoding ({y_encoding}).")

# ytr_label = ydata_label[tr_id]
# yvl_label = ydata_label[vl_id]
# yte_label = ydata_label[te_id]
# ytr_label = tr_meta[args.target[0]].values
# yvl_label = vl_meta[args.target[0]].values
# yte_label = te_meta[args.target[0]].values    


# ----------------------
# Define model
# ----------------------

# import ipdb; ipdb.set_trace()

# Calc output bias
# neg, pos = np.bincount(tr_meta[args.target[0]].values)
from sf_utils import get_categories_from_manifest
categories = get_categories_from_manifest(train_tfr_files, manifest, outcomes)
neg = categories[0]["num_tiles"]
pos = categories[1]["num_tiles"]

total = neg + pos
print("Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(total, pos, 100 * pos / total))
output_bias = np.log([pos/neg])
print("Output bias:", output_bias)
# output_bias = None


if args.target[0] == "Response":
    if params.use_tile is True:
        build_model_kwargs = {"base_image_model": params.base_image_model,
                              "use_ge": params.use_ge,
                              "use_dd1": params.use_dd1,
                              "use_dd2": params.use_dd2,
                              "use_tile": params.use_tile,
                              "ge_shape": ge_shape,
                              "dd_shape": dd_shape,
                              "dense1_ge": params.dense1_ge,
                              "dense1_dd1": params.dense1_dd1,
                              "dense1_dd2": params.dense1_dd2,
                              "dense1_top": params.dense1_top,
                              "top_hidden_dropout1": params.top_hidden_dropout1,
                              # "model_type": params.model_type,
                              "output_bias": output_bias,
                              "loss": loss,
                              "optimizer": optimizers.Adam(learning_rate=params.learning_rate)
        }
        model = build_model_rsp(**build_model_kwargs)
                                
    else:
        model = build_model_rsp_baseline(use_ge=params.use_ge,
                                         use_dd1=params.use_dd1,
                                         use_dd2=params.use_dd2,
                                         ge_shape=ge_shape,
                                         dd_shape=dd_shape,
                                         model_type=params.model_type)
        x = {"ge_data": tr_ge.values, "dd_data": tr_dd.values}
        y = {"Response": ytr}
        validation_data = ({"ge_data": vl_ge.values,
                            "dd_data": vl_dd.values},
                           {"Response": yvl})
else:
    raise NotImplementedError("Need to check this method")
    model = build_model_rna()

print_fn("")
# print_fn(model.summary())
model.summary(print_fn=print_fn)

# import ipdb; ipdb.set_trace()
# steps = 40
# res = model.evaluate(train_data,
#                      steps=params.batch_size * steps // params.batch_size,
#                      verbose=1)
# print("Loss: {:0.4f}".format(res[0]))

# -------------
# Train model
# -------------

# import ipdb; ipdb.set_trace()

print_fn(f"Train steps:      {train_steps}")
print_fn(f"Validation steps: {validation_steps}")

if args.trn_phase == "train":

    # # Base model
    # base_model_path = prjdir/"base_multimodal_model"
    # if base_model_path.exists():
    #     # load model
    #     model = tf.keras.models.load_model(base_model_path, compile=True)
    # else:
    #     # save model
    #     model.save(base_model_path)

    # # Dump/log performance measures of the base model
    # aa = model.evaluate(val_data, steps=validation_steps, verbose=1)
    # print_fn("Base model val_loss: {}".format(aa[0]))

    t = time()

    if params.use_tile is True:
        timer = Timer()
        history = model.fit(x=train_data,
                            # batch_size=params.batch_size,
                            # steps_per_epoch=total_train_tiles//params.batch_size,
                            steps_per_epoch=train_steps,
                            # steps_per_epoch=640//params.batch_size,
                            validation_data=val_data,
                            # validation_steps=total_val_tiles//eval_batch_size,
                            validation_steps=validation_steps,
                            # validation_steps=640//params.batch_size,
                            class_weight=class_weight,
                            epochs=params.epochs,
                            shuffle=False,
                            verbose=0,
                            callbacks=callbacks)

    elif params.use_tile is False:
        history = model.fit(x=x,
                            y=y,
                            epochs=total_epochs,
                            verbose=1,
                            validation_data=validation_data,
                            callbacks=callbacks)

    print_fn("")
    timer.display_timer(print_fn)
    import ipdb; ipdb.set_trace()
    res = pd.DataFrame(results)
    res.to_csv(outdir/"results.csv", index=False)

    # --------------------------
    # Save model
    # --------------------------
    # import ipdb; ipdb.set_trace()
    # The key difference between HDF5 and SavedModel is that HDF5 uses object
    # configs to save the model architecture, while SavedModel saves the execution
    # graph. Thus, SavedModels are able to save custom objects like subclassed
    # models and custom layers without requiring the original code.

    # Save the entire model as a SavedModel.
    final_model_fpath = outdir/f"final_model_for_split_id_{split_id}"
    model.save(final_model_fpath)

else:
    model = tf.keras.models.load_model(final_model_fpath, compile=True)

del train_data, val_data


# --------------------------
# Evaluate
# --------------------------
# import ipdb; ipdb.set_trace()

def calc_per_tile_preds(data_with_meta, model, outdir, verbose=True):
    """ ... """
    # meta_keys = ["smp", "Group", "grp_name", "Response"]
    meta_keys = ["smp", "image_id", "tile_id"]
    # meta_keys = ["smp"]
    meta_agg = {k: None for k in meta_keys}
    y_true, y_pred_prob, y_pred_label = [], [], []

    # import ipdb; ipdb.set_trace()
    for i, batch in enumerate(data_with_meta):
        # print(i)
        if (i+1) % 50 == 0:
            print(f"\rbatch {i+1}", end="")

        fea = batch[0]
        label = batch[1]
        # smp = batch[2]
        meta = batch[2]

        # Predictions
        preds = model.predict(fea)
        y_pred_prob.append(preds)
        preds = np.squeeze(preds)
        if np.ndim(preds) > 1:
            y_pred_label.extend( np.argmax(preds, axis=1).tolist() )  # SparseCategoricalCrossentropy
        else:
            y_pred_label.extend( [0 if p<0.5 else 1 for p in preds] )  # BinaryCrossentropy

        # True labels
        # y_true.extend( label[args.target[0]].numpy().tolist() )  # when batch[1] is dict
        y_true.extend( label.numpy().tolist() )  # when batch[1] is array

        # Meta
        # smp_list.extend( [smp_bytes.decode('utf-8') for smp_bytes in batch[2].numpy().tolist()] )
        for k in meta_keys:
            vv = [val_bytes.decode("utf-8") for val_bytes in meta[k].numpy().tolist()]
            if meta_agg[k] is None:
                meta_agg[k] = vv
            else:
                meta_agg[k].extend(vv)

        del batch, fea, label, meta

    # Meta
    df_meta = pd.DataFrame(meta_agg)
    # print("\ndf memory {:.2f} GB".format( df_meta.memory_usage().sum()/1e9 ))

    # Predictions
    y_pred_prob = np.vstack(y_pred_prob)
    if np.ndim(np.squeeze(y_pred_prob)) > 1:
        df_y_pred_prob = pd.DataFrame(y_pred_prob, columns=[f"prob_{c}" for c in range(y_pred_prob.shape[1])])
    else:
        df_y_pred_prob = pd.DataFrame(y_pred_prob, columns=["prob"])

    # True labels
    df_labels = pd.DataFrame({"y_true": y_true, "y_pred_label": y_pred_label})

    # Combine
    prd = pd.concat([df_meta, df_y_pred_prob, df_labels], axis=1)
    return prd


def agg_per_smp_preds(prd, id_name, outdir):
    """ Agg predictions per smp. """
    aa = []
    for sample in prd[id_name].unique():
        dd = {id_name: sample}
        df = prd[prd[id_name] == sample]
        dd["y_true"] = df["y_true"].unique()[0]
        dd["y_pred_label"] = np.argmax(np.bincount(df.y_pred_label))
        dd["smp_acc"] = sum(df.y_true == df.y_pred_label)/df.shape[0]
        aa.append(dd)

    agg_preds = pd.DataFrame(aa).sort_values(args.id_name).reset_index(drop=True)
    # agg_preds.to_csv(outdir/'test_preds_per_smp.csv', index=False)

    # Efficient use of groupby().apply() !!
    # xx = prd.groupby('smp').apply(lambda x: pd.Series({
    #     'y_true': x['y_true'].unique()[0],
    #     'y_pred_label': np.argmax(np.bincount(x['y_pred_label'])),
    #     'pred_acc': sum(x['y_true'] == x['y_pred_label'])/x.shape[0]
    # })).reset_index().sort_values(args.id_name).reset_index(drop=True)
    # xx = xx.astype({'y_true': int, 'y_pred_label': int})
    # print(agg_preds.equals(xx))
    return agg_preds


def get_preds(tf_data, meta, model, outdir, args, name):
    """ ... """
    # Predictions per tile
    timer = Timer()
    tile_preds = calc_per_tile_preds(tf_data, model=model, outdir=outdir)
    timer.display_timer(print_fn)

    # Aggregate predictions per sample
    smp_preds = agg_per_smp_preds(tile_preds, id_name=args.id_name, outdir=outdir)
    smp_preds = smp_preds.merge(meta, on="smp", how="inner")

    # Save predictions
    tile_preds.to_csv(outdir/f"{name}_preds_per_tile.csv", index=False)
    smp_preds.to_csv(outdir/f"{name}_preds_per_smp.csv", index=False)

    # Scores
    # tile_scores = calc_scores(test_tile_preds["y_true"].values, test_tile_preds["prob_1"].values, mltype="cls")
    tile_scores = calc_scores(tile_preds["y_true"].values, tile_preds["prob"].values, mltype="cls")
    smp_scores = calc_scores(smp_preds["y_true"].values, smp_preds["y_pred_label"].values, mltype="cls")

    dump_dict(tile_scores, outdir/f"{name}_tile_scores.txt")
    dump_dict(smp_scores, outdir/f"{name}_smp_scores.txt")

    import ipdb; ipdb.set_trace()

    # Confusion
    print_fn("\nPer-sample confusion:")
    smp_cnf_mtrx = confusion_matrix(smp_preds["y_true"], smp_preds["y_pred_label"])
    save_confusion_matrix(true_labels=smp_preds["y_true"].values,
                          predictions=smp_preds["y_pred_label"].values,
                          labels=["Non-response", "Response"],
                          outpath=outdir/f"{name}_smp_confusion.png")
    print_fn(smp_cnf_mtrx)

    print_fn("Per-tile confusion:")
    tile_cnf_mtrx = confusion_matrix(tile_preds["y_true"], tile_preds["y_pred_label"])
    save_confusion_matrix(true_labels=tile_preds["y_true"].values,
                          predictions=tile_preds["prob"].values,
                          labels=["Non-response", "Response"],
                          outpath=outdir/f"{name}_tile_confusion.png")
    print_fn(tile_cnf_mtrx)


import ipdb; ipdb.set_trace()
timer = Timer()
print_fn("\n{}".format(green("Calculating predictions.")))

print_fn("\n{}".format(bold("Test set predictions.")))
get_preds(test_data, te_meta, model, outdir, args, name="test")
del test_data

print_fn("\n{}".format(bold("Validation set predictions.")))
get_preds(eval_val_data, vl_meta, model, outdir, args, name="test")
del eval_val_data

print_fn("\n{}".format(bold("Train set predictions.")))
get_preds(eval_train_data, tr_meta, model, outdir, args, name="train")
del eval_train_data

timer.display_timer(print_fn)
print_fn("\nDone.")
