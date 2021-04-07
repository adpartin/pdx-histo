"""
Prediction of drug response with TFRecords.
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
import glob
import shutil
import tempfile
from pathlib import Path
from pprint import pprint
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping

fdir = Path(__file__).resolve().parent
# from config import cfg
sys.path.append(str(fdir/".."))
import src
from src.config import cfg
from src.models import build_model_rsp, build_model_rsp_baseline, keras_callbacks
from src.ml.scale import get_scaler
from src.ml.evals import calc_scores, calc_preds, dump_preds, save_confusion_matrix
from src.utils.utils import Params, dump_dict, read_lines, cast_list, Timer
from src.datasets.tidy import split_data_and_extract_fea, extract_fea
from src.tf_utils import get_tfr_files, calc_records_in_tfr_files, count_data_items
from src.sf_utils import (green, interleave_tfrecords, create_tf_data, calc_class_weights,
                          parse_tfrec_fn_rsp, parse_tfrec_fn_rna,
                          create_manifest)

# Seed
# seed = 42
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

args, other_args = parser.parse_known_args()
pprint(args)

single_drug = False

# Load dataframe (annotations)
prjdir = cfg.MAIN_PRJDIR/args.prjname
annotations_file = cfg.DATA_PROCESSED_DIR/args.dataname/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

# Create outdir
split_on = "none" if args.split_on is (None or "none") else args.split_on
outdir = prjdir/f"multimodal/split_on_{split_on}"
os.makedirs(outdir, exist_ok=True)

# Import hyper-parameters
# import ipdb; ipdb.set_trace()
prm_file_path = prjdir/"multimodal/params.json"
if prm_file_path.exists() is False:
    shutil.copy(fdir/"../default_params/default_params_multimodal.json", prm_file_path)
params = Params(prm_file_path)

# import ipdb; ipdb.set_trace()
print("\nFull dataset:")
if args.target[0] == "Response":
    pprint(data.groupby(["ctype", "Response"]).agg({split_on: "nunique", "smp": "nunique"}).reset_index().rename(
        columns={split_on: f"{split_on}_unq", "smp": "smp_unq"}))
else:
    pprint(data[args.target[0]].value_counts())

ge_cols  = [c for c in data.columns if c.startswith("ge_")]
dd1_cols = [c for c in data.columns if c.startswith("dd1_")]
dd2_cols = [c for c in data.columns if c.startswith("dd2_")]
data = data.astype({"image_id": str, "slide": str})

# Scalers for each feature set
ge_scaler, dd1_scaler, dd2_scaler = None, None, None

if params.use_ge and len(ge_cols) > 0:
    ge_scaler = get_scaler(data[ge_cols])

if params.use_dd1 and len(dd1_cols) > 0:
    dd1_scaler = get_scaler(data[dd1_cols])

if params.use_dd2 and len(dd2_cols) > 0:
    dd2_scaler = get_scaler(data[dd2_cols])



# loss={'csite_label': tf.keras.losses.categorical_crossentropy,
#     'ctype_label': tf.keras.losses.categorical_crossentropy},
# loss = {'ctype': tf.keras.losses.SparseCategoricalCrossentropy()}

# outcome_header = ['Response']
outcome_header = args.target
# aux_headers = ['ctype', 'csite']  # (ap)

label = f"{params.tile_px}px_{params.tile_um}um"


if args.target[0] == "Response":
    if params.single_drug:
        tfr_dir = cfg.SF_TFR_DIR_RSP
        parse_fn = parse_tfrec_fn_rsp
    else:
        tfr_dir = cfg.SF_TFR_DIR_RSP_DRUG_PAIR
        parse_fn = parse_tfrec_fn_rsp

elif args.target[0] == "ctype":
    tfr_dir = cfg.SF_TFR_DIR_RNA_NEW
    parse_fn = parse_tfrec_fn_rna
tfr_dir = tfr_dir/label


# (ap) Create outcomes (for drug response)
# __init__ --> _trainer --> training_dataset.get_outcomes_from_annotations
outcomes = {}
unique_outcomes = list(set(data[args.target[0]].values))
unique_outcomes.sort()

for smp, o in zip(data[args.id_name], data[args.target[0]]):
    outcomes[smp] = {"outcome": unique_outcomes.index(o)}
    # outcomes[smp] = {'outcome': unique_outcomes.index(o), 'submitter_id': smp}

print("\n'outcomes':")
print(type(outcomes))
print(len(outcomes))
print(list(outcomes.keys())[:3])
print(outcomes[list(outcomes.keys())[3]])


# -----------------------------------------------
# Create manifest
# -----------------------------------------------
timer = Timer()
manifest = create_manifest(directory=tfr_dir, n_files=None)
timer.display_timer()
print('\nmanifest:')
print(type(manifest))
print(len(manifest))
print(list(manifest.keys())[:3])
print(manifest[list(manifest.keys())[3]])


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
print("\nTrain:")
pprint(tr_meta.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
print("\nVal:")
pprint(vl_meta.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())
print("\nTest:")
pprint(te_meta.groupby(["ctype", "Response"]).agg({"grp_name": "nunique", "smp": "nunique"}).reset_index())

# Make sure indices do not overlap
assert len( set(tr_id).intersection(set(vl_id)) ) == 0, "Overlapping indices btw tr and vl"
assert len( set(tr_id).intersection(set(te_id)) ) == 0, "Overlapping indices btw tr and te"
assert len( set(vl_id).intersection(set(te_id)) ) == 0, "Overlapping indices btw tr and vl"

# Print split ratios
print("")
print("Train samples {} ({:.2f}%)".format( len(tr_id), 100*len(tr_id)/data.shape[0] ))
print("Val   samples {} ({:.2f}%)".format( len(vl_id), 100*len(vl_id)/data.shape[0] ))
print("Test  samples {} ({:.2f}%)".format( len(te_id), 100*len(te_id)/data.shape[0] ))

tr_grp_unq = set(tr_meta[split_on].values)
vl_grp_unq = set(vl_meta[split_on].values)
te_grp_unq = set(te_meta[split_on].values)
print("")
print(f"Total intersects on {split_on} btw tr and vl: {len(tr_grp_unq.intersection(vl_grp_unq))}")
print(f"Total intersects on {split_on} btw tr and te: {len(tr_grp_unq.intersection(te_grp_unq))}")
print(f"Total intersects on {split_on} btw vl and te: {len(vl_grp_unq.intersection(te_grp_unq))}")
print(f"Unique {split_on} in tr: {len(tr_grp_unq)}")
print(f"Unique {split_on} in vl: {len(vl_grp_unq)}")
print(f"Unique {split_on} in te: {len(te_grp_unq)}")



# List of sample names for T/V/E
tr_smp_names = list(tr_meta[args.id_name].values)
vl_smp_names = list(vl_meta[args.id_name].values)
te_smp_names = list(te_meta[args.id_name].values)

# TFRecords filenames
train_tfr_files = get_tfr_files(tfr_dir, tr_smp_names)  # training_tfrecords
val_tfr_files = get_tfr_files(tfr_dir, vl_smp_names)  # validation_tfrecords
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
# timers = []

# timer = Timer()
# total_test_tiles = calc_records_in_tfr_files(test_tfr_files) # 268028
# t_end = timer.timer_end()
# print(t_end)
# timers.append(t_end)

# timer = Timer()
# total_val_tiles = calc_records_in_tfr_files(val_tfr_files) # 261300
# t_end = timer.timer_end()
# print(t_end)
# timers.append(t_end)

# timer = Timer()
# total_train_tiles = calc_records_in_tfr_files(train_tfr_files) # 2129951
# t_end = timer.timer_end()
# print(t_end)
# timers.append(t_end)

# total_trn_tiles = count_data_items(train_tfr_files)
tile_cnts = pd.read_csv(cfg.SF_TFR_DIR_RSP_DRUG_PAIR/label/"tile_counts_per_slide.csv")
# tr_tile_cnts = tile_cnts.merge(tr_meta[["smp", "Group", "grp_name", "Response"]], on="smp", how="inner")

total_test_tiles = tile_cnts[tile_cnts[args.id_name].isin(te_smp_names)]["max_tiles"].sum()
total_val_tiles = tile_cnts[tile_cnts[args.id_name].isin(vl_smp_names)]["max_tiles"].sum()
total_train_tiles = tile_cnts[tile_cnts[args.id_name].isin(tr_smp_names)]["max_tiles"].sum()


# import ipdb; ipdb.set_trace()
print(len(outcomes))
print(len(manifest))
print(outcomes[list(outcomes.keys())[3]])
print(manifest[list(manifest.keys())[3]])


# -------------------------------
# create SlideflowModel model SFM
# -------------------------------
# import ipdb; ipdb.set_trace()

#DATA_DIR = '/vol/ml/apartin/projects/slideflow-proj/sf_pdx_bin_rsp2/project/models/ctype-Xception_v0-kfold1';
DATA_DIR = outdir
MANIFEST = manifest
# IMAGE_SIZE = image_size = params.tile_px
DTYPE = "float16" if params.use_fp16 else "float32"
SLIDE_ANNOTATIONS = slide_annotations = outcomes
#SLIDES = list(slide_annotations.keys())
SAMPLES = list(slide_annotations.keys())
DATASETS = {}

#outcomes_ = [slide_annotations[slide]['outcome'] for slide in SLIDES]
outcomes_ = [slide_annotations[smp]['outcome'] for smp in SAMPLES]

if params.model_type == 'categorical':
    NUM_CLASSES = len(list(set(outcomes_)))  # infer this from other variables

#ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SLIDES, outcomes_), -1)]
ANNOTATIONS_TABLES = [tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(SAMPLES, outcomes_), -1)]


# import ipdb; ipdb.set_trace()

if args.target[0] == 'Response':
    # Response
    parse_fn = parse_tfrec_fn_rsp
    parse_fn_kwargs_train = {
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
    parse_fn_kwargs_train = {
        'use_tile': params.use_tile,
        'use_ge': params.use_ge,
        'ge_scaler': ge_scaler,
        'id_name': args.id_name,
        'MODEL_TYPE': params.model_type,
        'ANNOTATIONS_TABLES': ANNOTATIONS_TABLES,
        'AUGMENT': params.augment,
    }
parse_fn_kwargs_non_train = parse_fn_kwargs_train.copy()
parse_fn_kwargs_non_train["augment"] = False

# class_weights_method = "BY_SAMPLE"
class_weights_method = "BY_TILE"
# class_weights_method = "NONE"

# import ipdb; ipdb.set_trace()
# from sklearn.utils.class_weight import compute_class_weight
# y = tr_meta["Response"].values
# class_weight = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weight = calc_class_weights(train_tfr_files,
                                  class_weights_method=class_weights_method,
                                  manifest=manifest,
                                  SLIDE_ANNOTATIONS=SLIDE_ANNOTATIONS,
                                  MODEL_TYPE=params.model_type)
# class_weight = {"Response": class_weight}

print("\nTraining TFRecods")

# import ipdb; ipdb.set_trace()
train_data = create_tf_data(
    tfrecords=train_tfr_files,
    n_concurrent_shards=32,
    shuffle_files=True,
    interleave=True,
    shuffle_size=8192,
    repeat=True,
    batch_size=params.batch_size,
    seed=cfg.seed,
    prefetch=2,
    parse_fn=parse_fn,
    include_meta=False,
    **parse_fn_kwargs_train)

# Determine feature shapes from data
bb = next(train_data.__iter__())

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

for i, rec in enumerate(train_data.take(4)):
    tf.print(rec[1])

val_data = create_tf_data(
    tfrecords=val_tfr_files,
    n_concurrent_shards=None,
    shuffle_files=False,
    interleave=False,
    shuffle_size=None,
    repeat=False,
    batch_size=params.batch_size, # 2048
    seed=None,
    prefetch=2,
    parse_fn=parse_fn,
    include_meta=False,
    **parse_fn_kwargs_non_train)

test_data = create_tf_data(
    tfrecords=test_tfr_files,
    n_concurrent_shards=None,
    shuffle_files=False,
    interleave=False,
    shuffle_size=None,
    repeat=False,
    batch_size=params.batch_size, # 2048
    seed=None,
    prefetch=2,
    parse_fn=parse_fn,
    include_meta=True,
    **parse_fn_kwargs_non_train)

# ----------------------
# Prep for training
# ----------------------
# import ipdb; ipdb.set_trace()

class PredictionAndEvaluationCallback(tf.keras.callbacks.Callback):
    pass


# #callbacks = [history_callback, PredictionAndEvaluationCallback(), cp_callback, tensorboard_callback]
# callbacks = [history_callback, cp_callback]

callbacks = keras_callbacks(outdir, monitor="val_loss")


# https://www.tensorflow.org/guide/mixed_precision
if DTYPE == "float16":
    # TF 2.4
    # from tensorflow.keras import mixed_precision
    # policy = tf.keras.mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_global_policy(policy)
    # TF 2.3
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    print("Training with mixed precision")
    policy = tf.keras.mixed_precision.experimental.Policy("mixed_float16")
    mixed_precision.set_policy(policy)
    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)


# import ipdb; ipdb.set_trace()
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
ytr_label = tr_meta[args.target[0]].values
yvl_label = vl_meta[args.target[0]].values
yte_label = te_meta[args.target[0]].values    


# -------------
# Train model
# -------------

# import ipdb; ipdb.set_trace()

if args.target[0] == "Response":
    if params.use_tile is True:
        model = build_model_rsp(use_ge=params.use_ge,
                                use_dd1=params.use_dd1,
                                use_dd2=params.use_dd2,
                                use_tile=params.use_tile,
                                ge_shape=ge_shape,
                                dd_shape=dd_shape,
                                model_type=params.model_type,
                                output_bias=None,
                                loss=loss)
    else:
        model = build_model_rsp_baseline(use_ge=params.use_ge,
                                         use_dd1=params.use_dd1,
                                         use_dd2=params.use_dd2,
                                         ge_shape=ge_shape,
                                         dd_shape=dd_shape,
                                         model_type=params.model_type,
                                         NUM_CLASSES=NUM_CLASSES)
        x = {"ge_data": tr_ge.values, "dd_data": tr_dd.values}
        y = {"Response": ytr}
        validation_data = ({"ge_data": vl_ge.values,
                            "dd_data": vl_dd.values},
                           {"Response": yvl})
else:
    raise NotImplementedError("Need to check this method")
    model = build_model_rna()

print()
print(model.summary())

# model.compile(loss=losses.BinaryCrossentropy(),
#               optimizer=Adam(learning_rate=params.learning_rate),
#               metrics=metrics)

final_model_fpath = outdir/f"final_model_for_split_id_{split_id}"
# train = True
# train = False
train = True if args.trn_phase == "train" else False

# import ipdb; ipdb.set_trace()

if train:
    t = time()
    print("Start training ...")
    if params.use_tile is True:
        timer = Timer()
        history = model.fit(x=train_data,
                            # batch_size=params.batch_size,
                            steps_per_epoch=total_train_tiles//params.batch_size,
                            # steps_per_epoch=640//params.batch_size,
                            validation_data=val_data,
                            validation_steps=total_val_tiles//params.batch_size,
                            # validation_steps=640//params.batch_size,
                            class_weight=class_weight,
                            epochs=params.epochs,
                            shuffle=False,
                            verbose=1,
                            callbacks=callbacks)
        # history = model.fit(
        #     x = get_dataset(files_train, augment=True, shuffle=True, repeat=True, dim=IMG_SIZES[fold], batch_size = BATCH_SIZES[fold]), 
        #     epochs = EPOCHS[fold],
        #     callbacks = [sv, get_lr_callback(BATCH_SIZES[fold])], 
        #     steps_per_epoch = count_data_items(files_train)/BATCH_SIZES[fold]//REPLICAS,
        #     validation_data = get_dataset(files_valid, augment=False, shuffle=False, repeat=False, dim=IMG_SIZES[fold]), #class_weight = {0:1,1:2},
        #     verbose=VERBOSE
        # )

    elif params.use_tile is False:
        history = model.fit(x=x,
                            y=y,
                            epochs=total_epochs,
                            verbose=1,
                            validation_data=validation_data,
                            callbacks=callbacks)
    timer.display_timer()


    # --------------------------
    # Save model
    # --------------------------
    # import ipdb; ipdb.set_trace()
    # The key difference between HDF5 and SavedModel is that HDF5 uses object
    # configs to save the model architecture, while SavedModel saves the execution
    # graph. Thus, SavedModels are able to save custom objects like subclassed
    # models and custom layers without requiring the original code.

    # Save the entire model as a SavedModel.
    model.save(final_model_fpath)

# --------------------------
# Save model
# --------------------------
if not train:
    model = tf.keras.models.load_model(final_model_fpath, compile=True)

# --------------------------
# Evaluate
# --------------------------
# import ipdb; ipdb.set_trace()

def calc_per_tile_preds(data_with_meta, model, outdir):
    """ ... """
    # meta_keys = ["smp", "Group", "grp_name", "Response"]
    meta_keys = ["smp", "image_id", "tile_id"]
    # meta_keys = ["smp"]
    meta_agg = {k: None for k in meta_keys}
    y_true, y_pred_prob, y_pred_label, smp_list = [], [], [], []

    import ipdb; ipdb.set_trace()

    for i, batch in enumerate(data_with_meta):
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

    # Meta
    df_meta = pd.DataFrame(meta_agg)

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
    # prd.to_csv(outdir/'test_preds_per_tiles.csv', index=False)
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


# Predictions per tile
# test_tile_preds = calc_per_tile_preds(test_data_with_smp_names, model=model, outdir=outdir)
test_tile_preds = calc_per_tile_preds(test_data, model=model, outdir=outdir)

# Aggregate predictions per sample
# import ipdb; ipdb.set_trace()
test_smp_preds = agg_per_smp_preds(test_tile_preds, id_name=args.id_name, outdir=outdir)
test_smp_preds = test_smp_preds.merge(te_meta, on="smp", how="inner")

test_tile_preds.to_csv(outdir/"test_preds_per_tile.csv", index=False)
test_smp_preds.to_csv(outdir/"test_preds_per_smp.csv", index=False)
# dump_preds(yte_label, yte_prd[:, 1], te_meta, outpath=outdir/"test_preds_per_smp.csv")

# Scores
# tile_scores = calc_scores(test_tile_preds["y_true"].values, test_tile_preds["prob_1"].values, mltype="cls")
tile_scores = calc_scores(test_tile_preds["y_true"].values, test_tile_preds["prob"].values, mltype="cls")
smp_scores = calc_scores(test_smp_preds["y_true"].values, test_smp_preds["y_pred_label"].values, mltype="cls")

dump_dict(tile_scores, outdir/"test_tile_scores.txt")
dump_dict(smp_scores, outdir/"test_smp_scores.txt")

# Confusion
cnf_mtrx = confusion_matrix(test_smp_preds["y_true"], test_smp_preds["y_pred_label"])
save_confusion_matrix(cnf_mtrx, labels=["Non-response", "Response"],
                      outpath=outdir/"confusion.png")
pprint(cnf_mtrx)

print("\nDone.")
