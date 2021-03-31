"""
Train baseline model for drug response using rna and dd data (w/o tfrecords).
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
import glob
import shutil
from time import time
from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
assert tf.__version__ >= "2.0"

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam

from models import build_model_rsp_baseline
from ml.scale import get_scaler
from ml.evals import calc_scores, calc_preds, dump_preds, save_confusion_matrix
from utils.utils import Params, dump_dict

from datasets.tidy import split_data_and_extract_fea

fdir = Path(__file__).resolve().parent
from config import cfg

# Seed
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)


parser = argparse.ArgumentParser("Train NN.")
parser.add_argument('-t', '--target',
                    type=str,
                    nargs='+',
                    # default=['ctype'],
                    default=['Response'],
                    choices=['Response', 'ctype', 'csite'],
                    help='Name of target output.')
parser.add_argument('--id_name',
                    type=str,
                    # default='slide',
                    default='smp',
                    choices=['slide', 'smp'],
                    help='Column name of the ID.')
parser.add_argument('--split_on',
                    type=str,
                    default=None,
                    choices=['Sample', 'slide', 'Group'],
                    help='Specify the hard split variable/column (default: None).')
parser.add_argument('--prjname',
                    type=str,
                    default='bin_rsp_all',
                    help='Project name (folder that contains the annotations.csv dataframe).')
parser.add_argument('--dataname',
                    type=str,
                    default='tidy_all',
                    help='Project name (folder that contains the annotations.csv dataframe).')

args, other_args = parser.parse_known_args()
pprint(args)

# import ipdb; ipdb.set_trace()

# Load dataframe (annotations)
prjdir = cfg.MAIN_PRJDIR/args.prjname
annotations_file = cfg.DATA_PROCESSED_DIR/args.dataname/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

# Outdir
split_on = 'none' if args.split_on is (None or 'none') else args.split_on
outdir = prjdir/f'baseline/split_on_{split_on}'
os.makedirs(outdir, exist_ok=True)

# Import parameters
prm_file_path = prjdir/"baseline/params.json"
if prm_file_path.exists() is False:
    shutil.copy(fdir/"../default_params/default_params_baseline.json", prm_file_path)
params = Params(prm_file_path)

# import ipdb; ipdb.set_trace()
print("\nFull dataset:")
if args.target[0] == 'Response':
    pprint(data.groupby(["ctype", "Response"]).agg({split_on: "nunique", "smp": "nunique"}).reset_index().rename(
        columns={split_on: f"{split_on}_unq", "smp": "smp_unq"}))
else:
    pprint(data[args.target[0]].value_counts())

ge_cols = [c for c in data.columns if c.startswith('ge_')]
dd_cols = [c for c in data.columns if c.startswith('dd_')]
data = data.astype({'image_id': str, 'slide': str})

# RNA scaler
if params.use_ge:
    ge_scaler = get_scaler(data[ge_cols])
else:
    ge_scaler = None

# Descriptors scaler
if params.use_dd:
    dd_scaler = get_scaler(data[dd_cols])
else:
    dd_scaler = None

# -----------------------------------------------
# Data splits
# -----------------------------------------------

def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines

def cast_list(ll, dtype=int):
    return [dtype(i) for i in ll]


# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Yitan's splits
# --------------
# import ipdb; ipdb.set_trace()
splitdir = cfg.DATADIR/"PDX_Transfer_Learning_Classification/Processed_Data/Data_For_MultiModal_Learning/Data_Partition"
# split_id = 0
split_id = 2

index_col_name = "index"
tr_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"TrainList.txt")), int)
vl_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"ValList.txt")), int)
te_id = cast_list(read_lines(str(splitdir/f"cv_{split_id}"/"TestList.txt")), int)

# Update ids
tr_id = sorted(set(data[index_col_name]).intersection(set(tr_id)))
vl_id = sorted(set(data[index_col_name]).intersection(set(vl_id)))
te_id = sorted(set(data[index_col_name]).intersection(set(te_id)))
# ------------------------------------------------------------

kwargs = {"ge_cols": ge_cols,
          "dd_cols": dd_cols,
          "ge_scaler": ge_scaler,
          "dd_scaler": dd_scaler,
          "ge_dtype": cfg.GE_DTYPE,
          "dd_dtype": cfg.DD_DTYPE,
          "index_col_name": index_col_name
}
tr_ge, tr_dd = split_data_and_extract_fea(data, ids=tr_id, **kwargs)
vl_ge, vl_dd = split_data_and_extract_fea(data, ids=vl_id, **kwargs)
te_ge, te_dd = split_data_and_extract_fea(data, ids=te_id, **kwargs)

ge_shape = (te_ge.shape[1],)
dd_shape = (tr_dd.shape[1],)

# Variables (dict/dataframes/arrays) that are passed as features to the NN
xtr = {"ge_data": tr_ge.values, "dd_data": tr_dd.values}
xvl = {"ge_data": vl_ge.values, "dd_data": vl_dd.values}
xte = {"ge_data": te_ge.values, "dd_data": te_dd.values}

# Extarct meta for T/V/E
if index_col_name in data.columns:
    tr_meta = data[data[index_col_name].isin(tr_id)].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
    vl_meta = data[data[index_col_name].isin(vl_id)].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
    te_meta = data[data[index_col_name].isin(te_id)].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
else:
    tr_meta = data.iloc[tr_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
    vl_meta = data.iloc[vl_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
    te_meta = data.iloc[te_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)

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


# Onehot encoding
# ydata = data[args.target[0]].values
# y_onehot = pd.get_dummies(ydata)
# ydata_label = np.argmax(y_onehot.values, axis=1)
# num_classes = len(np.unique(ydata_label))

def keras_callbacks(outdir, monitor='val_loss'):
    """ ... """
    from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
    checkpointer = ModelCheckpoint(str(outdir/'model_best_at_{epoch}.ckpt'), monitor='val_loss',
                                   verbose=0, save_weights_only=False, save_best_only=True)
    csv_logger = CSVLogger(outdir/'training.log')
    reduce_lr = ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=10,
                                  verbose=1, mode='auto', min_delta=0.0001,
                                  cooldown=0, min_lr=0)
    early_stop = EarlyStopping(monitor=monitor, patience=20, verbose=1, mode='auto')

    return [checkpointer, csv_logger, early_stop, reduce_lr]

callbacks = keras_callbacks(outdir, monitor='val_loss')

# https://www.tensorflow.org/guide/mixed_precision
# Mixed precision
# if DTYPE == 'float16':
#     # TF 2.3
#     from tensorflow.keras.mixed_precision import experimental as mixed_precision
#     print("Training with mixed precision")
#     policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#     mixed_precision.set_policy(policy)
#     print('Compute dtype: %s' % policy.compute_dtype)
#     print('Variable dtype: %s' % policy.variable_dtype)

# y_encoding = 'onehot'
y_encoding = 'label'

import ipdb; ipdb.set_trace()

if y_encoding == 'onehot':
    if index_col_name in data.columns:
        # print(te_meta[["index", "Group", "grp_name", "Response"]])
        ytr = pd.get_dummies(tr_meta[args.target[0]].values)
        yvl = pd.get_dummies(vl_meta[args.target[0]].values)
        yte = pd.get_dummies(te_meta[args.target[0]].values)
    else:
        ytr = y_onehot.iloc[tr_id, :].reset_index(drop=True)
        yvl = y_onehot.iloc[vl_id, :].reset_index(drop=True)
        yte = y_onehot.iloc[te_id, :].reset_index(drop=True)

    loss = losses.CategoricalCrossentropy()

elif y_encoding == 'label':
    if index_col_name in data.columns:
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
    raise ValueError(f'Unknown value for y_encoding ({y_encoding}).')
    
# ytr_label = ydata_label[tr_id]
# yvl_label = ydata_label[vl_id]
# yte_label = ydata_label[te_id]
ytr_label = tr_meta[args.target[0]].values
yvl_label = vl_meta[args.target[0]].values
yte_label = te_meta[args.target[0]].values

# TODO: doesn't work with class_weight!
# ytr = {"Response": ytr}
# yvl = {"Response": yvl}
# yte = {"Response": yte}

# --------------------------
# Define and Train
# --------------------------
# import ipdb; ipdb.set_trace()
neg, pos = np.bincount(data["Response"])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
output_bias = np.log([pos/neg])
print(output_bias)
# output_bias = None

model = build_model_rsp_baseline(use_ge=params.use_ge,
                                 use_dd=params.use_dd,
                                 ge_shape=ge_shape,
                                 dd_shape=dd_shape,
                                 model_type=params.model_type,
                                 output_bias=output_bias)
print()
print(model.summary())

# METRICS = [
#       keras.metrics.TruePositives(name='tp'),
#       keras.metrics.FalsePositives(name='fp'),
#       keras.metrics.TrueNegatives(name='tn'),
#       keras.metrics.FalseNegatives(name='fn'),
#       keras.metrics.BinaryAccuracy(name='accuracy'),
#       keras.metrics.Precision(name='precision'),
#       keras.metrics.Recall(name='recall'),
#       keras.metrics.AUC(name='auc'),
# ]

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.AUC(name='auc'),
]

model.compile(loss=loss,
              optimizer=Adam(learning_rate=params.learning_rate),
              metrics=METRICS)

# import ipdb; ipdb.set_trace()

# With output_bias initialized properly, the initial loss is much smaller
# results = model.evaluate(xtr, ytr, batch_size=params.batch_size, verbose=0)
# print("Loss: {:0.4f}".format(results[0]))

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weighted = True
if weighted:
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
else:
    class_weight = None

t = time()
history = model.fit(x=xtr,
                    y=ytr,
                    batch_size=params.batch_size,
                    epochs=params.epochs,
                    verbose=1,
                    validation_data=(xvl, yvl),
                    class_weight=class_weight,
                    shuffle=True,
                    callbacks=callbacks)
print('Runtime: {:.2f} mins'.format( (time() - t)/60) )

# --------------------------
# Evaluate
# --------------------------
import ipdb; ipdb.set_trace()

# Predictions
p = 0.5
yte_prd = model.predict(xte)
if yte_prd.shape[1] > 1:
    yte_prd_label = np.argmax(yte_prd, axis=1)
    dump_preds(yte_label, yte_prd[:, 1], te_meta, outpath=outdir/"test_preds.csv")
    scores = calc_scores(yte_label, yte_prd[:, 1], mltype="cls")
    dump_dict(scores, outdir/"test_scores.txt")

else:
    yte_prd = yte_prd.reshape(-1,)
    # yte_prd_label = yte_prd > p
    yte_prd_label = [1 if ii > p else 0 for ii in yte_prd]
    dump_preds(yte_label, yte_prd, te_meta, outpath=outdir/"test_preds.csv")
    scores = calc_scores(yte_label, yte_prd, mltype="cls")
    dump_dict(scores, outdir/"test_scores.txt")

# Confusion
cnf_mtrx = confusion_matrix(yte_label, yte_prd_label)
save_confusion_matrix(cnf_mtrx, labels=["Non-response", "Response"],
                      outpath=outdir/"confusion.png")
pprint(cnf_mtrx)

print('\nDone.')
