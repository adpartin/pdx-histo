"""
Train baseline model for drug response using rna and dd data (w/o tfrecords).
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
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
assert tf.__version__ >= "2.0"

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.models import Sequential, Model, load_model
# from tensorflow.keras.utils import plot_model

from models import build_model_rsp_baseline, keras_callbacks
from ml.scale import get_scaler
from ml.evals import calc_scores, calc_preds, dump_preds, save_confusion_matrix
from utils.utils import Params, dump_dict, read_lines, cast_list
from datasets.tidy import split_data_and_extract_fea, extract_fea

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
                    default=['Response'],
                    choices=['Response', 'ctype', 'csite'],
                    help='Name of target output.')
parser.add_argument('--id_name',
                    type=str,
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
split_on = "none" if args.split_on is (None or "none") else args.split_on
outdir = prjdir/f"baseline/split_on_{split_on}"
os.makedirs(outdir, exist_ok=True)

# Import parameters
prm_file_path = prjdir/"baseline/params.json"
if prm_file_path.exists() is False:
    shutil.copy(fdir/"../default_params/default_params_baseline.json", prm_file_path)
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

# -----------------------------------------------
# Data splits
# -----------------------------------------------

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

# # Obtain the relevant ids
# if index_col_name in data.columns:
#     tr = data[data[index_col_name].isin(tr_id)].sort_values([split_on], ascending=True).reset_index(drop=True)
#     vl = data[data[index_col_name].isin(vl_id)].sort_values([split_on], ascending=True).reset_index(drop=True)
#     te = data[data[index_col_name].isin(te_id)].sort_values([split_on], ascending=True).reset_index(drop=True)
# else:
#     tr = data.iloc[tr_id, :].sort_values([split_on], ascending=True).reset_index(drop=True)
#     vl = data.iloc[vl_id, :].sort_values([split_on], ascending=True).reset_index(drop=True)
#     te = data.iloc[te_id, :].sort_values([split_on], ascending=True).reset_index(drop=True)
# kwargs = {"ge_cols": ge_cols,
#           "dd1_cols": dd1_cols,
#           "dd2_cols": dd2_cols,
#           "ge_scaler": ge_scaler,
#           "dd1_scaler": dd1_scaler,
#           "dd2_scaler": dd2_scaler,
#           "ge_dtype": cfg.GE_DTYPE,
#           "dd_dtype": cfg.DD_DTYPE,
# }
# import ipdb; ipdb.set_trace()
# tr_ge, tr_dd1, tr_dd2 = extract_fea(tr, **kwargs)
# vl_ge, vl_dd1, vl_dd2 = extract_fea(vl, **kwargs)
# te_ge, te_dd1, te_dd2 = extract_fea(te, **kwargs)

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


# Onehot encoding
# ydata = data[args.target[0]].values
# y_onehot = pd.get_dummies(ydata)
# ydata_label = np.argmax(y_onehot.values, axis=1)
# num_classes = len(np.unique(ydata_label))

callbacks = keras_callbacks(outdir, monitor="val_loss")

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

# TODO: doesn't work with class_weight!
# ytr = {"Response": ytr}
# yvl = {"Response": yvl}
# yte = {"Response": yte}

# --------------------------
# Define and Train
# --------------------------

# Note! When I put METRICS in model.py, it immediately occupies a lot of the GPU memory!
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

neg, pos = np.bincount(data["Response"])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
output_bias = np.log([pos/neg])
print(output_bias)
# output_bias = None

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weighted = True
import ipdb; ipdb.set_trace()
if weighted:
    # y = data["Response"].values
    # weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    # class_weight = {0: weights[0], 1: weights[1]}
    weight_for_0 = (1 / neg) * (total) / 2.0
    weight_for_1 = (1 / pos) * (total) / 2.0
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
else:
    class_weight = None

# import ipdb; ipdb.set_trace()
model = build_model_rsp_baseline(use_ge=params.use_ge,
                                 use_dd1=params.use_dd1,
                                 use_dd2=params.use_dd2,
                                 ge_shape=ge_shape,
                                 dd_shape=dd_shape,
                                 model_type=params.model_type,
                                 output_bias=output_bias)
print()
print(model.summary())

# Save weights
# initial_weights = os.path.join(tempfile.mkdtemp(), "initial_weights")
initial_weights = str(outdir/"initial_weights")
print(initial_weights)
model.save_weights(initial_weights)
model.load_weights(initial_weights)

model.compile(loss=loss,
              optimizer=Adam(learning_rate=params.learning_rate),
              metrics=METRICS)

# import ipdb; ipdb.set_trace()

# With output_bias initialized properly, the initial loss is much smaller
# results = model.evaluate(xtr, ytr, batch_size=params.batch_size, verbose=0)
# print("Loss: {:0.4f}".format(results[0]))

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
# import ipdb; ipdb.set_trace()

# Predictions
yte_prd = model.predict(xte)
yte_prd = np.around(yte_prd, 3)

p = 0.5
if yte_prd.shape[1] > 1:
    # cross-entropy
    yte_prd_label = np.argmax(yte_prd, axis=1)
    dump_preds(yte_label, yte_prd[:, 1], te_meta, outpath=outdir/"test_preds.csv")
    scores = calc_scores(yte_label, yte_prd[:, 1], mltype="cls")
    dump_dict(scores, outdir/"test_scores.txt")
else:
    # binary cross-entropy
    yte_prd = yte_prd.reshape(-1,)
    yte_prd_label = [1 if ii > p else 0 for ii in yte_prd]

    # dump_preds(yte_label, yte_prd, te_meta, outpath=outdir/"test_preds.csv")
    y_true = pd.Series(yte_label, name='y_true')
    y_pred = pd.Series(yte_prd, name='y_pred')
    y_pred_label = pd.Series(yte_prd_label, name='y_pred_label')
    test_pred = pd.concat([te_meta, y_true, y_pred, y_pred_label], axis=1)
    test_pred = test_pred.sort_values(split_on, ascending=True)
    test_pred.to_csv(outdir/"test_preds.csv", index=False)

    scores = calc_scores(yte_label, yte_prd, mltype="cls")
    dump_dict(scores, outdir/"test_scores.txt")

# Confusion
cnf_mtrx = confusion_matrix(yte_label, yte_prd_label)
pprint(cnf_mtrx)

save_confusion_matrix(true_labels=yte_label, predictions=yte_prd, p=p,
                      labels=["Non-response", "Response"],
                      outpath=outdir/"confusion.png")
pprint(scores)
print('\nDone.')
