"""
Train drug response using rna and dd data (w/o tfrecords).
"""
import os
import sys
assert sys.version_info >= (3, 5)

import argparse
import glob
from time import time
from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
assert tf.__version__ >= "2.0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential, Model


# from tf_utils import get_tfr_files
# from sf_utils import (green, interleave_tfrecords,
#                       parse_tfrec_fn_rsp, parse_tfrec_fn_rna,
#                       create_manifest)
from models import build_model_rsp, build_model_rna, build_model_rsp_simple
from ml.scale import get_scaler
from ml.evals import calc_scores, calc_preds
from utils.utils import Params

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
                    choices=['Sample', 'slide'],
                    help='Specify the hard split variable/column (default: None).')
parser.add_argument('--prjname',
                    type=str,
                    default='bin_rsp_balance_02',
                    help='Project name (folder that contains the annotations.csv dataframe).')

args, other_args = parser.parse_known_args()
pprint(args)


# Load dataframe (annotations)
prjdir = cfg.MAIN_PRJDIR/args.prjname
annotations_file = prjdir/cfg.SF_ANNOTATIONS_FILENAME
data = pd.read_csv(annotations_file)
print(data.shape)

# import ipdb; ipdb.set_trace()

# Args
epochs = 20
batch_size = 32
# y_encoding = 'onehot'
y_encoding = 'label'

# Import parameters
prm_dir = prjdir/'params.json'
params = Params(prm_dir)

# Outdir
split_on = 'none' if args.split_on is (None or 'none') else args.split_on
# outdir = prjdir/f'results/split_on_{split_on}'
# os.makedirs(outdir, exist_ok=True)

if args.target[0] == 'Response':
    pprint(data.groupby(['ctype', 'Response']).agg({
        'index': 'nunique'}).reset_index().rename(columns={'index': 'count'}))
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

# import ipdb; ipdb.set_trace()

# T/V/E filenames
splitdir = prjdir/f'annotations.splits/split_on_{split_on}'
split_id = 0

split_pattern = f'1fold_s{split_id}_*_id.txt'
single_split_files = glob.glob(str(splitdir/split_pattern))
# single_split_files = list(splitdir.glob(split_pattern))

def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines

def cast_list(ll, dtype=int):
    return [dtype(i) for i in ll]

# Get indices for the split
assert len(single_split_files) >= 2, f'Split {split_id} contains only one file.'
for id_file in single_split_files:
    if 'tr_id' in id_file:
        tr_id = cast_list(read_lines(id_file), int)
    elif 'vl_id' in id_file:
        vl_id = cast_list(read_lines(id_file), int)
    elif 'te_id' in id_file:
        te_id = cast_list(read_lines(id_file), int)

# Dataframes of T/V/E samples
tr_df = data.iloc[tr_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
vl_df = data.iloc[vl_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
te_df = data.iloc[te_id, :].sort_values(args.id_name, ascending=True).reset_index(drop=True)
print('Total samples {}'.format(tr_df.shape[0] + vl_df.shape[0] + te_df.shape[0]))

tr_ge_df, tr_dd_df = tr_df[ge_cols], tr_df[dd_cols]
vl_ge_df, vl_dd_df = vl_df[ge_cols], vl_df[dd_cols]
te_ge_df, te_dd_df = te_df[ge_cols], te_df[dd_cols]

tr_dd_df = pd.DataFrame(dd_scaler.transform(tr_dd_df), columns=dd_cols, dtype=cfg.DD_DTYPE)
vl_dd_df = pd.DataFrame(dd_scaler.transform(vl_dd_df), columns=dd_cols, dtype=cfg.DD_DTYPE)
te_dd_df = pd.DataFrame(dd_scaler.transform(te_dd_df), columns=dd_cols, dtype=cfg.DD_DTYPE)

tr_ge_df = pd.DataFrame(ge_scaler.transform(tr_ge_df), columns=ge_cols, dtype=cfg.GE_DTYPE)
vl_ge_df = pd.DataFrame(ge_scaler.transform(vl_ge_df), columns=ge_cols, dtype=cfg.GE_DTYPE)
te_ge_df = pd.DataFrame(ge_scaler.transform(te_ge_df), columns=ge_cols, dtype=cfg.GE_DTYPE)

ge_shape = (tr_ge_df.shape[1], )
dd_shape = (tr_dd_df.shape[1], )

xtr = {"ge_data": tr_ge_df.values, "dd_data": tr_dd_df.values}
xvl = {"ge_data": vl_ge_df.values, "dd_data": vl_dd_df.values}
xte = {"ge_data": te_ge_df.values, "dd_data": te_dd_df.values}

# ydata = data[args.target].values
# ytr = ydata[tr_id]
# yvl = ydata[vl_id]
# yte = ydata[te_id]

# Onehot encoding
ydata = data[args.target[0]].values
# y_onehot = pd.get_dummies(ydata_label)
y_onehot = pd.get_dummies(ydata)
ydata_label = np.argmax(y_onehot.values, axis=1)
num_classes = len(np.unique(ydata_label))

# # Scale RNA
# xdata = data[ge_cols]
# x_scaler = StandardScaler()
# x1 = pd.DataFrame(x_scaler.fit_transform(xdata), columns=ge_cols, dtype=np.float32)
# xdata = x1; del x1

# # Split
# tr_ids, te_ids = train_test_split(range(xdata.shape[0]), test_size=0.2, random_state=seed, stratify=ydata)
# xtr = xdata.iloc[tr_ids, :].reset_index(drop=True)
# xte = xdata.iloc[te_ids, :].reset_index(drop=True)


# Create callbacks for early stopping, checkpoint saving, summaries, and history
history_callback = tf.keras.callbacks.History()
checkpoint_path = os.path.join(prjdir, "cp.ckpt")
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=1)
callbacks = [history_callback, cp_callback]

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

# import ipdb; ipdb.set_trace()

if y_encoding == 'onehot':
    ytr = y_onehot.iloc[tr_id, :].reset_index(drop=True)
    yvl = y_onehot.iloc[vl_id, :].reset_index(drop=True)
    yte = y_onehot.iloc[te_id, :].reset_index(drop=True)
    loss = losses.CategoricalCrossentropy()
elif y_encoding == 'label':
    ytr = ydata_label[tr_id]
    yvl = ydata_label[vl_id]
    yte = ydata_label[te_id]
    loss = losses.SparseCategoricalCrossentropy()
else:
    raise ValueError(f'Unknown value for y_encoding ({y_encoding}).')
    
ytr_label = ydata_label[tr_id]
yvl_label = ydata_label[vl_id]
yte_label = ydata_label[te_id]

ytr = {"Response": ytr}
yvl = {"Response": yvl}
yte = {"Response": yte}

model = build_model_rsp_simple(use_ge=params.use_ge, use_dd=params.use_dd,
                               ge_shape=ge_shape, dd_shape=dd_shape,
                               model_type=params.model_type, NUM_CLASSES=num_classes)
print()
print(model.summary())

model.compile(loss=loss,
              optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
              metrics=["accuracy"])

t = time()
history = model.fit(x=xtr,
                    y=ytr,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(xvl, yvl),
                    callbacks=callbacks)
print('Runtime: {:.2f} mins'.format( (time() - t)/60) )

import ipdb; ipdb.set_trace()

# Predictions
yte_prd = model.predict(xte)
yte_prd_label = np.argmax(yte_prd, axis=1)
# yte_true_label = np.argmax(yte.values, axis=1)

cnf_mtrx = confusion_matrix(yte_label, yte_prd_label)
# disp = ConfusionMatrixDisplay(cnf_mtrx, display_labels=list(y_onehot.columns))
# disp.plot(xticks_rotation=65);
pprint(cnf_mtrx)

print('\nDone.')
