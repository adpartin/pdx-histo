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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
assert tf.__version__ >= "2.0"
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam

from models import build_model_rsp_baseline
from ml.scale import get_scaler
from ml.evals import calc_scores, calc_preds, dump_preds
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
                    choices=['Sample', 'slide'],
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

if args.target[0] == 'Response':
    pprint(data.reset_index().groupby(['ctype', 'Response']).agg({
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

# T/V/E filenames
splitdir = cfg.DATA_PROCESSED_DIR/args.dataname/f'annotations.splits/split_on_{split_on}'
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

kwargs = {"ge_cols": ge_cols,
          "dd_cols": dd_cols,
          "ge_scaler": ge_scaler,
          "dd_scaler": dd_scaler,
          "ge_dtype": cfg.GE_DTYPE,
          "dd_dtype": cfg.DD_DTYPE
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
tr_meta = data.iloc[tr_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
vl_meta = data.iloc[vl_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)
te_meta = data.iloc[te_id, :].drop(columns=ge_cols + dd_cols).reset_index(drop=True)

# Onehot encoding
ydata = data[args.target[0]].values
# y_onehot = pd.get_dummies(ydata_label)
y_onehot = pd.get_dummies(ydata)
ydata_label = np.argmax(y_onehot.values, axis=1)
num_classes = len(np.unique(ydata_label))

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

# --------------------------
# Define and Train
# --------------------------
model = build_model_rsp_baseline(use_ge=params.use_ge,
                                 use_dd=params.use_dd,
                                 ge_shape=ge_shape,
                                 dd_shape=dd_shape,
                                 model_type=params.model_type,
                                 NUM_CLASSES=num_classes)
print()
print(model.summary())

model.compile(loss=loss,
              optimizer=Adam(learning_rate=params.learning_rate),
              metrics=["accuracy"])

t = time()
history = model.fit(x=xtr,
                    y=ytr,
                    batch_size=params.batch_size,
                    epochs=params.epochs,
                    verbose=1,
                    validation_data=(xvl, yvl),
                    callbacks=callbacks)
print('Runtime: {:.2f} mins'.format( (time() - t)/60) )

# --------------------------
# Evaluate
# --------------------------
# import ipdb; ipdb.set_trace()

# Predictions
yte_prd = model.predict(xte)
yte_prd_label = np.argmax(yte_prd, axis=1)
dump_preds(yte_label, yte_prd[:, 1], te_meta, outpath=outdir/"test_preds.csv")

scores = calc_scores(yte_label, yte_prd[:, 1], mltype="cls")
dump_dict(scores, outdir/"test_scores.txt")

# Confusion
cnf_mtrx = confusion_matrix(yte_label, yte_prd_label)
pprint(cnf_mtrx)

fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cnf_mtrx, annot=True, cmap='Blues', linewidths=0.1, linecolor='white')
ax.set_xticklabels(["Non-response", "Response"])
ax.set_yticklabels(["Non-response", "Response"])
ax.set(ylabel="True", xlabel="Predicted")
plt.savefig(outdir/"confusion.png", bbox_inches='tight', dpi=150)

print('\nDone.')
